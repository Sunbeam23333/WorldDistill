"""Training-loop integration for explicit tensor/pipeline ownership."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from training.parallel.topology import create_parallel_topology
from training.parallel.state import local_partition_metadata
from training.parallel.plans import apply_model_parallel_plan


@torch.no_grad()
def _broadcast_initial_model(model, device):
    # A model may still be CPU staged while the global backend is NCCL. Move
    # only one tensor at a time; never materialize the unpartitioned CUDA model.
    for value in list(model.parameters()) + list(model.buffers()):
        staged = value.detach().to(device).contiguous()
        dist.broadcast(staged, src=0)
        value.copy_(staged.to(value.device))


def initialize_model_parallel(trainer):
    args = trainer.args
    if getattr(trainer, "_model_parallel_initialized", False) or args.tp_size * args.pp_size == 1:
        return
    if trainer.parallel_mode != "ddp":
        raise ValueError("Explicit TP/PP uses data-parallel gradient reduction; select parallel_mode=ddp")
    if not trainer.is_distributed:
        raise ValueError("TP/PP training requires torchrun workers")
    if trainer.optimizer.state:
        raise ValueError("Apply model parallelism before restoring optimizer state")
    from training.trainers.base_distill_trainer import EMAModel
    from training.utils.optimizers import build_optimizer
    from training.utils.schedulers import build_lr_scheduler
    from training.utils.sharded_state import build_training_grad_scaler

    roles = [("student", trainer.student_model), ("teacher", trainer.teacher_model)]
    if getattr(trainer, "ema_model", None) is not None:
        roles.append(("target", trainer.ema_model))
    runtime = trainer.runtime
    old_offloader = getattr(runtime, "_teacher_offloader", None)
    if old_offloader is not None:
        old_offloader.close()
        runtime._teacher_offloader = None
    trainer._model_parallel_roles = {}
    for role, model in roles:
        # Independent role groups make teacher/student CUDA-stream overlap
        # independent of each other on every model-parallel collective.
        topology = create_parallel_topology(args.tp_size, args.pp_size)
        _broadcast_initial_model(model, trainer.device)
        model._worlddistill_trainable_schema = {
            name: tuple(parameter.shape) for name, parameter in model.named_parameters()
            if parameter.requires_grad}
        if role == "teacher" and args.distill_method == "progressive_distill" and args.use_lora:
            from training.parallel.state import capture_unpartitioned_meta_model
            trainer._unpartitioned_teacher_meta = capture_unpartitioned_meta_model(model)
        apply_model_parallel_plan(model, tp_group=topology.tp_group, pp_group=topology.pp_group)
        if role != "teacher" or args.runtime_teacher_offload == "none":
            model.to(trainer.device)
        trainer._model_parallel_roles[role] = topology
    trainer.model_parallel = trainer._model_parallel_roles["student"]
    trainer._manual_replicated_sync = True
    trainer.optimizer = build_optimizer(
        trainer.student_model, optimizer_type=args.optimizer, lr=args.learning_rate,
        weight_decay=args.weight_decay, adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2, adam_epsilon=args.adam_epsilon)
    trainer.lr_scheduler = build_lr_scheduler(
        trainer.optimizer, scheduler_type=args.lr_scheduler, warmup_steps=args.warmup_steps,
        total_steps=args.max_train_steps, min_lr_ratio=args.lr_min_ratio)
    if trainer.ema is not None:
        trainer.ema = EMAModel(trainer.student_model, decay=args.ema_decay, warmup_steps=args.ema_warmup_steps)
    if getattr(trainer, "consistency_ema", None) is not None:
        trainer.consistency_ema = EMAModel(trainer.student_model, decay=args.ema_decay,
                                         warmup_steps=args.ema_warmup_steps)
        trainer.consistency_ema.apply_to(trainer.ema_model)
    if args.mixed_precision == "fp16":
        # Every parameter partition must agree before taking an optimizer step.
        trainer.scaler = build_training_grad_scaler("fsdp", "fp16", trainer.device)
    if runtime is not None:
        runtime.teacher_model = trainer.teacher_model
        runtime.student_model = trainer.student_model
        runtime.context_parallel_group = trainer.model_parallel.mp_group
        if args.runtime_teacher_offload != "none":
            from training.runtime.teacher_offload import TeacherParameterOffloader
            runtime._teacher_offloader = TeacherParameterOffloader(
                trainer.teacher_model, trainer.device, target=args.runtime_teacher_offload,
                directory=args.runtime_teacher_offload_dir or os.path.join(args.output_dir, "runtime_teacher_weights"),
                pin_memory=bool(getattr(args, "runtime_teacher_offload_pin_memory", True)))
    trainer._model_parallel_initialized = True


def synchronize_model_parallel_batch(trainer, batch):
    topology = getattr(trainer, "model_parallel", None)
    if topology is None:
        return batch
    from training.utils.context_parallel import synchronize_context_batch, synchronize_context_rng
    batch = synchronize_context_batch(batch, topology.mp_group, trainer.device)
    synchronize_context_rng(topology.mp_group, trainer.device)
    return batch


@torch.no_grad()
def model_parallel_grad_norm(trainer, max_norm=float("inf")):
    """L2 norm over unique global parameters, excluding replicated copies."""
    topology = trainer.model_parallel
    model = trainer._unwrap_model(trainer.student_model)
    parameters = dict(model.named_parameters())
    squared = torch.zeros((), device=trainer.device, dtype=torch.float64)
    for spec in local_partition_metadata(model).values():
        parameter = parameters.get(spec["local_name"])
        if parameter is None or parameter.grad is None:
            continue
        replicas = (topology.tp_size if spec["axis"] is None else 1)
        replicas *= topology.pp_size if spec["pp_owner"] is None else 1
        squared.add_(parameter.grad.detach().double().square().sum() / replicas)
    dist.all_reduce(squared, group=topology.mp_group)
    norm = squared.sqrt().float()
    if max_norm != float("inf"):
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.)
        for parameter in parameters.values():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient.to(parameter.grad.device))
    return norm
