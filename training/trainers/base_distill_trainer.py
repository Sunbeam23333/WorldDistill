"""Base Distillation Trainer.

Provides the common training loop infrastructure shared by all distillation methods.
Handles:
- Teacher/student model management
- Flow Matching forward process (noising)
- Gradient accumulation + distributed sync
- EMA (Exponential Moving Average) of student weights
- Checkpoint save/load (DDP/FSDP/DeepSpeed compatible)
- FSDP / DeepSpeed ZeRO / DDP parallel strategies
- Sequence Parallelism for long-sequence video DiT models
- Logging (console + optional WandB)

References:
- HY-WorldPlay TrainingPipeline: OOP training loop with train_one_step()
- Open-Sora train.py: Flow matching noising, bucket sampling, ColossalAI
- Progressive Distillation (Salimans & Ho, 2022): EMA teacher
- Consistency Models (Song et al., 2023): EMA target network
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from training.runtime.distill_cache import DistillCache
from training.runtime.fused_supervision import fused_masked_mse_loss, fused_supervision_available
from training.runtime.teacher_student_runtime import PendingTeacherForward, TeacherStudentRuntime
from training.trainer_args import TrainerArgs
from training.utils.distributed import (
    wrap_model,
    wrap_model_fsdp,
    init_deepspeed,
    get_deepspeed_config,
    init_sequence_parallel,
    scatter_sequence,
    gather_sequence,
)
from training.utils.experiment_tracking import ExperimentTracker


@contextlib.contextmanager
def _nullcontext():
    """Backport of contextlib.nullcontext for compatibility."""
    yield


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters updated as:
        ema_param = decay * ema_param + (1 - decay) * param

    Supports warmup: decay ramps from 0 to target_decay over warmup_steps.

    For FSDP models: EMA must operate on the unwrapped (local shard) parameters.
    We handle this by accepting the raw model (not FSDP-wrapped).

    References:
    - Consistency Models: EMA for stable target network
    - Improved DDPM: EMA for sampling quality
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        # Store shadow parameters
        self.shadow = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def get_decay(self) -> float:
        """Get current EMA decay, with optional warmup ramp."""
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters with current model parameters."""
        decay = self.get_decay()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
        self.step_count += 1

    def apply_to(self, model: nn.Module):
        """Copy shadow parameters into model (for inference / target network)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "step_count": self.step_count, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]):
        self.shadow = state["shadow"]
        self.step_count = state.get("step_count", 0)
        self.decay = state.get("decay", self.decay)


class BaseDistillTrainer(ABC):
    """Abstract base trainer for all distillation methods.

    Subclasses must implement:
    - compute_distill_loss(): Define the distillation loss
    - prepare_teacher_input(): Prepare inputs for teacher forward pass

    Optionally override:
    - prepare_student_input(): If student input differs from teacher
    - on_train_step_end(): Custom per-step logic

    Supported parallelism:
    - DDP: Standard DistributedDataParallel (default for <~14B param models)
    - FSDP: FullyShardedDataParallel (for large models, shards params/grads/optim)
    - DeepSpeed: ZeRO Stage 1/2/3 (alternative to FSDP, with CPU offload)
    - Sequence Parallelism: Splits long sequences across GPUs within a group
    """

    def __init__(
        self,
        args: TrainerArgs,
        teacher_model: nn.Module,
        student_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: torch.device = None,
        batch_encoder: Optional[Any] = None,
        runtime: Optional[TeacherStudentRuntime] = None,
        distill_cache: Optional[DistillCache] = None,
    ):
        self.args = args
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_encoder = batch_encoder
        self.runtime = runtime
        self.distill_cache = distill_cache
        self.enable_fused_supervision = bool(getattr(args, "enable_fused_supervision_kernel", False))
        self.fused_supervision_backend = getattr(args, "fused_supervision_backend", "auto")
        self._fused_supervision_warned = False

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Parallel strategy
        self.parallel_mode = getattr(args, "parallel_mode", "ddp")  # ddp | fsdp | deepspeed
        self._deepspeed_engine = None  # Set during train() if using DeepSpeed

        # Mixed precision
        self.use_amp = args.mixed_precision != "no"
        self.amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        # Note: When using DeepSpeed or FSDP, they handle their own mixed precision.
        # GradScaler is only needed for DDP + fp16.
        self._use_scaler = (args.mixed_precision == "fp16" and self.parallel_mode == "ddp")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_scaler)

        # Flow Matching parameters
        self.num_train_timesteps = 1000
        self.sigma_min = 0.0
        self.prediction_type = getattr(args, "prediction_type", "velocity")  # velocity | epsilon | x0

        # Distributed
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = self.rank == 0

        # Sequence Parallelism
        self.sp_size = getattr(args, "sp_size", 1)
        self.sp_group = None
        if self.sp_size > 1 and self.is_distributed:
            self.sp_group = init_sequence_parallel(self.sp_size)
            if self.sp_group is not None:
                logger.info(f"Sequence parallelism enabled: sp_size={self.sp_size}")

        # Freeze teacher
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # EMA (optional, enabled by subclasses or args)
        self.ema: Optional[EMAModel] = None
        if getattr(args, "use_ema", False):
            self.ema = EMAModel(
                self._unwrap_model(self.student_model)
                if hasattr(self.student_model, "module")
                else self.student_model,
                decay=getattr(args, "ema_decay", 0.9999),
                warmup_steps=getattr(args, "ema_warmup_steps", 0),
            )

        self._tracker: Optional[ExperimentTracker] = None
        self._optimizer_skipped_steps = 0
        self._train_start_time = time.perf_counter()
        if self.is_main_process:
            self._tracker = ExperimentTracker(args)

    # ==================== Abstract Methods ====================

    @abstractmethod
    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the distillation loss."""
        ...

    @abstractmethod
    def prepare_teacher_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        """Prepare input kwargs for teacher model forward pass."""
        ...

    # ==================== Helpers ====================

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP/FSDP/DeepSpeed to get the raw module."""
        # DeepSpeed engine
        if hasattr(model, "module"):
            return model.module
        return model

    def _v_to_x0(self, v: torch.Tensor, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Convert velocity prediction to x0 prediction.

        In flow matching: x_t = (1-sigma)*x_0 + sigma*eps
        velocity v = eps - x_0 = dx/dsigma
        Therefore: x_0 = x_t - sigma * v
        """
        return x_t.float() - sigma * v.float()

    def _v_to_eps(self, v: torch.Tensor, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Convert velocity prediction to epsilon prediction.

        eps = x_t + (1-sigma) * v  (derived from v = eps - x_0)
        """
        return x_t.float() + (1 - sigma) * v.float()

    def _get_no_sync_context(self):
        """Get the appropriate no_sync context for the current parallel mode.

        - DDP: model.no_sync()
        - FSDP: model.no_sync()
        - DeepSpeed: handled internally, but we still use model.no_sync() if available
        """
        model = self.student_model
        if hasattr(model, "no_sync"):
            return model.no_sync()
        return _nullcontext()

    # ==================== Core Training Loop ====================

    def train(self):
        """Main training loop."""
        logger.info(
            f"Starting {self.args.distill_method} training | "
            f"Steps: {self.args.max_train_steps} | "
            f"LR: {self.args.learning_rate} | "
            f"Batch: {self.args.batch_size} | "
            f"Grad Accum: {self.args.gradient_accumulation_steps} | "
            f"Parallel: {self.parallel_mode}"
        )

        if self._tracker is not None:
            self._tracker.log_config(vars(self.args))

        if self.args.resume_from:
            self.load_checkpoint(self.args.resume_from)

        # ---- Wrap model with chosen parallel strategy ----
        if self.parallel_mode == "deepspeed":
            self._init_deepspeed()
        elif self.parallel_mode == "fsdp":
            self._init_fsdp()
        else:
            # DDP (default)
            if self.is_distributed and not isinstance(
                self.student_model, torch.nn.parallel.DistributedDataParallel
            ):
                self.student_model = torch.nn.parallel.DistributedDataParallel(
                    self.student_model,
                    device_ids=[self.rank],
                    find_unused_parameters=False,
                )

        self.student_model.train()
        self._data_iter = iter(self.train_dataloader)

        try:
            while self.global_step < self.args.max_train_steps:
                # Get next batch
                batch = self._next_batch()

                # Move batch to device and encode raw inputs if needed
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)

                # Sequence parallel: scatter video latents along temporal dim
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

                step_start_time = time.perf_counter()
                step_metrics = self.train_step(batch)
                self.global_step += 1
                step_metrics = self._augment_step_metrics(step_metrics, batch, step_start_time)

                # EMA update
                if self.ema is not None:
                    self.ema.update(self._unwrap_model(self.student_model))

                # Logging
                if self.is_main_process and self.global_step % self.args.log_every == 0:
                    self._log_metrics(step_metrics)

                # Evaluation
                if (
                    self.val_dataloader is not None
                    and self.args.eval_every > 0
                    and self.global_step % self.args.eval_every == 0
                ):
                    eval_metrics = self.evaluate()
                    if self.is_main_process and eval_metrics:
                        self._log_metrics(eval_metrics)

                # Save checkpoint
                if self.is_main_process and self.global_step % self.args.save_every == 0:
                    self.save_checkpoint(self.global_step, self.args.output_dir)

                # Custom per-step hook
                self.on_train_step_end(step_metrics)

            logger.info(f"Training completed at step {self.global_step}.")
            if self.is_main_process:
                self.save_checkpoint(self.global_step, self.args.output_dir)
        finally:
            if self._tracker is not None:
                self._tracker.close()

    def _init_deepspeed(self):
        """Initialize DeepSpeed engine for training."""
        ds_stage = getattr(self.args, "deepspeed_stage", 2)
        ds_config = get_deepspeed_config(
            stage=ds_stage,
            train_batch_size=self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            learning_rate=self.args.learning_rate,
            max_grad_norm=self.args.max_grad_norm,
            cpu_offload=self.args.cpu_offload,
        )

        self.student_model, self.optimizer, self.lr_scheduler = init_deepspeed(
            model=self.student_model,
            optimizer=self.optimizer,
            ds_config=ds_config,
            lr_scheduler=self.lr_scheduler,
        )
        self._deepspeed_engine = self.student_model
        logger.info(f"DeepSpeed ZeRO-{ds_stage} initialized.")

    def _init_fsdp(self):
        """Initialize FSDP wrapping for training."""
        if self.is_distributed and not self._is_fsdp_wrapped(self.student_model):
            self.student_model = wrap_model_fsdp(
                self.student_model,
                shard_strategy=getattr(self.args, "fsdp_shard_strategy", "full"),
                cpu_offload=self.args.cpu_offload,
                mixed_precision=self.args.mixed_precision,
            )
            logger.info("Student model wrapped with FSDP.")

            # Re-create optimizer after FSDP wrapping (FSDP reshards parameters)
            from training.utils.optimizers import build_optimizer
            from training.utils.schedulers import build_lr_scheduler
            self.optimizer = build_optimizer(
                self.student_model,
                optimizer_type=self.args.optimizer,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
            )
            self.lr_scheduler = build_lr_scheduler(
                self.optimizer,
                scheduler_type=self.args.lr_scheduler,
                warmup_steps=self.args.warmup_steps,
                total_steps=self.args.max_train_steps,
                min_lr_ratio=self.args.lr_min_ratio,
            )
            logger.info("Optimizer/scheduler re-created for FSDP-wrapped model.")

    @staticmethod
    def _is_fsdp_wrapped(model: nn.Module) -> bool:
        """Check if model is already FSDP-wrapped."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return isinstance(model, FSDP)
        except ImportError:
            return False

    def _next_batch(self) -> Dict[str, Any]:
        """Get the next batch from dataloader, handling epoch transitions."""
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(self.epoch)
            if hasattr(self.train_dataloader, "batch_sampler") and hasattr(self.train_dataloader.batch_sampler, "set_epoch"):
                self.train_dataloader.batch_sampler.set_epoch(self.epoch)
            self._data_iter = iter(self.train_dataloader)
            batch = next(self._data_iter)
        return batch

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step with gradient accumulation.

        Handles DDP, FSDP, and DeepSpeed uniformly.
        """
        if self._deepspeed_engine is not None:
            return self._train_step_deepspeed(batch)
        else:
            return self._train_step_native(batch)

    def _train_step_native(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Native PyTorch training step (DDP / FSDP)."""
        self.optimizer.zero_grad()
        total_loss = 0.0

        for accum_idx in range(self.args.gradient_accumulation_steps):
            if accum_idx > 0:
                batch = self._next_batch()
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

            # Sync gradients only on last accumulation step
            is_last_accum = accum_idx == self.args.gradient_accumulation_steps - 1
            ctx = (
                self._get_no_sync_context()
                if self.is_distributed and not is_last_accum
                else _nullcontext()
            )

            with ctx:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self._forward_and_loss(batch)
                    loss = loss / self.args.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            total_loss += loss.item()

        # Gradient clipping
        if self.args.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            if self._is_fsdp_wrapped(self.student_model):
                # FSDP requires using its own clip_grad_norm_
                grad_norm = self.student_model.clip_grad_norm_(self.args.max_grad_norm)
            else:
                params = self._unwrap_model(self.student_model).parameters()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.args.max_grad_norm)
        else:
            grad_norm = self._compute_grad_norm()

        # All-reduce loss for accurate logging
        if self.is_distributed:
            loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item() / self.world_size

        # Skip step if gradient is too large (following HY-WorldPlay)
        grad_skip_threshold = self.args.grad_skip_threshold
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        optimizer_skipped = 0.0
        if grad_norm_val < grad_skip_threshold:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
        else:
            optimizer_skipped = 1.0
            logger.warning(
                f"Step {self.global_step}: grad_norm={grad_norm_val:.2f} > {grad_skip_threshold}, skipping."
            )
            self.optimizer.zero_grad()

        return {
            "loss": total_loss,
            "grad_norm": grad_norm_val,
            "lr": self.optimizer.param_groups[0]["lr"],
            "optimizer_skipped": optimizer_skipped,
        }

    def _train_step_deepspeed(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """DeepSpeed training step.

        DeepSpeed handles gradient accumulation, mixed precision, and
        gradient clipping internally through its engine.
        """
        engine = self._deepspeed_engine
        assert engine is not None, "DeepSpeed engine should be initialized before calling _train_step_deepspeed"
        total_loss = 0.0

        for accum_idx in range(self.args.gradient_accumulation_steps):
            if accum_idx > 0:
                batch = self._next_batch()
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

            loss = self._forward_and_loss(batch)
            loss = loss / self.args.gradient_accumulation_steps

            # DeepSpeed handles backward + gradient sync + accumulation
            engine.backward(loss)
            total_loss += loss.item()

        # DeepSpeed handles optimizer step + gradient clipping + LR scheduling
        engine.step()

        # Get grad norm from DeepSpeed
        grad_norm = 0.0
        if hasattr(engine, "get_global_grad_norm"):
            grad_norm = engine.get_global_grad_norm()

        return {
            "loss": total_loss,
            "grad_norm": grad_norm if isinstance(grad_norm, float) else 0.0,
            "lr": engine.get_lr()[0] if hasattr(engine, "get_lr") else self.args.learning_rate,
            "optimizer_skipped": 0.0,
        }

    def _augment_step_metrics(
        self,
        metrics: Dict[str, float],
        batch: Dict[str, Any],
        step_start_time: float,
    ) -> Dict[str, float]:
        enriched = dict(metrics)
        step_time = max(time.perf_counter() - step_start_time, 1e-6)
        effective_batch_size = (
            self._infer_batch_size(batch)
            * self.args.gradient_accumulation_steps
            * self.world_size
        )
        self._optimizer_skipped_steps += int(enriched.get("optimizer_skipped", 0.0))

        enriched["epoch"] = float(self.epoch)
        enriched["progress"] = self.global_step / max(1, self.args.max_train_steps)
        enriched["step_time_sec"] = step_time
        enriched["steps_per_sec"] = 1.0 / step_time
        enriched["elapsed_hours"] = max(time.perf_counter() - self._train_start_time, 0.0) / 3600.0
        enriched["eta_hours"] = max(self.args.max_train_steps - self.global_step, 0) * step_time / 3600.0
        enriched["optimizer_skipped_total"] = float(self._optimizer_skipped_steps)
        if effective_batch_size > 0:
            enriched["samples_per_sec"] = effective_batch_size / step_time

        if self.device.type == "cuda" and torch.cuda.is_available():
            enriched["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            enriched["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            enriched["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        if self.runtime is not None:
            runtime_stats = self.runtime.stats()
            for key, value in runtime_stats.items():
                enriched[f"runtime/{key}"] = float(value)
            hits = int(runtime_stats.get("hits", 0))
            misses = int(runtime_stats.get("misses", 0))
            if hits + misses > 0:
                enriched["runtime/cache_hit_rate"] = hits / (hits + misses)

        return enriched

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run teacher + student forward and compute distillation loss.

        Standard flow matching process:
        1. Sample timesteps t ~ logit-normal
        2. Create noisy latents: x_t = (1-sigma)*x_0 + sigma*noise
        3. Teacher forward (no grad) -> teacher prediction
        4. Student forward -> student prediction
        5. Compute distillation loss
        """
        latents = batch["latents"]  # (B, C, T, H, W) or (B, C, H, W)
        bs = latents.shape[0]

        # Sample timesteps
        timesteps = self._sample_timesteps(bs)
        sigmas = timesteps / self.num_train_timesteps

        # Create noisy latents via flow matching: x_t = (1-sigma)*x_0 + sigma*noise
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise

        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        teacher_output, student_output = self._run_teacher_student_pair(
            teacher_input=teacher_input,
            student_input=student_input,
            batch=batch,
            cache_extra={"timesteps": timesteps},
        )

        # Compute loss
        loss = self.compute_distill_loss(teacher_output, student_output, batch, timesteps)
        return loss

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for flow matching.

        Uses logit-normal distribution (following SD3/Open-Sora) for better
        coverage of the noise schedule. Mean/std are configurable.
        """
        logit_mean = getattr(self.args, "logit_normal_mean", 0.0)
        logit_std = getattr(self.args, "logit_normal_std", 1.0)
        u = torch.randn(batch_size, device=self.device) * logit_std + logit_mean
        t = torch.sigmoid(u)  # Maps to (0, 1)
        # Clamp to avoid numerical issues at boundaries
        t = t.clamp(min=0.001, max=0.999)
        timesteps = t * self.num_train_timesteps
        return timesteps.to(self.device)

    def prepare_student_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        """Prepare input for student model. Default: same as teacher."""
        return self.prepare_teacher_input(batch, noisy_latents, timesteps)

    def _augment_model_input(self, input_kwargs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("encoder_hidden_states", "image_cond", "camera_poses", "actions"):
            if key in batch and batch[key] is not None:
                input_kwargs[key] = batch[key]
        return input_kwargs

    def run_teacher(
        self,
        input_kwargs: Dict[str, Any],
        batch: Dict[str, Any],
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[Dict[str, Any]] = None,
        allow_cache: bool = True,
    ) -> torch.Tensor:
        if self.runtime is not None:
            return self.runtime.run_teacher(
                input_kwargs=input_kwargs,
                batch=batch,
                global_step=self.global_step,
                cache_namespace=cache_namespace,
                cache_extra=cache_extra,
                allow_cache=allow_cache,
            )

        with torch.no_grad():
            output = self.teacher_model(**input_kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def launch_teacher(
        self,
        input_kwargs: Dict[str, Any],
        batch: Dict[str, Any],
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[Dict[str, Any]] = None,
        allow_cache: bool = True,
    ) -> Optional[PendingTeacherForward]:
        if self.runtime is None or not self.runtime.can_pipeline_teacher_student():
            return None
        return self.runtime.launch_teacher(
            input_kwargs=input_kwargs,
            batch=batch,
            global_step=self.global_step,
            cache_namespace=cache_namespace,
            cache_extra=cache_extra,
            allow_cache=allow_cache,
        )

    def wait_teacher(self, pending_teacher: Optional[PendingTeacherForward]) -> Optional[torch.Tensor]:
        if pending_teacher is None:
            return None
        assert self.runtime is not None
        return self.runtime.wait_teacher(pending_teacher)

    def run_student(
        self,
        input_kwargs: Dict[str, Any],
        batch: Dict[str, Any],
        model: Optional[nn.Module] = None,
        tag: str = "student",
    ) -> torch.Tensor:
        student_model = model if model is not None else self.student_model
        if self.runtime is not None:
            return self.runtime.run_student(
                model=student_model,
                input_kwargs=input_kwargs,
                batch=batch,
                global_step=self.global_step,
                tag=tag,
            )
        output = student_model(**input_kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def select_runtime_memory_frames(
        self,
        all_frames: torch.Tensor,
        current_chunk_idx: int,
        chunk_size: int,
        memory_frames: int,
    ) -> Optional[torch.Tensor]:
        if self.runtime is not None:
            return self.runtime.select_memory_frames(
                all_frames=all_frames,
                current_chunk_idx=current_chunk_idx,
                chunk_size=chunk_size,
                memory_frames=memory_frames,
            )

        start_frame = current_chunk_idx * chunk_size
        mem_start = max(0, start_frame - memory_frames)
        mem_end = start_frame
        if mem_end <= mem_start:
            return None
        return all_frames[:, :, mem_start:mem_end]

    def _run_teacher_student_pair(
        self,
        teacher_input: Dict[str, Any],
        student_input: Dict[str, Any],
        batch: Dict[str, Any],
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[Dict[str, Any]] = None,
        allow_cache: bool = True,
        student_model: Optional[nn.Module] = None,
        student_tag: str = "student",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pending_teacher = self.launch_teacher(
            teacher_input,
            batch,
            cache_namespace=cache_namespace,
            cache_extra=cache_extra,
            allow_cache=allow_cache,
        )
        student_output = self.run_student(
            student_input,
            batch,
            model=student_model,
            tag=student_tag,
        )
        if pending_teacher is not None:
            teacher_output = self.wait_teacher(pending_teacher)
        else:
            teacher_output = self.run_teacher(
                teacher_input,
                batch,
                cache_namespace=cache_namespace,
                cache_extra=cache_extra,
                allow_cache=allow_cache,
            )
        assert teacher_output is not None
        return teacher_output, student_output

    def compute_supervision_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_fused = self.enable_fused_supervision and self._should_use_fused_supervision(prediction, target)
        return fused_masked_mse_loss(
            prediction=prediction,
            target=target,
            mask=mask,
            enabled=use_fused,
        )

    def _should_use_fused_supervision(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> bool:
        if self.fused_supervision_backend == "none":
            return False
        if self.fused_supervision_backend not in {"auto", "triton"}:
            return False
        if not fused_supervision_available() and not self._fused_supervision_warned:
            logger.warning("请求启用 fused supervision kernel，但当前环境不可用，将自动回退到 PyTorch loss。")
            self._fused_supervision_warned = True
            return False
        if prediction.shape != target.shape:
            return False
        return fused_supervision_available()

    def _prepare_batch_for_model(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.batch_encoder is None or "latents" in batch:
            return batch
        if "pixel_values" not in batch:
            return batch

        prepared_batch = dict(batch)
        prepared_batch = self.batch_encoder.encode_batch(prepared_batch)
        return prepared_batch

    # ==================== Evaluation ====================

    def _forward_and_loss_eval(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Evaluation forward pass (no updates, no auxiliary losses)."""
        latents = batch["latents"]
        bs = latents.shape[0]

        timesteps = self._sample_timesteps(bs)
        sigmas = timesteps / self.num_train_timesteps

        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise

        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        teacher_output = self.run_teacher(
            teacher_input,
            batch,
            cache_extra={"timesteps": timesteps, "mode": "eval"},
        )

        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        student_output = self.run_student(student_input, batch)

        loss = self.compute_distill_loss(teacher_output, student_output, batch, timesteps)
        return loss

    def validation_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Validation step (override in subclasses if needed)."""
        return self._forward_and_loss_eval(batch)

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation dataloader and return metrics."""
        if self.val_dataloader is None:
            return {}

        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

                loss = self.validation_step(batch)
                total_loss += loss.item()
                num_batches += 1

                if num_batches >= self.args.eval_batches:
                    break

        if num_batches == 0:
            self.student_model.train()
            return {}

        avg_loss = total_loss / max(1, num_batches)
        if self.is_distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size

        self.student_model.train()
        return {"eval_loss": avg_loss}

    # ==================== Hooks ====================

    def on_train_step_end(self, metrics: Dict[str, float]):
        """Hook called after each training step. Override for custom logic."""
        pass

    # ==================== Checkpoint ====================

    def save_checkpoint(self, step: int, output_dir: str):
        """Save training checkpoint (DDP/FSDP/DeepSpeed compatible)."""
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if self._deepspeed_engine is not None:
            # DeepSpeed has its own checkpoint mechanism
            self._deepspeed_engine.save_checkpoint(ckpt_dir, tag=f"step-{step}")
            logger.info(f"DeepSpeed checkpoint saved to {ckpt_dir}")
        elif self._is_fsdp_wrapped(self.student_model):
            # FSDP: use full_state_dict or sharded_state_dict
            try:
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    StateDictType,
                    FullStateDictConfig,
                )
                # Save full state dict on rank 0
                full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.student_model, StateDictType.FULL_STATE_DICT, full_cfg):
                    model_state = self.student_model.state_dict()
                    if self.is_main_process:
                        state = {
                            "step": step,
                            "epoch": self.epoch,
                            "student_model": model_state,
                            "optimizer": {},  # FSDP optimizer state is complex; save separately if needed
                            "best_loss": self.best_loss,
                        }
                        if self.ema is not None:
                            state["ema"] = self.ema.state_dict()
                        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
                        logger.info(f"FSDP checkpoint saved to {ckpt_dir}")
            except Exception as e:
                logger.warning(f"FSDP checkpoint save failed: {e}. Falling back to DDP-style save.")
                self._save_checkpoint_ddp(step, ckpt_dir)
        else:
            self._save_checkpoint_ddp(step, ckpt_dir)

    def _save_checkpoint_ddp(self, step: int, ckpt_dir: str):
        """Standard DDP checkpoint save."""
        model_state = self._unwrap_model(self.student_model).state_dict()
        state = {
            "step": step,
            "epoch": self.epoch,
            "student_model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
        logger.info(f"Checkpoint saved to {ckpt_dir}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint (DDP/FSDP/DeepSpeed compatible)."""
        if self._deepspeed_engine is not None:
            # DeepSpeed handles its own checkpoint loading
            _, client_state = self._deepspeed_engine.load_checkpoint(path)
            if client_state:
                self.global_step = client_state.get("step", 0)
                self.epoch = client_state.get("epoch", 0)
            logger.info(f"DeepSpeed checkpoint resumed from {path}")
            return

        ckpt_path = os.path.join(path, "trainer_state.pt") if os.path.isdir(path) else path
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Load model state into unwrapped model
        raw_model = self._unwrap_model(self.student_model)
        raw_model.load_state_dict(state["student_model"])

        if "optimizer" in state and state["optimizer"]:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")

        if "lr_scheduler" in state:
            try:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            except Exception as e:
                logger.warning(f"Could not load lr_scheduler state: {e}")

        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        self.global_step = state["step"]
        self.epoch = state.get("epoch", 0)
        self.best_loss = state.get("best_loss", float("inf"))

        # Restore EMA if present
        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
            logger.info("EMA state restored from checkpoint.")

        logger.info(f"Resumed from step {self.global_step}")

    # ==================== Utilities ====================

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor values in batch dict to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            else:
                result[k] = v
        return result

    @staticmethod
    def _infer_batch_size(batch: Dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
        return 0

    def _compute_grad_norm(self) -> torch.Tensor:
        """Compute total gradient norm across all parameters."""
        total_norm = torch.tensor(0.0, device=self.device)
        for p in self._unwrap_model(self.student_model).parameters():
            if p.grad is not None:
                total_norm += p.grad.data.float().norm(2) ** 2
        return total_norm.sqrt()

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        msg_parts = [f"Step {self.global_step}/{self.args.max_train_steps}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                msg_parts.append(f"{k}={v:.6f}" if "loss" in k else f"{k}={v:.4f}")
            else:
                msg_parts.append(f"{k}={v}")
        logger.info(" | ".join(msg_parts))
        if self._tracker is not None:
            self._tracker.log_metrics(metrics, step=self.global_step)
