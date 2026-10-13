"""Full training, accumulation, canonical checkpoint, and deterministic restart."""
import copy
import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from training.trainer_args import TrainerArgs
from training.train_distill import _build_training_optimizer, set_seed
from training.utils.schedulers import build_lr_scheduler
from training.trainers import build_trainer


def _train(rank, world, root, tp, pp, limit, resume="", method="step_distill", extras=None):
    from training.tests.test_context_parallel_training import _tiny_wan
    set_seed(371)
    teacher = _tiny_wan()
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    extra = dict(extras or {})
    checkpointing = extra.pop("gradient_checkpointing", False)
    if extra.get("use_lora"):
        from peft import LoraConfig, get_peft_model
        student = get_peft_model(student, LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_v"]))
    args = TrainerArgs(distill_method=method, mixed_precision="no", tp_size=tp, pp_size=pp,
        output_dir=str(root), max_train_steps=limit, batch_size=1, num_workers=0,
        gradient_accumulation_steps=2, gradient_checkpointing=checkpointing, use_ema=True,
        warmup_steps=0, lr_scheduler="constant", learning_rate=0.002, max_grad_norm=0.4,
        save_every=20, log_every=20, report_to="none", resume_from=str(resume),
        progressive_stages=[8, 4, 2], progressive_stage_steps=1,
        window_size=3, overlap_frames=1, denoising_steps_per_frame=1, causal_attention=False,
        num_frames=3, memory_frames=1, temporal_context_size=2, curriculum_training=False,
        adversarial_latent_channels=2, adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1,
        adversarial_r1_weight=0., dmd_latent_channels=2, dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1,
        dmd_fake_score_updates=1, dmd_teacher_steps=2, dmd_student_steps=2, **extra)
    if args.use_lora:
        args.lora_rank, args.lora_alpha, args.lora_target_modules = 2, 4, ["to_q", "to_v"]
    generator = torch.Generator().manual_seed(9)
    data = [{"latents": torch.randn(2, 3, 2, 2, generator=generator),
             "encoder_hidden_states": torch.randn(3, 8, generator=generator)} for _ in range(8)]
    dp = world // (tp * pp)
    if args.elastic_resume:
        from training.utils.elastic import ElasticBatchSampler
        loader = DataLoader(data, batch_sampler=ElasticBatchSampler(data, batch_size=1,
            rank=rank // (tp * pp), world_size=dp, seed=41, shuffle=True, drop_last=True))
    else:
        loader = DataLoader(data, batch_size=1, sampler=DistributedSampler(data,
            num_replicas=dp, rank=rank // (tp * pp), seed=41, shuffle=True))
    optimizer = _build_training_optimizer(student, args)
    scheduler = build_lr_scheduler(optimizer, scheduler_type="constant", warmup_steps=0, total_steps=limit)
    runtime = None
    if args.runtime_teacher_offload != "none":
        from training.runtime.teacher_student_runtime import TeacherStudentRuntime
        runtime = TeacherStudentRuntime(args, teacher, student, torch.device("cpu"))
    trainer = build_trainer(method=method, args=args, teacher_model=teacher, student_model=student,
        optimizer=optimizer, lr_scheduler=scheduler, train_dataloader=loader, device=torch.device("cpu"), runtime=runtime)
    set_seed(618 + rank)
    if resume and args.elastic_resume:
        original_step = trainer.train_step
        def checked_first_step(batch):
            from training.parallel.checkpoint import collect_model_parallel_checkpoint
            expected = torch.load(Path(resume) / "trainer_state.pt", weights_only=False)
            actual = collect_model_parallel_checkpoint(trainer, trainer.global_step)
            for key in ("student_model", "optimizer", "ema", "lr_scheduler", "scaler"):
                _equal_tree(actual[key], expected[key])
            trainer.train_step = original_step
            return original_step(batch)
        trainer.train_step = checked_first_step
    trainer.train()
    return trainer


def _equal_tree(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal_tree(a, b)
    else:
        assert actual == expected


def _worker(rank, world, rendezvous, root, tp, pp, method, extras=None):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=90))
    try:
        root = Path(root)
        _train(rank, world, root / "continuous", tp, pp, 2, method=method, extras=extras)
        _train(rank, world, root / "split", tp, pp, 1, method=method, extras=extras)
        trainer = _train(rank, world, root / "resumed", tp, pp, 2,
                        root / "split/checkpoint-1", method=method, extras=extras)
        assert trainer.model_parallel.dp_size == world // (tp * pp)
        assert dist.get_backend(trainer.model_parallel.checkpoint_group) == "gloo"
        if rank == 0:
            reference = torch.load(root / "continuous/checkpoint-2/trainer_state.pt", weights_only=False)
            restored = torch.load(root / "resumed/checkpoint-2/trainer_state.pt", weights_only=False)
            for name in ("student_model", "optimizer", "ema", "lr_scheduler", "scaler"):
                _equal_tree(restored[name], reference[name])
            assert any("blocks.0" in name for name in restored["student_model"])
            assert all(".local." not in name for name in restored["student_model"])
            from training.student_export import export_student
            from training.model_adapter import load_diffusers_training_model
            from training.tests.test_context_parallel_training import _tiny_wan
            base = root / "base"
            _tiny_wan().model.save_pretrained(base)
            export_student(str(base), str(root / "resumed/checkpoint-2"), str(root / "export"), num_steps=2)
            loaded = load_diffusers_training_model(root / "export")
            reference_model = _tiny_wan()
            if (extras or {}).get("use_lora"):
                from peft import LoraConfig, get_peft_model
                reference_model = get_peft_model(reference_model,
                    LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_v"]))
            if (extras or {}).get("use_dual_model"):
                from training.model_adapter import NoiseRoutedDenoiser
                reference_model = NoiseRoutedDenoiser(reference_model, copy.deepcopy(reference_model), .5, 1000)
            reference_model.load_state_dict(restored["student_model"], strict=True)
            if (extras or {}).get("use_lora"):
                reference_model = reference_model.merge_and_unload(safe_merge=True)
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, reference_model.state_dict()[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="Diffusers model integration dependency")
@pytest.mark.parametrize("world,tp,pp,method", [(2, 2, 1, "step_distill"),
    (2, 1, 2, "step_distill"), (4, 2, 2, "step_distill"), (4, 2, 1, "step_distill"),
    (2, 1, 2, "progressive_distill"), (2, 2, 1, "consistency_distill"),
    (2, 1, 2, "stream_distill"), (2, 2, 1, "context_forcing"),
    (2, 1, 2, "dmd_distill"), (2, 2, 1, "adversarial_distill")])
def test_model_parallel_full_training_resume(tmp_path, world, tp, pp, method):
    mp.spawn(_worker, args=(world, str(tmp_path / "rdzv"), str(tmp_path), tp, pp, method),
             nprocs=world, join=True)


@pytest.mark.parametrize("tp,pp,method,extras", [
    (2, 1, "step_distill", {"use_lora": True, "gradient_checkpointing": True}),
    (1, 2, "step_distill", {"runtime_teacher_offload": "cpu", "gradient_checkpointing": True}),
    (2, 2, "progressive_distill", {"use_lora": True, "runtime_teacher_offload": "nvme", "gradient_checkpointing": True}),
    (2, 1, "step_distill", {"use_dual_model": True}),
])
def test_model_parallel_lora_checkpointing_and_offload(tmp_path, tp, pp, method, extras):
    pytest.importorskip("diffusers")
    if extras.get("use_lora"):
        pytest.importorskip("peft")
    world = tp * pp
    mp.spawn(_worker, args=(world, str(tmp_path / "rdzv"), str(tmp_path), tp, pp, method, extras),
             nprocs=world, join=True)


def _resize_worker(rank, world, rendezvous, directory, resume, limit):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=90))
    try:
        trainer = _train(rank, world, Path(directory), 2, 1, limit, resume,
                         extras={"elastic_resume": True})
        assert trainer.global_step == limit
    finally:
        dist.destroy_process_group()


def test_model_parallel_elastic_dp_resize_preserves_all_states(tmp_path):
    pytest.importorskip("diffusers")
    from training.utils.elastic import latest_elastic_checkpoint
    resume = ""
    for limit, world in enumerate((4, 2, 4), 1):
        output = tmp_path / f"phase{limit}"
        mp.spawn(_resize_worker, args=(world, str(tmp_path / f"rdzv{limit}"), str(output), resume, limit),
                 nprocs=world, join=True)
        resume = latest_elastic_checkpoint(output)
        assert resume == str(output / f"checkpoint-{limit}")
    state = torch.load(Path(resume) / "trainer_state.pt", weights_only=False)
    assert state["resume_state"]["elastic"]["generation"] == 2
