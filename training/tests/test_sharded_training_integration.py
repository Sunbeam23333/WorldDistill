"""EMA lifecycle integration plus a two-device CUDA train/save/resume gate."""

from datetime import timedelta
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training.trainer_args import TrainerArgs
from training.trainers import TRAINER_REGISTRY
from training.trainers import base_distill_trainer as base
from training.utils.sharded_state import ShardedEMA


class Flow(torch.nn.Module):
    def __init__(self, value=0.4):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.projection.weight.fill_(value)

    def forward(self, hidden_states, timestep, **kwargs):
        return self.projection(hidden_states)


def construct(tmp_path, **fields):
    args = TrainerArgs(mixed_precision="no", report_to="none", output_dir=str(tmp_path), **fields)
    student = Flow()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    return TRAINER_REGISTRY[args.distill_method](
        args=args, teacher_model=Flow(0.8), student_model=student, optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        train_dataloader=[], device=torch.device("cpu"),
    )


@pytest.mark.parametrize("parallel,stage", [("fsdp", 2), ("deepspeed", 3)])
def test_sharded_base_ema_is_deferred_until_partitioning(tmp_path, parallel, stage):
    trainer = construct(tmp_path, parallel_mode=parallel, deepspeed_stage=stage, use_ema=True)
    assert trainer.ema is None
    if parallel == "fsdp":
        trainer._initialize_sharded_ema()
        assert isinstance(trainer.ema, ShardedEMA)
        assert trainer.ema.parallel_mode == "fsdp"


def test_sharded_base_checkpoint_marker_and_rank_sidecar_roundtrip(tmp_path, monkeypatch):
    trainer = construct(tmp_path, parallel_mode="fsdp", use_ema=True)
    trainer._initialize_sharded_ema()
    with torch.no_grad():
        trainer.student_model.projection.weight.add_(1)
    trainer.ema.update(trainer.student_model)
    directory = tmp_path / "checkpoint-1"
    directory.mkdir()
    trainer.ema.save(directory)
    expected = trainer.ema.state_dict()
    trainer.ema.shadow["projection.weight"].zero_()
    trainer._restore_sharded_ema(str(directory))
    torch.testing.assert_close(trainer.ema.shadow["projection.weight"], expected["shadow"]["projection.weight"])
    assert trainer.ema.step_count == 1
    state = {"step": 1, "epoch": 0, "student_model": {}, "optimizer": {}, "lr_scheduler": {},
             "scaler": {}, "best_loss": float("inf"), "sharded_ema": "wrong"}
    with pytest.raises(ValueError, match="marker"):
        trainer._validate_standard_checkpoint_state(state)


def test_fsdp_consistency_shadow_initializes_from_student_and_applies_to_frozen_target(tmp_path, monkeypatch):
    from training.trainers import consistency_distill_trainer as consistency

    trainer = construct(tmp_path, distill_method="consistency_distill", parallel_mode="fsdp")
    assert trainer.consistency_ema is None
    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, **kwargs):
            return self.module(**kwargs)

    trainer.student_model = Wrapper(trainer.student_model)
    monkeypatch.setattr(trainer, "_is_fsdp_wrapped", lambda model: isinstance(model, Wrapper))
    monkeypatch.setattr(consistency, "wrap_model_fsdp", lambda model, **kwargs: Wrapper(model))
    trainer._initialize_parallel_targets()
    assert isinstance(trainer.consistency_ema, ShardedEMA)
    with torch.no_grad():
        trainer.student_model.module.projection.weight.add_(2)
    trainer.on_train_step_end({"optimizer_skipped": 0})
    trainer.consistency_ema.apply_to(trainer.ema_model.module)
    torch.testing.assert_close(trainer.ema_model.module.projection.weight,
                               trainer.consistency_ema.shadow["projection.weight"])
    before = trainer.consistency_ema.step_count
    trainer.on_train_step_end({"optimizer_skipped": 1})
    assert trainer.consistency_ema.step_count == before
    state = trainer._local_target_state()
    trainer._restore_local_target_state(state)
    state["parameters"]["projection.weight"] = torch.ones(10)
    with pytest.raises(ValueError, match="shape differs"):
        trainer._restore_local_target_state(state)


def _cuda_worker(rank, init_file, output_root, parallel, precision):
    from tools.check_distributed_training import TinyDenoiser, assert_state_close, checkpoint_payload
    from training.runtime.teacher_student_runtime import TeacherStudentRuntime

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        for method in ("step_distill", "consistency_distill"):
            def train(name, steps, resume=""):
                torch.manual_seed(431)
                teacher, student = TinyDenoiser(0.8), TinyDenoiser(0.4)
                args = TrainerArgs(
                    distill_method=method, parallel_mode=parallel, deepspeed_stage=3,
                    mixed_precision=precision, use_ema=True, report_to="none",
                    max_train_steps=steps, warmup_steps=0, lr_scheduler="constant",
                    learning_rate=0.001, output_dir=str(Path(output_root) / method / name),
                    save_every=100, log_every=100, resume_from=resume,
                    runtime_teacher_offload="cpu", runtime_enable_dpp=True,
                )
                optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
                runtime = TeacherStudentRuntime(args, teacher, student, device)
                generator = torch.Generator().manual_seed(13 + rank)
                loader = [{"latents": torch.randn(1, 2, 4, 4, 4, generator=generator)} for _ in range(3)]
                trainer = TRAINER_REGISTRY[method](
                    args=args, teacher_model=teacher, student_model=student, optimizer=optimizer,
                    lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
                    train_dataloader=loader, device=device, runtime=runtime,
                )
                torch.manual_seed(711 + rank)
                trainer.train()
                assert isinstance(trainer.ema, ShardedEMA)
                assert trainer.ema.step_count > 0
                return Path(args.output_dir) / f"checkpoint-{steps}"

            full = train("full", 4)
            interrupted = train("interrupted", 2)
            resumed = train("resumed", 4, str(interrupted))
            dist.barrier()
            expected, expected_state = checkpoint_payload(full, parallel)
            actual, actual_state = checkpoint_payload(resumed, parallel)
            assert_state_close(actual_state, expected_state)
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], atol=1e-7, rtol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(), reason="Two CUDA devices with NCCL required")
@pytest.mark.parametrize("parallel", ["fsdp", "deepspeed"])
def test_cuda_sharded_fp16_ema_consistency_and_exact_restart(tmp_path, parallel):
    if parallel == "deepspeed" and importlib.util.find_spec("deepspeed") is None:
        pytest.skip("DeepSpeed installation required")
    context = mp.spawn(_cuda_worker, args=(str(tmp_path / "nccl-init"), str(tmp_path), parallel, "fp16"), nprocs=2, join=False)
    try:
        import time
        deadline = time.monotonic() + 240
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("CUDA distributed EMA test did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
