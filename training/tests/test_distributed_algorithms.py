"""Two-process CPU/Gloo correctness gate, including exact algorithm resume."""

import datetime
import os
import random
import socket
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_algorithm_regressions import TinyFlow, make_trainer


def _seed(rank):
    random.seed(123 + rank)
    np.random.seed(123 + rank)
    torch.manual_seed(123 + rank)


def _gloo_worker(rank, rendezvous, output_root):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        for method, extra in [
            ("step_distill", {}), ("progressive_distill", {}), ("consistency_distill", {}),
            ("stream_distill", {}), ("context_forcing", {}),
            ("adversarial_distill", {}), ("dmd_distill", {"dmd_variant": "dmd2"}),
        ]:
            def construct(name, max_steps, resume=""):
                _seed(rank)
                return make_trainer(method, os.path.join(output_root, method, name),
                                    max_train_steps=max_steps, resume_from=resume, **extra)

            reference = construct("reference", 4)
            reference.train()
            expected = reference._unwrap_model(reference.student_model).state_dict()

            interrupted = construct("interrupted", 2)
            interrupted.train()
            checkpoint = os.path.join(output_root, method, "interrupted", "checkpoint-2")
            restored = construct("restored", 4, checkpoint)
            restored.train()
            actual = restored._unwrap_model(restored.student_model).state_dict()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                rank_zero = actual[name].clone()
                dist.broadcast(rank_zero, src=0)
                torch.testing.assert_close(actual[name], rank_zero, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _routed_worker(rank, rendezvous, output_root):
    from training.model_adapter import NoiseRoutedDenoiser
    from training.trainer_args import TrainerArgs
    from training.trainers import TRAINER_REGISTRY
    from training.utils.replicated_gradients import synchronize_replicated_gradients
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        # Exercise absent gradients, mixed dtypes, bounded large-parameter
        # splitting, and cross-rank Inf propagation before GradScaler checking.
        parameters = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(6)), torch.nn.Parameter(torch.zeros(2)),
            torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)), torch.nn.Parameter(torch.zeros(1)),
        ])
        parameters[0].grad = torch.ones(6) if rank == 0 else None
        parameters[2].grad = torch.full((3,), float(rank + 1), dtype=torch.float64)
        parameters[3].grad = torch.tensor([float("inf") if rank else 1.0])
        synchronize_replicated_gradients(parameters, bucket_bytes=8)
        torch.testing.assert_close(parameters[0].grad, torch.full((6,), 0.5))
        assert parameters[1].grad is None
        torch.testing.assert_close(parameters[2].grad, torch.full((3,), 1.5, dtype=torch.float64))
        assert torch.isinf(parameters[3].grad).all()
        for method in ("step_distill", "stream_distill", "progressive_distill", "consistency_distill",
                       "context_forcing", "adversarial_distill", "dmd_distill"):
            _seed(rank)
            teacher = NoiseRoutedDenoiser(TinyFlow(0.8), TinyFlow(0.7), 0.5)
            student = NoiseRoutedDenoiser(TinyFlow(0.4), TinyFlow(0.3), 0.5)
            optimizer = torch.optim.AdamW(student.parameters(), lr=0.001)
            args = TrainerArgs(
                distill_method=method, output_dir=os.path.join(output_root, method), mixed_precision="no",
                report_to="none", max_train_steps=2, log_every=100, save_every=100,
                gradient_checkpointing=False, window_size=4, overlap_frames=2, denoising_steps_per_frame=4,
                curriculum_training=False, temporal_context_size=2, memory_frames=2,
                dmd_variant="dmd2", dmd_fake_score_updates=1, dmd_student_steps=4,
                dmd_latent_channels=2, dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1,
                adversarial_latent_channels=2, adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1,
            )
            trainer = TRAINER_REGISTRY[method](
                args=args, teacher_model=teacher, student_model=student, optimizer=optimizer,
                lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
                train_dataloader=[{"latents": torch.randn(1, 2, 6, 8, 8)}], device=torch.device("cpu"),
            )
            trainer.train()
            for model in (trainer.student_model, getattr(trainer, "fake_score_model", None),
                          getattr(trainer, "discriminator", None)):
                if model is None:
                    continue
                for parameter in model.parameters():
                    reference = parameter.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_all_trainers_two_rank_gloo_and_exact_resume(tmp_path):
    # A sandbox may forbid even local IPC sockets. Report that environment gate
    # explicitly instead of crashing a spawned C++ libuv process with SIGABRT.
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Local loopback sockets are forbidden by this sandbox; rerun with IPC permission")
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_noise_routed_backbones_two_rank_gloo(tmp_path):
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Local loopback sockets are forbidden by this sandbox; rerun with IPC permission")
    mp.spawn(_routed_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
