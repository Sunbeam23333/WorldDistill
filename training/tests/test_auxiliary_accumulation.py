"""Alternating optimizer semantics, loss scaling, and accumulation resume."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import pytest
import torch
from torch import nn
from training.trainer_args import TrainerArgs
from training.trainers import TRAINER_REGISTRY
from training.utils.auxiliary_optimization import finish_auxiliary_step


class TinyDenoiser(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
    def forward(self, hidden_states, timestep, **kwargs):
        return self.scale * hidden_states


def make_trainer(path, method, steps=4, resume="", amp="no"):
    torch.manual_seed(13)
    teacher, student = TinyDenoiser(0.8), TinyDenoiser(0.4)
    args = TrainerArgs(distill_method=method, output_dir=str(path), mixed_precision=amp,
        max_train_steps=steps, resume_from=resume, report_to="none", gradient_accumulation_steps=2,
        batch_size=1, num_workers=0, log_every=100, save_every=100, gradient_checkpointing=False,
        dmd_variant="dmd2", dmd_fake_score_updates=2, dmd_student_steps=2, dmd_teacher_steps=2,
        dmd_latent_channels=2, dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1,
        adversarial_latent_channels=2, adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    result = TRAINER_REGISTRY[method](args=args, teacher_model=teacher, student_model=student,
        optimizer=optimizer, lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        train_dataloader=[{"latents": torch.randn(1, 2, 4, 4, 4)} for _ in range(3)], device=torch.device("cpu"))
    torch.manual_seed(61)
    return result


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", ["adversarial_distill", "dmd_distill"])
def test_aux_gas2_four_steps_equal_two_plus_two_resume(tmp_path, method):
    torch.set_num_threads(1)
    continuous = make_trainer(tmp_path / "continuous", method)
    continuous.train()
    rng = torch.rand(4)
    partial = make_trainer(tmp_path / "partial", method, steps=2)
    partial.train()
    resumed = make_trainer(tmp_path / "resumed", method, resume=str(tmp_path / "partial" / "checkpoint-2"))
    resumed.train()
    for attribute in ("student_model", "optimizer", "discriminator", "disc_optimizer", "disc_scaler"):
        assert_nested_equal(getattr(continuous, attribute).state_dict(), getattr(resumed, attribute).state_dict())
    if method == "dmd_distill":
        for attribute in ("fake_score_model", "fake_score_optimizer", "fake_score_scaler"):
            assert_nested_equal(getattr(continuous, attribute).state_dict(), getattr(resumed, attribute).state_dict())
        # K=2 score turns per student optimizer step, not K*GAS=4.
        assert {int(state["step"]) for state in resumed.fake_score_optimizer.state.values()} == {8}
    assert {int(state["step"]) for state in resumed.disc_optimizer.state.values()} == {4}
    torch.testing.assert_close(torch.rand(4), rng, atol=0, rtol=0)
    assert not getattr(resumed, "_auxiliary_prepared", False)


def test_auxiliary_scaler_unscales_before_clip_and_preserves_overflow_weights():
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    loss = model(torch.ones(1, 2)).sum()
    scaler.scale(loss).backward()
    before = model.weight.detach().clone()
    assert finish_auxiliary_step(model, optimizer, scaler, max_grad_norm=1)
    torch.testing.assert_close(model.weight, before - 0.1 / (2 ** 0.5), atol=1e-6, rtol=1e-6)
    optimizer.zero_grad(set_to_none=True)
    before = model.weight.detach().clone()
    scaler.scale(model(torch.ones(1, 2)).sum() * float("inf")).backward()
    assert not finish_auxiliary_step(model, optimizer, scaler, max_grad_norm=1)
    torch.testing.assert_close(before, model.weight, rtol=0, atol=0)
    assert scaler.get_scale() == 64


def test_auxiliary_fp16_constructs_independent_scalers_on_cpu(tmp_path):
    # This checks ownership/configuration, not GPU arithmetic.
    trainer = make_trainer(tmp_path, "dmd_distill", amp="fp16")
    assert trainer.fake_score_scaler is not trainer.scaler
    assert trainer.disc_scaler is not trainer.fake_score_scaler


def free_port():
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
    except PermissionError:
        pytest.skip("Local process-group sockets are unavailable")


def test_two_rank_auxiliary_gas2_resume_qualification(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
        "--rdzv-backend=static", "--master-addr=127.0.0.1", f"--master-port={free_port()}",
        "-m", "tools.check_distributed_training", "--device", "cpu", "--methods", "adversarial_distill,dmd_distill",
        "--gradient-accumulation-steps", "2", "--output-dir", str(tmp_path / "evidence")]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=150)
    assert completed.returncode == 0, completed.stdout + completed.stderr
