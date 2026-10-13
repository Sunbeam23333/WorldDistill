"""Analytic MeanFlow identity, actual gradients, and full training resume."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import os
import socket
import subprocess
import sys
import pytest
import torch
from torch import nn

from training.trainer_args import TrainerArgs
from training.trainers.mean_flow_distill_trainer import MeanFlowDistillTrainer
from training.utils.meanflow import MeanFlowTimeAdapter, meanflow_jvp, meanflow_target, require_meanflow_model


class DualTimeLinear(nn.Module):
    supports_meanflow = True

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.4))
        self.b = nn.Parameter(torch.tensor(0.2))
        self.c = nn.Parameter(torch.tensor(-0.1))

    def forward(self, hidden_states, timestep, timestep_r):
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        return self.a * hidden_states + self.b * timestep.reshape(shape) / 1000 + self.c * timestep_r.reshape(shape) / 1000


class VelocityTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, hidden_states, timestep, **kwargs):
        return self.scale * hidden_states


def test_exact_jvp_time_scaling_and_stop_gradient_target():
    model = DualTimeLinear()
    x = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    t, r = torch.tensor([0.8, 0.6]), torch.tensor([0.2, 0.6])
    derivative = meanflow_jvp(model, x, t, r, v, {}, 1000)
    torch.testing.assert_close(derivative, model.a.detach() * v.detach() + model.b.detach())
    target = meanflow_target(v, derivative, t, r)
    assert not target.requires_grad
    torch.testing.assert_close(target[1], v.detach()[1])  # diagonal is velocity matching
    prediction = model(x, t * 1000, r * 1000)
    loss = (prediction - target).square().mean()
    expected_a = (2 * (prediction.detach() - target) * x).mean()
    loss.backward()
    torch.testing.assert_close(model.a.grad, expected_a)
    assert model.b.grad is not None and model.c.grad is not None
    assert v.grad is None


def test_jvp_differentiates_nonlinearity_not_finite_difference():
    class Nonlinear(DualTimeLinear):
        def forward(self, hidden_states, timestep, timestep_r):
            return self.a * hidden_states.square() + (timestep / 1000).sin()[:, None] + timestep_r[:, None] / 1000
    model = Nonlinear()
    x = torch.tensor([[0.4, 0.8]])
    t, r, v = torch.tensor([0.7]), torch.tensor([0.2]), torch.tensor([[0.9, 0.1]])
    actual = meanflow_jvp(model, x, t, r, v, {}, 1000)
    torch.testing.assert_close(actual, 2 * model.a.detach() * x * v + t.cos()[:, None])


def test_adapter_retains_initial_function_and_learns_distinct_endpoint():
    base = VelocityTeacher()
    model = MeanFlowTimeAdapter(base, latent_channels=2)
    x = torch.randn(1, 2, 2, 2, 2)
    t, r = torch.tensor([800.0]), torch.tensor([100.0])
    torch.testing.assert_close(model(x, t, r), base(x, t))
    with torch.no_grad():
        model.time_conditioner[-1].weight.fill_(0.1)
    assert not torch.equal(model(x, t, r), model(x, t, torch.tensor([500.0])))
    model(x, t, r).square().mean().backward()
    assert model.time_conditioner[-1].weight.grad.abs().sum() > 0
    assert base.scale.grad is not None
    with pytest.raises(ValueError, match="dual-time"):
        require_meanflow_model(base)


def build_trainer(directory, steps=4, resume="", model=None, diagonal=0):
    torch.manual_seed(1703)
    teacher = VelocityTeacher()
    student = model or DualTimeLinear()
    args = TrainerArgs(distill_method="mean_flow_distill", max_train_steps=steps,
        output_dir=str(directory), resume_from=resume, meanflow_diagonal_ratio=diagonal,
        meanflow_norm_power=0, mixed_precision="no", report_to="none", batch_size=1,
        num_workers=0, log_every=100, save_every=100, gradient_checkpointing=False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    trainer = MeanFlowDistillTrainer(args=args, teacher_model=teacher, student_model=student,
        optimizer=optimizer, lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        train_dataloader=[{"latents": torch.randn(1, 2, 2, 2, 2)} for _ in range(3)],
        device=torch.device("cpu"))
    torch.manual_seed(103)
    return trainer


def test_training_resume_matches_continuous_weights_optimizer_and_rng(tmp_path):
    continuous = build_trainer(tmp_path / "continuous")
    continuous.train()
    expected_next = torch.rand(4)
    interrupted = build_trainer(tmp_path / "interrupted", steps=2)
    interrupted.train()
    resumed = build_trainer(tmp_path / "resumed", resume=str(tmp_path / "interrupted" / "checkpoint-2"))
    resumed.train()
    assert resumed.global_step == 4
    for key, value in continuous.student_model.state_dict().items():
        torch.testing.assert_close(value, resumed.student_model.state_dict()[key], atol=0, rtol=0)
    for left, right in zip(continuous.optimizer.state.values(), resumed.optimizer.state.values()):
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
    torch.testing.assert_close(torch.rand(4), expected_next, atol=0, rtol=0)
    assert not torch.equal(continuous.student_model.a, torch.tensor(0.4))


def test_validation_has_true_jvp_and_leaves_parameter_grads_untouched(tmp_path):
    trainer = build_trainer(tmp_path)
    with torch.no_grad():
        loss = trainer.validation_step({"latents": torch.randn(2, 2, 2, 2, 2)})
    assert torch.isfinite(loss) and not loss.requires_grad
    assert all(parameter.grad is None for parameter in trainer.student_model.parameters())
    generated = trainer.generate_samples({"latents": torch.zeros(2, 2, 2, 2, 2)}, num_steps=2)
    assert generated.shape == (2, 2, 2, 2, 2) and torch.isfinite(generated).all()


def test_zero3_detached_jvp_context_restores_hooks_before_trainable_forward(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path)
    raw = trainer.student_model
    calls = []
    offload = SimpleNamespace(module=raw, forward_hooks=[], backward_hooks=[])
    def register(module):
        calls.append("register")
        offload.forward_hooks.append(module.register_forward_pre_hook(lambda *_: calls.append("hook")))
    offload._register_deepspeed_module = register
    register(raw)
    trainer.args.deepspeed_stage = 3
    trainer._deepspeed_engine = SimpleNamespace(optimizer=SimpleNamespace(parameter_offload=offload))
    @contextmanager
    def gather(parameters, modifier_rank):
        calls.append("gather")
        yield
        assert offload.forward_hooks  # restore before repartition, including failures
        calls.append("release")
    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(zero=SimpleNamespace(GatheredParameters=gather)))
    x, t, r, v = torch.ones(1, 2), torch.tensor([0.8]), torch.tensor([0.2]), torch.ones(1, 2)
    with trainer._full_jvp_student() as student:
        derivative = meanflow_jvp(student, x, t, r, v, {}, 1000)
        assert "hook" not in calls
    assert calls == ["register", "gather", "register", "release"]
    raw(x, t * 1000, r * 1000).sum().backward()
    assert calls[-1] == "hook" and raw.a.grad is not None
    torch.testing.assert_close(derivative, torch.full_like(x, 0.6))


@pytest.mark.parametrize("field,value", [("meanflow_diagonal_ratio", 1.1), ("meanflow_norm_power", -1),
    ("meanflow_norm_eps", 0), ("joint_av_audio_weight", float("nan")), ("num_train_timesteps", -1)])
def test_objective_argument_validation(field, value):
    with pytest.raises(ValueError, match=field):
        TrainerArgs(**{field: value})


def test_routed_meanflow_different_experts_across_two_ranks(tmp_path):
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    except PermissionError:
        pytest.skip("Local process-group sockets are unavailable")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
        "--rdzv-backend=static", "--master-addr=127.0.0.1", f"--master-port={port}",
        str(Path(__file__).with_name("meanflow_routed_probe.py")), str(tmp_path / "routed")]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=100)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "routed-meanflow-two-rank-passed" in completed.stdout
