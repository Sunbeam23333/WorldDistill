"""Independent auxiliary replicas with student engine/sharding ownership."""
import importlib.util
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


class TinyFlow(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))
    def forward(self, hidden_states, timestep, **kwargs):
        return hidden_states * self.scale


def construct(method, mode, path):
    torch.manual_seed(31)
    student, teacher = TinyFlow(0.4), TinyFlow(0.8)
    args = TrainerArgs(distill_method=method, parallel_mode=mode, mixed_precision="no",
        report_to="none", output_dir=str(path), gradient_accumulation_steps=2,
        dmd_variant="dmd2", dmd_fake_score_updates=2, dmd_student_steps=2,
        dmd_latent_channels=2, dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1,
        adversarial_latent_channels=2, adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    return TRAINER_REGISTRY[method](args=args, student_model=student, teacher_model=teacher,
        optimizer=optimizer, lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        train_dataloader=[{"latents": torch.randn(1, 2, 4, 4, 4)} for _ in range(3)], device=torch.device("cpu"))


class Engine(nn.Module):
    """API control: only engine.backward/step may update the main student."""
    def __init__(self, module, optimizer):
        super().__init__()
        self.module, self.optimizer = module, optimizer
        self.backward_calls = self.step_calls = self.updates = 0
    def forward(self, **kwargs):
        return self.module(**kwargs)
    def backward(self, loss):
        self.backward_calls += 1
        (loss / 2).backward()
    def step(self):
        self.step_calls += 1
        if self.step_calls % 2 == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
    def get_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]


class ForbiddenStudentScaler:
    def scale(self, *args):
        raise AssertionError("DeepSpeed student scaling must remain engine-owned")
    def step(self, *args):
        raise AssertionError("Auxiliary optimizers cannot use the student's scaler")


@pytest.mark.parametrize("method", ["adversarial_distill", "dmd_distill"])
def test_deepspeed_student_and_replicated_auxiliary_updates_have_disjoint_ownership(tmp_path, method):
    torch.set_num_threads(1)
    instance = construct(method, "deepspeed", tmp_path)
    student = instance.student_model
    engine = Engine(student, instance.optimizer)
    instance.student_model = instance._deepspeed_engine = engine
    instance.scaler = ForbiddenStudentScaler()
    instance.disc_scaler = torch.amp.GradScaler("cpu")
    if method == "dmd_distill":
        instance.fake_score_scaler = torch.amp.GradScaler("cpu")
    instance._initialize_data_iterator(resuming=False)
    metrics = instance.train_step(instance._next_batch())
    assert engine.backward_calls == engine.step_calls == 2 and engine.updates == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(student.scale.detach(), torch.tensor(0.4))
    assert {int(state["step"]) for state in instance.disc_optimizer.state.values()} == {1}
    report = instance.auxiliary_placement_report()
    assert report["parallel_mode"] == "replicated" and report["student_parallel_mode"] == "deepspeed"
    assert report["replicas"]["discriminator"]["optimizer_tensor_bytes"] > 0
    if method == "dmd_distill":
        assert {int(state["step"]) for state in instance.fake_score_optimizer.state.values()} == {2}
        assert report["replicas"]["fake_score_model"]["parameter_bytes"] == 4


@pytest.mark.parametrize("method", ["adversarial_distill", "dmd_distill"])
def test_fsdp_configuration_keeps_auxiliary_replicas_separate_and_places_fake_on_device(tmp_path, method):
    instance = construct(method, "fsdp", tmp_path)
    report = instance.auxiliary_placement_report()
    assert report["parallel_mode"] == "replicated" and report["student_parallel_mode"] == "fsdp"
    student_ids = {id(p) for p in instance.student_model.parameters()}
    for model_name in report["replicas"]:
        model = getattr(instance, model_name)
        assert all(parameter.device == instance.device and parameter.dtype == torch.float32 for parameter in model.parameters())
        assert not student_ids.intersection(id(p) for p in model.parameters())
    if method == "dmd_distill":
        assert next(instance.fake_score_model.parameters()).data_ptr() != next(instance.teacher_model.parameters()).data_ptr()


@pytest.mark.parametrize("parallel,stage", [("fsdp", 2), ("deepspeed", 1), ("deepspeed", 2), ("deepspeed", 3)])
@pytest.mark.parametrize("precision", ["no", "fp16", "bf16"])
def test_cuda_two_rank_sharded_student_replicated_auxiliary_resume(tmp_path, parallel, stage, precision):
    if not torch.version.cuda or torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible CUDA devices")
    if precision == "bf16" and any(torch.cuda.get_device_capability(index)[0] < 8 for index in (0, 1)):
        pytest.skip("BF16 case requires native BF16 on both selected devices")
    if parallel == "deepspeed" and importlib.util.find_spec("deepspeed") is None:
        pytest.skip("DeepSpeed backend is not installed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
        "--rdzv-backend=static", "--master-addr=127.0.0.1", f"--master-port={port}",
        "-m", "tools.check_distributed_training", "--device", "cuda", "--parallel", parallel,
        "--zero-stage", str(stage), "--mixed-precision", precision, "--gradient-accumulation-steps", "2",
        "--methods", "adversarial_distill,dmd_distill", "--output-dir", str(tmp_path / "run")]
    result = subprocess.run(command, cwd=root, env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
