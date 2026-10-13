"""One dual-expert optimizer: accumulation, native schedules, and migration."""
import copy
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import pytest
import torch
from torch import nn
from training.model_adapter import NoiseRoutedDenoiser
from training.trainer_args import TrainerArgs
from training.trainers.step_distill_trainer import StepDistillTrainer
from training.tests.test_auxiliary_accumulation import assert_nested_equal
from training.tests.test_sharded_student_replicated_auxiliary import Engine, ForbiddenStudentScaler


class Expert(nn.Module):
    def __init__(self, value=0.4, process="flow"):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))
        self.num_train_timesteps = 20
        self.training_noise_process = process
        self.training_prediction_type = "velocity" if process == "flow" else "epsilon"
        if process == "vp":
            self.register_buffer("training_alphas_cumprod", torch.linspace(0.99, 0.2, 20))
    def forward(self, hidden_states, timestep, **kwargs):
        return self.scale * hidden_states


def construct(path, *, steps=4, resume="", mode="ddp", process="flow", gas=2, ema=True):
    torch.manual_seed(301)
    student, teacher = Expert(0.4, process), Expert(0.8, process)
    args = TrainerArgs(distill_method="step_distill", use_dual_model=True, use_ema=ema,
        parallel_mode=mode, mixed_precision="no", report_to="none", output_dir=str(path),
        gradient_accumulation_steps=gas, max_train_steps=steps, resume_from=resume,
        num_workers=0, batch_size=1, gradient_checkpointing=False,
        log_every=100, save_every=100, warmup_steps=0, learning_rate=0.02, lr_scheduler="constant")
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.02)
    result = StepDistillTrainer(args=args, student_model=student, teacher_model=teacher,
        optimizer=optimizer, lr_scheduler=None, device=torch.device("cpu"),
        train_dataloader=[{"latents": torch.randn(1, 2, 2, 2, 2)} for _ in range(3)])
    counter = [0]
    def alternate(size):
        value = result.distill_timesteps[0 if counter[0] % 2 == 0 else -1]
        counter[0] += 1
        return value.expand(size)
    result._sample_timesteps = alternate
    torch.manual_seed(83)
    return result


def test_gas2_crosses_both_experts_and_steps_each_parameter_once(tmp_path):
    trainer = construct(tmp_path)
    trainer._initialize_data_iterator(resuming=False)
    trainer.train_step(trainer._next_batch())
    assert isinstance(trainer.student_model, NoiseRoutedDenoiser)
    assert trainer.student_high.scale != 0.4 and trainer.student_low.scale != 0.4
    assert {int(state["step"]) for state in trainer.optimizer.state.values()} == {1}
    assert set(trainer.ema.shadow) == {"high.scale", "low.scale"}


def test_deepspeed_engine_owns_both_experts_and_scaling(tmp_path):
    trainer = construct(tmp_path, mode="deepspeed", ema=False)
    model = trainer.student_model
    trainer.student_model = trainer._deepspeed_engine = Engine(model, trainer.optimizer)
    trainer.scaler = ForbiddenStudentScaler()
    trainer._initialize_data_iterator(resuming=False)
    trainer.train_step(trainer._next_batch())
    assert trainer._deepspeed_engine.backward_calls == 2 and trainer._deepspeed_engine.updates == 1
    assert model.high.scale != 0.4 and model.low.scale != 0.4


@pytest.mark.parametrize("process", ["flow", "vp"])
def test_non1000_native_flow_and_vp_boundary_matches_old_step_routing(tmp_path, process):
    trainer = construct(tmp_path, process=process)
    router = trainer.student_model
    with torch.no_grad():
        router.high.scale.fill_(1)
        router.low.scale.fill_(2)
    timesteps = trainer.distill_timesteps
    expected = torch.where(timesteps >= timesteps[trainer.boundary_step_index], 1.0, 2.0)
    prediction = router(torch.ones(4, 1), timesteps).flatten()
    torch.testing.assert_close(prediction, expected)
    assert trainer.num_train_timesteps == 20


def test_unified_gas2_resume_has_exact_both_expert_and_ema_state(tmp_path):
    full = construct(tmp_path / "full")
    full.train()
    rng = torch.rand(4)
    partial = construct(tmp_path / "partial", steps=2)
    partial.train()
    resumed = construct(tmp_path / "resumed", resume=str(tmp_path / "partial" / "checkpoint-2"))
    resumed.train()
    for attribute in ("student_model", "optimizer", "lr_scheduler", "ema"):
        assert_nested_equal(getattr(full, attribute).state_dict(), getattr(resumed, attribute).state_dict())
    torch.testing.assert_close(torch.rand(4), rng, rtol=0, atol=0)
    assert not (tmp_path / "full" / "checkpoint-4" / "student_low.pt").exists()


def test_legacy_high_plus_low_checkpoint_migrates_and_initializes_low_ema(tmp_path):
    trainer = construct(tmp_path / "current", gas=1)
    trainer._initialize_data_iterator(resuming=False)
    trainer.train_step(trainer._next_batch())
    trainer.global_step = 1
    trainer.save_checkpoint(1, str(tmp_path / "current"))
    state = torch.load(tmp_path / "current" / "checkpoint-1" / "trainer_state.pt", weights_only=False)
    # Reproduce the original DDP checkpoint schema (no low EMA existed).
    low = copy.deepcopy(trainer.student_low.state_dict())
    state["student_model"] = {k[len("high."):]: v for k, v in state["student_model"].items() if k.startswith("high.")}
    state["args"].pop("dual_student_layout")
    state["ema"]["shadow"] = {k[len("high."):]: v for k, v in state["ema"]["shadow"].items() if k.startswith("high.")}
    directory = tmp_path / "legacy"
    directory.mkdir()
    torch.save(state, directory / "trainer_state.pt")
    torch.save(low, directory / "student_low.pt")
    restored = construct(tmp_path / "restored", gas=1)
    restored.load_checkpoint(str(directory))
    assert_nested_equal(restored.student_model.state_dict(), trainer.student_model.state_dict())
    assert_nested_equal(restored.optimizer.state_dict(), trainer.optimizer.state_dict())
    torch.testing.assert_close(restored.ema.shadow["low.scale"], low["scale"])
    state["args"]["dual_student_layout"] = "unknown_v99"
    torch.save(state, directory / "trainer_state.pt")
    with pytest.raises(RuntimeError, match="Unknown dual_student_layout"):
        restored.load_checkpoint(str(directory))


def test_unified_resume_rejects_missing_routing_metadata_before_loading_weights(tmp_path):
    trainer = construct(tmp_path / "source")
    trainer.save_checkpoint(0, str(tmp_path / "source"))
    # Create a deliberately incomplete copy without deleting user artifacts.
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    state = torch.load(tmp_path / "source" / "checkpoint-0" / "trainer_state.pt", weights_only=False)
    torch.save(state, incomplete / "trainer_state.pt")
    restored = construct(tmp_path / "restored")
    before = copy.deepcopy(restored.student_model.state_dict())
    with pytest.raises(RuntimeError, match="routing metadata"):
        restored.load_checkpoint(str(incomplete))
    assert_nested_equal(restored.student_model.state_dict(), before)


@pytest.mark.parametrize("strategy", ["ddp", "fsdp", "zero1", "zero2", "zero3"])
@pytest.mark.parametrize("precision", ["no", "fp16", "bf16"])
def test_two_rank_unified_dual_gas2_checkpoint_qualification(tmp_path, strategy, precision):
    cpu = strategy == "ddp" and precision == "no"
    if not cpu and torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    if strategy.startswith("zero") and importlib.util.find_spec("deepspeed") is None:
        pytest.skip("DeepSpeed package is required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
        "--rdzv-backend=static", "--master-addr=127.0.0.1", f"--master-port={port}",
        "-m", "tools.check_distributed_training", "--device", "cpu" if cpu else "cuda",
        "--methods", "step_distill", "--dual-student", "--gradient-accumulation-steps", "2",
        "--mixed-precision", precision, "--parallel", "deepspeed" if strategy.startswith("zero") else strategy,
        "--output-dir", str(tmp_path / "qualification")]
    if strategy.startswith("zero"):
        command += ["--zero-stage", strategy[-1]]
    env = dict(os.environ, OMP_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                               capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
