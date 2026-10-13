import pytest
import torch
from torch import nn
from training.trainer_args import TrainerArgs
from training.utils.noise_objective import NoiseObjective, resolve_noise_objectives


@pytest.mark.parametrize("process,kinds", [("flow", ["velocity", "epsilon", "x0"]),
                                         ("vp", ["epsilon", "v_prediction", "sample"])])
def test_prediction_conversions_recover_exact_clean_and_noise(process, kinds):
    alpha_bar = torch.linspace(0.99, 0.01, 37) if process == "vp" else None
    objective = NoiseObjective(process, 37, kinds[0], alpha_bar)
    t = torch.tensor([8, 28])
    x0, epsilon = torch.randn(2, 2, 3), torch.randn(2, 2, 3)
    noisy = objective.add_noise(x0, epsilon, t)
    alpha, sigma = objective.coefficients(t, x0.ndim)
    values = {"x0": x0, "sample": x0, "epsilon": epsilon, "velocity": epsilon - x0,
              "v_prediction": alpha * epsilon - sigma * x0}
    for source in kinds:
        restored_x0, restored_eps = objective.to_x0_epsilon(values[source], noisy, t, source)
        torch.testing.assert_close(restored_x0, x0)
        torch.testing.assert_close(restored_eps, epsilon)
        for target in kinds:
            torch.testing.assert_close(objective.convert_prediction(values[source], noisy, t, source, target), values[target])


@pytest.mark.parametrize("scheduler_name", ["CogVideoXDDIMScheduler", "CogVideoXDPMScheduler"])
def test_vp_noise_and_velocity_match_real_diffusers_scheduler(scheduler_name):
    diffusers = pytest.importorskip("diffusers")
    scheduler = getattr(diffusers, scheduler_name)(num_train_timesteps=37)
    objective = NoiseObjective("vp", 37, "v_prediction", scheduler.alphas_cumprod)
    clean, noise = torch.randn(2, 2, 3, 2, 2), torch.randn(2, 2, 3, 2, 2)
    t = torch.tensor([9, 30])
    actual = objective.add_noise(clean, noise, t)
    torch.testing.assert_close(actual, scheduler.add_noise(clean, noise, t))
    velocity = scheduler.get_velocity(clean, noise, t)
    x0, eps = objective.to_x0_epsilon(velocity, actual, t)
    torch.testing.assert_close(x0, clean)
    torch.testing.assert_close(eps, noise)


def test_schedule_resolution_non1000_and_explicit_mismatch():
    teacher, student = nn.Linear(2, 2), nn.Linear(2, 2)
    teacher.num_train_timesteps = student.num_train_timesteps = 37
    left, right = resolve_noise_objectives(TrainerArgs(), teacher, student)
    assert left.num_train_timesteps == right.num_train_timesteps == 37
    with pytest.raises(ValueError, match="disagrees"):
        resolve_noise_objectives(TrainerArgs(num_train_timesteps=1000), teacher, student)


@pytest.mark.parametrize("alphas", [[0.8, 0.9], [0.9, -0.1], [0.9, float("nan")]])
def test_vp_rejects_invalid_scheduler_coefficients(alphas):
    with pytest.raises(ValueError, match="alphas_cumprod"):
        NoiseObjective("vp", 2, "epsilon", torch.tensor(alphas))


def test_vp_missing_schedule_and_fractional_indices_fail_closed():
    with pytest.raises(ValueError, match="checkpoint scheduler"):
        NoiseObjective("vp", 2, "epsilon")
    objective = NoiseObjective("vp", 2, "epsilon", torch.tensor([0.9, 0.3]))
    with pytest.raises(ValueError, match="integer"):
        objective.coefficients(torch.tensor([0.5]), 2)


def test_native_vp_step_training_has_nonzero_gradients(tmp_path):
    from training.trainers.step_distill_trainer import StepDistillTrainer
    class TinyVP(nn.Module):
        training_noise_process = "vp"
        training_prediction_type = "v_prediction"
        num_train_timesteps = 37
        def __init__(self, scale):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(scale))
            self.register_buffer("training_alphas_cumprod", torch.linspace(0.99, 0.01, 37))
        def forward(self, hidden_states, timestep):
            return self.scale * hidden_states
    teacher, student = TinyVP(0.8), TinyVP(0.4)
    trainer = StepDistillTrainer(args=TrainerArgs(report_to="none", mixed_precision="no"),
        teacher_model=teacher, student_model=student, optimizer=torch.optim.AdamW(student.parameters()),
        lr_scheduler=None, train_dataloader=[], device=torch.device("cpu"))
    loss = trainer._forward_and_loss({"latents": torch.randn(2, 2, 3, 2, 2)})
    loss.backward()
    assert torch.isfinite(loss) and student.scale.grad.abs() > 0
    assert teacher.scale.grad is None
    assert trainer.distill_timesteps.tolist() == [36, 28, 18, 9]
