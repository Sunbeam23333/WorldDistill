import pytest
import torch

from training.model_adapter import DiffusersTrainingAdapter, NoiseRoutedDenoiser
from training.trainer_args import TrainerArgs
from training.trainers import TRAINER_REGISTRY


@pytest.mark.parametrize("role", ["teacher", "student"])
def test_nonstandard_router_time_scale_is_not_silently_trained(role):
    teacher = NoiseRoutedDenoiser(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), 0.5)
    student = NoiseRoutedDenoiser(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), 0.5)
    {"teacher": teacher, "student": student}[role].num_train_timesteps = 500
    with pytest.raises(ValueError, match=f"{role} declares num_train_timesteps=500"):
        TRAINER_REGISTRY["step_distill"](
            args=TrainerArgs(mixed_precision="no", report_to="none"),
            teacher_model=teacher, student_model=student,
            optimizer=torch.optim.AdamW(student.parameters()), lr_scheduler=None,
            train_dataloader=[], device=torch.device("cpu"),
        )


class CogVideoXTransformer3DModel(torch.nn.Linear):
    """Small architecture-name fixture; no pretrained model download."""


@pytest.mark.parametrize("role", ["teacher", "student"])
def test_cogvideox_vp_process_is_rejected_by_flow_trainers(role):
    models = {"teacher": torch.nn.Linear(2, 2), "student": torch.nn.Linear(2, 2)}
    models[role] = DiffusersTrainingAdapter(CogVideoXTransformer3DModel(2, 2))
    assert models[role].training_noise_process == "vp"
    with pytest.raises(ValueError, match=f"{role} declares training_noise_process='vp'"):
        TRAINER_REGISTRY["step_distill"](
            args=TrainerArgs(mixed_precision="no", report_to="none"),
            teacher_model=models["teacher"], student_model=models["student"],
            optimizer=torch.optim.AdamW(models["student"].parameters()), lr_scheduler=None,
            train_dataloader=[], device=torch.device("cpu"),
        )


def test_unknown_denoiser_family_is_not_newly_blocked_by_noise_guard():
    teacher = DiffusersTrainingAdapter(torch.nn.Linear(2, 2))
    student = DiffusersTrainingAdapter(torch.nn.Linear(2, 2))
    assert getattr(teacher, "training_noise_process", "flow") == "flow"
    trainer = TRAINER_REGISTRY["step_distill"](
        args=TrainerArgs(mixed_precision="no", report_to="none"),
        teacher_model=teacher, student_model=student,
        optimizer=torch.optim.AdamW(student.parameters()), lr_scheduler=None,
        train_dataloader=[], device=torch.device("cpu"),
    )
    assert trainer.num_train_timesteps == 1000


@pytest.mark.parametrize("scheduler_name", ["CogVideoXDDIMScheduler", "CogVideoXDPMScheduler"])
def test_real_cogvideox_scheduler_noise_is_vp_not_flow(scheduler_name):
    diffusers = pytest.importorskip("diffusers")
    scheduler = getattr(diffusers, scheduler_name)()
    timestep = torch.tensor([500])
    clean, noise = torch.ones(1, 1), torch.zeros(1, 1)
    actual = scheduler.add_noise(clean, noise, timestep)
    vp = scheduler.alphas_cumprod[timestep].to(clean.dtype).sqrt().reshape(1, 1)
    flow = (1 - timestep.float() / scheduler.config.num_train_timesteps).reshape(1, 1)
    torch.testing.assert_close(actual, vp)
    assert not torch.allclose(actual, flow)
