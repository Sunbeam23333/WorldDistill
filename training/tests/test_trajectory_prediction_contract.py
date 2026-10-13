import pytest
import torch

from training.trainer_args import TrainerArgs
from training.trainers import TRAINER_REGISTRY


class Flow(torch.nn.Module):
    training_noise_process = "flow"

    def __init__(self, prediction):
        super().__init__()
        self.training_prediction_type = prediction
        self.weight = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, hidden_states, timestep, **kwargs):
        return hidden_states * self.weight


@pytest.mark.parametrize("method", ["progressive_distill", "stream_distill", "consistency_distill",
                                    "context_forcing", "dmd_distill", "adversarial_distill", "mean_flow_distill"])
def test_velocity_trajectory_requires_velocity_teacher(method, tmp_path):
    teacher, student = Flow("epsilon"), Flow("velocity")
    if method == "mean_flow_distill":
        from training.utils.meanflow import MeanFlowTimeAdapter
        student = MeanFlowTimeAdapter(student, latent_channels=2)
    args = TrainerArgs(distill_method=method, output_dir=str(tmp_path), mixed_precision="no", report_to="none")
    with pytest.raises(ValueError, match="teacher_prediction_type"):
        TRAINER_REGISTRY[method](args=args, teacher_model=teacher, student_model=student,
            optimizer=torch.optim.AdamW(student.parameters()), lr_scheduler=None,
            train_dataloader=[], device=torch.device("cpu"))
