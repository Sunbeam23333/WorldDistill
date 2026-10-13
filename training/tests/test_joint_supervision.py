import pytest
import torch

from tools.check_distributed_training import TinyAVDenoiser
from training.trainer_args import TrainerArgs
from training.trainers.joint_av_distill_trainer import JointAVDistillTrainer


@pytest.mark.parametrize("loss_type", ["mse", "huber"])
def test_joint_supervision_respects_both_masks_and_loss_type(tmp_path, loss_type):
    student, teacher = TinyAVDenoiser(0.4), TinyAVDenoiser(0.8)
    args = TrainerArgs(distill_method="joint_av_distill", mixed_precision="no", report_to="none",
                       output_dir=str(tmp_path), loss_type=loss_type, huber_c=0.5)
    trainer = JointAVDistillTrainer(args=args, teacher_model=teacher, student_model=student,
        optimizer=torch.optim.AdamW(student.parameters()), lr_scheduler=None,
        train_dataloader=[], device=torch.device("cpu"))
    video = torch.full((1, 2, 2, 1, 1), 2., requires_grad=True)
    audio = torch.full((1, 2, 3), 2., requires_grad=True)
    batch = {"loss_mask": torch.tensor([0., 1.]).reshape(1, 1, 2, 1, 1),
             "audio_loss_mask": torch.tensor([0., 1.]).reshape(1, 2, 1)}
    actual = trainer.compute_distill_loss(
        {"sample": torch.zeros_like(video), "audio_sample": torch.zeros_like(audio)},
        {"sample": video, "audio_sample": audio}, batch, torch.tensor([500.]))
    expected = 8 if loss_type == "mse" else 1.75
    assert actual.item() == pytest.approx(expected)
    actual.backward()
    assert video.grad[:, :, 0].count_nonzero() == 0
    assert audio.grad[:, 0].count_nonzero() == 0
    assert video.grad[:, :, 1].abs().sum() > 0
    assert audio.grad[:, 1].abs().sum() > 0
