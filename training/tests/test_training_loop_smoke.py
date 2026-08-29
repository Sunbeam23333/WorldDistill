from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    DataLoader = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from training.trainer_args import TrainerArgs

import numpy as np

if _TORCH_IMPORT_ERROR is None:
    from training.trainers.step_distill_trainer import StepDistillTrainer


@unittest.skipIf(_TORCH_IMPORT_ERROR is not None, f"torch is unavailable: {_TORCH_IMPORT_ERROR}")
class TrainingLoopSmokeTests(unittest.TestCase):
    def test_step_distillation_runs_two_cpu_steps_and_resumes(self) -> None:
        class TinyFlow(nn.Module):
            def __init__(self, scale: float) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(scale))

            def forward(self, hidden_states, timestep, **kwargs):
                del timestep, kwargs
                return hidden_states * self.scale

        with tempfile.TemporaryDirectory() as tmpdir:
            teacher = TinyFlow(2.0)
            student = TinyFlow(1.0)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            dataloader = DataLoader(
                [{"latents": torch.ones(2, 3)}],
                batch_size=1,
                shuffle=False,
            )
            args = TrainerArgs(
                distill_method="step_distill",
                output_dir=tmpdir,
                max_train_steps=2,
                mixed_precision="no",
                num_workers=0,
                log_every=1,
                save_every=100,
                report_to="none",
                grad_skip_threshold=1_000.0,
            )
            trainer = StepDistillTrainer(
                args=args,
                teacher_model=teacher,
                student_model=student,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                train_dataloader=dataloader,
                device=torch.device("cpu"),
            )

            initial_scale = student.scale.detach().clone()
            trainer.train()
            checkpoint = Path(tmpdir) / "checkpoint-2" / "trainer_state.pt"

            expected_python_random = random.random()
            expected_numpy_random = float(np.random.rand())
            expected_torch_random = torch.rand(3)

            self.assertEqual(trainer.global_step, 2)
            self.assertFalse(torch.equal(student.scale.detach(), initial_scale))
            self.assertTrue(checkpoint.exists())

            restored_teacher = TinyFlow(2.0)
            restored_student = TinyFlow(0.0)
            restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=1e-2)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
            restored = StepDistillTrainer(
                args=args,
                teacher_model=restored_teacher,
                student_model=restored_student,
                optimizer=restored_optimizer,
                lr_scheduler=restored_scheduler,
                train_dataloader=dataloader,
                device=torch.device("cpu"),
            )
            restored.load_checkpoint(str(checkpoint.parent))

            self.assertEqual(restored.global_step, 2)
            self.assertEqual(restored.epoch, 1)
            self.assertEqual(restored._batches_in_epoch, 1)
            self.assertTrue(torch.allclose(restored_student.scale, student.scale))
            self.assertEqual(random.random(), expected_python_random)
            self.assertEqual(float(np.random.rand()), expected_numpy_random)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch_random))

            restored._initialize_data_iterator(resuming=True)
            next_batch = restored._next_batch()
            self.assertEqual(restored.epoch, 2)
            self.assertEqual(restored._batches_in_epoch, 1)
            self.assertTrue(torch.equal(next_batch["latents"], torch.ones(1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
