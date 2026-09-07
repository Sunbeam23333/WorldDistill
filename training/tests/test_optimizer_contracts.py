from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from training.trainer_args import TrainerArgs
from training.trainers.step_distill_trainer import StepDistillTrainer
from training.trainers.adversarial_distill_trainer import AdversarialDistillTrainer
from training.trainers.dmd_distill_trainer import DMDDistillTrainer
from training.utils.optimizers import build_optimizer
from training.utils.checkpoint_io import load_model_state_file


class OptimizerContractTests(unittest.TestCase):
    def test_muon_roundtrips_nested_adamw_momentum_and_tracks_lr(self) -> None:
        model = nn.Linear(3, 2)
        optimizer = build_optimizer(model, optimizer_type="muon", lr=0.02)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()

        state = optimizer.state_dict()
        self.assertIn("adamw_optimizer", state)
        self.assertTrue(state["adamw_optimizer"]["state"])

        restored_model = nn.Linear(3, 2)
        restored = build_optimizer(restored_model, optimizer_type="muon", lr=0.02)
        restored.load_state_dict(state)
        self.assertEqual(
            len(restored.adamw_optimizer.state),
            len(optimizer.adamw_optimizer.state),
        )

        adamw_group = next(group for group in restored.param_groups if group["type"] == "adamw")
        adamw_group["lr"] = 1.25e-4
        restored_model(torch.ones(1, 3)).sum().backward()
        restored.step()
        self.assertEqual(restored.adamw_optimizer.param_groups[0]["lr"], 1.25e-4)

    def test_dual_model_rebuilds_optimizer_scheduler_for_both_students(self) -> None:
        teacher = nn.Linear(2, 2)
        student = nn.Linear(2, 2)
        original_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        original_scheduler = torch.optim.lr_scheduler.LambdaLR(original_optimizer, lambda _: 1.0)
        args = TrainerArgs(
            distill_method="step_distill",
            use_dual_model=True,
            optimizer="adamw",
            mixed_precision="no",
            report_to="none",
            max_train_steps=2,
        )
        trainer = StepDistillTrainer(
            args=args,
            teacher_model=teacher,
            student_model=student,
            optimizer=original_optimizer,
            lr_scheduler=original_scheduler,
            train_dataloader=[],
            device=torch.device("cpu"),
        )

        optimized_ids = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        expected_ids = {id(parameter) for parameter in trainer._trainable_parameters()}
        self.assertEqual(optimized_ids, expected_ids)
        self.assertIs(trainer.lr_scheduler.optimizer, trainer.optimizer)

    def test_muon_and_dual_model_reject_unsupported_sharded_modes(self) -> None:
        for parallel_mode in ("fsdp", "deepspeed"):
            with self.subTest(kind="muon", parallel_mode=parallel_mode):
                args = TrainerArgs(
                    optimizer="muon",
                    parallel_mode=parallel_mode,
                    mixed_precision="no",
                    report_to="none",
                )
                with self.assertRaisesRegex(ValueError, "serial/DDP"):
                    StepDistillTrainer(
                        args=args,
                        teacher_model=nn.Linear(2, 2),
                        student_model=nn.Linear(2, 2),
                        optimizer=torch.optim.AdamW(nn.Linear(2, 2).parameters()),
                        lr_scheduler=None,
                        train_dataloader=[],
                        device=torch.device("cpu"),
                    )

            with self.subTest(kind="dual", parallel_mode=parallel_mode):
                teacher = nn.Linear(2, 2)
                student = nn.Linear(2, 2)
                optimizer = torch.optim.AdamW(student.parameters())
                args = TrainerArgs(
                    optimizer="adamw",
                    use_dual_model=True,
                    parallel_mode=parallel_mode,
                    mixed_precision="no",
                    report_to="none",
                )
                with self.assertRaisesRegex(ValueError, "Dual high/low-noise"):
                    StepDistillTrainer(
                        args=args,
                        teacher_model=teacher,
                        student_model=student,
                        optimizer=optimizer,
                        lr_scheduler=None,
                        train_dataloader=[],
                        device=torch.device("cpu"),
                    )

    def test_fsdp_fp16_fails_before_unscaled_training(self) -> None:
        student = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "ShardedGradScaler"):
            StepDistillTrainer(
                args=TrainerArgs(
                    parallel_mode="fsdp",
                    mixed_precision="fp16",
                    report_to="none",
                ),
                teacher_model=nn.Linear(2, 2),
                student_model=student,
                optimizer=torch.optim.AdamW(student.parameters()),
                lr_scheduler=None,
                train_dataloader=[],
                device=torch.device("cpu"),
            )

    def test_sequence_parallel_requires_model_layer_adapter(self) -> None:
        student = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "model-layer sequence-parallel adapters"):
            StepDistillTrainer(
                args=TrainerArgs(
                    sp_size=2,
                    mixed_precision="no",
                    report_to="none",
                ),
                teacher_model=nn.Linear(2, 2),
                student_model=student,
                optimizer=torch.optim.AdamW(student.parameters()),
                lr_scheduler=None,
                train_dataloader=[],
                device=torch.device("cpu"),
            )

    def test_auxiliary_optimizer_trainers_reject_unscaled_fp16(self) -> None:
        from training.trainers.adversarial_distill_trainer import AdversarialDistillTrainer
        from training.trainers.dmd_distill_trainer import DMDDistillTrainer

        for trainer_cls, method in (
            (AdversarialDistillTrainer, "adversarial_distill"),
            (DMDDistillTrainer, "dmd_distill"),
        ):
            with self.subTest(trainer=trainer_cls.__name__):
                student = nn.Linear(2, 2)
                with self.assertRaisesRegex(ValueError, "GradScaler"):
                    trainer_cls(
                        args=TrainerArgs(
                            distill_method=method,
                            mixed_precision="fp16",
                            report_to="none",
                        ),
                        teacher_model=nn.Linear(2, 2),
                        student_model=student,
                        optimizer=torch.optim.AdamW(student.parameters()),
                        lr_scheduler=None,
                        train_dataloader=[],
                        device=torch.device("cpu"),
                    )

    def test_deepspeed_accumulation_steps_once_per_microbatch(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.backward_values = []
                self.step_calls = 0

            def backward(self, loss) -> None:
                self.backward_values.append(float(loss.item()))

            def step(self) -> None:
                self.step_calls += 1

            def get_global_grad_norm(self):
                return 0.0

            def get_lr(self):
                return [1e-3]

        student = nn.Linear(2, 2)
        trainer = StepDistillTrainer(
            args=TrainerArgs(
                gradient_accumulation_steps=3,
                mixed_precision="no",
                report_to="none",
            ),
            teacher_model=nn.Linear(2, 2),
            student_model=student,
            optimizer=torch.optim.AdamW(student.parameters()),
            lr_scheduler=None,
            train_dataloader=[],
            device=torch.device("cpu"),
        )
        engine = FakeEngine()
        trainer._deepspeed_engine = engine
        trainer._data_iter = iter(
            [{"value": torch.tensor(6.0)}, {"value": torch.tensor(9.0)}]
        )
        trainer._forward_and_loss = lambda batch: batch["value"]

        metrics = trainer._train_step_deepspeed({"value": torch.tensor(3.0)})

        self.assertEqual(engine.backward_values, [3.0, 6.0, 9.0])
        self.assertEqual(engine.step_calls, 3)
        self.assertEqual(metrics["loss"], 6.0)

    def test_student_state_loader_fails_for_missing_or_directory_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            with self.assertRaisesRegex(IsADirectoryError, "must be a file"):
                load_model_state_file(str(directory))
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                load_model_state_file(str(directory / "missing.pt"))

    def test_dual_student_rejects_missing_low_noise_checkpoint(self) -> None:
        student = nn.Linear(2, 2)
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            StepDistillTrainer(
                args=TrainerArgs(
                    use_dual_model=True,
                    student_low_model="/definitely/missing/student-low.pt",
                    mixed_precision="no",
                    report_to="none",
                ),
                teacher_model=nn.Linear(2, 2),
                student_model=student,
                optimizer=torch.optim.AdamW(student.parameters()),
                lr_scheduler=None,
                train_dataloader=[],
                device=torch.device("cpu"),
            )

    def test_auxiliary_trainers_reject_ambiguous_accumulation_cadence(self) -> None:
        for trainer_cls, method in (
            (AdversarialDistillTrainer, "adversarial_distill"),
            (DMDDistillTrainer, "dmd_distill"),
        ):
            student = nn.Linear(2, 2)
            with self.subTest(method=method), self.assertRaisesRegex(
                ValueError, "gradient_accumulation_steps=1"
            ):
                trainer_cls(
                    args=TrainerArgs(
                        distill_method=method,
                        gradient_accumulation_steps=2,
                        mixed_precision="no",
                        report_to="none",
                    ),
                    teacher_model=nn.Linear(2, 2),
                    student_model=student,
                    optimizer=torch.optim.AdamW(student.parameters()),
                    lr_scheduler=None,
                    train_dataloader=[],
                    device=torch.device("cpu"),
                )

    def test_dmd2_enables_gan_path_by_variant(self) -> None:
        student = nn.Linear(2, 2)
        trainer = DMDDistillTrainer(
            args=TrainerArgs(
                distill_method="dmd_distill",
                dmd_variant="dmd2",
                dmd_use_gan=False,
                dmd_latent_channels=2,
                dmd_disc_hidden_dim=8,
                dmd_disc_num_blocks=1,
                mixed_precision="no",
                report_to="none",
            ),
            teacher_model=nn.Linear(2, 2),
            student_model=student,
            optimizer=torch.optim.AdamW(student.parameters()),
            lr_scheduler=None,
            train_dataloader=[],
            device=torch.device("cpu"),
        )

        self.assertTrue(trainer.use_gan)
        self.assertTrue(trainer.enable_fake_score)
