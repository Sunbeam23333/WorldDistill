from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

if _TORCH_IMPORT_ERROR is None:
    from training.trainers.base_distill_trainer import BaseDistillTrainer, EMAModel


    class _CheckpointTrainer(BaseDistillTrainer):
        def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
            del teacher_output, batch, timesteps
            return student_output.sum()

        def prepare_teacher_input(self, batch, noisy_latents, timesteps):
            del batch
            return {"hidden_states": noisy_latents, "timestep": timesteps}


    class _Stateful:
        def __init__(self, state=None, load_error: Exception | None = None):
            self.state = {} if state is None else state
            self.load_error = load_error

        def state_dict(self):
            return self.state

        def load_state_dict(self, state):
            if self.load_error is not None:
                raise self.load_error
            self.state = state


@unittest.skipIf(_TORCH_IMPORT_ERROR is not None, f"torch is unavailable: {_TORCH_IMPORT_ERROR}")
class CheckpointContractTests(unittest.TestCase):
    def _bare_trainer(self):
        trainer = object.__new__(_CheckpointTrainer)
        trainer.device = torch.device("cpu")
        trainer.args = SimpleNamespace(deepspeed_stage=2)
        trainer.parallel_mode = "ddp"
        trainer._deepspeed_engine = None
        trainer.ema = None
        trainer.global_step = 0
        trainer.epoch = 0
        trainer.best_loss = float("inf")
        return trainer

    def _standard_trainer(self):
        trainer = self._bare_trainer()
        trainer.student_model = nn.Linear(2, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.student_model.parameters(), lr=1e-3)
        trainer.lr_scheduler = _Stateful({"last_epoch": 3})
        trainer.scaler = _Stateful({"scale": 1024.0})
        return trainer

    @staticmethod
    def _standard_state(trainer):
        return {
            "step": 9,
            "epoch": 2,
            "student_model": trainer.student_model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "lr_scheduler": {"last_epoch": 9},
            "scaler": {"scale": 512.0},
            "best_loss": 0.25,
        }

    def test_rank0_io_failure_is_broadcast_before_raise(self):
        trainer = self._bare_trainer()
        broadcast = Mock()

        with (
            patch.object(trainer, "_checkpoint_distributed", return_value=True),
            patch.object(trainer, "_checkpoint_is_main_process", return_value=True),
            patch(
                "training.trainers.base_distill_trainer.dist.broadcast_object_list",
                broadcast,
            ),
            self.assertRaisesRegex(RuntimeError, "EOFError: truncated"),
        ):
            trainer._run_rank0_or_raise(
                lambda: (_ for _ in ()).throw(EOFError("truncated")),
                "Load checkpoint",
            )

        payload = broadcast.call_args.args[0]
        self.assertIn("rank 0", payload[0])

    def test_nonzero_rank_receives_rank0_failure_without_running_action(self):
        trainer = self._bare_trainer()
        action = Mock()

        def _receive_error(payload, src):
            self.assertEqual(src, 0)
            payload[0] = "rank-zero disk failure"

        with (
            patch.object(trainer, "_checkpoint_distributed", return_value=True),
            patch.object(trainer, "_checkpoint_is_main_process", return_value=False),
            patch(
                "training.trainers.base_distill_trainer.dist.broadcast_object_list",
                side_effect=_receive_error,
            ),
            self.assertRaisesRegex(RuntimeError, "rank-zero disk failure"),
        ):
            trainer._run_rank0_or_raise(action, "Load checkpoint")

        action.assert_not_called()

    def test_all_rank_helper_propagates_a_remote_failure(self):
        trainer = self._bare_trainer()

        def _gather(errors, local_error):
            self.assertIsNone(local_error)
            errors[:] = [None, "Restore failed on rank 1: ValueError: corrupt"]

        with (
            patch.object(trainer, "_checkpoint_distributed", return_value=True),
            patch.object(trainer, "_checkpoint_rank", return_value=0),
            patch("training.trainers.base_distill_trainer.dist.get_world_size", return_value=2),
            patch(
                "training.trainers.base_distill_trainer.dist.all_gather_object",
                side_effect=_gather,
            ),
            self.assertRaisesRegex(RuntimeError, "failed on rank 1"),
        ):
            trainer._run_all_ranks_or_raise(lambda: "ok", "Restore")

    def test_atomic_save_preserves_old_checkpoint_after_partial_write(self):
        trainer = self._bare_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "state.pt"
            trainer._atomic_save_rank0(
                str(checkpoint_path),
                lambda: {"version": "old"},
                "Write checkpoint",
            )

            def _partial_write_then_fail(state, path):
                del state
                Path(path).write_bytes(b"partial")
                raise OSError("disk full")

            with (
                patch(
                    "training.trainers.base_distill_trainer.torch.save",
                    side_effect=_partial_write_then_fail,
                ),
                self.assertRaisesRegex(RuntimeError, "disk full"),
            ):
                trainer._atomic_save_rank0(
                    str(checkpoint_path),
                    lambda: {"version": "new"},
                    "Write checkpoint",
                )

            self.assertEqual(
                torch.load(checkpoint_path, map_location="cpu", weights_only=False),
                {"version": "old"},
            )
            self.assertEqual(list(Path(tmpdir).glob("state.pt.tmp-rank0-*")), [])

    def test_missing_and_incomplete_standard_resume_are_fatal(self):
        trainer = self._standard_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.pt"
            with self.assertRaisesRegex(RuntimeError, "Checkpoint file not found"):
                trainer.load_checkpoint(str(missing))

            incomplete = Path(tmpdir) / "incomplete.pt"
            torch.save({"step": 1}, incomplete)
            with self.assertRaisesRegex(RuntimeError, "missing required keys"):
                trainer.load_checkpoint(str(incomplete))

    def test_optimizer_and_scheduler_restore_failures_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer_trainer = self._standard_trainer()
            optimizer_state = self._standard_state(optimizer_trainer)
            optimizer_state["optimizer"] = {}
            optimizer_path = Path(tmpdir) / "bad_optimizer.pt"
            torch.save(optimizer_state, optimizer_path)
            with self.assertRaisesRegex(RuntimeError, "Restore DDP checkpoint"):
                optimizer_trainer.load_checkpoint(str(optimizer_path))

            scheduler_trainer = self._standard_trainer()
            scheduler_state = self._standard_state(scheduler_trainer)
            scheduler_path = Path(tmpdir) / "bad_scheduler.pt"
            torch.save(scheduler_state, scheduler_path)
            scheduler_trainer.lr_scheduler = _Stateful(
                load_error=ValueError("scheduler state rejected")
            )
            with self.assertRaisesRegex(RuntimeError, "scheduler state rejected"):
                scheduler_trainer.load_checkpoint(str(scheduler_path))

    def test_ema_restore_is_strict_and_preserves_target_device_and_dtype(self):
        model = nn.Linear(2, 1, dtype=torch.float32)
        ema = EMAModel(model)
        state = ema.state_dict()
        state["shadow"] = type(state["shadow"])(
            (name, value.to(dtype=torch.float64)) for name, value in state["shadow"].items()
        )
        state["step_count"] = 11
        ema.load_state_dict(state)
        self.assertEqual(ema.step_count, 11)
        self.assertTrue(all(value.dtype == torch.float32 for value in ema.shadow.values()))

        broken = ema.state_dict()
        broken["shadow"] = {}
        with self.assertRaisesRegex(KeyError, "parameter names do not match"):
            ema.load_state_dict(broken)

    def test_deepspeed_zero2_saves_and_restores_base_ema(self):
        trainer = self._bare_trainer()
        trainer.args = SimpleNamespace(deepspeed_stage=2)
        trainer.epoch = 4
        trainer.best_loss = 0.125
        trainer.ema = EMAModel(nn.Linear(2, 1))
        trainer.ema.step_count = 8

        class _FakeEngine:
            def __init__(self):
                self.saved_client_state = None
                self.load_result = None

            def save_checkpoint(self, path, tag, client_state):
                del path, tag
                self.saved_client_state = client_state
                return True

            def load_checkpoint(self, path):
                del path
                return self.load_result

        engine = _FakeEngine()
        trainer._deepspeed_engine = engine
        trainer._save_checkpoint_deepspeed(12, "/unused")
        self.assertEqual(engine.saved_client_state["ema"]["step_count"], 8)

        restored_ema = EMAModel(nn.Linear(2, 1))
        trainer.ema = restored_ema
        engine.load_result = ("/loaded/step-12", engine.saved_client_state)
        trainer._load_checkpoint_deepspeed("/unused")
        self.assertEqual(trainer.global_step, 12)
        self.assertEqual(trainer.epoch, 4)
        self.assertEqual(trainer.best_loss, 0.125)
        self.assertEqual(trainer.ema.step_count, 8)

    def test_deepspeed_rejects_missing_load_path_or_client_state(self):
        trainer = self._bare_trainer()
        trainer.args = SimpleNamespace(deepspeed_stage=2)

        engine = Mock()
        trainer._deepspeed_engine = engine
        for result, message in (
            ((None, {}), "could not load"),
            (("/loaded", None), "client_state must be a dict"),
            (("/loaded", {}), "missing required keys"),
        ):
            with self.subTest(result=result):
                engine.load_checkpoint.return_value = result
                with self.assertRaisesRegex(RuntimeError, message):
                    trainer._load_checkpoint_deepspeed("/checkpoint")

        engine.save_checkpoint.return_value = False
        with self.assertRaisesRegex(RuntimeError, "returned False"):
            trainer._save_checkpoint_deepspeed(1, "/checkpoint")

    def test_fsdp_uses_full_cpu_state_for_save_and_rank0_broadcast_for_load(self):
        trainer = self._bare_trainer()
        trainer.student_model = object()
        trainer.optimizer = object()
        trainer.lr_scheduler = _Stateful({"last_epoch": 5})
        trainer.scaler = _Stateful({"scale": 1.0})
        trainer.args = SimpleNamespace(deepspeed_stage=2, marker="test")
        trainer.epoch = 3
        trainer.best_loss = 0.5

        model_state = {"weight": torch.tensor([1.0])}
        optimizer_state = {"state": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "torch.distributed.checkpoint.state_dict.get_state_dict",
                return_value=(model_state, optimizer_state),
            ) as get_state:
                trainer._save_checkpoint_fsdp(7, tmpdir)

            save_options = get_state.call_args.kwargs["options"]
            self.assertTrue(save_options.full_state_dict)
            self.assertTrue(save_options.cpu_offload)
            saved = torch.load(
                Path(tmpdir) / "trainer_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(saved["student_model"]["weight"].device.type, "cpu")

            with patch(
                "torch.distributed.checkpoint.state_dict.set_state_dict",
            ) as set_state:
                trainer._load_checkpoint_fsdp(tmpdir)

            load_options = set_state.call_args.kwargs["options"]
            self.assertTrue(load_options.full_state_dict)
            self.assertTrue(load_options.broadcast_from_rank0)
            self.assertTrue(load_options.strict)
            restored_model_state = set_state.call_args.kwargs["model_state_dict"]
            restored_optimizer_state = set_state.call_args.kwargs["optim_state_dict"]
            self.assertEqual(set(restored_model_state), {"weight"})
            self.assertTrue(torch.equal(restored_model_state["weight"], model_state["weight"]))
            self.assertEqual(restored_optimizer_state, optimizer_state)

    def test_ema_and_consistency_sharding_contract_matrix(self):
        rejected = (
            ("fsdp", 2, True, "step_distill"),
            ("deepspeed", 3, True, "step_distill"),
            ("fsdp", 2, False, "consistency_distill"),
            ("deepspeed", 3, False, "consistency_distill"),
        )
        allowed = (
            ("ddp", 2, True, "step_distill"),
            ("deepspeed", 1, True, "step_distill"),
            ("deepspeed", 2, True, "step_distill"),
            ("ddp", 2, False, "consistency_distill"),
        )

        for parallel_mode, stage, use_ema, method in rejected:
            with self.subTest(contract="rejected", parallel_mode=parallel_mode, stage=stage):
                trainer = self._bare_trainer()
                trainer.parallel_mode = parallel_mode
                trainer.args = SimpleNamespace(
                    deepspeed_stage=stage,
                    use_ema=use_ema,
                    distill_method=method,
                )
                with self.assertRaises(ValueError):
                    trainer._validate_ema_parallel_contract()

        for parallel_mode, stage, use_ema, method in allowed:
            with self.subTest(contract="allowed", parallel_mode=parallel_mode, stage=stage):
                trainer = self._bare_trainer()
                trainer.parallel_mode = parallel_mode
                trainer.args = SimpleNamespace(
                    deepspeed_stage=stage,
                    use_ema=use_ema,
                    distill_method=method,
                )
                trainer._validate_ema_parallel_contract()

    def test_stateful_hook_runs_before_checkpoint(self):
        from training.trainer_args import TrainerArgs

        events = []

        class _EventTrainer(_CheckpointTrainer):
            def train_step(self, batch):
                del batch
                return {
                    "loss": 0.0,
                    "grad_norm": 0.0,
                    "lr": 1e-3,
                    "optimizer_skipped": 0.0,
                }

            def on_train_step_end(self, metrics):
                del metrics
                events.append("hook")

            def save_checkpoint(self, step, output_dir):
                del step, output_dir
                events.append("checkpoint")

        with tempfile.TemporaryDirectory() as tmpdir:
            teacher = nn.Linear(2, 1)
            student = nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            args = TrainerArgs(
                output_dir=tmpdir,
                max_train_steps=1,
                mixed_precision="no",
                log_every=100,
                save_every=1,
                report_to="none",
            )
            trainer = _EventTrainer(
                args=args,
                teacher_model=teacher,
                student_model=student,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                train_dataloader=[{"latents": torch.ones(1, 2)}],
                device=torch.device("cpu"),
            )
            trainer.train()

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[:2], ["hook", "checkpoint"])

    def test_stateful_subclass_checkpoint_roundtrips(self):
        from training.trainer_args import TrainerArgs
        from training.trainers.adversarial_distill_trainer import AdversarialDistillTrainer
        from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer
        from training.trainers.context_forcing_trainer import ContextForcingTrainer
        from training.trainers.dmd_distill_trainer import DMDDistillTrainer
        from training.trainers.progressive_distill_trainer import ProgressiveDistillTrainer
        from training.trainers.step_distill_trainer import StepDistillTrainer

        class _TinyFlow(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))

            def forward(self, hidden_states, timestep, **kwargs):
                del timestep, kwargs
                return hidden_states * self.weight

        def _make_trainer(trainer_cls, output_dir, method, **overrides):
            args = TrainerArgs(
                distill_method=method,
                output_dir=str(output_dir),
                mixed_precision="no",
                report_to="none",
                num_workers=0,
                **overrides,
            )
            teacher = _TinyFlow(2.0)
            student = _TinyFlow(1.0)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            return trainer_cls(
                args=args,
                teacher_model=teacher,
                student_model=student,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                train_dataloader=[{"latents": torch.ones(1, 1)}],
                device=torch.device("cpu"),
            )

        cases = (
            (StepDistillTrainer, "step_distill", {"use_dual_model": True}),
            (ConsistencyDistillTrainer, "consistency_distill", {}),
            (
                DMDDistillTrainer,
                "dmd_distill",
                {"dmd_variant": "dmd", "dmd_use_ema_fake_score": True},
            ),
            (
                AdversarialDistillTrainer,
                "adversarial_distill",
                {
                    "adversarial_latent_channels": 2,
                    "adversarial_disc_hidden_dim": 8,
                    "adversarial_disc_num_blocks": 1,
                },
            ),
            (
                ProgressiveDistillTrainer,
                "progressive_distill",
                {"progressive_stages": [8, 4, 2]},
            ),
            (
                ContextForcingTrainer,
                "context_forcing",
                {"curriculum_stages": [2, 4], "num_frames": 4},
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for index, (trainer_cls, method, overrides) in enumerate(cases):
                with self.subTest(trainer=trainer_cls.__name__):
                    output_dir = Path(tmpdir) / str(index)
                    trainer = _make_trainer(trainer_cls, output_dir, method, **overrides)
                    if isinstance(trainer, ProgressiveDistillTrainer):
                        trainer.current_stage = 1
                        trainer.current_teacher_steps = 4
                        trainer.current_student_steps = 2
                        trainer._reset_stage_optimizer()
                    if isinstance(trainer, ContextForcingTrainer):
                        trainer.current_curriculum_stage = 1
                        trainer.current_num_frames = 4
                        trainer._curriculum_advanced = {1}

                    trainer.global_step = 3
                    trainer.save_checkpoint(3, str(output_dir))

                    restored = _make_trainer(trainer_cls, output_dir, method, **overrides)
                    restored.load_checkpoint(str(output_dir / "checkpoint-3"))
                    self.assertEqual(restored.global_step, 3)
                    if isinstance(restored, ProgressiveDistillTrainer):
                        self.assertEqual(restored.current_stage, 1)
                        self.assertEqual(restored.current_teacher_steps, 4)
                        self.assertEqual(restored.current_student_steps, 2)
                        self.assertEqual(
                            restored.lr_scheduler.lr_lambdas[0].keywords["total"],
                            restored.stage_steps,
                        )
                    if isinstance(restored, ContextForcingTrainer):
                        self.assertEqual(restored.current_curriculum_stage, 1)
                        self.assertEqual(restored.current_num_frames, 4)
                        self.assertEqual(restored._curriculum_advanced, {1})


if __name__ == "__main__":
    unittest.main()
