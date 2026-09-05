import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.env_compat import EXPECTED_TRANSFORMERS_VERSION
from training.model_catalog import resolve_model_metadata
from training.trainer_args import TrainerArgs, build_training_arg_parser, parse_training_args
from training.train_distill import _bucket_sampler_drop_last
from training.utils.model_output import extract_prediction_tensor

_EXPERIMENT_TRACKING_PATH = PROJECT_ROOT / "training" / "utils" / "experiment_tracking.py"
_spec = importlib.util.spec_from_file_location("worlddistill_experiment_tracking", _EXPERIMENT_TRACKING_PATH)
assert _spec is not None and _spec.loader is not None
_experiment_tracking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_experiment_tracking)
ExperimentTracker = _experiment_tracking.ExperimentTracker

_INFERENCE_MODEL_CATALOG_PATH = PROJECT_ROOT / "inference" / "lightx2v" / "utils" / "model_catalog.py"
_inference_spec = importlib.util.spec_from_file_location("worlddistill_inference_model_catalog", _INFERENCE_MODEL_CATALOG_PATH)
assert _inference_spec is not None and _inference_spec.loader is not None
_inference_model_catalog = importlib.util.module_from_spec(_inference_spec)
_inference_spec.loader.exec_module(_inference_model_catalog)
resolve_inference_metadata = _inference_model_catalog.resolve_model_metadata
resolve_default_inference_config = _inference_model_catalog.resolve_default_config_path

if _TORCH_IMPORT_ERROR is None:
    from training.runtime import build_runtime
    from training.runtime.distill_cache import MemoryDistillCache
    from training.runtime.fused_supervision import fused_masked_mse_loss
    from training.trainers.context_forcing_trainer import ContextForcingTrainer
    from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer
    from training.trainers.progressive_distill_trainer import ProgressiveDistillTrainer
    from training.trainers.step_distill_trainer import StepDistillTrainer

    class IdentityTeacher(nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            return latents * 2.0


    class IdentityStudent(nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

else:
    build_runtime = None
    fused_masked_mse_loss = None

    class IdentityTeacher:  # pragma: no cover - used only when torch is unavailable
        pass


    class IdentityStudent:  # pragma: no cover - used only when torch is unavailable
        pass


class PresetSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.step_preset = PROJECT_ROOT / "configs" / "distill_presets" / "step_distill_4step.json"
        self.context_preset = PROJECT_ROOT / "configs" / "distill_presets" / "context_forcing.json"
        self.runtime_preset = PROJECT_ROOT / "configs" / "distill_presets" / "world_model_runtime.json"

    def test_step_preset_enables_runtime_flags(self) -> None:
        args = TrainerArgs(distill_preset=str(self.step_preset))

        self.assertEqual(args.distill_method, "step_distill")
        self.assertTrue(args.use_dual_model)
        self.assertEqual(args.loss_type, "mse")
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertTrue(args.enable_runtime)
        self.assertTrue(args.runtime_enable_dpp)
        self.assertTrue(args.enable_fused_supervision_kernel)
        self.assertEqual(args.fused_supervision_backend, "triton")

    def test_combined_presets_merge_in_order(self) -> None:
        args = TrainerArgs(distill_preset=f"{self.context_preset},{self.runtime_preset}")

        self.assertEqual(args.distill_method, "context_forcing")
        self.assertEqual(args.runtime_name, "world_model")
        self.assertEqual(args.runtime_cache_backend, "hybrid")
        self.assertTrue(args.enable_runtime)
        self.assertTrue(args.runtime_enable_dpp)
        self.assertTrue(args.enable_fused_supervision_kernel)
        self.assertEqual(args.curriculum_stages, [32, 64, 96, 128, 160])

    def test_report_to_normalizes_legacy_wandb_flag(self) -> None:
        args = TrainerArgs(use_wandb=True)
        self.assertEqual(args.report_to, "console,wandb")

    def test_experiment_tracker_handles_disabled_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(TrainerArgs(output_dir=tmpdir, report_to="none"))
            self.assertEqual(tracker.active_backends(), [])
            tracker.close()

    def test_parse_training_args_accepts_tracking_flags(self) -> None:
        argv = [
            "train_distill.py",
            "--teacher_model_path", "/tmp/teacher",
            "--data_json", "/tmp/train.json",
            "--model_cls", "wan-2.2-moe",
            "--report_to", "console,tensorboard,wandb",
            "--tensorboard_log_dir", "/tmp/tensorboard",
            "--wandb_project", "demo-project",
            "--wandb_entity", "demo-team",
            "--wandb_run_name", "run-001",
            "--wandb_tags", "smoke,tracking",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_training_args()

        self.assertEqual(args.model_cls, "wan2.2_moe")
        self.assertEqual(args.report_to, "console,tensorboard,wandb")
        self.assertEqual(args.tensorboard_log_dir, "/tmp/tensorboard")
        self.assertEqual(args.wandb_project, "demo-project")
        self.assertEqual(args.wandb_entity, "demo-team")
        self.assertEqual(args.wandb_run_name, "run-001")
        self.assertEqual(args.wandb_tags, "smoke,tracking")

    def test_parse_training_args_preserves_true_defaults(self) -> None:
        argv = [
            "train_distill.py",
            "--teacher_model_path", "/tmp/teacher",
            "--data_json", "/tmp/train.json",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_training_args()

        self.assertTrue(args.gradient_checkpointing)
        self.assertTrue(args.use_bucket_sampler)
        self.assertEqual(args.required_transformers_version, EXPECTED_TRANSFORMERS_VERSION)

    def test_loss_cli_exposes_only_implemented_objectives(self) -> None:
        parser = build_training_arg_parser()
        loss_action = next(action for action in parser._actions if action.dest == "loss_type")
        consistency_action = next(
            action for action in parser._actions if action.dest == "consistency_loss_type"
        )

        self.assertEqual(loss_action.choices, ["mse", "huber"])
        self.assertEqual(consistency_action.choices, ["mse", "huber"])
        with self.assertRaisesRegex(ValueError, "Unsupported supervision loss_type"):
            TrainerArgs(loss_type="lpips")

    def test_distributed_bucket_sampling_drops_incomplete_batches(self) -> None:
        self.assertFalse(_bucket_sampler_drop_last(world_size=1))
        self.assertTrue(_bucket_sampler_drop_last(world_size=2))

    @unittest.skipIf(_TORCH_IMPORT_ERROR is not None, f"torch is unavailable: {_TORCH_IMPORT_ERROR}")
    def test_diffusers_style_output_extracts_sample_tensor(self) -> None:
        class DiffusersOutput:
            def __init__(self, sample):
                self.sample = sample

        expected = torch.ones(2, 3)
        self.assertIs(extract_prediction_tensor(DiffusersOutput(expected)), expected)
        with self.assertRaisesRegex(TypeError, "unsupported output type"):
            extract_prediction_tensor({"hidden_states": expected}, tag="student")

    def test_parse_training_args_accepts_cuda_compile_flags(self) -> None:
        argv = [
            "train_distill.py",
            "--teacher_model_path", "/tmp/teacher",
            "--data_json", "/tmp/train.json",
            "--enable_tf32",
            "--float32_matmul_precision", "medium",
            "--enable_torch_compile",
            "--torch_compile_scope", "both",
            "--torch_compile_mode", "max-autotune-no-cudagraphs",
            "--torch_compile_fullgraph",
            "--torch_compile_dynamic",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_training_args()

        self.assertTrue(args.enable_tf32)
        self.assertEqual(args.float32_matmul_precision, "medium")
        self.assertTrue(args.enable_torch_compile)
        self.assertEqual(args.torch_compile_scope, "both")
        self.assertEqual(args.torch_compile_mode, "max-autotune-no-cudagraphs")
        self.assertTrue(args.torch_compile_fullgraph)
        self.assertTrue(args.torch_compile_dynamic)

    def test_explicit_cli_values_override_preset_defaults(self) -> None:
        argv = [
            "train_distill.py",
            "--teacher_model_path", "/tmp/teacher",
            "--data_json", "/tmp/train.json",
            "--config", str(self.step_preset),
            "--learning_rate", "0.123",
            "--parallel_mode", "fsdp",
            "--no-use_dual_model",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_training_args()

        self.assertEqual(args.learning_rate, 0.123)
        self.assertEqual(args.parallel_mode, "fsdp")
        self.assertFalse(args.use_dual_model)
        self.assertTrue(args.enable_runtime)

    def test_inference_catalog_resolves_distill_pair_and_image_defaults(self) -> None:
        metadata = resolve_inference_metadata("wan2.1_distill", task="t2v")
        default_config = resolve_default_inference_config("qwen-image-edit-2511", task="i2i")

        self.assertEqual(metadata["canonical_model_cls"], "wan2.1_distill")
        self.assertEqual(metadata["distill_stage"], "student")
        self.assertEqual(metadata["teacher_model_cls"], "wan2.1")
        self.assertTrue(default_config.endswith("qwen_image_i2i_2511.json"))

    def test_inference_catalog_resolves_audio_and_audio_video_tasks(self) -> None:
        seko_metadata = resolve_inference_metadata("seko-talk", task="s2v")
        ltx2_metadata = resolve_inference_metadata("ltx2", task="i2av")
        seko_config = resolve_default_inference_config("seko_talk", task="rs2v")

        self.assertTrue(seko_metadata["supports_task"])
        self.assertIn("audio", seko_metadata["input_modalities"])
        self.assertEqual(seko_metadata["model_family"], "audio_video")
        self.assertTrue(ltx2_metadata["supports_task"])
        self.assertIn("audio", ltx2_metadata["output_modalities"])
        self.assertIn("video", ltx2_metadata["output_modalities"])
        self.assertTrue(seko_config.endswith("seko_talk/shot/rs2v/main.json"))


@unittest.skipIf(_TORCH_IMPORT_ERROR is not None, f"torch is unavailable: {_TORCH_IMPORT_ERROR}")
class RuntimeSmokeTests(unittest.TestCase):
    def test_runtime_builds_when_dpp_requested(self) -> None:
        args = TrainerArgs(distill_method="step_distill", runtime_enable_dpp=True)
        runtime = build_runtime(
            args=args,
            teacher_model=IdentityTeacher(),
            student_model=IdentityStudent(),
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertFalse(runtime.can_pipeline_teacher_student())

    def test_runtime_teacher_forward_runs_without_cuda(self) -> None:
        args = TrainerArgs(distill_method="step_distill", runtime_enable_dpp=True)
        runtime = build_runtime(
            args=args,
            teacher_model=IdentityTeacher(),
            student_model=IdentityStudent(),
            device=torch.device("cpu"),
        )

        assert runtime is not None
        latents = torch.ones(2, 3)
        output = runtime.run_teacher(
            input_kwargs={"latents": latents},
            batch={"latents": latents},
            global_step=0,
        )
        self.assertTrue(torch.equal(output, latents * 2.0))

    def test_fused_supervision_matches_reference_on_cpu(self) -> None:
        prediction = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
        target = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        expected = ((prediction - target).square() * mask).sum() / mask.sum()
        fallback_loss = fused_masked_mse_loss(prediction, target, mask=mask, enabled=False)
        auto_loss = fused_masked_mse_loss(prediction, target, mask=mask, enabled=True)

        self.assertTrue(torch.allclose(fallback_loss, expected))
        self.assertTrue(torch.allclose(auto_loss, expected))

    def test_masked_huber_supervision_matches_reference(self) -> None:
        teacher = nn.Linear(2, 2)
        student = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        args = TrainerArgs(
            distill_method="step_distill",
            loss_type="huber",
            huber_c=0.5,
            mixed_precision="no",
            report_to="none",
        )
        trainer = StepDistillTrainer(
            args=args,
            teacher_model=teacher,
            student_model=student,
            optimizer=optimizer,
            lr_scheduler=None,
            train_dataloader=[],
            device=torch.device("cpu"),
        )
        prediction = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
        target = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        per_element = torch.nn.functional.huber_loss(
            prediction,
            target,
            reduction="none",
            delta=0.5,
        )
        expected = (per_element * mask).sum() / mask.sum()

        self.assertTrue(
            torch.allclose(
                trainer.compute_supervision_loss(prediction, target, mask=mask),
                expected,
            )
        )

    def test_hybrid_sparse_memory_uses_full_history_and_keeps_recent_tail(self) -> None:
        args = TrainerArgs(
            distill_method="context_forcing",
            enable_runtime=True,
            runtime_name="world_model",
            runtime_memory_policy="hybrid_sparse",
            runtime_memory_budget_frames=5,
            runtime_memory_recent_ratio=0.4,
        )
        runtime = build_runtime(
            args=args,
            teacher_model=IdentityTeacher(),
            student_model=IdentityStudent(),
            device=torch.device("cpu"),
        )
        assert runtime is not None

        indices = runtime.select_memory_indices(
            total_frames=16,
            current_chunk_idx=3,
            chunk_size=4,
            memory_frames=5,
            device=torch.device("cpu"),
        )

        assert indices is not None
        self.assertEqual(indices.tolist()[-2:], [10, 11])
        self.assertEqual(indices.numel(), 5)
        self.assertLess(indices.tolist()[0], 7)
        self.assertEqual(indices.tolist(), sorted(set(indices.tolist())))

    def test_cache_key_hashes_tensor_content_beyond_prefix(self) -> None:
        args = TrainerArgs(distill_method="step_distill", enable_runtime=True)
        runtime = build_runtime(
            args=args,
            teacher_model=IdentityTeacher(),
            student_model=IdentityStudent(),
            device=torch.device("cpu"),
        )
        assert runtime is not None

        first = torch.zeros(32)
        second = first.clone()
        second[-1] = 1.0
        first_key = runtime.build_cache_key(
            namespace="teacher_output",
            batch={"sample_id": ["sample-1"]},
            input_kwargs={"latents": first},
        )
        second_key = runtime.build_cache_key(
            namespace="teacher_output",
            batch={"sample_id": ["sample-1"]},
            input_kwargs={"latents": second},
        )

        self.assertNotEqual(first_key, second_key)

    def test_cache_key_accepts_scalar_tensors(self) -> None:
        args = TrainerArgs(distill_method="step_distill", enable_runtime=True)
        runtime = build_runtime(
            args=args,
            teacher_model=IdentityTeacher(),
            student_model=IdentityStudent(),
            device=torch.device("cpu"),
        )
        assert runtime is not None

        key = runtime.build_cache_key(
            namespace="teacher_output",
            batch={"sample_id": ["sample-1"]},
            input_kwargs={"timestep": torch.tensor(0.5)},
        )

        self.assertIsInstance(key, str)
        self.assertTrue(key)

    def test_cache_identity_separates_teacher_revisions(self) -> None:
        first = build_runtime(
            args=TrainerArgs(
                distill_method="step_distill",
                enable_runtime=True,
                runtime_cache_identity="teacher-revision-a",
            ),
            teacher_model=IdentityTeacher(),
            student_model=IdentityStudent(),
            device=torch.device("cpu"),
        )
        second = build_runtime(
            args=TrainerArgs(
                distill_method="step_distill",
                enable_runtime=True,
                runtime_cache_identity="teacher-revision-b",
            ),
            teacher_model=IdentityTeacher(),
            student_model=IdentityStudent(),
            device=torch.device("cpu"),
        )
        assert first is not None and second is not None

        first_key = first.build_cache_key("teacher_output", {"sample_id": ["clip"]})
        second_key = second.build_cache_key("teacher_output", {"sample_id": ["clip"]})

        self.assertNotEqual(first_key, second_key)

    def test_cache_rejects_entries_from_a_later_timeline(self) -> None:
        cache = MemoryDistillCache(freshness_steps=0)
        cache.put("key", torch.tensor([1.0]), current_step=10)

        self.assertIsNone(cache.get("key", current_step=1))

    def test_context_chunk_ranges_keep_non_divisible_tail(self) -> None:
        self.assertEqual(
            ContextForcingTrainer._chunk_ranges(num_frames=10, chunk_size=4),
            [(0, 4), (4, 8), (8, 10)],
        )

    def test_context_temporal_conditions_follow_packed_frame_indices(self) -> None:
        batch = {
            "camera_poses": torch.arange(2 * 6 * 4 * 4).reshape(2, 6, 4, 4),
            "actions": torch.arange(2 * 6 * 3).reshape(2, 6, 3),
            "encoder_hidden_states": torch.ones(2, 4, 8),
        }
        indices = torch.tensor([0, 3, 4])

        selected = ContextForcingTrainer._slice_temporal_conditions(batch, indices, 6)

        self.assertTrue(torch.equal(selected["actions"], batch["actions"][:, indices]))
        self.assertTrue(torch.equal(selected["camera_poses"], batch["camera_poses"][:, indices]))
        self.assertIs(selected["encoder_hidden_states"], batch["encoder_hidden_states"])

    def test_context_temporal_condition_rejects_ambiguous_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot infer the temporal axis"):
            ContextForcingTrainer._slice_temporal_conditions(
                {"actions": torch.zeros(2, 6, 6)},
                torch.tensor([1, 3]),
                6,
            )

    def test_context_curriculum_activates_final_stage_before_training_ends(self) -> None:
        student = nn.Linear(2, 2)
        trainer = ContextForcingTrainer(
            args=TrainerArgs(
                distill_method="context_forcing",
                curriculum_stages=[20, 40, 60, 80, 100],
                max_train_steps=100,
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

        trainer.global_step = 20
        trainer.on_train_step_end({})
        self.assertEqual((trainer.current_curriculum_stage, trainer.current_num_frames), (1, 40))
        trainer.global_step = 80
        trainer.on_train_step_end({})
        self.assertEqual((trainer.current_curriculum_stage, trainer.current_num_frames), (4, 100))

    def test_progressive_schedule_ends_at_declared_target(self) -> None:
        student = nn.Linear(2, 2)
        trainer = ProgressiveDistillTrainer(
            args=TrainerArgs(
                distill_method="progressive_distill",
                progressive_stages=[64, 32, 16, 8, 4],
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

        self.assertEqual(trainer._stage_pair(0), (64, 32))
        self.assertEqual(trainer._stage_pair(3), (8, 4))
        with self.assertRaises(IndexError):
            trainer._stage_pair(4)

    def test_velocity_only_methods_reject_other_prediction_types(self) -> None:
        student = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "velocity-space"):
            ConsistencyDistillTrainer(
                args=TrainerArgs(
                    distill_method="consistency_distill",
                    prediction_type="x0",
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


if __name__ == "__main__":
    unittest.main()
