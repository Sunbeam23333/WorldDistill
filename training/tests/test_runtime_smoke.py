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
from training.trainer_args import TrainerArgs, parse_training_args

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
    from training.runtime.fused_supervision import fused_masked_mse_loss

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
        self.assertTrue(seko_config.endswith("seko_talk/shot/rs2v/rs2v.json"))


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


if __name__ == "__main__":
    unittest.main()
