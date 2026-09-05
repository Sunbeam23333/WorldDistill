import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill_capabilities import REGISTRY_BACKED_DISTILL_METHODS
from training.model_catalog import resolve_model_metadata as resolve_training_metadata
from training.trainer_args import build_training_arg_parser

_INFERENCE_MODEL_CATALOG_PATH = PROJECT_ROOT / "inference" / "lightx2v" / "utils" / "model_catalog.py"
_inference_spec = importlib.util.spec_from_file_location(
    "worlddistill_inference_model_catalog_consistency",
    _INFERENCE_MODEL_CATALOG_PATH,
)
assert _inference_spec is not None and _inference_spec.loader is not None
_inference_model_catalog = importlib.util.module_from_spec(_inference_spec)
_inference_spec.loader.exec_module(_inference_model_catalog)
resolve_inference_metadata = _inference_model_catalog.resolve_model_metadata


def _distill_method_choices() -> tuple[str, ...]:
    parser = build_training_arg_parser()
    for action in parser._actions:
        if "--distill_method" in action.option_strings:
            return tuple(action.choices or ())
    raise AssertionError("Could not find --distill_method action in training parser")


class CapabilityConsistencyTests(unittest.TestCase):
    def test_cli_choices_match_registry_backed_method_contract(self) -> None:
        self.assertEqual(_distill_method_choices(), REGISTRY_BACKED_DISTILL_METHODS)

    def test_mean_flow_student_is_cataloged_but_not_registry_backed(self) -> None:
        for resolver in (resolve_training_metadata, resolve_inference_metadata):
            metadata = resolver("wan2.1_mean_flow_distill", task="t2v")
            self.assertEqual(metadata["distill_methods"], ["mean_flow_distill"])
            self.assertEqual(metadata["registry_backed_distill_methods"], [])
            self.assertEqual(metadata["non_registry_distill_methods"], ["mean_flow_distill"])
            self.assertEqual(metadata["catalog_only_distill_methods"], ["mean_flow_distill"])
            self.assertFalse(metadata["supports_registry_training_entry"])
            self.assertFalse(metadata["supports_opd_like_runtime"])

    def test_training_and_inference_catalogs_agree_on_capability_fields(self) -> None:
        shared_fields = (
            "distill_methods",
            "registry_backed_distill_methods",
            "non_registry_distill_methods",
            "catalog_only_distill_methods",
            "supports_registry_training_entry",
            "supports_opd_like_runtime",
        )
        for model_cls, task in (
            ("wan2.1", "t2v"),
            ("wan2.1_mean_flow_distill", "t2v"),
            ("wan2.2_moe", "t2v"),
            ("worldplay_distill", "game"),
        ):
            training_metadata = resolve_training_metadata(model_cls, task=task)
            inference_metadata = resolve_inference_metadata(model_cls, task=task)
            for field in shared_fields:
                self.assertEqual(
                    training_metadata[field],
                    inference_metadata[field],
                    msg=f"Capability field {field} drifted for {model_cls}",
                )

    def test_teacher_catalog_splits_registry_backed_and_non_registry_method_tags(self) -> None:
        metadata = resolve_training_metadata("wan2.1", task="t2v")
        self.assertEqual(metadata["registry_backed_distill_methods"], ["step_distill"])
        self.assertEqual(metadata["non_registry_distill_methods"], ["lora_distill", "mean_flow_distill"])
        self.assertEqual(metadata["catalog_only_distill_methods"], [])
        self.assertTrue(metadata["supports_registry_training_entry"])


if __name__ == "__main__":
    unittest.main()
