import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = PROJECT_ROOT / "tools" / "validate_paper_artifacts.py"

_validator_spec = importlib.util.spec_from_file_location("worlddistill_paper_artifact_validator", VALIDATOR_PATH)
assert _validator_spec is not None and _validator_spec.loader is not None
_validator = importlib.util.module_from_spec(_validator_spec)
sys.modules[_validator_spec.name] = _validator
_validator_spec.loader.exec_module(_validator)


def _touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub", encoding="utf-8")


class PaperArtifactValidationTests(unittest.TestCase):
    def test_fewstep_block_accepts_minimum_required_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "fewstep_t2v" / "train").mkdir(parents=True, exist_ok=True)
            _touch(root, "fewstep_t2v/teacher_samples/teacher.mp4")
            _touch(root, "fewstep_t2v/student_samples/student.mp4")

            audit = _validator.audit_block(root, "fewstep_t2v")

            self.assertFalse(audit.has_missing_required)
            self.assertIn("run_card.md", audit.missing_recommended)
            self.assertIn("metrics.json", audit.missing_recommended)
            self.assertIn("runtime_stats.json", audit.missing_recommended)
            self.assertIn("config_snapshot.json", audit.missing_recommended)

    def test_camera_control_block_flags_missing_required_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _touch(root, "camera_control/poses/orbit.json")
            _touch(root, "camera_control/poses/pan_left.json")
            _touch(root, "camera_control/samples/orbit.mp4")
            _touch(root, "camera_control/samples/pan_left.mp4")
            _touch(root, "camera_control/samples/zoom_in.mp4")

            audit = _validator.audit_block(root, "camera_control")

            self.assertTrue(audit.has_missing_required)
            self.assertIn("poses/zoom_in.json", audit.missing_required)
            self.assertIn("samples/pan_left.mp4", [finding.matches[0] for finding in audit.findings if finding.exists])


if __name__ == "__main__":
    unittest.main()
