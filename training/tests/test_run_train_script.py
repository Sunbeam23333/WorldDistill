from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_TRAIN = PROJECT_ROOT / "scripts" / "run_train.sh"


class RunTrainScriptTests(unittest.TestCase):
    def test_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(RUN_TRAIN)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_help_lists_cuda_compile_controls_without_unbound_variables(self) -> None:
        result = subprocess.run(
            ["bash", str(RUN_TRAIN), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--enable_tf32", result.stdout)
        self.assertIn("--enable_torch_compile", result.stdout)
        self.assertIn("--torch_compile_dynamic", result.stdout)


if __name__ == "__main__":
    unittest.main()
