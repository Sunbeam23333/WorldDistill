from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_TRAIN = PROJECT_ROOT / "scripts" / "run_train.sh"


class RunTrainScriptTests(unittest.TestCase):
    @staticmethod
    def _captured_argv(
        temp_root: Path,
        *script_args: str,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        bin_dir = temp_root / "fake bin"
        bin_dir.mkdir()
        python_shim = bin_dir / "python3"
        torchrun_shim = bin_dir / "torchrun"
        capture_path = temp_root / "captured argv"
        python_shim.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        torchrun_shim.write_text(
            '#!/bin/bash\nprintf \'%s\\0\' "$@" > "$CAPTURE_ARGV"\n',
            encoding="utf-8",
        )
        python_shim.chmod(0o755)
        torchrun_shim.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        env["CAPTURE_ARGV"] = str(capture_path)
        result = subprocess.run(
            ["bash", str(RUN_TRAIN), *script_args],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        argv = []
        if capture_path.exists():
            argv = [
                part.decode("utf-8")
                for part in capture_path.read_bytes().split(b"\0")
                if part
            ]
        return result, argv

    @staticmethod
    def _value_after(argv: list[str], option: str) -> str:
        index = argv.index(option)
        return argv[index + 1]

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

    def test_paths_and_run_name_with_spaces_remain_single_argv_values(self) -> None:
        with tempfile.TemporaryDirectory(prefix="worlddistill shell test ") as temp_dir:
            temp_root = Path(temp_dir)
            config_path = temp_root / "preset with spaces.json"
            config_path.write_text("{}\n", encoding="utf-8")
            teacher_path = temp_root / "teacher weights"
            student_path = temp_root / "student weights"
            data_path = temp_root / "training data.json"
            output_path = temp_root / "training results"
            run_name = "context forcing experiment 01"

            result, argv = self._captured_argv(
                temp_root,
                "--method",
                "context_forcing",
                "--teacher_model",
                str(teacher_path),
                "--student_model",
                str(student_path),
                "--data_json",
                str(data_path),
                "--output_dir",
                str(output_path),
                "--config",
                str(config_path),
                "--wandb_run_name",
                run_name,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(self._value_after(argv, "--teacher_model_path"), str(teacher_path))
            self.assertEqual(self._value_after(argv, "--student_model_path"), str(student_path))
            self.assertEqual(self._value_after(argv, "--data_json"), str(data_path))
            self.assertEqual(self._value_after(argv, "--output_dir"), str(output_path))
            self.assertEqual(self._value_after(argv, "--config"), str(config_path))
            self.assertEqual(self._value_after(argv, "--wandb_run_name"), run_name)
            self.assertEqual(self._value_after(argv, "--num_frames"), "160")

    def test_non_context_default_and_sharded_step_dual_fallback_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="worlddistill shell test ") as temp_dir:
            temp_root = Path(temp_dir)
            config_path = temp_root / "step preset.json"
            config_path.write_text("{}\n", encoding="utf-8")
            result, argv = self._captured_argv(
                temp_root,
                "--method",
                "step_distill",
                "--teacher_model",
                str(temp_root / "teacher model"),
                "--data_json",
                str(temp_root / "train manifest.json"),
                "--config",
                str(config_path),
                "--parallel",
                "fsdp",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(self._value_after(argv, "--num_frames"), "49")
            self.assertIn("--no-use_dual_model", argv)


if __name__ == "__main__":
    unittest.main()
