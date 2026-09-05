from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_INFER = PROJECT_ROOT / "scripts" / "run_infer.sh"
INPUT_INFO_MODULE = PROJECT_ROOT / "inference" / "lightx2v" / "utils" / "input_info.py"


def _run_dry(*arguments: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    result = subprocess.run(
        ["bash", str(RUN_INFER), *arguments, "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    command_line = next(
        (
            line.removeprefix("Resolved command: ")
            for line in result.stdout.splitlines()
            if line.startswith("Resolved command: ")
        ),
        "",
    )
    return result, shlex.split(command_line)


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


class RunInferScriptTests(unittest.TestCase):
    def test_shell_syntax_and_help(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(RUN_INFER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, msg=syntax.stderr)

        help_result = subprocess.run(
            ["bash", str(RUN_INFER), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
        for option in (
            "--audio_path",
            "--last_frame_path",
            "--image_strength",
            "--src_ref_images",
            "--src_video",
            "--src_mask",
            "--src_pose_path",
            "--src_face_path",
            "--src_bg_path",
            "--src_mask_path",
            "--dry-run",
        ):
            self.assertIn(option, help_result.stdout)

    def test_audio_tasks_forward_image_and_audio_paths(self) -> None:
        result, command = _run_dry(
            "--model_cls",
            "wan2.2_audio",
            "--task",
            "s2v",
            "--model_path",
            "/models/wan-audio",
            "--image_path",
            "portrait input.png",
            "--audio_path",
            "speech clip.wav",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Dry run: dependency compatibility check skipped.", result.stdout)
        self.assertEqual(_option_value(command, "--image_path"), "portrait input.png")
        self.assertEqual(_option_value(command, "--audio_path"), "speech clip.wav")
        self.assertTrue(
            _option_value(command, "--config_json").endswith(
                "wan22/wan_moe_i2v_audio.json"
            )
        )

    def test_rs2v_uses_the_stateful_shot_pipeline(self) -> None:
        result, command = _run_dry(
            "--model_cls",
            "seko_talk",
            "--task",
            "rs2v",
            "--model_path",
            "/models/seko-talk",
            "--image_path",
            "portrait input.png",
            "--audio_path",
            "speech clip.wav",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("lightx2v.shot_runner.rs2v_infer", command)
        self.assertNotIn("--task", command)
        self.assertNotIn("--image_strength", command)
        self.assertEqual(_option_value(command, "--image_path"), "portrait input.png")
        self.assertEqual(_option_value(command, "--audio_path"), "speech clip.wav")
        self.assertTrue(
            _option_value(command, "--config_json").endswith(
                "seko_talk/shot/rs2v/main.json"
            )
        )

    def test_flf2v_and_i2av_specific_arguments_are_forwarded(self) -> None:
        flf_result, flf_command = _run_dry(
            "--model_cls",
            "wan2.2_moe",
            "--task",
            "flf2v",
            "--image_path",
            "first.png",
            "--last_frame_path",
            "last.png",
        )
        self.assertEqual(flf_result.returncode, 0, msg=flf_result.stderr)
        self.assertEqual(_option_value(flf_command, "--last_frame_path"), "last.png")

        i2av_result, i2av_command = _run_dry(
            "--model_cls",
            "ltx2",
            "--task",
            "i2av",
            "--image_path",
            "reference.png",
            "--image_strength",
            "0.35",
        )
        self.assertEqual(i2av_result.returncode, 0, msg=i2av_result.stderr)
        self.assertEqual(_option_value(i2av_command, "--image_strength"), "0.35")

    def test_vace_inputs_are_forwarded(self) -> None:
        result, command = _run_dry(
            "--model_cls",
            "wan2.1_vace",
            "--task",
            "vace",
            "--src_ref_images",
            "ref one.png,ref two.png",
            "--src_video",
            "source video.mp4",
            "--src_mask",
            "source mask.png",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(_option_value(command, "--src_ref_images"), "ref one.png,ref two.png")
        self.assertEqual(_option_value(command, "--src_video"), "source video.mp4")
        self.assertEqual(_option_value(command, "--src_mask"), "source mask.png")

    def test_animate_inputs_are_forwarded(self) -> None:
        result, command = _run_dry(
            "--model_cls",
            "wan2.2_animate",
            "--task",
            "animate",
            "--src_pose_path",
            "pose.mp4",
            "--src_face_path",
            "face.mp4",
            "--src_ref_images",
            "reference.png",
            "--src_bg_path",
            "background.mp4",
            "--src_mask_path",
            "mask.mp4",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(_option_value(command, "--src_pose_path"), "pose.mp4")
        self.assertEqual(_option_value(command, "--src_face_path"), "face.mp4")
        self.assertEqual(_option_value(command, "--src_ref_images"), "reference.png")
        self.assertEqual(_option_value(command, "--src_bg_path"), "background.mp4")
        self.assertEqual(_option_value(command, "--src_mask_path"), "mask.mp4")

    def test_i2i_and_game_commands_keep_required_conditioning(self) -> None:
        i2i_result, i2i_command = _run_dry(
            "--model_cls",
            "qwen_image",
            "--task",
            "i2i",
            "--image_path",
            "one.png,two.png",
        )
        self.assertEqual(i2i_result.returncode, 0, msg=i2i_result.stderr)
        self.assertEqual(_option_value(i2i_command, "--image_path"), "one.png,two.png")

        for model_cls in ("worldplay_distill", "matrix_game_2"):
            with self.subTest(model_cls=model_cls):
                result, command = _run_dry(
                    "--model_cls",
                    model_cls,
                    "--task",
                    "game",
                    "--image_path",
                    "world.png",
                    "--pose",
                    "w-3, right-0.5",
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertEqual(_option_value(command, "--image_path"), "world.png")
                self.assertEqual(_option_value(command, "--pose"), "w-3, right-0.5")

    def test_gpu_count_rejects_unsafe_values(self) -> None:
        result = subprocess.run(
            ["bash", str(RUN_INFER), "--gpus", "0", "--dry-run"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--gpus must be a positive integer", result.stdout)

    def test_gpu_count_must_match_configured_parallel_topology(self) -> None:
        mismatched, _ = _run_dry(
            "--model_cls",
            "wan2.2_moe",
            "--task",
            "t2v",
            "--gpus",
            "8",
            "--prompt",
            "topology check",
        )
        self.assertNotEqual(mismatched.returncode, 0)
        self.assertIn("conflicts with config parallel world size 1", mismatched.stdout)

        matching, command = _run_dry(
            "--model_cls",
            "wan2.2_moe",
            "--task",
            "t2v",
            "--gpus",
            "8",
            "--prompt",
            "topology check",
            "--config_json",
            str(PROJECT_ROOT / "inference/configs/wan22/wan_moe_t2v_h20_8gpu.json"),
        )
        self.assertEqual(matching.returncode, 0, msg=matching.stderr)
        self.assertIn("--nproc_per_node=8", command)

    def test_unknown_model_task_pair_needs_an_explicit_config(self) -> None:
        result, _ = _run_dry(
            "--model_cls",
            "not_a_model",
            "--task",
            "t2v",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No public default config", result.stdout)

    def test_required_argument_presence_is_checked_without_touching_files(self) -> None:
        result, _ = _run_dry(
            "--model_cls",
            "wan2.2_animate",
            "--task",
            "animate",
            "--src_pose_path",
            "pose.mp4",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --src_pose_path, --src_face_path, and --src_ref_images", result.stdout)

    def test_missing_explicit_config_has_a_clean_error(self) -> None:
        result, _ = _run_dry(
            "--model_cls",
            "wan2.2_moe",
            "--task",
            "t2v",
            "--prompt",
            "missing config check",
            "--config_json",
            "/definitely/missing/config.json",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Config JSON not found", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


class InputInfoTaskMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "worlddistill_input_info_for_test", INPUT_INFO_MODULE
        )
        if spec is None or spec.loader is None:  # pragma: no cover - import machinery guard
            raise RuntimeError(f"Cannot load {INPUT_INFO_MODULE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        # input_info only needs torch.Tensor for annotations/default placeholders;
        # keep this CLI contract test independent of a local CUDA/PyTorch runtime.
        torch_module = types.ModuleType("torch")
        torch_module.Tensor = object
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = torch_module
        try:
            spec.loader.exec_module(module)
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch
        cls.input_info = module

    def test_validated_tasks_have_matching_input_types_and_required_fields(self) -> None:
        expected = {
            "i2i": (self.input_info.I2IInputInfo, {"image_path"}),
            "s2v": (self.input_info.S2VInputInfo, {"image_path", "audio_path"}),
            "rs2v": (self.input_info.RS2VInputInfo, {"image_path", "audio_path"}),
            "game": (self.input_info.GameInputInfo, {"image_path", "pose", "action_path"}),
        }
        for task, (input_type, fields) in expected.items():
            with self.subTest(task=task):
                instance = self.input_info.init_empty_input_info(task)
                self.assertIsInstance(instance, input_type)
                self.assertTrue(fields.issubset(instance.__dataclass_fields__))


if __name__ == "__main__":
    unittest.main()
