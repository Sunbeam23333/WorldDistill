"""Match the public VBench custom-input call and validate its output inventory."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.evaluate_generated import evaluate_vbench, trajectory_metrics


def test_official_vbench_call_preserves_prompts_dimensions_and_results(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clip.mp4").write_bytes(b"fixture video")
    info = tmp_path / "full_info.json"
    info.write_text("[]")
    captured = {}

    class VBench:
        def __init__(self, device, full_info, output):
            captured.update(device=device, full_info=full_info)
            self.output = output
            from pathlib import Path
            Path(output).mkdir()

        def build_full_dimension_list(self):
            return ["subject_consistency"]

        def check_dimension_requires_extra_info(self, dimensions):
            captured["checked_dimensions"] = dimensions

        def evaluate(self, **kwargs):
            from pathlib import Path
            captured.update(kwargs)
            (Path(self.output) / "worlddistill_eval_results.json").write_text(
                json.dumps({"subject_consistency": [0.8, []]}))

    monkeypatch.setitem(sys.modules, "vbench", SimpleNamespace(VBench=VBench))
    monkeypatch.setattr("importlib.metadata.version", lambda _: "test-version")
    result = evaluate_vbench(videos, info, tmp_path / "result", ["subject_consistency"],
                             device="cpu", prompts={"clip.mp4": "A real task prompt"})
    assert captured["mode"] == "custom_input" and captured["local"] is True
    assert captured["prompt_list"] == {"clip.mp4": "A real task prompt"}
    assert result["vbench"]["subject_consistency"][0] == 0.8
    assert len(result["full_info_sha256"]) == 64
    with pytest.raises(ValueError, match="map video filenames"):
        evaluate_vbench(videos, info, tmp_path / "bad", ["subject_consistency"], prompts=["wrong"])


def test_integer_trajectories_do_not_subtract_in_unsigned_storage():
    pred = torch.tensor([[[0, 0]]], dtype=torch.uint8)
    target = torch.tensor([[[3, 4]]], dtype=torch.uint8)
    assert trajectory_metrics(pred, target)["ade"] == 5
