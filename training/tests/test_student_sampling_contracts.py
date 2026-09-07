import json
from types import SimpleNamespace

import pytest

from tools.sample_student import resolve_sampling_options


def options(**overrides):
    args = dict(conditions=None, prompt="test", num_frames=None, height=None, width=None, num_steps=None)
    args.update(overrides)
    return SimpleNamespace(**args)


def test_json_controls_and_horizon_are_not_overwritten_by_cli_defaults(tmp_path):
    path = tmp_path / "conditions.json"
    path.write_text(json.dumps({"num_frames": 125, "height": 720, "width": 1280,
                               "num_inference_steps": 2, "guidance_scale": 1.0, "actions": [1, 2]}))
    actual = resolve_sampling_options(options(conditions=str(path)), {"num_inference_steps": 4})
    assert actual == {"prompt": "test", "num_frames": 125, "height": 720, "width": 1280,
                      "num_inference_steps": 2, "guidance_scale": 1.0, "actions": [1, 2]}
    explicit = resolve_sampling_options(options(conditions=str(path), num_frames=49, num_steps=8),
                                        {"num_inference_steps": 4})
    assert explicit["num_frames"] == 49 and explicit["num_inference_steps"] == 8


def test_sampler_defaults_use_export_step_count():
    actual = resolve_sampling_options(options(), {"num_inference_steps": 1})
    assert actual == {"prompt": "test", "num_frames": 81, "height": 480, "width": 832, "num_inference_steps": 1}


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, None])
def test_invalid_json_horizon_is_rejected(tmp_path, invalid):
    path = tmp_path / "conditions.json"
    path.write_text(json.dumps({"num_frames": invalid}))
    with pytest.raises(ValueError, match="positive integer"):
        resolve_sampling_options(options(conditions=str(path)), {"num_inference_steps": 4})
