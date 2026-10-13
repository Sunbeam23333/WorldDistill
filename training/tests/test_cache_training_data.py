import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import default_collate

from tools.cache_training_data import build_parser, cache_training_data
from training.data.video_dataset import CachedLatentDataset, VideoDataset
from training.utils.batch_encoder import DiffusersRawBatchEncoder


@pytest.fixture
def raw_cache_inputs(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text('{"_class_name":"LTX2Pipeline"}')
    (model / "config.json").write_text('{"test_config":1}')
    source = tmp_path / "raw.json"
    torch.save(torch.arange(21).reshape(7, 3).float(), tmp_path / "actions.pt")
    torch.save(torch.arange(28).reshape(7, 4).float(), tmp_path / "camera.pt")
    torch.save(torch.ones(1, 6, 2), tmp_path / "audio.pt")
    source.write_text(json.dumps([{"path": "input.mp4", "sample_id": "clip", "text": "prompt", "num_frames": 5,
        "fps": 24, "action_path": "actions.pt", "camera_path": "camera.pt", "audio_feature_path": "audio.pt",
        "audio_alignment": "sampled_video"}]))
    def video(self, path, frames, return_metadata=False):
        assert path == str(tmp_path / "input.mp4") and frames == 5
        pixels = torch.arange(60).reshape(3, 5, 2, 2).float() / 30 - 1
        return pixels, torch.tensor([0, 1, 3, 4, 6]), 7
    monkeypatch.setattr(VideoDataset, "_load_video", video)
    # Exercise the real encoder's temporal-control mapping and AV packing.
    encoder = DiffusersRawBatchEncoder.__new__(DiffusersRawBatchEncoder)
    encoder.device, encoder.dtype = torch.device("cpu"), torch.float32
    encoder.temporal_compression_ratio = 2
    encoder.vae_scaling_factor = 1.
    encoder.vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=1.), encode=lambda pixels:
        SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: pixels[:, :2, ::2] + torch.rand(()))))
    encoder.pipe = type("LTX2Pipeline", (), {})()
    encoder.pipe.audio_vae = SimpleNamespace(encode=lambda values: values, latents_mean=torch.zeros(2),
                                            latents_std=torch.ones(2))
    encoder._encode_text = lambda _: {"encoder_hidden_states": torch.ones(1, 3, 4),
        "encoder_attention_mask": torch.tensor([[True, True, False]]), "encoder_hidden_states_2": torch.ones(1, 2, 4),
        "pooled_projections": torch.arange(4).reshape(1, 4), "audio_encoder_hidden_states": torch.full((1, 3, 4), 2.),
        "audio_encoder_attention_mask": torch.tensor([[True, True, False]]), "guidance": torch.tensor([3.])}
    return source, model, encoder


def test_raw_cache_roundtrip_preserves_all_modalities_controls_and_provenance(raw_cache_inputs, tmp_path):
    source, model, encoder = raw_cache_inputs
    state, python_state = torch.random.get_rng_state(), random.getstate()
    report = cache_training_data(source, model, tmp_path / "cache", seed=7, encoder=encoder)
    assert torch.equal(state, torch.random.get_rng_state()) and random.getstate() == python_state
    dataset = CachedLatentDataset(report["manifest"], cache_dir=report["cache_dir"])
    batch = default_collate([dataset[0]])
    assert batch["latents"].shape == (1, 2, 3, 2, 2)
    torch.testing.assert_close(batch["actions"][0], torch.arange(21).reshape(7, 3).float()[[0, 3, 6]])
    torch.testing.assert_close(batch["camera_poses"][0], torch.arange(28).reshape(7, 4).float()[[0, 3, 6]])
    assert batch["audio_latents"].shape == (1, 6, 2) and batch["fps"].item() == 24
    for key, value in encoder._encode_text(["prompt"]).items():
        torch.testing.assert_close(batch[key], value)
    row = json.loads((tmp_path / "cache/manifest.json").read_text())[0]
    assert row["source_frame_indices"] == [0, 1, 3, 4, 6]
    assert row["latent_source_frame_indices"] == [0, 3, 6]
    assert row["num_frames"] == 3 and row["encoding_seed"] == 7
    assert row["collation_signature"] == dataset.get_metadata(0)["collation_signature"]
    assert row["collation_signature"]["encoder_hidden_states_2"]["shape"] == [2, 4]
    metadata = json.loads((tmp_path / "cache/cache_metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["completed_samples"] == 1
    assert metadata["configuration_files"].keys() == {"model_index.json", "config.json"}
    assert len(metadata["configuration_sha256"]) == 64 and len(metadata["artifacts"]) == 3
    assert not (tmp_path / "cache/cache_incomplete.json").exists()
    second = cache_training_data(source, model, tmp_path / "same-seed", seed=7, encoder=encoder)
    torch.testing.assert_close(dataset[0]["latents"],
        CachedLatentDataset(second["manifest"], cache_dir=second["cache_dir"])[0]["latents"], rtol=0, atol=0)


@pytest.mark.parametrize("existing_file", [False, True])
def test_cache_never_overwrites_existing_output(raw_cache_inputs, tmp_path, existing_file):
    source, model, encoder = raw_cache_inputs
    output = tmp_path / "existing"
    output.mkdir()
    if existing_file:
        (output / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="existing"):
        cache_training_data(source, model, output, encoder=encoder)
    assert list(output.iterdir()) == ([output / "user.txt"] if existing_file else [])


def test_cache_failure_keeps_diagnostic_not_a_completed_manifest(raw_cache_inputs, tmp_path):
    source, model, encoder = raw_cache_inputs
    encoder._encode_text = lambda _: {"unsupported_nested": [torch.ones(1, 3)]}
    output = tmp_path / "failed"
    with pytest.raises(TypeError, match="non-tensor condition"):
        cache_training_data(source, model, output, encoder=encoder)
    assert not (output / "manifest.json").exists()
    metadata = json.loads((output / "cache_incomplete.json").read_text())
    assert metadata["status"] == "failed" and metadata["completed_samples"] == 0


def test_provenance_write_failure_cannot_publish_consumable_manifest(raw_cache_inputs, tmp_path, monkeypatch):
    from tools import cache_training_data as tool
    source, model, encoder = raw_cache_inputs
    original = tool._atomic_json
    def write(path, value):
        if path.name == "cache_metadata.json":
            raise OSError("provenance disk failure")
        return original(path, value)
    monkeypatch.setattr(tool, "_atomic_json", write)
    output = tmp_path / "metadata-failed"
    with pytest.raises(OSError, match="provenance disk"):
        cache_training_data(source, model, output, encoder=encoder)
    assert not (output / "manifest.json").exists()
    assert json.loads((output / "cache_incomplete.json").read_text())["status"] == "failed"


def test_progress_cleanup_failure_does_not_unpublish_complete_cache(raw_cache_inputs, tmp_path, monkeypatch):
    from pathlib import Path
    source, model, encoder = raw_cache_inputs
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path.name == "cache_incomplete.json":
            raise OSError("cleanup denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.warns(RuntimeWarning, match="Cache is complete"):
        result = cache_training_data(source, model, tmp_path / "cleanup-warning", encoder=encoder)
    assert json.loads(Path(result["metadata"]).read_text())["status"] == "complete"
    assert len(CachedLatentDataset(result["manifest"], cache_dir=result["cache_dir"])) == 1


def test_cache_cli_exposes_local_encoder_and_audio_contract():
    parser = build_parser()
    args = parser.parse_args(["--data_json", "raw.json", "--model_path", "pipeline", "--output_dir", "cache",
                              "--seed", "9", "--device", "cpu"])
    assert args.seed == 9 and args.device == "cpu" and args.dtype == "float32"
    help_text = " ".join(parser.format_help().split())
    assert "log-mel" in help_text and "never overwritten" in help_text
