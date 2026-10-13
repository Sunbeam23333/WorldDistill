import json

import pytest
import torch
from torch.utils.data import DataLoader

from training.data.bucket_sampler import BucketSampler
from training.data.video_dataset import CachedLatentDataset, VideoDataset, collation_signature


def _cached_rows(tmp_path, specifications):
    rows = []
    for index, spec in enumerate(specifications):
        latent = torch.zeros(2, spec.get("frames", 3), 2, spec.get("width", 2))
        torch.save(latent, tmp_path / f"latent-{index}.pt")
        conditions = {"encoder_hidden_states": torch.ones(spec.get("text1", 3), 4),
                      "encoder_hidden_states_2": torch.ones(spec.get("text2", 2), 4),
                      "encoder_attention_mask": torch.ones(spec.get("text1", 3), dtype=torch.bool),
                      "guidance": torch.tensor(spec.get("guidance", 3.), dtype=spec.get("dtype", torch.float32))}
        if spec.get("image"):
            conditions["image_cond"] = torch.zeros(2, 1, 2, 2)
        if spec.get("scalar_vector"):
            conditions["guidance"] = conditions["guidance"].reshape(1)
        torch.save(conditions, tmp_path / f"conditions-{index}.pt")
        # Deliberately stale dimensions must not determine batching.
        row = {"latent_path": f"latent-{index}.pt", "conditioning_path": f"conditions-{index}.pt",
               "resolution": [999, 999], "num_frames": 99, "sample_id": str(index)}
        if "fps" in spec:
            row["fps"] = spec["fps"]
        if "audio" in spec:
            torch.save(torch.zeros(spec["audio"], 4), tmp_path / f"audio-{index}.pt")
            row["audio_latent_path"] = f"audio-{index}.pt"
        rows.append(row)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows))
    return CachedLatentDataset(str(path), cache_dir=str(tmp_path))


@pytest.mark.parametrize("configured", [False, True])
def test_real_dataloader_separates_exact_video_audio_and_multiencoder_shapes(tmp_path, configured):
    specs = [{}, {"frames": 5}, {"width": 3}, {"text1": 4}, {"text2": 4}, {"guidance": 4.},
             {"scalar_vector": True}, {"dtype": torch.float64}, {"image": True}, {"fps": 24},
             {"fps": 24, "audio": 4}, {"fps": 30, "audio": 4}, {"fps": 24, "audio": 6}]
    dataset = _cached_rows(tmp_path, [spec for spec in specs for _ in range(2)])
    config = {"nearby": {"resolution": [2, 2], "num_frames": 4}} if configured else None
    sampler = BucketSampler(dataset, batch_size=2, shuffle=False, bucket_config=config)
    assert len(sampler.buckets) == len(specs)
    seen = []
    for batch in DataLoader(dataset, batch_sampler=sampler):
        assert batch["latents"].shape[0] == 2
        assert batch["num_frames"].unique().numel() == 1
        assert batch["num_frames"][0].item() == batch["latents"].shape[2]
        if "fps" in batch:
            assert batch["fps"].unique().numel() == 1
        seen.extend(batch["manifest_idx"].tolist())
    assert sorted(seen) == list(range(len(dataset)))
    assert dataset.get_metadata(0)["resolution"] == [2, 2]
    assert dataset.get_metadata(0)["num_frames"] == 3


def test_header_inspection_uses_mmap_and_retains_no_tensor_storage(tmp_path, monkeypatch):
    dataset = _cached_rows(tmp_path, [{"audio": 4, "fps": 24}])
    original, calls = torch.load, []
    def load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    metadata = dataset.get_metadata(0)
    assert calls and all(call.get("mmap") is True and call["weights_only"] for call in calls)
    count = len(calls)
    assert dataset.get_metadata(0) == metadata and len(calls) == count
    # JSON serialization cannot succeed if a storage-bearing tensor leaked.
    json.dumps(dataset._metadata_cache)


def test_declared_cache_signature_is_checked_against_artifact_headers(tmp_path):
    dataset = _cached_rows(tmp_path, [{}])
    signature = collation_signature(dataset[0])
    dataset.data[0]["collation_signature"] = signature
    assert dataset.get_metadata(0)["collation_signature"] == signature
    torch.save(torch.ones(2, 7, 2, 2), tmp_path / "latent-0.pt")
    dataset._metadata_cache.clear()
    with pytest.raises(ValueError, match="collation_signature does not match"):
        dataset.get_metadata(0)


def test_legacy_serialization_header_fallback_is_storage_free_after_inspection(tmp_path):
    dataset = _cached_rows(tmp_path, [{}])
    torch.save(torch.zeros(2, 3, 2, 2), tmp_path / "latent-0.pt", _use_new_zipfile_serialization=False)
    assert dataset.get_metadata(0)["num_frames"] == 3
    json.dumps(dataset._metadata_cache)


def test_raw_dataloader_uses_decoder_size_and_exact_controls_audio_fps(tmp_path, monkeypatch):
    specs = [(3, 24, 4), (5, 24, 4), (3, 30, 4), (3, 24, 6)]
    torch.save(torch.arange(14).reshape(2, 7), tmp_path / "controls.pt")
    rows = []
    for index, (frames, fps, length) in enumerate(specs * 2):
        torch.save(torch.ones(1, length, 2), tmp_path / f"audio-{index}.pt")
        rows.append({"path": f"{index}.mp4", "num_frames": frames, "fps": fps, "resolution": [999, 999],
                     "action_path": "controls.pt", "action_temporal_axis": 1,
                     "audio_feature_path": f"audio-{index}.pt", "audio_alignment": "sampled_video"})
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(rows))
    dataset = VideoDataset(str(path), video_dir=str(tmp_path))
    dataset.target_h = dataset.target_w = 2
    def video(self, path, frames, return_metadata=False):
        return torch.zeros(3, frames, 2, 2), torch.arange(frames), 7
    monkeypatch.setattr(VideoDataset, "_load_video", video)
    sampler = BucketSampler(dataset, batch_size=2, shuffle=False,
                            bucket_config={"all": {"resolution": [2, 2], "num_frames": 4}})
    assert len(sampler.buckets) == len(specs)
    for batch in DataLoader(dataset, batch_sampler=sampler):
        assert batch["pixel_values"].shape[0] == 2
        assert batch["actions"].shape[1] == batch["pixel_values"].shape[2]
        assert batch["fps"].unique().numel() == 1
        assert all(value.tolist() == [2, 2] for value in batch["resolution"])


def test_metadata_only_dataset_keeps_nearby_frame_lengths_separate():
    class MetadataDataset:
        data = [{"resolution": [2, 2], "num_frames": frames} for frames in (3, 5)]
        def __len__(self):
            return len(self.data)
    sampler = BucketSampler(MetadataDataset(), batch_size=2, shuffle=False)
    assert list(sampler) == [[0], [1]]
