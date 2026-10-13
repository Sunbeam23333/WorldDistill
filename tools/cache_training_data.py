"""Encode a raw video manifest into the native CachedLatentDataset format.

Uses the checkpoint's actual VAE, text/image encoders and AV connectors. LTX2
audio inputs remain checkpoint-preprocessed, sampled-video-aligned log-mel
features; this command does not guess a waveform/STFT preprocessing recipe.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import default_collate

from training.data.video_dataset import CachedLatentDataset, VideoDataset, collation_signature
from training.student_export import file_sha256
from training.utils.batch_encoder import DiffusersRawBatchEncoder


def model_configuration_fingerprint(model_path):
    root = Path(model_path).resolve()
    if not (root / "model_index.json").is_file():
        raise ValueError("Raw cache encoding requires a local Diffusers pipeline with model_index.json")
    files = {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob("*.json"))
             if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"model_path": str(root), "configuration_sha256": digest, "configuration_files": files}


@contextmanager
def sample_seed(seed):
    """Give each row a reproducible VAE sample without leaking global RNG state."""
    python_state = random.getstate()
    try:
        import numpy as np
    except ImportError:
        np = None
    numpy_state = np.random.get_state() if np is not None else None
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            if np is not None:
                np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        if np is not None:
            np.random.set_state(numpy_state)


def _atomic_json(path, value):
    pending = path.with_suffix(path.suffix + ".incomplete")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def _save_tensor_artifact(root, relative, value, artifacts):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pt.incomplete")
    torch.save(value, pending)
    os.replace(pending, path)
    artifacts[relative] = file_sha256(path)


def _unbatch_tensor(value, name):
    if not torch.is_tensor(value) or value.ndim < 1 or value.shape[0] != 1:
        raise ValueError(f"Encoded {name} must retain the explicit singleton batch dimension")
    if value.is_floating_point() or value.is_complex():
        if not torch.isfinite(value).all():
            raise ValueError(f"Encoded {name} contains non-finite values")
    return value[0].detach().cpu().contiguous().clone()


def cache_training_data(data_json, model_path, output_dir, *, video_dir=None, resolution="480p",
                        num_frames=49, device="cpu", dtype="float32", seed=42, encoder=None):
    """Create a new cache directory; on failure leave an explicit incomplete record.

    ``encoder`` is an optional programmatic encoder implementing encode_batch;
    the CLI always constructs DiffusersRawBatchEncoder from the local pipeline.
    Training consumes manifest.json with cache_dir set to the output directory.
    """
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16 or bfloat16")
    if resolution not in {"360p", "480p", "720p", "1080p"}:
        raise ValueError("resolution must name a VideoDataset resolution preset")
    source, destination = Path(data_json).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite any existing cache directory: {destination}")
    rows = json.loads(source.read_text())
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Raw manifest must contain a nonempty JSON list of sample objects")
    for row in rows:
        count = row.get("num_frames", num_frames)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Each raw sample num_frames must be a positive integer")
    model_record = model_configuration_fingerprint(model_path)
    dataset = VideoDataset(str(source), video_dir=str(Path(video_dir).resolve()) if video_dir else str(source.parent),
                           resolution=resolution, num_frames=num_frames)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    progress_path = destination / "cache_incomplete.json"
    metadata = {"schema_version": 1, "status": "incomplete", "source_manifest": str(source),
        "source_manifest_sha256": file_sha256(source), **model_record, "seed": seed,
        "seed_policy": "(seed + manifest index) modulo 2**63", "device": str(device), "dtype": dtype,
        "resolution": resolution, "requested_num_frames": num_frames, "torch_version": str(torch.__version__),
        "sample_count": len(dataset), "completed_samples": 0, "artifacts": {}}
    _atomic_json(progress_path, metadata)
    manifest = []
    try:
        if encoder is None:
            encoder = DiffusersRawBatchEncoder(str(model_path), torch.device(device), getattr(torch, dtype))
        metadata["encoder_class"] = type(encoder).__name__
        metadata["pipeline_class"] = type(getattr(encoder, "pipe", None)).__name__
        ignored = {"pixel_values", "conditioning_pixel_values", "audio_pixel_values", "source_frame_indices",
                   "text", "resolution", "num_frames", "video_path", "sample_id", "manifest_idx", "fps",
                   "latents", "audio_latents"}
        for index in range(len(dataset)):
            per_sample_seed = (seed + index) % (2 ** 63)
            with sample_seed(per_sample_seed), torch.no_grad():
                raw = dataset[index]
                encoded = encoder.encode_batch(default_collate([raw]))
            if not isinstance(encoded, dict):
                raise TypeError("Raw encoder must return a dictionary of batched tensors and metadata")
            latent = _unbatch_tensor(encoded.get("latents"), "latents")
            if latent.ndim != 4 or min(latent.shape) < 1:
                raise ValueError("Video cache latents must have nonempty C,T,H,W layout")
            name = f"{index:08d}"
            entry = {"sample_id": raw["sample_id"], "latent_path": f"latents/{name}.pt",
                "num_frames": latent.shape[1], "resolution": list(latent.shape[-2:]), "text": raw["text"],
                "source_manifest_idx": index, "source_video": str(Path(raw["video_path"]).resolve()),
                "source_frame_indices": raw["source_frame_indices"].tolist(),
                "sampled_pixel_num_frames": raw["num_frames"], "encoding_seed": per_sample_seed,
                "model_configuration_sha256": model_record["configuration_sha256"]}
            ratio = int(getattr(encoder, "temporal_compression_ratio", 1))
            step = 1 if latent.shape[1] == raw["num_frames"] else ratio
            anchors = torch.arange(latent.shape[1]) * step
            if step < 1 or anchors[-1] >= len(raw["source_frame_indices"]):
                raise ValueError("Encoded latent frames have no valid source-frame alignment")
            entry["latent_source_frame_indices"] = raw["source_frame_indices"][anchors].tolist()
            if "fps" in raw:
                entry["fps"] = raw["fps"]
            conditions = {}
            for key, value in encoded.items():
                if key in ignored or value is None:
                    continue
                if not torch.is_tensor(value):
                    raise TypeError(f"Cannot cache non-tensor condition {key!r} without its model-specific collation contract")
                conditions[key] = _unbatch_tensor(value, key)
            for key in ("actions", "camera_poses"):
                if key in conditions and (conditions[key].ndim < 1 or conditions[key].shape[0] != latent.shape[1]):
                    raise ValueError(f"Cached {key} must align with the latent temporal dimension")
            _save_tensor_artifact(destination, entry["latent_path"], latent, metadata["artifacts"])
            if "audio_latents" in encoded:
                audio = _unbatch_tensor(encoded["audio_latents"], "audio_latents")
                if audio.ndim != 2:
                    raise ValueError("Cached audio must have normalized packed L,D layout")
                entry["audio_latent_path"] = f"audio_latents/{name}.pt"
                _save_tensor_artifact(destination, entry["audio_latent_path"], audio, metadata["artifacts"])
            if conditions:
                entry["conditioning_path"] = f"conditioning/{name}.pt"
                _save_tensor_artifact(destination, entry["conditioning_path"], conditions, metadata["artifacts"])
            sample = {"latents": latent, "num_frames": latent.shape[1], "sample_id": raw["sample_id"],
                      "manifest_idx": index, **conditions}
            if "fps" in entry:
                sample["fps"] = entry["fps"]
            if "audio_latent_path" in entry:
                sample["audio_latents"] = audio
            entry["collation_signature"] = collation_signature(sample)
            manifest.append(entry)
            metadata["completed_samples"] = len(manifest)
            # Avoid rewriting an ever-growing hash inventory after every row.
            # The complete inventory is published once in the final record.
            _atomic_json(progress_path, {key: value for key, value in metadata.items() if key != "artifacts"})
        # Validate the exact interchange consumer before declaring completion.
        pending_manifest = destination / "manifest.pending.json"
        _atomic_json(pending_manifest, manifest)
        cache = CachedLatentDataset(str(pending_manifest), cache_dir=str(destination))
        for index in range(len(cache)):
            cache[index]
            cache.get_metadata(index)
        metadata.update(status="complete", manifest="manifest.json",
                        manifest_sha256=file_sha256(pending_manifest))
        _atomic_json(destination / "cache_metadata.json", metadata)
        # The consumable manifest is the final commit marker, after both tensor
        # artifacts and their provenance record have been written successfully.
        os.replace(pending_manifest, destination / "manifest.json")
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        _atomic_json(progress_path, metadata)
        raise
    try:
        progress_path.unlink(missing_ok=True)
    except OSError as error:
        warnings.warn(f"Cache is complete; could not remove its progress marker: {error}", RuntimeWarning)
    return {"manifest": str(destination / "manifest.json"), "cache_dir": str(destination),
            "metadata": str(destination / "cache_metadata.json"), "samples": len(manifest)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_json", required=True, help="Raw VideoDataset manifest")
    parser.add_argument("--model_path", required=True, help="Local full Diffusers pipeline including auxiliary encoders")
    parser.add_argument("--output_dir", required=True, help="New directory; existing directories are never overwritten")
    parser.add_argument("--video_dir", help="Video/control path root; defaults to the source manifest directory")
    parser.add_argument("--resolution", choices=["360p", "480p", "720p", "1080p"], default="480p")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    print(json.dumps(cache_training_data(**vars(build_parser().parse_args())), indent=2))


if __name__ == "__main__":
    main()
