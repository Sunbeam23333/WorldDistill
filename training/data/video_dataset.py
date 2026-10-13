"""Video Dataset for distillation training.

Supports two modes:
1. Raw video mode: Load videos + encode to latents on the fly
2. Cached latent mode: Load pre-computed latent/text embeddings (much faster)

The cached mode is recommended for distillation training, as encoding
is expensive and the teacher model already provides the targets.

References:
- HY-WorldPlay CameraJsonWMemDataset: JSON-indexed pre-computed latents
- Open-Sora datasets: Bucket-aware video loading
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from loguru import logger


def collation_signature(sample: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe structure used for exact-shape, field-preserving batching.

    Text and sample identities may vary within a batch. Scalar conditioning
    values (including fps) must agree: some native pipelines consume one scalar
    for the whole batch rather than a per-example vector.
    """
    def describe(value, *, identity=False):
        if torch.is_tensor(value):
            spec = {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
            if value.numel() == 1 and not value.is_meta:
                scalar = value.item()
                spec["scalar"] = str(scalar) if isinstance(scalar, complex) else scalar
            return spec
        if isinstance(value, dict):
            return {key: describe(child) for key, child in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return {"type": type(value).__name__, "items": [describe(child) for child in value]}
        if isinstance(value, str):
            return {"type": "str"}
        if isinstance(value, (bool, int, float)):
            return {"type": type(value).__name__, **({} if identity else {"value": value})}
        raise TypeError(f"Cannot determine collation structure for {type(value).__name__}")

    return {key: describe(value, identity=key == "manifest_idx") for key, value in sorted(sample.items())}


def _load_tensor_headers(path):
    """Memory-map storage; inspect shapes and only touch scalar tensor values.

    Pre-zip torch.save files cannot be memory-mapped. They are read one file at
    a time and immediately released after producing the small JSON signature;
    dataset metadata never retains tensor storage.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


class VideoDataset(Dataset):
    """Generic video dataset that loads raw video files.

    Expects a JSON manifest with entries like:
    {
        "path": "video_001.mp4",
        "text": "A cat playing...",
        "resolution": [480, 854],
        "num_frames": 49,
        "fps": 24
    }

    NOTE: Requires either `decord` or `torchvision.io` for video decoding.
    Install decord via: `pip install decord`.
    """

    def __init__(
        self,
        data_json: str,
        video_dir: str = "",
        resolution: str = "480p",
        num_frames: int = 49,
        transform=None,
    ):
        self.video_dir = video_dir
        self.resolution = resolution
        self.num_frames = num_frames
        self.transform = transform
        self._metadata_cache = {}

        # Parse target resolution
        res_map = {"360p": (360, 640), "480p": (480, 854), "720p": (720, 1280), "1080p": (1080, 1920)}
        self.target_h, self.target_w = res_map.get(resolution, (480, 854))

        with open(data_json, "r") as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples from {data_json}")

    def __len__(self):
        return len(self.data)

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        if idx in self._metadata_cache:
            return self._metadata_cache[idx]
        if self.transform is not None:
            raise ValueError("Shape-safe video bucketing requires fixed decoder output; "
                             "cache transformed videos before using a custom transform with BucketSampler")
        item = self.data[idx]
        frames = int(item.get("num_frames", self.num_frames))
        sample = {"pixel_values": torch.empty((3, frames, self.target_h, self.target_w), device="meta"),
                  "text": "", "resolution": [self.target_h, self.target_w], "num_frames": frames,
                  "video_path": "", "sample_id": "", "manifest_idx": idx,
                  "source_frame_indices": torch.empty(frames, dtype=torch.int64, device="meta")}
        if "fps" in item:
            sample["fps"] = float(item["fps"])
        for path_key, key, prefix in (("audio_feature_path", "audio_pixel_values", None),
                ("action_path", "actions", "action"), ("camera_path", "camera_poses", "camera")):
            if not item.get(path_key):
                continue
            path = item[path_key]
            path = os.path.join(self.video_dir, path) if self.video_dir and not os.path.isabs(path) else path
            value = _load_tensor_headers(path)
            if not torch.is_tensor(value):
                raise TypeError(f"Declared {key} must be a tensor")
            if prefix is not None:
                axis = int(item.get(f"{prefix}_temporal_axis", 0))
                if not value.ndim or not -value.ndim <= axis < value.ndim:
                    raise ValueError(f"Invalid {key} temporal axis")
                shape = list(value.movedim(axis, 0).shape)
                shape[0] = frames
                value = torch.empty(shape, dtype=value.dtype, device="meta")
            sample[key] = value
        metadata = {"resolution": [self.target_h, self.target_w], "num_frames": frames,
                    "collation_signature": collation_signature(sample)}
        self._metadata_cache[idx] = metadata
        return metadata

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        video_key = item.get("path") or item.get("video_path")
        if not video_key:
            raise KeyError("Raw video manifest requires `path` or `video_path` for each sample.")

        video_path = os.path.join(self.video_dir, video_key) if self.video_dir and not os.path.isabs(video_key) else video_key

        frames, frame_indices, source_frames = self._load_video(
            video_path, item.get("num_frames", self.num_frames), return_metadata=True
        )

        result = {
            "pixel_values": frames,  # (C, T, H, W) float32 in [-1, 1]
            "text": item.get("text", ""),
            "resolution": list(frames.shape[-2:]),
            "num_frames": frames.shape[1],
            "video_path": video_path,
            "sample_id": str(item.get("sample_id") or video_path),
            "manifest_idx": idx,
            "source_frame_indices": frame_indices,
        }
        if "fps" in item:
            fps = float(item["fps"])
            if not 0 < fps < float("inf"):
                raise ValueError("Raw video frame rate must be finite and positive")
            result["fps"] = fps
        if item.get("audio_feature_path"):
            if item.get("audio_alignment") != "sampled_video":
                raise ValueError("audio_feature_path must declare audio_alignment='sampled_video': features "
                                 "must match the sampled frames and fps, not an unsampled source soundtrack")
            if "fps" not in result:
                raise ValueError("Audiovisual raw samples require an explicit sampled-video fps")
            audio = self._load_optional_tensor(item["audio_feature_path"], "audio features")
            if audio.ndim != 3 or not torch.isfinite(audio).all():
                raise ValueError("Audio features must be finite C,L,M log-mel tensors from the checkpoint preprocessor")
            result["audio_pixel_values"] = audio

        if "camera_path" in item and item.get("camera_path"):
            result["camera_poses"] = self._load_optional_tensor(item["camera_path"], "camera")

        if "action_path" in item and item.get("action_path"):
            result["actions"] = self._load_optional_tensor(item["action_path"], "action")

        for key, prefix in (("actions", "action"), ("camera_poses", "camera")):
            if key not in result:
                continue
            value = result[key]
            axis = int(item.get(f"{prefix}_temporal_axis", 0))
            if value.ndim == 0 or not -value.ndim <= axis < value.ndim or value.shape[axis] != source_frames:
                raise ValueError(
                    f"Raw {key} must declare one entry per source video frame ({source_frames}); "
                    f"got {tuple(value.shape)} on temporal axis {axis}. Resample timestamped controls before loading."
                )
            # Standardize to T,... before DataLoader adds the batch dimension.
            result[key] = value.index_select(axis, frame_indices).movedim(axis, 0)

        return result

    def _load_optional_tensor(self, tensor_path: str, tensor_name: str) -> torch.Tensor:
        """Load conditioning that was explicitly declared in the manifest."""

        resolved_path = tensor_path
        if self.video_dir and not os.path.isabs(tensor_path):
            resolved_path = os.path.join(self.video_dir, tensor_path)
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"Declared {tensor_name} tensor does not exist: {resolved_path}"
            )
        try:
            value = torch.load(resolved_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load declared {tensor_name} tensor from {resolved_path}: {exc}"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Declared {tensor_name} conditioning must be a tensor, "
                f"got {type(value).__name__}: {resolved_path}"
            )
        return value

    def _load_video(self, video_path: str, num_frames: int, *, return_metadata: bool = False):
        """Load video frames using decord or torchvision fallback.

        Returns:
            Tensor of shape (C, T, H, W) in [-1, 1] range.
        """
        try:
            import decord

            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(video_path, width=self.target_w, height=self.target_h)
            total = len(vr)
            indices = self._sample_frame_indices(total, num_frames)
            frames = vr.get_batch(indices.tolist())  # (T, H, W, C)
            frames = frames.permute(3, 0, 1, 2).float() / 127.5 - 1.0
        except ImportError:
            try:
                import torch.nn.functional as F
                import torchvision.io

                frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
                total = frames.shape[0]
                indices = self._sample_frame_indices(total, num_frames)
                frames = frames[indices]  # (T, H, W, C)
                frames = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # (T, C, H, W)
                if frames.shape[-2:] != (self.target_h, self.target_w):
                    frames = F.interpolate(frames, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
                frames = frames.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot load video '{video_path}'. "
                    f"Install decord (`pip install decord`) or torchvision with video support. "
                    f"Error: {e}"
                )

        frames = self._pad_or_trim_frames(frames, num_frames)
        if indices.numel() < num_frames:
            indices = torch.cat((indices, indices[-1:].expand(num_frames - indices.numel())))

        if self.transform is not None:
            frames = self.transform(frames)

        return (frames, indices, total) if return_metadata else frames

    @staticmethod
    def _sample_frame_indices(total_frames: int, target_frames: int) -> torch.Tensor:
        if total_frames <= 0:
            raise RuntimeError("Video contains no frames.")
        if total_frames >= target_frames:
            return torch.linspace(0, total_frames - 1, target_frames).long()
        return torch.arange(total_frames)

    @staticmethod
    def _pad_or_trim_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
        current_frames = frames.shape[1]
        if current_frames == target_frames:
            return frames
        if current_frames > target_frames:
            return frames[:, :target_frames]
        pad_count = target_frames - current_frames
        pad_frame = frames[:, -1:].repeat(1, pad_count, 1, 1)
        return torch.cat([frames, pad_frame], dim=1)


class CachedLatentDataset(Dataset):
    """Dataset loading pre-computed latents and text embeddings.

    This is the recommended dataset for distillation training.
    Expects pre-computed files:
    - latents: .pt files with shape (C, T, H, W)
    - text_embeds: .pt files with text encoder outputs
    - (optional) image_cond: .pt files with image conditioning

    Manifest format:
    {
        "latent_path": "latents/video_001.pt",
        "text_embed_path": "text_embeds/video_001.pt",
        "image_cond_path": "image_conds/video_001.pt",  (optional)
        "num_frames": 49,
        "resolution": [60, 107],  (latent resolution)
        "text": "A cat playing..."  (for logging)
    }
    """

    def __init__(
        self,
        data_json: str,
        cache_dir: str = "",
        validate_paths: bool = True,
    ):
        self.cache_dir = cache_dir
        self._metadata_cache = {}

        with open(data_json, "r") as f:
            self.data = json.load(f)

        if not isinstance(self.data, list):
            raise ValueError(f"Cached manifest must contain a JSON list: {data_json}")
        if validate_paths:
            self._validate_manifest_paths(data_json)

        logger.info(f"Loaded {len(self.data)} cached samples from {data_json}")

    @staticmethod
    def _sample_id(item: Dict[str, Any], idx: int) -> str:
        return str(item.get("sample_id") or item.get("id") or item.get("latent_path") or idx)

    def _resolve_cache_path(self, path: str) -> str:
        if self.cache_dir and not os.path.isabs(path):
            return os.path.join(self.cache_dir, path)
        return path

    def _validate_manifest_paths(self, manifest_path: str) -> None:
        path_keys = (
            "latent_path",
            "text_embed_path",
            "image_cond_path",
            "camera_path",
            "action_path",
            "conditioning_path",
            "audio_latent_path",
        )
        for idx, item in enumerate(self.data):
            if not isinstance(item, dict):
                raise ValueError(f"Cached manifest entry {idx} is not an object: {manifest_path}")
            sample_id = self._sample_id(item, idx)
            latent_path = item.get("latent_path")
            if not latent_path:
                raise ValueError(f"Cached sample '{sample_id}' is missing required latent_path")
            for path_key in path_keys:
                declared_path = item.get(path_key)
                if not declared_path:
                    continue
                resolved_path = self._resolve_cache_path(str(declared_path))
                if not os.path.isfile(resolved_path):
                    raise FileNotFoundError(
                        f"Cached sample '{sample_id}' declares missing {path_key}: {resolved_path}"
                    )

    def _load_cached_tensor(self, item: Dict[str, Any], idx: int, path_key: str, *, metadata_only=False):
        declared_path = item.get(path_key)
        if not declared_path:
            return None
        resolved_path = self._resolve_cache_path(str(declared_path))
        sample_id = self._sample_id(item, idx)
        try:
            if metadata_only:
                return _load_tensor_headers(resolved_path)
            return torch.load(resolved_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cached {path_key} for sample '{sample_id}' "
                f"from '{resolved_path}': {exc}"
            ) from exc

    def __len__(self):
        return len(self.data)

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Inspect actual tensor headers once; do not trust stale manifest sizes."""
        if idx not in self._metadata_cache:
            sample = self._read_sample(idx, metadata_only=True)
            signature = collation_signature(sample)
            declared = self.data[idx].get("collation_signature")
            if declared is not None and declared != signature:
                raise ValueError(f"Cached sample '{self._sample_id(self.data[idx], idx)}' "
                                 "collation_signature does not match its tensor artifacts")
            self._metadata_cache[idx] = {"resolution": list(sample["latents"].shape[-2:]),
                                        "num_frames": sample["num_frames"], "collation_signature": signature}
        return self._metadata_cache[idx]

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self._read_sample(idx)

    def _read_sample(self, idx, *, metadata_only=False) -> Dict[str, Any]:
        item = self.data[idx]
        def load(path_key):
            return self._load_cached_tensor(item, idx, path_key, metadata_only=metadata_only)

        # Load pre-computed latent
        latents = load("latent_path")
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                f"Cached latent for sample '{self._sample_id(item, idx)}' must be a tensor, "
                f"got {type(latents).__name__}"
            )
        if latents.ndim < 2:
            raise ValueError(
                f"Cached latent for sample '{self._sample_id(item, idx)}' must have at least 2 dimensions, "
                f"got shape {tuple(latents.shape)}"
            )

        sample_id = self._sample_id(item, idx)
        result = {
            "latents": latents,  # (C, T, H, W)
            "num_frames": latents.shape[1] if latents.dim() == 4 else 1,
            "sample_id": sample_id,
            "manifest_idx": idx,
        }
        if "fps" in item:
            fps = float(item["fps"])
            if not 0 < fps < float("inf"):
                raise ValueError("Cached video frame rate must be finite and positive")
            result["fps"] = fps
        if item.get("audio_latent_path"):
            audio = load("audio_latent_path")
            if not torch.is_tensor(audio) or audio.ndim != 2 or (not metadata_only and not torch.isfinite(audio).all()):
                raise ValueError("Cached audio latents must be normalized packed finite L,D tensors")
            result["audio_latents"] = audio

        # Model-specific multi-encoder tensors (e.g. Hunyuan pooled/text2/masks)
        # travel together, rather than being silently reduced to one embedding.
        if item.get("conditioning_path"):
            conditioning = load("conditioning_path")
            if not isinstance(conditioning, dict) or any(not isinstance(k, str) or not torch.is_tensor(v)
                                                         for k, v in conditioning.items()):
                raise TypeError("conditioning_path must contain a dict of named tensors")
            reserved = {"latents", "num_frames", "sample_id", "manifest_idx", "fps"}
            if "audio_latents" in result:
                reserved.add("audio_latents")
            if reserved.intersection(conditioning):
                raise ValueError("conditioning_path cannot override latent/sample metadata")
            result.update(conditioning)

        # Load text embeddings
        if "text_embed_path" in item:
            text_embeds = load("text_embed_path")
            if text_embeds is not None:
                result["encoder_hidden_states"] = text_embeds

        # Load image conditioning (for i2v tasks)
        if "image_cond_path" in item and item.get("image_cond_path"):
            result["image_cond"] = load("image_cond_path")

        # Load camera/action data (for world models)
        if "camera_path" in item and item.get("camera_path"):
            result["camera_poses"] = load("camera_path")

        if "action_path" in item and item.get("action_path"):
            result["actions"] = load("action_path")

        return result
