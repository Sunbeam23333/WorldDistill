"""Bucket Sampler for video data.

Groups samples by resolution and/or frame count into buckets, then samples
batches from each bucket. This ensures all samples in a batch have the
same spatial/temporal dimensions, avoiding expensive padding.

Supports distributed training: each rank gets a disjoint partition of indices.

References:
- Open-Sora VariableVideoBatchSampler: Resolution-aware bucket sampling
- AspectRatioBucketManager: Group by aspect ratio
"""

import math
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Sampler
from loguru import logger


class BucketSampler(Sampler):
    """Sampler that groups samples by resolution/frame-count buckets.

    Each bucket contains samples with similar resolution and frame count.
    Batches are drawn from within a single bucket to ensure uniform dimensions.

    Args:
        data_source: Dataset with `get_metadata(idx)` or JSON-based `data` attribute.
        batch_size: Number of samples per batch.
        bucket_config: Dict mapping bucket_name -> {resolution, num_frames, max_batch_size}.
            If None, auto-generates buckets from data.
        shuffle: Whether to shuffle within and across buckets.
        drop_last: Whether to drop incomplete batches.
        seed: Random seed for reproducibility.
        rank: Current process rank (for distributed training).
        world_size: Total number of processes.
    """

    def __init__(
        self,
        data_source,
        batch_size: int = 1,
        bucket_config: Optional[Dict] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size

        # Build buckets
        if bucket_config is not None:
            self.buckets = self._build_from_config(bucket_config)
        else:
            self.buckets = self._auto_bucket()

        self._log_bucket_stats()

    def _get_item_metadata(self, idx: int) -> Dict:
        """Get metadata for a sample without loading heavy tensors."""
        # Prefer lightweight metadata accessor
        if hasattr(self.data_source, "get_metadata"):
            return self.data_source.get_metadata(idx)
        # Fallback: read from .data list (JSON manifest) directly
        if hasattr(self.data_source, "data") and isinstance(self.data_source.data, list):
            item = self.data_source.data[idx]
            return {
                "resolution": item.get("resolution", [480, 854]),
                "num_frames": item.get("num_frames", 49),
            }
        # Last resort: call __getitem__ (may be slow for CachedLatentDataset)
        item = self.data_source[idx]
        return {
            "resolution": item.get("resolution", [480, 854]),
            "num_frames": item.get("num_frames", 49),
        }

    def _auto_bucket(self) -> Dict[str, List[int]]:
        """Automatically group samples into buckets by resolution and frames."""
        buckets: Dict[str, List[int]] = {}

        for idx in range(len(self.data_source)):
            meta = self._get_item_metadata(idx)
            res = tuple(meta.get("resolution", [480, 854]))
            nf = meta.get("num_frames", 49)

            # Quantize frames to nearest 16
            nf_bucket = max(1, (nf // 16) * 16)
            key = f"{res[0]}x{res[1]}_f{nf_bucket}"

            if key not in buckets:
                buckets[key] = []
            buckets[key].append(idx)

        return buckets

    def _build_from_config(self, config: Dict) -> Dict[str, List[int]]:
        """Build buckets from explicit configuration."""
        buckets: Dict[str, List[int]] = {}
        for bucket_name in config:
            buckets[bucket_name] = []

        # Assign samples to nearest bucket
        for idx in range(len(self.data_source)):
            meta = self._get_item_metadata(idx)
            res = tuple(meta.get("resolution", [480, 854]))
            nf = meta.get("num_frames", 49)

            best_bucket = None
            best_dist = float("inf")
            for bname, bspec in config.items():
                bres = tuple(bspec.get("resolution", [480, 854]))
                bnf = bspec.get("num_frames", 49)
                dist = abs(res[0] - bres[0]) + abs(res[1] - bres[1]) + abs(nf - bnf) * 10
                if dist < best_dist:
                    best_dist = dist
                    best_bucket = bname

            if best_bucket:
                buckets[best_bucket].append(idx)

        return buckets

    def _log_bucket_stats(self):
        """Log bucket distribution."""
        total = sum(len(v) for v in self.buckets.values())
        for name, indices in sorted(self.buckets.items()):
            pct = len(indices) / total * 100 if total > 0 else 0
            logger.info(f"  Bucket '{name}': {len(indices)} samples ({pct:.1f}%)")

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices, each batch from a single bucket.

        For distributed training, each rank gets a disjoint subset of batches.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Collect all batches from all buckets
        all_batches: List[List[int]] = []

        for bucket_name, indices in self.buckets.items():
            if len(indices) == 0:
                continue

            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                shuffled = [indices[i] for i in perm]
            else:
                shuffled = list(indices)

            # Create batches from this bucket
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batches across buckets
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        # Distributed: partition batches across ranks
        if self.world_size > 1:
            # Pad to make divisible by world_size
            remainder = len(all_batches) % self.world_size
            if remainder != 0:
                padding = self.world_size - remainder
                repeats = math.ceil(padding / len(all_batches)) if all_batches else 0
                all_batches += (all_batches * repeats)[:padding]
            all_batches = all_batches[self.rank :: self.world_size]

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for indices in self.buckets.values():
            n = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n += 1
            total += n
        if self.world_size > 1:
            total = math.ceil(total / self.world_size)
        return total

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
