"""Opt-in, topology-independent data cursors and elastic checkpoint publication.

The cursor counts committed *global batches*, not prefetched local batches.
Changing the data-parallel degree changes batch grouping and random streams;
this is a resumable training policy, not bitwise-equivalent replay.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import Sampler


class ElasticBatchSampler(Sampler):
    SCHEMA = "worlddistill-elastic-batches-v1"

    def __init__(self, dataset, *, batch_size, rank=0, world_size=1, seed=42,
                 shuffle=True, drop_last=False, bucket_sampler=None):
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Elastic sampler requires positive batch size/world size and valid rank")
        if bucket_sampler is not None and (bucket_sampler.world_size != 1 or bucket_sampler.rank != 0):
            raise ValueError("Elastic bucket source must contain the unpartitioned world-size-one batch plan")
        self.dataset = dataset
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.seed, self.shuffle, self.drop_last = seed, shuffle, drop_last
        self.bucket_sampler = bucket_sampler
        self.epoch = self.global_cursor = 0
        self._plan = None
        # Manifest identity protects against silently resuming a reordered data
        # source of the same length. Tensor payloads are deliberately not read.
        manifest = getattr(dataset, "data", None)
        identity = {"length": len(dataset), "manifest": manifest,
                    "batch_size": batch_size, "seed": seed, "shuffle": shuffle,
                    "drop_last": drop_last,
                    "buckets": getattr(bucket_sampler, "buckets", None)}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()

    def _batches(self):
        if self._plan is None:
            if self.bucket_sampler is not None:
                self.bucket_sampler.set_epoch(self.epoch)
                self._plan = list(self.bucket_sampler)
            else:
                generator = torch.Generator().manual_seed(self.seed + self.epoch)
                indices = (torch.randperm(len(self.dataset), generator=generator).tolist()
                           if self.shuffle else list(range(len(self.dataset))))
                self._plan = [indices[start:start + self.batch_size]
                              for start in range(0, len(indices), self.batch_size)
                              if not self.drop_last or start + self.batch_size <= len(indices)]
            if not self._plan:
                raise ValueError("Elastic training requires a nonempty global batch plan")
        return self._plan

    def __iter__(self):
        plan = self._batches()
        # Snapshot the committed cursor. DataLoader may prefetch from this
        # iterator but only the trainer may advance the persistent cursor.
        start = self.global_cursor
        for group_start in range(start, len(plan), self.world_size):
            position = group_start + self.rank
            # Equal collective counts require tail padding. Only unconsumed
            # final batches are repeated; prior committed data is never replayed.
            if position >= len(plan):
                position = group_start + (self.rank % (len(plan) - group_start))
            yield list(plan[position])

    def __len__(self):
        return math.ceil((len(self._batches()) - self.global_cursor) / self.world_size)

    def commit_batch(self):
        size = len(self._batches())
        if self.global_cursor >= size:
            raise RuntimeError("Elastic sampler committed beyond the epoch plan")
        self.global_cursor = min(size, self.global_cursor + self.world_size)

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch, self.global_cursor, self._plan = int(epoch), 0, None

    def state_dict(self):
        return {"schema": self.SCHEMA, "epoch": self.epoch,
                "global_cursor": self.global_cursor, "global_batches": len(self._batches()),
                "fingerprint": self.fingerprint}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("schema") != self.SCHEMA:
            raise ValueError("Missing valid elastic global data cursor")
        if state.get("fingerprint") != self.fingerprint:
            raise ValueError("Elastic dataset/batch plan differs from the checkpoint")
        for key in ("epoch", "global_cursor", "global_batches"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f"Invalid elastic data cursor {key}")
        self.set_epoch(state["epoch"])
        if state["global_batches"] != len(self._batches()) or state["global_cursor"] > len(self._batches()):
            raise ValueError("Elastic data cursor exceeds or differs from the epoch plan")
        self.global_cursor = state["global_cursor"]


def elastic_seed(seed, step, data_rank, generation):
    payload = f"worlddistill-elastic:{seed}:{step}:{data_rank}:{generation}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def latest_elastic_checkpoint(output_dir):
    """Discover only fully published checkpoints; ignore partial writes."""
    candidates = []
    for path in Path(output_dir).glob("checkpoint-*/elastic_complete.json"):
        state = json.loads(path.read_text())
        if state.get("schema") != "worlddistill-elastic-complete-v1":
            raise ValueError(f"Invalid elastic completion marker: {path}")
        if type(state.get("step")) is not int or state["step"] < 0:
            raise ValueError(f"Invalid elastic checkpoint step: {path}")
        for name, size in state.get("files", {}).items():
            target = path.parent / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not target.is_file() or target.stat().st_size != size:
                raise ValueError(f"Incomplete elastic checkpoint file: {target}")
        if not state.get("files"):
            raise ValueError(f"Empty elastic checkpoint manifest: {path}")
        candidates.append((state["step"], str(path.parent)))
    return max(candidates, default=(None, None))[1]


def publish_elastic_checkpoint(directory, step):
    """Publish after subclass sidecars have completed, using atomic rename."""
    directory = Path(directory)
    files = {str(path.relative_to(directory)): path.stat().st_size
             for path in directory.rglob("*") if path.is_file()
             and path.name != "elastic_complete.json" and ".tmp" not in path.name}
    if not files:
        raise ValueError("Cannot publish an empty elastic checkpoint")
    target = directory / "elastic_complete.json"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps({"schema": "worlddistill-elastic-complete-v1", "step": step, "files": files}))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
