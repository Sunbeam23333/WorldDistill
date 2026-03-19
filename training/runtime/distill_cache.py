"""Distillation cache primitives for training runtime.

This module intentionally keeps the implementation lightweight so it can serve
as a safe skeleton for future research features such as freshness-aware
supervision caching, heterogeneous cache placement, and chunk-level world model
state reuse.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry with minimal metadata for freshness control."""

    payload: Any
    created_step: int
    created_time: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class DistillCache:
    """Base cache with common stats + freshness helpers."""

    def __init__(self, max_entries: int = 256, freshness_steps: int = 0):
        self.max_entries = max(1, max_entries)
        self.freshness_steps = max(0, freshness_steps)
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0

    def get(self, key: str, current_step: Optional[int] = None) -> Any:
        raise NotImplementedError

    def put(
        self,
        key: str,
        payload: Any,
        current_step: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def invalidate(self, key: str) -> None:
        raise NotImplementedError

    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "puts": self._puts,
            "evictions": self._evictions,
        }

    def _is_stale(self, entry: CacheEntry, current_step: Optional[int]) -> bool:
        if current_step is None or self.freshness_steps <= 0:
            return False
        return (current_step - entry.created_step) > self.freshness_steps

    def _record_hit(self) -> None:
        self._hits += 1

    def _record_miss(self) -> None:
        self._misses += 1

    def _record_put(self) -> None:
        self._puts += 1

    def _record_eviction(self) -> None:
        self._evictions += 1


class MemoryDistillCache(DistillCache):
    """Simple in-memory LRU cache.

    By default payloads are detached and materialized on CPU. This keeps the
    skeleton safe for training even before a dedicated heterogeneous placement
    policy is implemented.
    """

    def __init__(
        self,
        max_entries: int = 256,
        freshness_steps: int = 0,
        store_on_cpu: bool = True,
        pin_memory: bool = False,
    ):
        super().__init__(max_entries=max_entries, freshness_steps=freshness_steps)
        self.store_on_cpu = store_on_cpu
        self.pin_memory = pin_memory
        self._storage: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str, current_step: Optional[int] = None) -> Any:
        entry = self._storage.get(key)
        if entry is None:
            self._record_miss()
            return None
        if self._is_stale(entry, current_step):
            self.invalidate(key)
            self._record_miss()
            return None
        self._storage.move_to_end(key)
        self._record_hit()
        return clone_payload(entry.payload, preserve_pin_memory=self.pin_memory)

    def put(
        self,
        key: str,
        payload: Any,
        current_step: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        stored_payload = detach_payload(
            payload,
            to_cpu=self.store_on_cpu,
            pin_memory=self.pin_memory,
        )
        if key in self._storage:
            self._storage.move_to_end(key)
        self._storage[key] = CacheEntry(
            payload=stored_payload,
            created_step=current_step,
            metadata=metadata or {},
        )
        while len(self._storage) > self.max_entries:
            self._storage.popitem(last=False)
            self._record_eviction()
        self._record_put()

    def invalidate(self, key: str) -> None:
        if key in self._storage:
            del self._storage[key]


class DiskDistillCache(DistillCache):
    """Disk-backed cache for teacher outputs or world-model chunk states."""

    def __init__(
        self,
        cache_dir: str,
        max_entries: int = 256,
        freshness_steps: int = 0,
    ):
        super().__init__(max_entries=max_entries, freshness_steps=freshness_steps)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._index: OrderedDict[str, str] = OrderedDict()

    def _path_for_key(self, key: str) -> str:
        hashed = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed}.pt")

    def get(self, key: str, current_step: Optional[int] = None) -> Any:
        path = self._path_for_key(key)
        if not os.path.exists(path):
            self._record_miss()
            return None
        try:
            entry_dict = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning(f"Failed to load distill cache entry {path}: {exc}")
            self._record_miss()
            return None

        entry = CacheEntry(
            payload=entry_dict.get("payload"),
            created_step=entry_dict.get("created_step", 0),
            created_time=entry_dict.get("created_time", time.time()),
            metadata=entry_dict.get("metadata", {}),
        )
        if self._is_stale(entry, current_step):
            self.invalidate(key)
            self._record_miss()
            return None

        self._index[key] = path
        self._index.move_to_end(key)
        self._record_hit()
        return clone_payload(entry.payload)

    def put(
        self,
        key: str,
        payload: Any,
        current_step: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        path = self._path_for_key(key)
        stored_payload = detach_payload(payload, to_cpu=True, pin_memory=False)
        torch.save(
            {
                "payload": stored_payload,
                "created_step": current_step,
                "created_time": time.time(),
                "metadata": metadata or {},
            },
            path,
        )
        self._index[key] = path
        self._index.move_to_end(key)
        while len(self._index) > self.max_entries:
            _, evict_path = self._index.popitem(last=False)
            if os.path.exists(evict_path):
                os.remove(evict_path)
            self._record_eviction()
        self._record_put()

    def invalidate(self, key: str) -> None:
        path = self._path_for_key(key)
        self._index.pop(key, None)
        if os.path.exists(path):
            os.remove(path)


class HybridDistillCache(DistillCache):
    """Two-tier cache with CPU-pinned hot entries and disk-backed cold entries.

    This intentionally keeps the policy simple and research-friendly:
    - hot tier: fast CPU memory for recently reused supervision/context
    - cold tier: persistent disk tier for longer-lived reuse across chunks/steps
    - promotion: disk hit is promoted into the hot tier
    """

    def __init__(
        self,
        cache_dir: str,
        hot_max_entries: int = 64,
        cold_max_entries: int = 256,
        freshness_steps: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__(max_entries=max(1, cold_max_entries), freshness_steps=freshness_steps)
        self.hot_cache = MemoryDistillCache(
            max_entries=max(1, hot_max_entries),
            freshness_steps=freshness_steps,
            store_on_cpu=True,
            pin_memory=pin_memory,
        )
        self.cold_cache = DiskDistillCache(
            cache_dir=cache_dir,
            max_entries=max(1, cold_max_entries),
            freshness_steps=freshness_steps,
        )
        self._hot_hits = 0
        self._cold_hits = 0
        self._promotions = 0

    def get(self, key: str, current_step: Optional[int] = None) -> Any:
        hot_value = self.hot_cache.get(key, current_step=current_step)
        if hot_value is not None:
            self._hot_hits += 1
            self._record_hit()
            return hot_value

        cold_value = self.cold_cache.get(key, current_step=current_step)
        if cold_value is None:
            self._record_miss()
            return None

        self._cold_hits += 1
        self._promotions += 1
        self._record_hit()
        self.hot_cache.put(
            key,
            cold_value,
            current_step=current_step if current_step is not None else 0,
            metadata={"promoted": True},
        )
        return cold_value

    def put(
        self,
        key: str,
        payload: Any,
        current_step: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}
        persist_to_disk = metadata.get("persist_to_disk", True)
        self.hot_cache.put(key, payload, current_step=current_step, metadata=metadata)
        if persist_to_disk:
            self.cold_cache.put(key, payload, current_step=current_step, metadata=metadata)
        self._record_put()

    def invalidate(self, key: str) -> None:
        self.hot_cache.invalidate(key)
        self.cold_cache.invalidate(key)

    def stats(self) -> dict[str, int]:
        hot_stats = self.hot_cache.stats()
        cold_stats = self.cold_cache.stats()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "puts": self._puts,
            "evictions": hot_stats["evictions"] + cold_stats["evictions"],
            "hot_hits": self._hot_hits,
            "cold_hits": self._cold_hits,
            "promotions": self._promotions,
        }


def detach_payload(payload: Any, to_cpu: bool = True, pin_memory: bool = False) -> Any:
    """Detach tensors recursively before putting them into cache."""

    if isinstance(payload, torch.Tensor):
        detached = payload.detach()
        if to_cpu:
            detached = detached.cpu()
            if pin_memory:
                detached = _maybe_pin_tensor(detached)
        return detached
    if isinstance(payload, dict):
        return {k: detach_payload(v, to_cpu=to_cpu, pin_memory=pin_memory) for k, v in payload.items()}
    if isinstance(payload, list):
        return [detach_payload(v, to_cpu=to_cpu, pin_memory=pin_memory) for v in payload]
    if isinstance(payload, tuple):
        return tuple(detach_payload(v, to_cpu=to_cpu, pin_memory=pin_memory) for v in payload)
    return payload


def move_payload_to_device(payload: Any, device: torch.device) -> Any:
    if isinstance(payload, torch.Tensor):
        return payload.to(device, non_blocking=True)
    if isinstance(payload, dict):
        return {k: move_payload_to_device(v, device) for k, v in payload.items()}
    if isinstance(payload, list):
        return [move_payload_to_device(v, device) for v in payload]
    if isinstance(payload, tuple):
        return tuple(move_payload_to_device(v, device) for v in payload)
    return payload


def clone_payload(payload: Any, preserve_pin_memory: bool = False) -> Any:
    if isinstance(payload, torch.Tensor):
        cloned = payload.clone()
        if preserve_pin_memory and cloned.device.type == "cpu" and payload.is_pinned():
            cloned = _maybe_pin_tensor(cloned)
        return cloned
    if isinstance(payload, dict):
        return {k: clone_payload(v, preserve_pin_memory=preserve_pin_memory) for k, v in payload.items()}
    if isinstance(payload, list):
        return [clone_payload(v, preserve_pin_memory=preserve_pin_memory) for v in payload]
    if isinstance(payload, tuple):
        return tuple(clone_payload(v, preserve_pin_memory=preserve_pin_memory) for v in payload)
    return payload


def _maybe_pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor
