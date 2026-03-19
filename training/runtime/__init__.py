"""Builders for distillation runtime components."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from loguru import logger

from training.runtime.distill_cache import (
    DistillCache,
    DiskDistillCache,
    HybridDistillCache,
    MemoryDistillCache,
)
from training.runtime.teacher_student_runtime import TeacherStudentRuntime
from training.runtime.world_model_runtime import WorldModelTeacherStudentRuntime


def build_distill_cache(args: Any) -> Optional[DistillCache]:
    backend = getattr(args, "runtime_cache_backend", "none")
    if backend == "none":
        return None

    max_entries = getattr(args, "runtime_cache_max_entries", 256)
    hot_entries = getattr(args, "runtime_cache_hot_entries", max(1, max_entries // 4))
    freshness_steps = getattr(args, "runtime_freshness_steps", 0)
    pin_memory = getattr(args, "runtime_cache_pin_memory", True)

    if backend == "memory":
        logger.info(
            "Building memory DistillCache | "
            f"max_entries={max_entries} | freshness_steps={freshness_steps} | pin_memory={pin_memory}"
        )
        return MemoryDistillCache(
            max_entries=max_entries,
            freshness_steps=freshness_steps,
            store_on_cpu=True,
            pin_memory=pin_memory,
        )

    cache_dir = getattr(args, "runtime_cache_dir", "") or os.path.join(
        getattr(args, "output_dir", "."),
        "runtime_cache",
    )

    if backend == "disk":
        logger.info(
            f"Building disk DistillCache | dir={cache_dir} | max_entries={max_entries} | freshness_steps={freshness_steps}"
        )
        return DiskDistillCache(
            cache_dir=cache_dir,
            max_entries=max_entries,
            freshness_steps=freshness_steps,
        )

    if backend == "hybrid":
        logger.info(
            "Building hybrid DistillCache | "
            f"dir={cache_dir} | hot_entries={hot_entries} | cold_entries={max_entries} | "
            f"freshness_steps={freshness_steps} | pin_memory={pin_memory}"
        )
        return HybridDistillCache(
            cache_dir=cache_dir,
            hot_max_entries=hot_entries,
            cold_max_entries=max_entries,
            freshness_steps=freshness_steps,
            pin_memory=pin_memory,
        )

    raise ValueError(f"Unsupported runtime_cache_backend: {backend}")


def build_runtime(
    args: Any,
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    device: torch.device,
    distill_cache: Optional[DistillCache] = None,
):
    runtime_name = getattr(args, "runtime_name", "auto")
    runtime_requested = (
        getattr(args, "enable_runtime", False)
        or getattr(args, "runtime_enable_dpp", False)
        or getattr(args, "runtime_cache_backend", "none") != "none"
        or getattr(args, "runtime_teacher_cache_mode", "disabled") != "disabled"
        or getattr(args, "runtime_prefetch_policy", "none") != "none"
        or getattr(args, "runtime_memory_budget_frames", 0) > 0
        or getattr(args, "runtime_enable_heterogeneous", False)
        or getattr(args, "runtime_teacher_offload", "none") != "none"
        or runtime_name not in {"auto", "noop", "none"}
    )

    if not runtime_requested:
        return None

    if runtime_name == "auto":
        runtime_name = "world_model" if getattr(args, "distill_method", "") == "context_forcing" else "teacher_student"

    if runtime_name in {"noop", "none"}:
        return None

    runtime_cls = TeacherStudentRuntime
    if runtime_name == "world_model":
        runtime_cls = WorldModelTeacherStudentRuntime
    elif runtime_name != "teacher_student":
        raise ValueError(f"Unsupported runtime_name: {runtime_name}")

    logger.info(
        "Building distillation runtime | "
        f"runtime={runtime_name} | cache_mode={getattr(args, 'runtime_teacher_cache_mode', 'disabled')} | "
        f"memory_policy={getattr(args, 'runtime_memory_policy', 'dense_recent')} | "
        f"heterogeneous={getattr(args, 'runtime_enable_heterogeneous', False)}"
    )
    return runtime_cls(
        args=args,
        teacher_model=teacher_model,
        student_model=student_model,
        device=device,
        distill_cache=distill_cache,
    )


__all__ = [
    "DistillCache",
    "TeacherStudentRuntime",
    "WorldModelTeacherStudentRuntime",
    "build_distill_cache",
    "build_runtime",
]
