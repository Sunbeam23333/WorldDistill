"""World-model specific runtime extensions.

This layer focuses on chunk/state oriented execution semantics used by
context-forcing style training. It intentionally implements a research-friendly
runtime for memory selection, teacher-context caching, and next-chunk prefetch
planning.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from training.runtime.teacher_student_runtime import TeacherStudentRuntime


class WorldModelTeacherStudentRuntime(TeacherStudentRuntime):
    """Teacher-student runtime with chunk-state helpers for world models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_policy = getattr(self.args, "runtime_memory_policy", "dense_recent")
        self.memory_budget_frames = max(0, getattr(self.args, "runtime_memory_budget_frames", 0))
        self.memory_recent_ratio = float(getattr(self.args, "runtime_memory_recent_ratio", 0.5))
        self.memory_recent_ratio = min(max(self.memory_recent_ratio, 0.0), 1.0)

    def select_memory_frames(
        self,
        all_frames: torch.Tensor,
        current_chunk_idx: int,
        chunk_size: int,
        memory_frames: int,
    ) -> Optional[torch.Tensor]:
        recent_context = super().select_memory_frames(
            all_frames=all_frames,
            current_chunk_idx=current_chunk_idx,
            chunk_size=chunk_size,
            memory_frames=memory_frames,
        )
        if recent_context is None:
            return None

        budget_frames = self.memory_budget_frames or memory_frames
        budget_frames = max(1, min(int(budget_frames), recent_context.shape[2]))

        if self.memory_policy == "dense_recent":
            return recent_context[:, :, -budget_frames:]

        if self.memory_policy == "strided_history":
            return self._compose_hybrid_context(
                recent_context=recent_context,
                budget_frames=budget_frames,
                keep_recent=max(1, budget_frames // 2),
            )

        if self.memory_policy == "hybrid_sparse":
            keep_recent = max(1, int(round(budget_frames * self.memory_recent_ratio)))
            return self._compose_hybrid_context(
                recent_context=recent_context,
                budget_frames=budget_frames,
                keep_recent=keep_recent,
            )

        return recent_context[:, :, -budget_frames:]

    def get_or_create_teacher_context(
        self,
        batch: dict[str, Any],
        global_step: int,
        chunk_start: int,
        chunk_end: int,
        producer: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        allow_cache = self.cache_mode in {"teacher_context", "hybrid"}
        return self.get_or_create_cached_value(
            namespace="teacher_context",
            batch=batch,
            global_step=global_step,
            producer=producer,
            extra={
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "prefetch_policy": self.prefetch_policy,
                "memory_policy": self.memory_policy,
                "memory_budget_frames": self.memory_budget_frames,
            },
            allow_cache=allow_cache,
        )

    def build_prefetch_request(
        self,
        cache_namespace: str,
        cache_extra: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        request = super().build_prefetch_request(
            cache_namespace=cache_namespace,
            cache_extra=cache_extra,
        )
        if request is None:
            return None
        request["memory_policy"] = self.memory_policy
        request["memory_budget_frames"] = self.memory_budget_frames
        return request

    @staticmethod
    def _compose_hybrid_context(
        recent_context: torch.Tensor,
        budget_frames: int,
        keep_recent: int,
    ) -> torch.Tensor:
        total_context_frames = recent_context.shape[2]
        if total_context_frames <= budget_frames:
            return recent_context

        keep_recent = max(1, min(keep_recent, budget_frames, total_context_frames))
        recent_tail = recent_context[:, :, -keep_recent:]
        history = recent_context[:, :, :-keep_recent]
        if history.shape[2] == 0:
            return recent_tail

        target_history = max(0, budget_frames - keep_recent)
        if target_history == 0:
            return recent_tail

        if history.shape[2] <= target_history:
            history_sparse = history
        else:
            indices = torch.linspace(
                0,
                history.shape[2] - 1,
                target_history,
                device=history.device,
            ).long()
            history_sparse = history.index_select(2, indices)
        return torch.cat([history_sparse, recent_tail], dim=2)
