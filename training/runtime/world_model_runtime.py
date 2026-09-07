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
        indices = self.select_memory_indices(
            total_frames=all_frames.shape[2],
            current_chunk_idx=current_chunk_idx,
            chunk_size=chunk_size,
            memory_frames=memory_frames,
            device=all_frames.device,
        )
        if indices is None:
            return None
        return all_frames.index_select(2, indices)

    def select_memory_indices(
        self,
        total_frames: int,
        current_chunk_idx: int,
        chunk_size: int,
        memory_frames: int,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Select memory from the complete prefix before the current chunk.

        ``hybrid_sparse`` preserves a contiguous recent tail and spends the
        remaining budget on chronological anchors from the older history.  This
        keeps the implementation aligned with the paper mechanism instead of
        sparsifying an already-truncated recent window.
        """

        history_end = min(max(0, current_chunk_idx * chunk_size), total_frames)
        if history_end == 0:
            return None

        budget_frames = self.memory_budget_frames or memory_frames
        if budget_frames <= 0:
            return None
        budget_frames = max(1, min(int(budget_frames), history_end))

        if self.memory_policy == "dense_recent":
            return torch.arange(
                history_end - budget_frames,
                history_end,
                device=device,
                dtype=torch.long,
            )

        if self.memory_policy == "strided_history":
            return self._uniform_history_indices(
                history_end=history_end,
                count=budget_frames,
                device=device,
            )

        if self.memory_policy == "hybrid_sparse":
            keep_recent = max(1, int(round(budget_frames * self.memory_recent_ratio)))
            return self._compose_hybrid_indices(
                history_end=history_end,
                budget_frames=budget_frames,
                keep_recent=keep_recent,
                device=device,
            )

        return torch.arange(
            history_end - budget_frames,
            history_end,
            device=device,
            dtype=torch.long,
        )

    def get_or_create_teacher_context(
        self,
        batch: dict[str, Any],
        global_step: int,
        chunk_start: int,
        chunk_end: int,
        frame_indices: Optional[list[int]],
        producer: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        allow_cache = self.cache_mode in {"teacher_context", "hybrid"}
        return self.get_or_create_cached_value(
            namespace="teacher_context",
            batch=batch,
            global_step=global_step,
            producer=producer,
            extra=self._teacher_context_cache_extra(batch, chunk_start, chunk_end, frame_indices),
            allow_cache=allow_cache,
        )

    def _teacher_context_cache_extra(self, batch, chunk_start, chunk_end, frame_indices):
        conditioning = {
            key: batch[key]
            for key in (
                "encoder_hidden_states", "encoder_attention_mask", "pooled_projections",
                "encoder_hidden_states_2", "encoder_attention_mask_2", "image_embeds",
                "encoder_hidden_states_image", "image_cond", "camera_poses", "actions",
                "guidance", "timestep_r", "image_rotary_emb", "rope_interpolation_scale",
            )
            if key in batch and batch[key] is not None
        }
        return {
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "frame_indices": frame_indices or [],
            "prefetch_policy": self.prefetch_policy,
            "memory_policy": self.memory_policy,
            "memory_budget_frames": self.memory_budget_frames,
            "conditioning": conditioning,
            # IDs alone do not identify an augmented/replaced source video.
            "source_latents": batch.get("latents"),
        }

    def prefetch_teacher_context(self, batch, global_step, frame_indices):
        if self.cache_mode not in {"teacher_context", "hybrid"} or not frame_indices:
            return False
        return self.prefetch_cached_value(
            namespace="teacher_context",
            batch=batch,
            global_step=global_step,
            extra=self._teacher_context_cache_extra(
                batch, frame_indices[0], frame_indices[-1] + 1, frame_indices
            ),
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
    def _compose_hybrid_indices(
        history_end: int,
        budget_frames: int,
        keep_recent: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if history_end <= budget_frames:
            return torch.arange(history_end, device=device, dtype=torch.long)

        keep_recent = max(1, min(keep_recent, budget_frames, history_end))
        recent_start = history_end - keep_recent
        recent_tail = torch.arange(recent_start, history_end, device=device, dtype=torch.long)
        if recent_start == 0:
            return recent_tail

        target_history = max(0, budget_frames - keep_recent)
        if target_history == 0:
            return recent_tail

        history_sparse = WorldModelTeacherStudentRuntime._uniform_history_indices(
            history_end=recent_start,
            count=target_history,
            device=device,
        )
        return torch.cat([history_sparse, recent_tail], dim=0)

    @staticmethod
    def _uniform_history_indices(
        history_end: int,
        count: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        count = min(max(1, count), history_end)
        if count == history_end:
            return torch.arange(history_end, device=device, dtype=torch.long)
        return torch.linspace(0, history_end - 1, count, device=device).round().long()
