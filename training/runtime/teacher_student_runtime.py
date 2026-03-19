"""Teacher-student runtime with cache-aware and stream-aware execution.

This runtime remains research-friendly, but now provides a concrete execution
substrate for distillation-aware overlap:
- teacher-output / teacher-context caching
- CUDA-stream based teacher-student overlap (DPP primitive)
- cache-backed supervision buffering
- future heterogeneous placement / prefetch planning
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from loguru import logger

from training.runtime.distill_cache import DistillCache, move_payload_to_device


@dataclass
class RuntimeStepState:
    global_step: int = -1
    batch_size: int = 0
    sample_ids: list[str] = field(default_factory=list)
    manifest_indices: list[int] = field(default_factory=list)
    planned_prefetches: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RuntimePlacementPlan:
    namespace: str
    execution_device: str
    offload_target: str
    cache_backend: str
    allow_async_prefetch: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "execution_device": self.execution_device,
            "offload_target": self.offload_target,
            "cache_backend": self.cache_backend,
            "allow_async_prefetch": self.allow_async_prefetch,
        }


@dataclass
class PendingTeacherForward:
    output: Any
    global_step: int
    cache_key: Optional[str] = None
    cache_enabled: bool = False
    cache_metadata: Optional[dict[str, Any]] = None
    ready_event: Optional[torch.cuda.Event] = None
    placement_plan: Optional[RuntimePlacementPlan] = None
    cache_namespace: str = "teacher_output"
    from_cache: bool = False
    cache_written: bool = False


class TeacherStudentRuntime:
    """Unified runtime boundary for teacher/student execution."""

    def __init__(
        self,
        args: Any,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: torch.device,
        distill_cache: Optional[DistillCache] = None,
    ):
        self.args = args
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.distill_cache = distill_cache

        self.runtime_name = getattr(args, "runtime_name", "teacher_student")
        self.cache_backend = getattr(args, "runtime_cache_backend", "none")
        self.cache_mode = getattr(args, "runtime_teacher_cache_mode", "disabled")
        self.prefetch_policy = getattr(args, "runtime_prefetch_policy", "none")
        self.enable_heterogeneous = getattr(args, "runtime_enable_heterogeneous", False)
        self.teacher_offload = getattr(args, "runtime_teacher_offload", "none")
        self.enable_dpp = bool(getattr(args, "runtime_enable_dpp", False))
        self.teacher_stream_priority = int(getattr(args, "runtime_teacher_stream_priority", 0))

        self._heterogeneous_notice_emitted = False
        self._placement_notice_emitted = False
        self._dpp_notice_emitted = False
        self._planned_prefetches = 0
        self._async_teacher_launches = 0
        self._async_teacher_waits = 0
        self._async_teacher_cache_hits = 0
        self.step_state = RuntimeStepState()
        self._teacher_stream: Optional[torch.cuda.Stream] = None
        self._maybe_init_dpp_stream()

    def begin_step(self, global_step: int, batch: dict[str, Any]) -> RuntimeStepState:
        self.step_state = RuntimeStepState(
            global_step=global_step,
            batch_size=self._infer_batch_size(batch),
            sample_ids=self._extract_sample_ids(batch),
            manifest_indices=self._extract_manifest_indices(batch),
        )
        return self.step_state

    def finish_step(self) -> None:
        return None

    def can_pipeline_teacher_student(self) -> bool:
        return self._teacher_stream is not None

    def launch_teacher(
        self,
        input_kwargs: dict[str, Any],
        batch: dict[str, Any],
        global_step: int,
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[dict[str, Any]] = None,
        allow_cache: bool = True,
    ) -> PendingTeacherForward:
        self.begin_step(global_step, batch)
        placement_plan = self._build_teacher_placement_plan(cache_namespace)
        use_cache = (
            allow_cache
            and self.distill_cache is not None
            and self.cache_mode in {"teacher_output", "hybrid"}
        )
        cache_key = None
        if use_cache:
            assert self.distill_cache is not None
            cache_key = self.build_cache_key(
                namespace=cache_namespace,
                batch=batch,
                input_kwargs=input_kwargs,
                extra=cache_extra,
            )
            cached = self.distill_cache.get(cache_key, current_step=global_step)
            if cached is not None:
                self._async_teacher_cache_hits += 1
                self._plan_prefetch(
                    cache_namespace=cache_namespace,
                    cache_extra=cache_extra,
                    placement_plan=placement_plan,
                )
                return PendingTeacherForward(
                    output=move_payload_to_device(cached, self.device),
                    global_step=global_step,
                    cache_key=cache_key,
                    cache_enabled=True,
                    cache_metadata=None,
                    ready_event=None,
                    placement_plan=placement_plan,
                    cache_namespace=cache_namespace,
                    from_cache=True,
                    cache_written=True,
                )

        self._maybe_log_heterogeneous_mode(placement_plan)
        cache_metadata = None
        if use_cache and cache_key is not None:
            cache_metadata = {
                **placement_plan.to_metadata(),
                "runtime": self.runtime_name,
                "sample_ids": self.step_state.sample_ids,
                "persist_to_disk": self._should_persist_to_disk(cache_namespace),
            }
            if cache_extra:
                cache_metadata["cache_extra"] = self._summarize_object(cache_extra)

        self._plan_prefetch(
            cache_namespace=cache_namespace,
            cache_extra=cache_extra,
            placement_plan=placement_plan,
        )

        if self.can_pipeline_teacher_student():
            teacher_stream = self._teacher_stream
            assert teacher_stream is not None
            current_stream = torch.cuda.current_stream(device=self.device)
            teacher_stream.wait_stream(current_stream)
            with torch.cuda.stream(teacher_stream):
                output = self._run_model(self.teacher_model, input_kwargs, no_grad=True, tag="teacher")
            ready_event = torch.cuda.Event(blocking=False)
            teacher_stream.record_event(ready_event)
            self._async_teacher_launches += 1
            self._maybe_log_dpp_mode()
            return PendingTeacherForward(
                output=output,
                global_step=global_step,
                cache_key=cache_key,
                cache_enabled=use_cache and cache_key is not None,
                cache_metadata=cache_metadata,
                ready_event=ready_event,
                placement_plan=placement_plan,
                cache_namespace=cache_namespace,
                from_cache=False,
                cache_written=False,
            )

        output = self._run_model(self.teacher_model, input_kwargs, no_grad=True, tag="teacher")
        return PendingTeacherForward(
            output=output,
            global_step=global_step,
            cache_key=cache_key,
            cache_enabled=use_cache and cache_key is not None,
            cache_metadata=cache_metadata,
            ready_event=None,
            placement_plan=placement_plan,
            cache_namespace=cache_namespace,
            from_cache=False,
            cache_written=False,
        )

    def wait_teacher(self, handle: PendingTeacherForward) -> Any:
        if handle.ready_event is not None and self.device.type == "cuda":
            current_stream = torch.cuda.current_stream(device=self.device)
            current_stream.wait_event(handle.ready_event)
            self._async_teacher_waits += 1

        if handle.cache_enabled and handle.cache_key is not None and not handle.cache_written:
            assert self.distill_cache is not None
            self.distill_cache.put(
                handle.cache_key,
                handle.output,
                current_step=handle.global_step,
                metadata=handle.cache_metadata or {},
            )
            handle.cache_written = True
        return handle.output

    def run_teacher(
        self,
        input_kwargs: dict[str, Any],
        batch: dict[str, Any],
        global_step: int,
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[dict[str, Any]] = None,
        allow_cache: bool = True,
    ) -> Any:
        handle = self.launch_teacher(
            input_kwargs=input_kwargs,
            batch=batch,
            global_step=global_step,
            cache_namespace=cache_namespace,
            cache_extra=cache_extra,
            allow_cache=allow_cache,
        )
        return self.wait_teacher(handle)

    def run_student(
        self,
        model: nn.Module,
        input_kwargs: dict[str, Any],
        batch: dict[str, Any],
        global_step: int,
        tag: str = "student",
    ) -> Any:
        self.begin_step(global_step, batch)
        return self._run_model(model, input_kwargs, no_grad=False, tag=tag)

    def get_or_create_cached_value(
        self,
        namespace: str,
        batch: dict[str, Any],
        global_step: int,
        producer: Callable[[], Any],
        extra: Optional[dict[str, Any]] = None,
        allow_cache: bool = True,
    ) -> Any:
        if not allow_cache or self.distill_cache is None:
            return producer()

        cache_key = self.build_cache_key(namespace=namespace, batch=batch, extra=extra)
        cached = self.distill_cache.get(cache_key, current_step=global_step)
        if cached is not None:
            return move_payload_to_device(cached, self.device)

        value = producer()
        self.distill_cache.put(
            cache_key,
            value,
            current_step=global_step,
            metadata={
                "namespace": namespace,
                "runtime": self.runtime_name,
                "persist_to_disk": self._should_persist_to_disk(namespace),
            },
        )
        return value

    def select_memory_frames(
        self,
        all_frames: torch.Tensor,
        current_chunk_idx: int,
        chunk_size: int,
        memory_frames: int,
    ) -> Optional[torch.Tensor]:
        start_frame = current_chunk_idx * chunk_size
        mem_start = max(0, start_frame - memory_frames)
        mem_end = start_frame
        if mem_end <= mem_start:
            return None
        return all_frames[:, :, mem_start:mem_end]

    def build_cache_key(
        self,
        namespace: str,
        batch: dict[str, Any],
        input_kwargs: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        payload = {
            "namespace": namespace,
            "sample_ids": self._extract_sample_ids(batch),
            "manifest_indices": self._extract_manifest_indices(batch),
            "input_signature": self._summarize_object(input_kwargs or {}),
            "extra": self._summarize_object(extra or {}),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def stats(self) -> dict[str, int]:
        base_stats = {
            "hits": 0,
            "misses": 0,
            "puts": 0,
            "evictions": 0,
        }
        if self.distill_cache is not None:
            base_stats.update(self.distill_cache.stats())
        base_stats["planned_prefetches"] = self._planned_prefetches
        base_stats["async_teacher_launches"] = self._async_teacher_launches
        base_stats["async_teacher_waits"] = self._async_teacher_waits
        base_stats["async_teacher_cache_hits"] = self._async_teacher_cache_hits
        return base_stats

    def _run_model(
        self,
        model: nn.Module,
        input_kwargs: dict[str, Any],
        no_grad: bool,
        tag: str = "model",
    ) -> Any:
        if no_grad:
            with torch.no_grad():
                output = model(**input_kwargs)
        else:
            output = model(**input_kwargs)
        return self._normalize_model_output(output, tag=tag)

    @staticmethod
    def _normalize_model_output(output: Any, tag: str = "model") -> Any:
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _build_teacher_placement_plan(self, namespace: str) -> RuntimePlacementPlan:
        execution_device = str(self.device)
        offload_target = "none"
        allow_async_prefetch = self.prefetch_policy != "none"
        if self.enable_heterogeneous:
            offload_target = self.teacher_offload
        return RuntimePlacementPlan(
            namespace=namespace,
            execution_device=execution_device,
            offload_target=offload_target,
            cache_backend=self.cache_backend,
            allow_async_prefetch=allow_async_prefetch,
        )

    def _should_persist_to_disk(self, namespace: str) -> bool:
        if self.cache_backend == "disk":
            return True
        if self.cache_backend != "hybrid":
            return False
        return namespace in {"teacher_output", "teacher_context", "teacher_context_forward"}

    def _plan_prefetch(
        self,
        cache_namespace: str,
        cache_extra: Optional[dict[str, Any]],
        placement_plan: RuntimePlacementPlan,
    ) -> None:
        if self.prefetch_policy == "none":
            return
        request = self.build_prefetch_request(cache_namespace=cache_namespace, cache_extra=cache_extra)
        if request is None:
            return
        request["allow_async_prefetch"] = placement_plan.allow_async_prefetch
        request["offload_target"] = placement_plan.offload_target
        self.step_state.planned_prefetches.append(request)
        self._planned_prefetches += 1
        if not self._placement_notice_emitted:
            logger.info(
                "TeacherStudentRuntime 记录 prefetch skeleton：当前会为后续 chunk/batch 生成计划，"
                "但尚未启动独立异步 worker 执行数据搬运。"
            )
            self._placement_notice_emitted = True

    def build_prefetch_request(
        self,
        cache_namespace: str,
        cache_extra: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not cache_extra:
            return None
        if self.prefetch_policy == "next_chunk":
            if "chunk_idx" in cache_extra:
                return {
                    "namespace": cache_namespace,
                    "mode": "next_chunk",
                    "next_chunk_idx": int(cache_extra["chunk_idx"]) + 1,
                }
            if "chunk_start" in cache_extra and "chunk_end" in cache_extra:
                chunk_span = int(cache_extra["chunk_end"]) - int(cache_extra["chunk_start"])
                return {
                    "namespace": cache_namespace,
                    "mode": "next_chunk",
                    "chunk_start": int(cache_extra["chunk_end"]),
                    "chunk_end": int(cache_extra["chunk_end"]) + max(chunk_span, 1),
                }
        if self.prefetch_policy == "next_batch":
            return {
                "namespace": cache_namespace,
                "mode": "next_batch",
                "sample_ids": list(self.step_state.sample_ids),
            }
        return None

    def _maybe_log_heterogeneous_mode(self, placement_plan: RuntimePlacementPlan) -> None:
        if not self.enable_heterogeneous or self._heterogeneous_notice_emitted:
            return
        logger.info(
            "TeacherStudentRuntime 异构模式当前为 research skeleton："
            f"teacher 仍在 {placement_plan.execution_device} 执行，"
            f"但已显式记录 offload_target={placement_plan.offload_target}、"
            f"cache_backend={placement_plan.cache_backend}、prefetch={self.prefetch_policy}。"
        )
        self._heterogeneous_notice_emitted = True

    def _maybe_init_dpp_stream(self) -> None:
        if not self.enable_dpp:
            return
        if self.device.type != "cuda":
            logger.warning("runtime_enable_dpp=True 但当前 device 不是 CUDA，DPP 将自动禁用。")
            return
        try:
            self._teacher_stream = torch.cuda.Stream(
                device=self.device,
                priority=self.teacher_stream_priority,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"初始化 teacher CUDA stream 失败，DPP 已回退到串行执行: {exc}")
            self._teacher_stream = None

    def _maybe_log_dpp_mode(self) -> None:
        if self._dpp_notice_emitted:
            return
        logger.info(
            "TeacherStudentRuntime 已启用 DPP：teacher forward 将在独立 CUDA stream 上发射，"
            "student forward 可在默认 stream 上并行推进，loss 计算前再显式同步 supervision buffer。"
        )
        self._dpp_notice_emitted = True

    @staticmethod
    def _infer_batch_size(batch: dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
        return 0

    @staticmethod
    def _extract_sample_ids(batch: dict[str, Any]) -> list[str]:
        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            manifest_idx = batch.get("manifest_idx")
            if manifest_idx is None:
                return []
            return [str(v) for v in TeacherStudentRuntime._to_python_list(manifest_idx)]
        return [str(v) for v in TeacherStudentRuntime._to_python_list(sample_ids)]

    @staticmethod
    def _extract_manifest_indices(batch: dict[str, Any]) -> list[int]:
        manifest_idx = batch.get("manifest_idx")
        if manifest_idx is None:
            return []
        result: list[int] = []
        for value in TeacherStudentRuntime._to_python_list(manifest_idx):
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _to_python_list(value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().view(-1).tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @classmethod
    def _summarize_object(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            flat = value.detach().float().cpu().reshape(-1)
            preview = flat[: min(16, flat.numel())].tolist()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "preview": [round(float(v), 6) for v in preview],
            }
        if isinstance(value, dict):
            return {k: cls._summarize_object(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._summarize_object(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
