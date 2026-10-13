"""Parameter offload for frozen teachers, independent of supervision caching.

The module's registered parameters and buffers stay on CPU. A functional call
temporarily supplies device copies, preserving tied parameters, parametrizations
and module hooks without permanently replacing the teacher. CUDA copies and the
forward run on the caller's stream; a lease records allocator-safe stream usage
and tracks its completion event. At most one full teacher launch is in flight.

``nvme`` uses a private, memory-mapped torch checkpoint as the CPU backing store.
This is ordinary filesystem paging plus H2D copies, not GPUDirect Storage. The
filesystem selected by ``directory`` determines whether the backing is NVMe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import threading
import uuid
from typing import Any

import torch
from torch import nn


def _preserve_partition_metadata(original, replacement):
    """Storage replacement must not erase canonical PP/TP buffer ownership."""
    for name in ("_worlddistill_name", "_worlddistill_pp_owner", "_worlddistill_partition"):
        if hasattr(original, name):
            value = getattr(original, name)
            setattr(replacement, name, dict(value) if isinstance(value, dict) else value)
    return replacement


@dataclass
class TeacherParameterLease:
    """Track allocator-safe weight release and completion of host writeback."""

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    ready_event: Any = None
    released: bool = False

    def release_storage_on_stream(self, stream) -> None:
        """Return weights to the allocator without a host-side CUDA wait.

        The allocator may reclaim storage only after this stream's queued uses
        finish. Outputs aliasing a weight still hold their own tensor reference.
        The completion event remains necessary for subsequent host writebacks.
        """
        for tensor in self.tensors.values():
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                tensor.record_stream(stream)
        self.tensors.clear()

    def release(self, *, wait: bool) -> bool:
        if self.released:
            return True
        if self.ready_event is not None:
            if wait:
                # CUDA failures are deliberately propagated, never converted to
                # a cache miss or a silent serial retry.
                self.ready_event.synchronize()
            elif not self.ready_event.query():
                return False
        self.tensors.clear()
        self.released = True
        return True


class TeacherParameterOffloader:
    """Bounded whole-teacher CPU/file-backed parameter offload.

    The teacher must fit on the execution device for one forward. This executor
    reduces its *between-forward* residency; it does not split individual layers
    or implicitly shard the teacher across GPUs. Its calls must be serialized on
    the training thread, including external teacher-weight replacements.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        target: str = "cpu",
        directory: str | Path | None = None,
        pin_memory: bool = True,
    ):
        if target not in {"cpu", "nvme"}:
            raise ValueError(f"Unsupported teacher parameter offload target: {target!r}")
        self.model = model
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Teacher parameter offload supports CPU and CUDA execution")
        self.target = target
        self.pin_memory = bool(pin_memory and target == "cpu" and self.device.type == "cuda")
        self._thread_id = threading.get_ident()
        self._active = False
        self._pending: TeacherParameterLease | None = None
        self._storage: tempfile.TemporaryDirectory | None = None
        self._backing_path: Path | None = None
        self._closed = False
        self._forwards = 0
        self._h2d_bytes = 0
        self._writebacks = 0

        for name, value in self._registered_state().items():
            if value.is_meta:
                raise ValueError(f"Teacher parameter {name!r} must be materialized before offload")
            if value.layout != torch.strided or value.is_quantized:
                raise ValueError(f"Teacher parameter {name!r} requires dense, non-quantized storage")
        # This boundary owns only a frozen teacher, never a student's optimizer
        # references. Freezing also permits safe in-place host-state writeback.
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to("cpu")
        if target == "nvme":
            base = Path(directory).expanduser() if directory else None
            if base is not None:
                base.mkdir(parents=True, exist_ok=True)
            self._storage = tempfile.TemporaryDirectory(prefix="teacher-weights-", dir=base)
            self.refresh_weights()
        elif self.pin_memory:
            pinned = {}
            aliases = {}
            for name, tensor in self._registered_state().items():
                if id(tensor) not in aliases:
                    aliases[id(tensor)] = tensor.detach().pin_memory()
                pinned[name] = aliases[id(tensor)]
            self._replace_host_storage(pinned)

    @property
    def storage_path(self) -> Path | None:
        return self._backing_path

    def refresh_weights(self) -> None:
        """Publish a new mmap backing after an external teacher-weight handoff."""
        self.synchronize()
        if self.target != "nvme" or self._storage is None:
            return
        path = Path(self._storage.name) / f"weights-{uuid.uuid4().hex}.pt"
        torch.save(self._registered_state(), path)
        mapped = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self._replace_host_storage(mapped)
        old_path = self._backing_path
        self._backing_path = path
        if old_path is not None:
            old_path.unlink()

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("Teacher parameter offload must run on its owning training thread")

    def _registered_state(self) -> dict[str, torch.Tensor]:
        # Include aliases and non-persistent buffers: state_dict() alone omits
        # the latter, yet functional_call must cover everything used by forward.
        return dict([
            *self.model.named_parameters(remove_duplicate=False),
            *self.model.named_buffers(remove_duplicate=False),
        ])

    def _replace_host_storage(self, state: dict[str, torch.Tensor]) -> None:
        parameters = dict(self.model.named_parameters(remove_duplicate=False))
        aliases = {}
        for name, original in self._registered_state().items():
            value = state[name]
            if id(original) in aliases:
                value = aliases[id(original)]
            else:
                aliases[id(original)] = value
            if name in parameters:
                original.data = value
            else:
                parent_name, _, local_name = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                parent._buffers[local_name] = _preserve_partition_metadata(original, value)

    def _host_buffer(self, name, value):
        # Cache buffers may be allocated lazily or resized by a model's forward.
        # A blocking copy is intentional on this uncommon shape-change path:
        # Python may inspect its new shape/value before another GPU launch.
        host = value.detach().to("cpu")
        if self.pin_memory and not host.is_pinned():
            host = host.pin_memory()
        parent_name, _, local_name = name.rpartition(".")
        parent = self.model.get_submodule(parent_name) if parent_name else self.model
        original = parent._buffers.get(local_name)
        parent._buffers[local_name] = _preserve_partition_metadata(original, host)
        return host

    def synchronize(self) -> None:
        """Drain queued uses before checkpointing or replacing teacher weights."""
        self._check_thread()
        if self._pending is not None:
            self._pending.release(wait=True)
            self._pending = None

    def forward(self, kwargs: dict[str, Any]) -> tuple[Any, TeacherParameterLease]:
        self._check_thread()
        if self._closed:
            raise RuntimeError("Teacher parameter offloader is closed")
        if self._active:
            raise RuntimeError("Reentrant teacher offload forward is not supported")
        self.synchronize()
        host_state = self._registered_state()
        if any(value.device.type != "cpu" for value in host_state.values()):
            raise RuntimeError("Offloaded teacher weights were moved externally; keep the teacher on CPU")
        self._active = True
        lease = TeacherParameterLease()
        versions = {}
        aliases = {}
        try:
            with torch.no_grad():
                for name, host in host_state.items():
                    if id(host) not in aliases:
                        value = host.detach().to(self.device, non_blocking=self.pin_memory)
                        aliases[id(host)] = value
                        if self.device.type == "cuda":
                            self._h2d_bytes += host.numel() * host.element_size()
                    lease.tensors[name] = aliases[id(host)]
                    versions[name] = (lease.tensors[name], lease.tensors[name]._version)
                output = torch.func.functional_call(
                    self.model, lease.tensors, args=(), kwargs=kwargs,
                    tie_weights=True, strict=True,
                )
                # Frozen eval modules may still update a registered counter or
                # cache buffer. Carry those changes back instead of resetting
                # them silently at every call. Parameter mutations are treated
                # identically, preserving ordinary Module forward semantics.
                written = set()
                parameter_names = set(dict(self.model.named_parameters(remove_duplicate=False)))
                for name, value in lease.tensors.items():
                    previous, version = versions[name]
                    if value is None and name in parameter_names:
                        raise RuntimeError(f"Teacher forward removed offloaded parameter {name!r}")
                    if value is None and name not in parameter_names:
                        parent_name, _, local_name = name.rpartition(".")
                        parent = self.model.get_submodule(parent_name) if parent_name else self.model
                        parent._buffers[local_name] = None
                        self._writebacks += 1
                        continue
                    if value is previous and value._version == version:
                        continue
                    host = host_state[name]
                    if id(host) in written:
                        continue
                    if value.shape != host.shape or value.dtype != host.dtype:
                        if name in parameter_names:
                            raise RuntimeError(f"Teacher forward changed shape/dtype of offloaded parameter {name!r}")
                        replacement = self._host_buffer(name, value)
                        for alias, original in host_state.items():
                            if alias != name and original is host:
                                parent_name, _, local_name = alias.rpartition(".")
                                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                                parent._buffers[local_name] = replacement
                    else:
                        host.copy_(value, non_blocking=self.pin_memory)
                    written.add(id(host))
                    self._writebacks += 1
                for name, value in self._registered_state().items():
                    if name in host_state:
                        continue
                    if name in parameter_names:
                        raise RuntimeError(f"Teacher forward registered a new parameter {name!r}; initialize weights before offload")
                    self._host_buffer(name, value)
                    # Include newly-created device buffers in allocator lifetime
                    # tracking even though functional_call did not know them.
                    lease.tensors[name] = value
                    self._writebacks += 1
                self._forwards += 1
            return output, lease
        finally:
            self._active = False
            if self.device.type == "cuda":
                # Keep both successful and exceptional partial launches alive
                # until queued copies/kernels/writebacks complete. Close or the
                # next forward observes asynchronous CUDA errors explicitly.
                lease.ready_event = torch.cuda.Event(blocking=False)
                stream = torch.cuda.current_stream(self.device)
                stream.record_event(lease.ready_event)
                lease.release_storage_on_stream(stream)
            else:
                lease.release(wait=True)
            self._pending = lease

    def stats(self) -> dict[str, int]:
        return {
            "teacher_offload_forwards": self._forwards,
            "teacher_offload_h2d_bytes": self._h2d_bytes,
            "teacher_offload_writebacks": self._writebacks,
            "teacher_offload_inflight": int(self._pending is not None and not self._pending.released),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.synchronize()
        if self._storage is not None:
            # Leave an ordinary usable CPU teacher after closing; unmap before
            # deleting our private temporary file, including on Windows.
            copies, aliases = {}, {}
            for name, tensor in self._registered_state().items():
                if id(tensor) not in aliases:
                    aliases[id(tensor)] = tensor.detach().clone()
                copies[name] = aliases[id(tensor)]
            self._replace_host_storage(copies)
            self._storage.cleanup()
            self._storage = None
            self._backing_path = None
        self._closed = True
