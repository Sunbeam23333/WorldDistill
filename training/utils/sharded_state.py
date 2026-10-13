"""AMP and EMA state with explicit distributed-shard ownership."""

from __future__ import annotations

import math
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist


def build_training_grad_scaler(parallel_mode, mixed_precision, device, *, process_group=None):
    """Use cross-rank overflow consensus for FSDP, native AMP otherwise.

    The default group is WORLD, including both axes of hybrid FSDP. An overflow
    on one replicated shard must prevent *every* replica from applying a step.
    DeepSpeed owns its own scaler and receives a disabled native scaler here.
    """
    enabled = mixed_precision == "fp16" and parallel_mode != "deepspeed"
    device_type = torch.device(device).type
    if parallel_mode == "fsdp" and enabled:
        if not dist.is_initialized():
            raise RuntimeError("FSDP FP16 requires an initialized process group for overflow consensus")
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        return ShardedGradScaler(
            device=device_type, enabled=True,
            process_group=process_group if process_group is not None else dist.group.WORLD,
        )
    return torch.amp.GradScaler(device_type, enabled=enabled)


class ShardedEMA:
    """Track local FSDP or ZeRO-3 FP32 parameter shards after optimizer steps.

    Construct *after* FSDP/DeepSpeed has partitioned parameters. FSDP updates
    must run outside forward/backward/summon_full_params scopes. ZeRO-3 uses
    DeepSpeed's public local FP32 parameter API, not empty placeholder .data.
    The checkpoint is rank-local and validates its partition geometry; retaining
    the same world size/wrapping is required for these rank-local EMA sidecars.
    """

    SCHEMA = "worlddistill-sharded-ema-v1"

    def __init__(self, model, *, parallel_mode="fsdp", decay=0.9999, warmup_steps=0):
        if parallel_mode not in {"fsdp", "deepspeed"}:
            raise ValueError("ShardedEMA requires fsdp or deepspeed")
        if not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError("EMA decay must be finite and in [0, 1]")
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("EMA warmup_steps must be a nonnegative integer")
        self.parallel_mode = parallel_mode
        self.decay = float(decay)
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in self._local_values(model).items()
        }

    @staticmethod
    def _parameter_names(model, *, include_frozen=False):
        result = {}
        for name, parameter in model.named_parameters():
            if not include_frozen and not parameter.requires_grad:
                continue
            # FSDP wrapper scopes are implementation details, not model names.
            name = name.replace("_fsdp_wrapped_module.", "")
            if name in result:
                raise ValueError(f"EMA parameter name collision: {name}")
            result[name] = parameter
        return result

    def _local_values(self, model):
        parameters = self._parameter_names(model, include_frozen=hasattr(self, "shadow"))
        if hasattr(self, "shadow"):
            parameters = {name: value for name, value in parameters.items() if name in self.shadow or value.requires_grad}
        if self.parallel_mode == "deepspeed":
            from deepspeed.utils import safe_get_local_fp32_param

            values = {name: safe_get_local_fp32_param(param) for name, param in parameters.items()}
            if any(value is None for value in values.values()):
                raise RuntimeError("ZeRO-3 local FP32 state is unavailable; initialize DeepSpeed before EMA")
            return values
        return parameters

    def _validate_partition(self, values):
        if values.keys() != self.shadow.keys():
            raise ValueError("EMA local parameter names changed after partitioning")
        for name, value in values.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(
                    f"EMA local shard shape changed for {name}: {tuple(value.shape)} != "
                    f"{tuple(self.shadow[name].shape)}; update only after resharding"
                )

    @torch.no_grad()
    def update(self, model):
        values = self._local_values(model)
        self._validate_partition(values)
        decay = self.decay
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            decay = min(decay, (1 + self.step_count) / (10 + self.step_count))
        for name, value in values.items():
            shadow = self.shadow[name]
            if shadow.device != value.device:
                shadow = shadow.to(value.device)
                self.shadow[name] = shadow
            shadow.mul_(decay).add_(value.detach().float(), alpha=1 - decay)
        self.step_count += 1

    @torch.no_grad()
    def copy_to(self, model):
        values = self._local_values(model)
        self._validate_partition(values)
        if self.parallel_mode == "deepspeed":
            from deepspeed.utils import safe_set_local_fp32_param

            for name, parameter in self._parameter_names(model, include_frozen=True).items():
                if name not in self.shadow:
                    continue
                safe_set_local_fp32_param(parameter, self.shadow[name].to(values[name].device))
        else:
            for name, parameter in values.items():
                parameter.copy_(self.shadow[name].to(parameter.device, dtype=parameter.dtype))

    apply_to = copy_to

    def state_dict(self):
        return {
            "schema": self.SCHEMA,
            "parallel_mode": self.parallel_mode,
            "rank": self.rank,
            "world_size": self.world_size,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
            "shadow": {name: value.detach().cpu().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("schema") != self.SCHEMA:
            raise ValueError("Invalid sharded EMA checkpoint schema")
        for key, expected in (("parallel_mode", self.parallel_mode), ("rank", self.rank), ("world_size", self.world_size)):
            if state.get(key) != expected or type(state.get(key)) is not type(expected):
                raise ValueError(f"Sharded EMA {key} mismatch: expected {expected}, got {state.get(key)!r}")
        for key in ("step_count", "warmup_steps"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f"Invalid sharded EMA {key}")
        decay = state.get("decay")
        if type(decay) not in (float, int) or not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError("Invalid sharded EMA decay")
        values = state.get("shadow")
        if not isinstance(values, dict) or any(not isinstance(value, torch.Tensor) for value in values.values()):
            raise ValueError("Invalid sharded EMA tensor state")
        self._validate_partition(values)
        if any(not value.is_floating_point() for value in values.values()):
            raise ValueError("Sharded EMA checkpoint tensors must be floating point")
        # Validate everything before modifying a live EMA.
        self.shadow = {name: value.to(device=self.shadow[name].device, dtype=torch.float32).clone() for name, value in values.items()}
        self.step_count = state["step_count"]
        self.warmup_steps = state["warmup_steps"]
        self.decay = float(decay)

    def save(self, checkpoint_directory):
        path = Path(checkpoint_directory) / f"ema_rank_{self.rank:05d}.pt"
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self, checkpoint_directory):
        path = Path(checkpoint_directory) / f"ema_rank_{self.rank:05d}.pt"
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def portable_state_dict(self, model):
        """Collect FP32 EMA tensors by canonical name, independent of DP size.

        Temporarily install local EMA values only during the full-state gather;
        restore the exact live student shards even if collection fails. This
        avoids retaining a second sharded model solely for EMA checkpointing.
        """
        if self.parallel_mode == "deepspeed" and hasattr(model, "module"):
            model = model.module
        local = self._local_values(model)
        self._validate_partition(local)
        backup = {name: value.detach().clone() for name, value in local.items()}
        try:
            self.copy_to(model)
            if self.parallel_mode == "fsdp":
                from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
                if any(value.dtype != torch.float32 for value in local.values()):
                    raise ValueError("Portable FSDP EMA requires FP32 original/master parameters")
                values = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=False))
                full = {name: values[name].detach().cpu().float().clone() for name in self.shadow}
            else:
                from deepspeed.utils import safe_get_full_fp32_param
                parameters = self._parameter_names(model, include_frozen=True)
                full = {}
                for name in self.shadow:
                    value = safe_get_full_fp32_param(parameters[name])
                    if value is None:
                        raise RuntimeError(f"ZeRO full EMA state unavailable for {name}")
                    full[name] = value.detach().cpu().float().clone()
        finally:
            self._restore_local_values(model, backup)
        return {"schema": "worlddistill-portable-ema-v1", "shadow": full,
                "decay": self.decay, "warmup_steps": self.warmup_steps, "step_count": self.step_count}

    @torch.no_grad()
    def _restore_local_values(self, model, values):
        if self.parallel_mode == "deepspeed":
            from deepspeed.utils import safe_set_local_fp32_param
            parameters = self._parameter_names(model, include_frozen=True)
            for name, value in values.items():
                safe_set_local_fp32_param(parameters[name], value)
        else:
            parameters = self._local_values(model)
            for name, value in values.items():
                parameters[name].copy_(value)

    @torch.no_grad()
    def load_portable_state_dict(self, model, state):
        """Repartition a full EMA into the current FSDP/ZeRO-3 shard geometry."""
        if self.parallel_mode == "deepspeed" and hasattr(model, "module"):
            model = model.module
        if not isinstance(state, dict) or state.get("schema") != "worlddistill-portable-ema-v1":
            raise ValueError("Invalid portable EMA schema")
        full = state.get("shadow")
        if not isinstance(full, dict) or full.keys() != self.shadow.keys():
            raise ValueError("Portable EMA parameter names differ")
        if any(not torch.is_tensor(value) or value.dtype != torch.float32 for value in full.values()):
            raise ValueError("Portable EMA must contain FP32 tensors")
        if (type(state.get("step_count")) is not int or state["step_count"] < 0
                or type(state.get("warmup_steps")) is not int or state["warmup_steps"] < 0
                or type(state.get("decay")) not in (float, int) or not math.isfinite(state["decay"])
                or not 0 <= state["decay"] <= 1):
            raise ValueError("Invalid portable EMA update metadata")
        local = self._local_values(model)
        if self.parallel_mode == "deepspeed":
            # safe_set_full_fp32_param in upstream ZeRO-3 assumes an already
            # padded full tensor and can overrun one-element bias tensors at
            # DP>1. Repartition into the public local FP32 view's geometry
            # directly, including zero padding, without changing the student.
            parameters = self._parameter_names(model, include_frozen=True)
            restored = {}
            for name, value in full.items():
                parameter = parameters[name]
                if tuple(parameter.ds_shape) != tuple(value.shape):
                    raise ValueError(f"Portable EMA shape differs for {name}")
                group = parameter.ds_process_group
                world, rank = dist.get_world_size(group), dist.get_rank(group)
                width = math.ceil(value.numel() / world)
                if local[name].numel() != width:
                    raise ValueError(f"ZeRO EMA local partition geometry differs for {name}")
                shard = torch.zeros_like(local[name], dtype=torch.float32)
                start = rank * width
                count = min(width, max(0, value.numel() - start))
                if count:
                    shard.reshape(-1)[:count].copy_(value.reshape(-1)[start:start + count])
                restored[name] = shard
            self._validate_partition(restored)
            self.shadow = restored
            self.decay, self.warmup_steps, self.step_count = float(state["decay"]), state["warmup_steps"], state["step_count"]
            return
        backup = {name: value.detach().clone() for name, value in local.items()}
        try:
            if self.parallel_mode == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                with FSDP.summon_full_params(model, writeback=True, rank0_only=False):
                    parameters = self._parameter_names(model, include_frozen=True)
                    for name, value in full.items():
                        if parameters[name].shape != value.shape or parameters[name].dtype != torch.float32:
                            raise ValueError(f"Portable EMA shape/master dtype differs for {name}")
                    for name, value in full.items():
                        parameters[name].copy_(value.to(parameters[name].device))
            restored = {name: value.detach().float().clone() for name, value in self._local_values(model).items()}
        finally:
            self._restore_local_values(model, backup)
        self._validate_partition(restored)
        self.shadow = restored
        self.decay, self.warmup_steps, self.step_count = float(state["decay"]), state["warmup_steps"], state["step_count"]
