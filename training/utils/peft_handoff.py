"""Merge a materialized LoRA student on CPU without mutating the live teacher."""
from __future__ import annotations
import copy
import torch
from torch import nn


def merge_lora_student_state(teacher, student, full_student_state):
    """Rebuild PEFT on a meta teacher skeleton, then safe-merge CPU state.

    The teacher's Parameter identities, offload mappings, hooks, and device are
    never changed. The full CPU student snapshot is required; additional merge
    workspace is allocated on CPU, not by deepcopying GPU model weights. PEFT
    owns adapter-specific math, including DoRA/rsLoRA/modules_to_save.
    """
    from peft import get_peft_model
    configs = getattr(student, "peft_config", None)
    if not isinstance(configs, dict) or not configs:
        raise ValueError("Progressive LoRA handoff requires a real PEFT student with named adapters")
    if getattr(teacher, "peft_config", None):
        raise ValueError("Progressive LoRA teacher must be a merged base model, not another PEFT wrapper")
    if any(str(getattr(config, "peft_type", "")).split(".")[-1] != "LORA" for config in configs.values()):
        raise ValueError("Progressive adapter handoff supports LoRA-family PEFT adapters")
    if any(torch.is_tensor(value) and value.device.type != "cpu" for value in full_student_state.values()):
        raise ValueError("Progressive LoRA merge requires a complete CPU student state")
    memo = {}
    for parameter in teacher.parameters():
        memo[id(parameter)] = nn.Parameter(torch.empty_like(parameter, device="meta"), requires_grad=False)
    for buffer in teacher.buffers():
        memo[id(buffer)] = torch.empty_like(buffer, device="meta")
    # Runtime hooks can close over locks, streams, or offload executors. This
    # temporary skeleton is never executed: only PEFT's state merge uses it.
    # Do not copy runtime ownership or invoke teacher checkpoint hooks here.
    for module in teacher.modules():
        for name, value in vars(module).items():
            if name.endswith("_hooks") and isinstance(value, dict):
                memo[id(value)] = type(value)()
            elif name in {"_parallel_config", "_context_parallel_group", "_diffusers_hook", "_hf_hook"} and value is not None:
                # CP configurations own DeviceMesh/ProcessGroup objects, and
                # Diffusers/Accelerate registries may own those configurations.
                # None belong in a non-executed CPU weight-merge skeleton.
                memo[id(value)] = None
    skeleton = copy.deepcopy(teacher, memo)
    adapter_names = list(configs)
    with torch.device("meta"):
        merged = get_peft_model(skeleton, copy.deepcopy(configs[adapter_names[0]]),
                               adapter_name=adapter_names[0], low_cpu_mem_usage=True)
        for name in adapter_names[1:]:
            merged.add_adapter(name, copy.deepcopy(configs[name]), low_cpu_mem_usage=True)
    merged.load_state_dict(full_student_state, strict=True, assign=True)
    active = getattr(student, "active_adapters", adapter_names[:1])
    active = [active] if isinstance(active, str) else list(active)
    if not active or not set(active).issubset(configs):
        raise ValueError("Progressive student has an invalid active LoRA adapter selection")
    merged = merged.merge_and_unload(safe_merge=True, adapter_names=active)
    return merged.state_dict()
