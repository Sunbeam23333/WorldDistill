"""Quiescent ZeRO-3 gathered-weight forwards without parameter-release hooks.

DeepSpeed's gathered generation context and TRL's model utilities use the same
offload-hook lifecycle. A detached input JVP also needs full weights through
both reverse-mode passes, so forward hooks must not partition them early.
Sources:
https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py
https://github.com/huggingface/trl/blob/main/trl/models/utils.py
"""
from contextlib import contextmanager


@contextmanager
def suspend_zero3_parameter_hooks(engine):
    """Temporarily disable ZeRO-3 module hooks inside GatheredParameters.

    Enter only between complete training forwards/backwards, never around a
    pending trainable graph. The caller owns the collective gather, and must
    keep it active until this context exits. Optimizer gradient-accumulator
    hooks, weights, existing gradients, and parameter identity are untouched.
    Re-registration and trace invalidation run even when the JVP raises.
    """
    optimizer = getattr(engine, "optimizer", None)
    offload = getattr(optimizer, "parameter_offload", optimizer)
    module = getattr(offload, "module", None)
    register = getattr(offload, "_register_deepspeed_module", None)
    forward = getattr(offload, "forward_hooks", None)
    backward = getattr(offload, "backward_hooks", None)
    if (module is None or not callable(register) or not isinstance(forward, list)
            or not isinstance(backward, list)
            or not all(callable(getattr(hook, "remove", None)) for hook in forward + backward)):
        raise RuntimeError("ZeRO-3 detached forward requires the parameter-offload module hook contract")
    marker = "_worlddistill_detached_hook_depth"
    depth = getattr(offload, marker, 0)
    setattr(offload, marker, depth + 1)
    if depth:
        try:
            yield module
        finally:
            setattr(offload, marker, depth)
        return

    # Stale module ownership would otherwise prevent GatheredParameters from
    # repartitioning on exit. Include external/tied parameters at every node.
    seen = set()
    for child in module.modules():
        parameters = list(child.parameters(recurse=False))
        external = getattr(child, "ds_external_parameters", None)
        if callable(external):
            parameters.extend(parameter for _, parameter in external())
        for parameter in parameters:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                active = getattr(parameter, "ds_active_sub_modules", None)
                if active is not None:
                    active.clear()

    try:
        for hook in forward + backward:
            hook.remove()
        offload.forward_hooks = []
        offload.backward_hooks = []
        yield module
    finally:
        try:
            coordinator = getattr(offload, "param_coordinator", None)
            if coordinator is not None:
                invalid = getattr(coordinator, "is_invalid_trace", None)
                invalidate = getattr(coordinator, "_invalidate_trace", None)
                if callable(invalidate) and (not callable(invalid) or not invalid()):
                    invalidate()
        finally:
            try:
                register(module)
            finally:
                delattr(offload, marker)
