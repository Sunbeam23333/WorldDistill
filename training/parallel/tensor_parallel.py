"""Trainable column/row tensor parallelism with canonical weight metadata.

The layouts follow PyTorch's ColwiseParallel/RowwiseParallel conventions:
https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
Explicit local Parameters make optimizer/checkpoint ownership inspectable.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value
    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        value = value.contiguous().clone()
        dist.all_reduce(value, group=group)
        return value
    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        value = value.contiguous().clone()
        dist.all_reduce(value, group=group)
        return value
    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


def _partition_parameter(parameter, axis, group):
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if parameter.shape[axis] % size:
        raise ValueError(f"TP dimension {tuple(parameter.shape)} axis={axis} must divide tp_size={size}")
    length = parameter.shape[axis] // size
    result = nn.Parameter(parameter.detach().narrow(axis, rank * length, length).clone(),
                          requires_grad=parameter.requires_grad)
    result._worlddistill_partition = {
        "global_shape": tuple(parameter.shape), "axis": axis,
        "start": rank * length, "end": (rank + 1) * length,
    }
    if hasattr(parameter, "_worlddistill_name"):
        result._worlddistill_name = parameter._worlddistill_name
    return result


class ColumnParallelLinear(nn.Module):
    """Replicated input -> output-feature shard; input gradients are summed."""
    def __init__(self, linear, group):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise ValueError("Column TP requires an unfused torch.nn.Linear; PEFT/quantized layers need an explicit plan")
        self.group = group
        self.in_features = linear.in_features
        self.out_features = linear.out_features // dist.get_world_size(group)
        self.weight = _partition_parameter(linear.weight, 0, group)
        self.bias = _partition_parameter(linear.bias, 0, group) if linear.bias is not None else None
    def forward(self, value):
        return F.linear(_Copy.apply(value, self.group), self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Input-feature shard -> replicated output; bias is added exactly once."""
    def __init__(self, linear, group):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise ValueError("Row TP requires an unfused torch.nn.Linear")
        self.group = group
        self.in_features = linear.in_features // dist.get_world_size(group)
        self.out_features = linear.out_features
        self.weight = _partition_parameter(linear.weight, 1, group)
        self.bias = linear.bias
    def forward(self, value):
        result = _Reduce.apply(F.linear(value, self.weight), self.group)
        return result if self.bias is None else result + self.bias


class _LocalColumnLinear(nn.Module):
    """LoRA B output shard; its replicated A explicitly sums gradients."""
    def __init__(self, linear, group):
        super().__init__()
        self.weight = _partition_parameter(linear.weight, 0, group)
        self.bias = _partition_parameter(linear.bias, 0, group) if linear.bias is not None else None
        self.in_features, self.out_features = linear.in_features, self.weight.shape[0]
    def forward(self, value):
        return F.linear(value, self.weight, self.bias)


class _ReplicatedLoRAInput(nn.Module):
    """Column LoRA A: both input and replicated weight receive SUM gradients."""
    def __init__(self, linear, group):
        super().__init__()
        self.group, self.weight, self.bias = group, linear.weight, linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features
    def forward(self, value):
        bias = _Copy.apply(self.bias, self.group) if self.bias is not None else None
        return F.linear(_Copy.apply(value, self.group), _Copy.apply(self.weight, self.group), bias)


def _parallel_linear(linear, group, style):
    constructor = ColumnParallelLinear if style == "column" else RowParallelLinear
    if type(linear) is nn.Linear:
        return constructor(linear, group)
    # Retain the actual PEFT class, configuration and adapter-control APIs. A
    # shallow module shell avoids copying full weights, but its module maps
    # must be independent so validation never mutates the live input model.
    try:
        from peft.tuners.lora.layer import Linear as LoRALinear
    except ImportError:
        LoRALinear = ()
    if not isinstance(linear, LoRALinear):
        raise ValueError("TP linear plans support torch.nn.Linear or unmerged PEFT LoRA Linear")
    if linear.merged or linear.lora_variant or any(linear.use_dora.values()):
        raise ValueError("TP requires unmerged vanilla/rsLoRA adapters; variant/DoRA layouts need a separate plan")
    if any(isinstance(dropout, nn.Dropout) and dropout.p for dropout in linear.lora_dropout.values()):
        raise ValueError("TP LoRA requires zero adapter dropout")
    if type(linear.base_layer) is not nn.Linear:
        raise ValueError("TP LoRA requires an unquantized, unfused torch.nn.Linear base layer")
    result = copy.copy(linear)
    result._modules = linear._modules.copy()
    result.base_layer = constructor(linear.base_layer, group)
    result.lora_A, result.lora_B = nn.ModuleDict(), nn.ModuleDict()
    for name, a in linear.lora_A.items():
        b = linear.lora_B[name]
        if type(a) is not nn.Linear or type(b) is not nn.Linear:
            raise ValueError("TP LoRA A/B factors must be ordinary Linear modules")
        if style == "column":
            result.lora_A[name] = _ReplicatedLoRAInput(a, group)
            result.lora_B[name] = _LocalColumnLinear(b, group)
        else:
            result.lora_A[name] = RowParallelLinear(a, group)
            result.lora_B[name] = b
    return result


class TensorParallelRMSNorm(nn.Module):
    """Wan normalizes Q/K across ALL heads, not separately per TP shard."""
    def __init__(self, norm, group):
        super().__init__()
        if type(norm) is not nn.RMSNorm or len(norm.normalized_shape) != 1:
            raise ValueError("Wan TP QK norm requires one-dimensional torch RMSNorm")
        self.group, self.eps = group, norm.eps
        self.global_width = norm.normalized_shape[0]
        self.weight = _partition_parameter(norm.weight, 0, group) if norm.weight is not None else None
    def forward(self, value):
        dtype = value.dtype
        x = value.float() if dtype in (torch.float16, torch.bfloat16) else value
        variance = _Sum.apply(x.square().sum(-1, keepdim=True), self.group) / self.global_width
        eps = torch.finfo(dtype).eps if self.eps is None else self.eps
        result = (x * torch.rsqrt(variance + eps)).to(dtype)
        return result if self.weight is None else result * self.weight


def mark_canonical_names(model):
    if not hasattr(model, "_worlddistill_global_state_schema"):
        # Capture before any TP/PP replacement removes remote weights. Strict
        # restore must validate every canonical key on every rank, not only
        # the subset owned by the receiving stage.
        aliases = {}
        for name, value in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
            previous = aliases.setdefault(id(value), name)
            if previous != name:
                raise ValueError(f"TP/PP requires explicit tied-weight ownership: {previous} is shared with {name}")
        model._worlddistill_global_state_schema = {name: tuple(value.shape)
                                                  for name, value in model.state_dict().items()}
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if not hasattr(value, "_worlddistill_name"):
            value._worlddistill_name = name


def apply_tensor_parallel_plan(model, group, plan):
    """Declarative native plan: {module.path: column|row|rms_norm}.

    Adjacent tensor layouts and head reshapes remain the architecture author's
    contract; arbitrary Linear replacement is not automatically a valid plan.
    """
    if not dist.is_initialized() or group is None:
        raise ValueError("Tensor parallelism requires an explicit initialized process group")
    mark_canonical_names(model)
    constructors = {"column": lambda module, group: _parallel_linear(module, group, "column"),
                    "row": lambda module, group: _parallel_linear(module, group, "row"),
                    "rms_norm": TensorParallelRMSNorm}
    replacements = []
    for path, style in plan.items():
        if style not in constructors or not path:
            raise ValueError(f"Invalid tensor-parallel plan entry {path!r}: {style!r}")
        old = model.get_submodule(path)
        replacements.append((path, constructors[style](old, group)))
    for path, replacement in replacements:
        parent, _, child = path.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, replacement)
    from training.parallel.state import cache_partition_metadata
    cache_partition_metadata(model)
    return model


def apply_wan_tensor_parallel(model, group):
    """Apply audited head/FFN sharding to raw Wan or an adapter/router tree."""
    mark_canonical_names(model)
    candidates = [(path, module) for path, module in model.named_modules()
                  if type(module).__name__ == "WanTransformer3DModel"]
    if not candidates:
        raise ValueError("Built-in TP plan requires WanTransformer3DModel; native models must declare a plan")
    size = dist.get_world_size(group)
    plan, attention = {}, []
    for prefix, wan in candidates:
        if getattr(wan, "_worlddistill_tp_applied", False) or getattr(wan, "_parallel_config", None) is not None:
            raise ValueError("Wan TP cannot be re-applied or combined with the context-parallel plan")
        if any(isinstance(module, nn.Dropout) and module.p for module in wan.modules()):
            raise ValueError("Wan TP requires zero dropout for replica-consistent residuals")
        for index, block in enumerate(wan.blocks):
            base = f"{prefix}.blocks.{index}".lstrip(".")
            for name in ("attn1", "attn2"):
                attn = getattr(block, name)
                if attn.heads % size or getattr(attn, "fused_projections", False):
                    raise ValueError("Wan TP needs divisible heads and unfused Q/K/V projections")
                attention.append(attn)
                for projection in ("to_q", "to_k", "to_v", "add_k_proj", "add_v_proj"):
                    if getattr(attn, projection, None) is not None:
                        plan[f"{base}.{name}.{projection}"] = "column"
                for norm in ("norm_q", "norm_k", "norm_added_k"):
                    if getattr(attn, norm, None) is not None:
                        plan[f"{base}.{name}.{norm}"] = "rms_norm"
                plan[f"{base}.{name}.to_out.0"] = "row"
            plan[f"{base}.ffn.net.0.proj"] = "column"
            plan[f"{base}.ffn.net.2"] = "row"
    apply_tensor_parallel_plan(model, group, plan)
    for attn in attention:
        attn.heads //= size
        attn.inner_dim //= size
        attn.kv_inner_dim //= size
    for _, wan in candidates:
        wan._worlddistill_tp_applied = True
    return model
