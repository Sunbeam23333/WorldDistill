"""Stage-owned module execution with a first-order autograd broadcast boundary.

This correctness-first schedule does not overlap pipeline microbatches. Only
the designated stage stores/executes a wrapped module. Replicated outputs let
existing iterative distillation objectives build ordinary PyTorch graphs.
Communication follows torch.distributed's group-local rank contract:
https://docs.pytorch.org/docs/stable/distributed.html
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.distributed as dist
from torch.autograd.function import once_differentiable
from torch.utils import _pytree
from torch.utils.checkpoint import checkpoint
from training.parallel.tensor_parallel import mark_canonical_names


@dataclass
class _Call:
    wrapper: object
    leaves: list
    spec: object
    tensor_positions: list
    grad_enabled: bool
    output_leaves: list | None = None
    output_spec: object = None


class _OwnerForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, call, anchor, *inputs_and_parameters):
        wrapper = call.wrapper
        count = len(call.tensor_positions)
        inputs = inputs_and_parameters[:count]
        ctx.call, ctx.count = call, count
        ctx.parameter_count = len(inputs_and_parameters) - count
        ctx.input_templates = [(tuple(t.shape), t.dtype, t.device, t.requires_grad) for t in inputs]
        ctx.device = anchor.device
        owner = wrapper.rank == wrapper.owner
        metadata, outputs = None, []
        if owner:
            try:
                leaves = list(call.leaves)
                local_inputs = [t.detach().requires_grad_(t.requires_grad) for t in inputs]
                for position, value in zip(call.tensor_positions, local_inputs):
                    leaves[position] = value
                args, kwargs = _pytree.tree_unflatten(leaves, call.spec)
                with torch.set_grad_enabled(call.grad_enabled):
                    if wrapper.activation_checkpointing and call.grad_enabled and wrapper.training:
                        # Recompute inside the owning stage only. Wrapping the
                        # distributed broadcast boundary would replay global
                        # collectives in rank-dependent autograd order.
                        result = checkpoint(wrapper.local, *args, use_reentrant=False, **kwargs)
                    else:
                        result = wrapper.local(*args, **kwargs)
                output_leaves, output_spec = _pytree.tree_flatten(result)
                positions = [i for i, value in enumerate(output_leaves) if torch.is_tensor(value)]
                outputs = [output_leaves[i] for i in positions]
                if not outputs:
                    raise ValueError("Pipeline-owned modules must return at least one tensor")
                metadata = {"spec": output_spec,
                    "leaves": [None if torch.is_tensor(value) else value for value in output_leaves],
                    "positions": positions,
                    "tensors": [(tuple(t.shape), t.dtype, t.requires_grad) for t in outputs]}
                ctx.local_inputs, ctx.local_outputs = local_inputs, outputs
                ctx.parameters = inputs_and_parameters[count:]
            except Exception as exc:
                metadata = {"error": f"{type(exc).__name__}: {exc}"}
        objects = [metadata]
        dist.broadcast_object_list(objects, src=wrapper.source_rank, group=wrapper.group, device=anchor.device)
        metadata = objects[0]
        if "error" in metadata:
            raise RuntimeError(f"Pipeline stage {wrapper.owner} forward failed: {metadata['error']}")
        call.output_leaves, call.output_spec = metadata["leaves"], metadata["spec"]
        ctx.output_templates = metadata["tensors"]
        result = []
        for index, (shape, dtype, differentiable) in enumerate(metadata["tensors"]):
            value = outputs[index].detach().contiguous().clone() if owner else torch.empty(shape, dtype=dtype, device=anchor.device)
            dist.broadcast(value, src=wrapper.source_rank, group=wrapper.group)
            result.append(value)
            if not differentiable:
                ctx.mark_non_differentiable(value)
        call.output_positions = metadata["positions"]
        return tuple(result)

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_gradients):
        wrapper = ctx.call.wrapper
        averaged = []
        # Every PP rank evaluates the same objective. Average those duplicate
        # losses, rather than multiplying every parameter gradient by pp_size.
        for gradient, (shape, dtype, differentiable) in zip(output_gradients, ctx.output_templates):
            value = torch.zeros(shape, dtype=dtype, device=ctx.device) if gradient is None else gradient.contiguous().clone()
            if differentiable:
                dist.all_reduce(value, group=wrapper.group)
                value.div_(dist.get_world_size(wrapper.group))
            averaged.append(value)
        input_gradients = [None] * ctx.count
        parameter_gradients = [None] * ctx.parameter_count
        error = None
        if wrapper.rank == wrapper.owner:
            try:
                variables = list(ctx.local_inputs) + list(ctx.parameters)
                selected = [(i, value) for i, value in enumerate(variables) if value.requires_grad]
                outputs = [(value, gradient) for value, gradient in zip(ctx.local_outputs, averaged) if value.requires_grad]
                with torch.enable_grad():
                    gradients = torch.autograd.grad([item[0] for item in outputs], [item[1] for item in selected],
                        grad_outputs=[item[1] for item in outputs], allow_unused=True, retain_graph=True) if outputs and selected else []
                for (index, _), gradient in zip(selected, gradients):
                    if index < ctx.count:
                        input_gradients[index] = gradient
                    else:
                        parameter_gradients[index - ctx.count] = gradient
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        status = [error]
        dist.broadcast_object_list(status, src=wrapper.source_rank, group=wrapper.group, device=ctx.device)
        if status[0]:
            raise RuntimeError(f"Pipeline stage {wrapper.owner} backward failed: {status[0]}")
        for index, (shape, dtype, device, required) in enumerate(ctx.input_templates):
            if not required:
                continue
            gradient = input_gradients[index]
            value = torch.zeros(shape, dtype=dtype, device=device) if gradient is None else gradient.contiguous().clone()
            dist.broadcast(value, src=wrapper.source_rank, group=wrapper.group)
            input_gradients[index] = value
        return (None, None, *input_gradients, *parameter_gradients)


class PipelineOwnedModule(nn.Module):
    """Non-owner ranks do not retain the module, its parameters, or its buffers."""
    def __init__(self, module, group, owner, activation_checkpointing=False):
        super().__init__()
        self.group, self.owner = group, int(owner)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.rank = dist.get_rank(group)
        if not 0 <= self.owner < dist.get_world_size(group):
            raise ValueError("Pipeline module owner is outside the PP group")
        self.source_rank = dist.get_global_rank(group, self.owner)
        self.local = module if self.rank == self.owner else None
        self.train(module.training)
        if self.local is not None:
            for value in list(self.local.parameters()) + list(self.local.buffers()):
                value._worlddistill_pp_owner = self.owner

    def forward(self, *args, **kwargs):
        leaves, spec = _pytree.tree_flatten((args, kwargs))
        positions = [i for i, value in enumerate(leaves) if torch.is_tensor(value)]
        tensors = [leaves[position] for position in positions]
        if not tensors:
            raise ValueError("Pipeline module calls require a tensor to define the communication device")
        # All stages need an autograd node even when their local inputs have
        # no gradient and all of this module's parameters live elsewhere.
        anchor = torch.zeros((), device=tensors[0].device, requires_grad=torch.is_grad_enabled())
        call = _Call(self, leaves, spec, positions, torch.is_grad_enabled())
        parameters = tuple(self.local.parameters()) if self.local is not None else ()
        outputs = _OwnerForward.apply(call, anchor, *tensors, *parameters)
        outputs = (outputs,) if torch.is_tensor(outputs) else outputs
        output_leaves = list(call.output_leaves)
        for position, value in zip(call.output_positions, outputs):
            output_leaves[position] = value
        return _pytree.tree_unflatten(output_leaves, call.output_spec)


def apply_pipeline_parallel_plan(model, group, plan, *, activation_checkpointing=False):
    """Native declarative {module.path: pp_owner}, preserving forward semantics."""
    if not dist.is_initialized() or group is None:
        raise ValueError("Pipeline parallelism requires an explicit initialized process group")
    mark_canonical_names(model)
    paths = list(plan)
    if any(not path or any(other != path and other.startswith(path + ".") for other in paths) for path in paths):
        raise ValueError("Pipeline module ownership paths must be nonempty and nonoverlapping")
    replacements = [(path, PipelineOwnedModule(model.get_submodule(path), group, owner,
                     activation_checkpointing=activation_checkpointing)) for path, owner in plan.items()]
    for path, replacement in replacements:
        parent, _, child = path.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, replacement)
    from training.parallel.state import cache_partition_metadata
    cache_partition_metadata(model)
    return model


def apply_wan_pipeline_parallel(model, group, *, activation_checkpointing=None):
    """Partition transformer blocks and expensive embedding/output modules.

    Tiny top-level scale/shift tables and parameterless RoPE remain replicated.
    All blocks are contiguous by stage; no full block survives on other stages.
    """
    candidates = [(path, module) for path, module in model.named_modules()
                  if type(module).__name__ == "WanTransformer3DModel"]
    if not candidates:
        raise ValueError("Built-in PP plan requires WanTransformer3DModel; native models must declare a plan")
    size = dist.get_world_size(group)
    plan = {}
    checkpointed = set()
    for prefix, wan in candidates:
        if getattr(wan, "_worlddistill_pp_applied", False) or getattr(wan, "_parallel_config", None) is not None:
            raise ValueError("Wan PP cannot be re-applied or combined with context parallelism")
        if len(wan.blocks) < size:
            raise ValueError("Wan PP needs at least one transformer block per pipeline stage")
        for index in range(len(wan.blocks)):
            path = f"{prefix}.blocks.{index}".lstrip(".")
            plan[path] = index * size // len(wan.blocks)
            if wan.gradient_checkpointing if activation_checkpointing is None else activation_checkpointing:
                checkpointed.add(path)
        for name, owner in (("patch_embedding", 0), ("condition_embedder", 0), ("norm_out", size - 1), ("proj_out", size - 1)):
            plan[f"{prefix}.{name}".lstrip(".")] = owner
    apply_pipeline_parallel_plan(model, group, plan)
    for path in checkpointed:
        model.get_submodule(path).activation_checkpointing = True
    for _, wan in candidates:
        # Move checkpoint ownership inside the stage; never checkpoint the
        # rank-collective wrapper from Wan's outer block loop.
        wan.gradient_checkpointing = False
        wan._worlddistill_pp_applied = True
    return model
