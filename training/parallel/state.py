"""Canonical CPU checkpoints for explicitly partitioned parameters/tensors."""
from __future__ import annotations
import copy
import torch
from torch import nn
import torch.distributed as dist


def capture_unpartitioned_meta_model(model):
    """Keep a weight-free full architecture for later LoRA stage handoffs.

    Call BEFORE partitioning. Keep the result on the trainer, not as a child
    module of the live model. Rebuilding PEFT against a stage-local or sharded
    teacher would lose remote layers and produce incorrect factor shapes.
    """
    if any(getattr(module, "_worlddistill_tp_applied", False) or
           getattr(module, "_worlddistill_pp_applied", False) or
           type(module).__name__ == "PipelineOwnedModule" for module in model.modules()):
        raise ValueError("Capture the full meta architecture before TP/PP partitioning")
    memo = {}
    for parameter in model.parameters():
        memo[id(parameter)] = nn.Parameter(torch.empty_like(parameter, device="meta"),
                                           requires_grad=parameter.requires_grad)
    for buffer in model.buffers():
        memo[id(buffer)] = torch.empty_like(buffer, device="meta")
    for module in model.modules():
        for name, value in vars(module).items():
            if name.endswith("_hooks") and isinstance(value, dict):
                memo[id(value)] = type(value)()
            elif name in {"_parallel_config", "_context_parallel_group", "_diffusers_hook", "_hf_hook"} and value is not None:
                memo[id(value)] = None
            elif torch.is_tensor(value) and id(value) not in memo:
                # Authors sometimes keep RoPE tables outside registered buffers.
                memo[id(value)] = torch.empty_like(value, device="meta")
    return copy.deepcopy(model, memo)


def _items(model):
    persistent = set(model.state_dict())
    yield from model.named_parameters()
    yield from ((name, value) for name, value in model.named_buffers() if name in persistent)


def local_partition_metadata(model):
    result = {}
    cached = getattr(model, "_worlddistill_local_partition_layout", {})
    for local_name, value in _items(model):
        previous = cached.get(local_name, {})
        name = getattr(value, "_worlddistill_name", previous.get("canonical_name", local_name))
        metadata = dict(getattr(value, "_worlddistill_partition", previous.get("partition", {})))
        metadata.setdefault("global_shape", tuple(value.shape))
        metadata.setdefault("axis", None)
        metadata.update(local_name=local_name, pp_owner=getattr(value, "_worlddistill_pp_owner", previous.get("pp_owner")),
                        local_shape=tuple(value.shape))
        if name in result:
            raise ValueError(f"Duplicated canonical parameter name {name}")
        result[name] = metadata
    return result


def cache_partition_metadata(model):
    """Keep layout outside Tensors, whose custom attrs do not survive .to()."""
    model._worlddistill_local_partition_layout = {
        spec["local_name"]: {"canonical_name": name, "pp_owner": spec["pp_owner"],
            "partition": {key: value for key, value in spec.items()
                          if key not in {"local_name", "pp_owner", "local_shape"}}}
        for name, spec in local_partition_metadata(model).items()
    }


def collect_full_named_tensors(model, tensor_by_parameter_name, group=None):
    """Collect same-shape named states (weights/EMA/moments) over one MP group.

    Input keys may be local module names or canonical names. Missing entries
    are preserved (e.g. unused-parameter optimizer moments). Replicas must
    agree; only explicitly partitioned axes are concatenated.
    """
    metadata = local_partition_metadata(model)
    local = {}
    for name, spec in metadata.items():
        value = tensor_by_parameter_name.get(name, tensor_by_parameter_name.get(spec["local_name"]))
        if value is None:
            continue
        if tuple(value.shape) != tuple(spec["local_shape"]):
            raise ValueError(f"Named tensor {name} must have its parameter's local shape")
        local[name] = (value.detach().cpu().clone(), spec)
    records = [local]
    if group is not None:
        records = [None] * dist.get_world_size(group)
        dist.all_gather_object(records, local, group=group)
    by_name = {}
    for record in records:
        for name, item in record.items():
            by_name.setdefault(name, []).append(item)
    result = {}
    for name, pieces in by_name.items():
        shape = tuple(pieces[0][1]["global_shape"])
        axis = pieces[0][1]["axis"]
        if any(tuple(spec["global_shape"]) != shape or spec["axis"] != axis for _, spec in pieces):
            raise ValueError(f"Inconsistent partition metadata for {name}")
        if axis is None:
            value = pieces[0][0]
            if any(not torch.equal(value, other) for other, _ in pieces[1:]):
                raise ValueError(f"Replicated tensor {name} differs across model-parallel ranks")
            result[name] = value
            continue
        unique = {}
        for value, spec in pieces:
            key = (spec["start"], spec["end"])
            if key in unique and not torch.equal(value, unique[key]):
                raise ValueError(f"Replicated shard {name} {key} differs across ranks")
            unique[key] = value
        position = 0
        values = []
        for (start, end), value in sorted(unique.items()):
            if start != position or end <= start:
                raise ValueError(f"Incomplete/overlapping shards for {name}")
            values.append(value)
            position = end
        if position != shape[axis]:
            raise ValueError(f"Missing shards for {name}: reconstructed {position}, expected {shape[axis]}")
        result[name] = torch.cat(values, dim=axis)
    return result


def collect_full_model_state(model, group=None, *, tp_group=None, pp_group=None):
    if tp_group is not None or pp_group is not None:
        if group is not None or (tp_group is not None and pp_group is not None):
            raise ValueError("Pass the complete MP group for combined TP×PP collection")
        group = tp_group if tp_group is not None else pp_group
    values = {name: value for name, value in _items(model)}
    return collect_full_named_tensors(model, values, group)


def shard_named_tensor_state(model, state, *, strict=False):
    """Full canonical tensor map -> local-name shards, without changing tensors."""
    output = {}
    for name, spec in local_partition_metadata(model).items():
        if name not in state:
            if strict:
                raise KeyError(f"Full state is missing {name}")
            continue
        value = state[name]
        if tuple(value.shape) != tuple(spec["global_shape"]):
            raise ValueError(f"Full tensor shape mismatch for {name}")
        if spec["axis"] is not None:
            value = value.narrow(spec["axis"], spec["start"], spec["end"] - spec["start"])
        output[spec["local_name"]] = value.clone()
    return output


@torch.no_grad()
def load_full_model_state(model, state, *, strict=True):
    if strict:
        schema = getattr(model, "_worlddistill_global_state_schema", None)
        if schema is None:
            schema = {name: spec["global_shape"] for name, spec in local_partition_metadata(model).items()}
        missing, unexpected = set(schema) - set(state), set(state) - set(schema)
        if missing or unexpected:
            raise KeyError(f"Canonical full state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for name, shape in schema.items():
            if not torch.is_tensor(state[name]) or tuple(state[name].shape) != tuple(shape):
                raise ValueError(f"Full tensor shape mismatch for {name}")
    shards = shard_named_tensor_state(model, state, strict=strict)
    for name, value in _items(model):
        if name in shards:
            value.copy_(shards[name].to(device=value.device, dtype=value.dtype))
    return model
