"""Canonical full checkpoints for TP/PP models, AdamW moments, and EMA."""
from __future__ import annotations

import copy

import torch
import torch.distributed as dist

from training.parallel.state import (
    collect_full_model_state, collect_full_named_tensors, local_partition_metadata,
    load_full_model_state, shard_named_tensor_state,
)


def _equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    return left == right


def collect_optimizer_state(model, optimizer, group):
    """Map parameter identities to full canonical names, never rank-local IDs."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("TP/PP checkpoints require AdamW optimizer state")
    metadata = local_partition_metadata(model)
    parameters = dict(model.named_parameters())
    names = {id(parameters[spec["local_name"]]): name for name, spec in metadata.items()
             if spec["local_name"] in parameters}
    local_options, local_scalars, tensor_fields = {}, {}, {}
    for parameter_group in optimizer.param_groups:
        options = {key: copy.deepcopy(value) for key, value in parameter_group.items() if key != "params"}
        for parameter in parameter_group["params"]:
            name = names[id(parameter)]
            local_options[name] = options
            if parameter not in optimizer.state:
                continue
            local_scalars[name] = {}
            for field, value in optimizer.state[parameter].items():
                if torch.is_tensor(value) and value.shape == parameter.shape and value.ndim:
                    tensor_fields.setdefault(field, {})[name] = value
                else:
                    local_scalars[name][field] = value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    records = [None] * dist.get_world_size(group)
    dist.all_gather_object(records, (local_options, local_scalars, sorted(tensor_fields)), group=group)
    options, states = {}, {}
    for record_options, record_scalars, _ in records:
        for name, value in record_options.items():
            if name in options and options[name] != value:
                raise ValueError(f"Optimizer hyperparameters disagree for replicated parameter {name}")
            options[name] = value
        for name, fields in record_scalars.items():
            previous = states.setdefault(name, {})
            for field, value in fields.items():
                if field in previous and not _equal(value, previous[field]):
                    raise ValueError(f"Optimizer scalar {name}/{field} differs across model-parallel replicas")
                previous[field] = value
    fields = sorted({field for _, _, names in records for field in names})
    for field in fields:
        values = collect_full_named_tensors(model, tensor_fields.get(field, {}), group)
        for name, value in values.items():
            states.setdefault(name, {})[field] = value
    return {"schema": "worlddistill-canonical-adamw-v1", "state": states, "options": options,
            "parameter_names": sorted(options), "initialized_state_names": sorted(states)}


def restore_optimizer_state(model, optimizer, state):
    if not isinstance(state, dict) or state.get("schema") != "worlddistill-canonical-adamw-v1":
        raise ValueError("TP/PP restore requires canonical AdamW state")
    expected_names = set(getattr(model, "_worlddistill_trainable_schema", state.get("parameter_names", [])))
    if set(state.get("parameter_names", [])) != expected_names or set(state.get("options", {})) != expected_names:
        raise ValueError("Canonical optimizer parameter inventory does not match the trainable model")
    initialized = set(state.get("initialized_state_names", []))
    if set(state.get("state", {})) != initialized or not initialized.issubset(expected_names):
        raise ValueError("Canonical optimizer initialized-state inventory is incomplete or unexpected")
    for name, fields in state["state"].items():
        required = {"step", "exp_avg", "exp_avg_sq"}
        if state["options"][name].get("amsgrad", False):
            required.add("max_exp_avg_sq")
        if not required.issubset(fields):
            raise ValueError(f"Incomplete AdamW moments for {name}")
    metadata = local_partition_metadata(model)
    parameters = dict(model.named_parameters())
    by_id = {id(parameters[spec["local_name"]]): (name, spec) for name, spec in metadata.items()
             if spec["local_name"] in parameters}
    template = optimizer.state_dict()
    restored = {"state": {}, "param_groups": []}
    for group, serialized_group in zip(optimizer.param_groups, template["param_groups"]):
        options = None
        for parameter, index in zip(group["params"], serialized_group["params"]):
            name, spec = by_id[id(parameter)]
            if name not in state["options"]:
                raise KeyError(f"Missing optimizer options for {name}")
            current_options = state["options"][name]
            if options is not None and options != current_options:
                raise ValueError("Local optimizer groups disagree with saved canonical parameter options")
            options = current_options
            if name not in state["state"]:
                continue
            fields = {}
            for field, value in state["state"][name].items():
                if torch.is_tensor(value) and value.ndim:
                    if tuple(value.shape) != tuple(spec["global_shape"]):
                        raise ValueError(f"Optimizer tensor shape mismatch for {name}/{field}")
                    if spec["axis"] is not None:
                        value = value.narrow(spec["axis"], spec["start"], spec["end"] - spec["start"])
                    value = value.clone()
                fields[field] = value
            restored["state"][index] = fields
        restored["param_groups"].append({**(options or {k: v for k, v in serialized_group.items() if k != "params"}),
                                          "params": serialized_group["params"]})
    optimizer.load_state_dict(restored)


def collect_ema_state(model, ema, group):
    state = ema.state_dict()
    return {**state, "shadow": collect_full_named_tensors(model, state["shadow"], group)}


def restore_ema_state(model, ema, state):
    shadows = shard_named_tensor_state(model, state["shadow"])
    missing = set(ema.shadow) - set(shadows)
    schema = getattr(model, "_worlddistill_trainable_schema", None)
    unexpected = set(state["shadow"]) - set(schema) if schema is not None else set()
    if missing or unexpected:
        raise ValueError(f"EMA state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    ema.load_state_dict({**state, "shadow": {name: value for name, value in shadows.items() if name in ema.shadow}})


def collect_model_parallel_checkpoint(trainer, step):
    model = trainer._unwrap_model(trainer.student_model)
    topology = trainer.model_parallel
    state = {
        "step": step, "epoch": trainer.epoch, "best_loss": trainer.best_loss,
        "student_model": collect_full_model_state(model, group=topology.checkpoint_group),
        "optimizer": collect_optimizer_state(model, trainer.optimizer, topology.checkpoint_group),
        "lr_scheduler": trainer.lr_scheduler.state_dict(), "scaler": trainer.scaler.state_dict(),
        "resume_state": trainer._collect_resume_state(), "args": vars(trainer.args),
        "model_parallel": {"schema": "worlddistill-model-parallel-v1", "tp_size": topology.tp_size,
                           "pp_size": topology.pp_size},
    }
    if trainer.ema is not None:
        state["ema"] = collect_ema_state(model, trainer.ema, topology.checkpoint_group)
    return state


def restore_model_parallel_checkpoint(trainer, state):
    topology = trainer.model_parallel
    expected = {"schema": "worlddistill-model-parallel-v1", "tp_size": topology.tp_size, "pp_size": topology.pp_size}
    if state.get("model_parallel") != expected:
        raise ValueError("Model-parallel resume requires the saved TP/PP plan degrees")
    trainer._validate_standard_checkpoint_state(state)
    model = trainer._unwrap_model(trainer.student_model)
    load_full_model_state(model, state["student_model"])
    restore_optimizer_state(model, trainer.optimizer, state["optimizer"])
    trainer.lr_scheduler.load_state_dict(state["lr_scheduler"])
    trainer.scaler.load_state_dict(state["scaler"])
    trainer.global_step, trainer.epoch = state["step"], state["epoch"]
    trainer.best_loss = state["best_loss"]
    trainer._restore_resume_state(state.get("resume_state"))
    if trainer.ema is not None:
        restore_ema_state(model, trainer.ema, state["ema"])
