"""DeepSpeed's real Universal Checkpoint conversion for DP-size changes.

Uses the upstream converter and loader, including FP32 master parameters and
Adam moments. ZeRO's deprecated ``elastic_checkpoint`` is deliberately unused.
"""
from pathlib import Path
from types import SimpleNamespace, MethodType
from contextlib import contextmanager
import copy

import torch


@torch.no_grad()
def _load_zero12_universal(optimizer, lp_groups_name, checkpoint_dir):
    """Reshard Universal Adam tensors, including ranks owning padding only.

    DeepSpeed 0.19.6's base loader indexes an empty ``steps`` list on those
    ranks and also rebinds the master inspection view to an Adam moment.
    Engine/client/scheduler loading and LP synchronization stay upstream;
    this local replacement only reads the official Universal tensor format.
    """
    from deepspeed.checkpoint.constants import PARAM
    from deepspeed.utils import lazy_init_hp_params_optimizer_state
    directory = Path(checkpoint_dir) / "zero"
    global_state = torch.load(directory / "optimizer_state.pt", map_location="cpu", weights_only=False)
    optimizer._load_global_state(global_state)
    if optimizer.mpu is not None:
        raise ValueError("Elastic ZeRO Universal resize requires the DDP/CP plan, not a DeepSpeed MPU")
    groups = getattr(optimizer, lp_groups_name)
    masters = optimizer.single_partition_of_fp32_groups
    saved_groups = global_state["param_groups"]
    if len(groups) != len(masters) or len(groups) != len(saved_groups):
        raise ValueError("Universal optimizer group inventory differs")
    for members, master, group, saved in zip(groups, masters, optimizer.optimizer.param_groups, saved_groups):
        # Padding is explicitly zeroed, including a rank with no real fragment.
        master.zero_()
        moments = {key: torch.zeros_like(master) for key in ("exp_avg", "exp_avg_sq")}
        steps = []
        for parameter in members:
            folder = directory / optimizer.param_names[parameter]
            step = torch.load(folder / "step.pt", map_location="cpu", weights_only=False)
            if step is None or torch.as_tensor(step).numel() != 1:
                raise ValueError("Universal Adam parameter is missing its scalar step")
            steps.append(step)
            mapping = parameter._hp_mapping
            for key in ("fp32", "exp_avg", "exp_avg_sq"):
                tensor = torch.load(folder / f"{key}.pt", map_location="cpu", weights_only=False)[PARAM]
                if tuple(tensor.shape) != tuple(parameter.shape) or tensor.dtype != torch.float32:
                    raise ValueError(f"Universal {key} shape/dtype differs for {optimizer.param_names[parameter]}")
                if mapping is None:
                    continue
                source = mapping.lp_fragment_address
                destination = mapping.get_hp_fragment_address()
                if source.numel != destination.numel:
                    raise ValueError("Universal parameter partition metadata is inconsistent")
                target = master if key == "fp32" else moments[key]
                target.narrow(0, destination.start, destination.numel).copy_(
                    tensor.flatten().narrow(0, source.start, source.numel).to(target.device))
        if not steps or any(not torch.equal(torch.as_tensor(value), torch.as_tensor(steps[0])) for value in steps):
            raise ValueError("Universal optimizer group has missing or unequal Adam steps")
        optimizer.optimizer.state[master] = {**moments, "step": copy.deepcopy(steps[0])}
        group.update({key: copy.deepcopy(value) for key, value in saved.items() if key != "params"})
        for parameter in members:
            mapping = parameter._hp_mapping
            if mapping is not None:
                address = mapping.get_hp_fragment_address()
                mapping.hp_fragment = master.narrow(0, address.start, address.numel)
        lazy_init_hp_params_optimizer_state(members, master, optimizer.optimizer.state)


@contextmanager
def deepspeed_universal_load_context(engine):
    """Scope the ZeRO-1/2 compatibility hook to one engine load operation."""
    if engine.zero_optimization_stage() == 3:
        yield
        return
    optimizer = engine.optimizer
    name = "load_hp_checkpoint_state_from_checkpoint_dir"
    owned = name in vars(optimizer)
    previous = getattr(optimizer, name)
    setattr(optimizer, name, MethodType(_load_zero12_universal, optimizer))
    try:
        yield
    finally:
        if owned:
            setattr(optimizer, name, previous)
        else:
            delattr(optimizer, name)


def _optimizer_layout(engine):
    """ZeRO 1/2 and 3 expose different master-group layouts in >=0.19.6."""
    zero = engine.optimizer
    if engine.zero_optimization_stage() == 3:
        return zero.fp16_groups, zero.fp32_partitioned_groups_flat, zero.sub_group_to_group_id
    groups = zero.bit16_groups
    return groups, zero.single_partition_of_fp32_groups, dict(enumerate(range(len(groups))))


def capture_deepspeed_supplement(engine):
    """Preserve state not fully handled by upstream's Universal loader.

    In 0.19.6 the ZeRO-3 loader copies group-zero options/step into every
    group and restores only optimizer-owned parameters. Preserve distinct
    weight-decay groups, steps, frozen tensors, and persistent buffers here.
    """
    from deepspeed import zero
    parameters = dict(engine.module.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    groups, masters, group_ids = _optimizer_layout(engine)
    mapping, steps = {}, {}
    for subgroup, (members, master) in enumerate(zip(groups, masters)):
        step = engine.optimizer.optimizer.state[master].get("step")
        if step is None:
            raise ValueError("Elastic ZeRO checkpoint requires initialized Adam state at an optimizer-step boundary")
        for parameter in members:
            name = names[id(parameter)]
            mapping[name] = group_ids[subgroup]
            steps[name] = step.detach().cpu().clone() if torch.is_tensor(step) else step
    options = [{key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
               for group in engine.optimizer.optimizer.param_groups]
    frozen = {}
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            with zero.GatheredParameters([parameter], enabled=hasattr(parameter, "ds_id")):
                frozen[name] = parameter.detach().cpu().clone()
    persistent = engine.module.state_dict().keys()
    buffers = {name: value.detach().cpu().clone() for name, value in engine.module.named_buffers() if name in persistent}
    return {"schema": "worlddistill-elastic-zero-supplement-v1", "groups": options,
            "parameter_groups": mapping, "steps": steps, "frozen": frozen, "buffers": buffers}


@torch.no_grad()
def restore_deepspeed_supplement(engine, state):
    from deepspeed import zero
    if not isinstance(state, dict) or state.get("schema") != "worlddistill-elastic-zero-supplement-v1":
        raise ValueError("Missing portable DeepSpeed optimizer-group/frozen state")
    parameters = dict(engine.module.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    groups, masters, group_ids = _optimizer_layout(engine)
    mapping = {names[id(parameter)]: group_ids[subgroup] for subgroup, members in enumerate(groups) for parameter in members}
    if mapping != state["parameter_groups"] or len(state["groups"]) != len(engine.optimizer.optimizer.param_groups):
        raise ValueError("Elastic ZeRO optimizer parameter-group ownership differs")
    if state["frozen"].keys() != {name for name, value in parameters.items() if not value.requires_grad}:
        raise ValueError("Elastic ZeRO frozen parameter inventory differs")
    persistent = engine.module.state_dict().keys()
    buffers = {name: value for name, value in engine.module.named_buffers() if name in persistent}
    if buffers.keys() != state["buffers"].keys():
        raise ValueError("Elastic ZeRO persistent buffer inventory differs")
    for subgroup, (members, master) in enumerate(zip(groups, masters)):
        values = [state["steps"][names[id(parameter)]] for parameter in members]
        if not values or any(not torch.equal(torch.as_tensor(value), torch.as_tensor(values[0])) for value in values):
            raise ValueError("Elastic ZeRO regrouping combined parameters with different Adam steps")
        current = engine.optimizer.optimizer.state[master]["step"]
        value = values[0]
        engine.optimizer.optimizer.state[master]["step"] = (torch.as_tensor(value).to(current.device).clone()
                                                            if torch.is_tensor(current) else int(value))
    for group, options in zip(engine.optimizer.optimizer.param_groups, state["groups"]):
        group.update(copy.deepcopy(options))
    if engine.zero_optimization_stage() < 3:
        # Universal's map_to_flat_opt_states (DeepSpeed 0.19.6) replaces
        # hp_fragment with the last Adam moment view. Re-link the actual FP32
        # master and both moment views after loading, before EMA/inspection can
        # use safe_get_full_fp32_param. This changes views, never tensor values.
        from deepspeed.utils import lazy_init_hp_params_optimizer_state
        for members, master in zip(groups, masters):
            for parameter in members:
                mapping = parameter._hp_mapping
                if mapping is not None:
                    address = mapping.get_hp_fragment_address()
                    mapping.hp_fragment = master.narrow(0, address.start, address.numel)
            lazy_init_hp_params_optimizer_state(members, master, engine.optimizer.optimizer.state)
    for name, value in state["frozen"].items():
        parameter = parameters[name]
        if tuple(getattr(parameter, "ds_shape", parameter.shape)) != tuple(value.shape):
            raise ValueError(f"Elastic ZeRO frozen shape differs for {name}")
        with zero.GatheredParameters([parameter], modifier_rank=0, enabled=hasattr(parameter, "ds_id")):
            parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))
    for name, value in state["buffers"].items():
        if buffers[name].shape != value.shape:
            raise ValueError(f"Elastic ZeRO buffer shape differs for {name}")
        buffers[name].copy_(value.to(buffers[name].device, dtype=buffers[name].dtype))


def configure_elastic_deepspeed(config, *, enabled):
    if not enabled:
        return config
    zero = config.get("zero_optimization", {})
    if int(zero.get("stage", 0)) < 1:
        raise ValueError("Elastic DeepSpeed uses Universal Checkpoints and requires ZeRO stage 1, 2 or 3")
    for key in ("offload_optimizer", "offload_param"):
        if zero.get(key, {}).get("device") == "nvme":
            raise ValueError("Elastic ZeRO resize does not support rank-local optimizer/parameter NVMe swap files; use CPU offload")
    config.setdefault("checkpoint", {})["load_universal"] = True
    return config


def convert_deepspeed_checkpoint(directory, tag):
    from deepspeed.checkpoint.ds_to_universal import main
    source = Path(directory) / tag
    destination = Path(directory) / "universal"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing native DeepSpeed checkpoint: {source}")
    main(SimpleNamespace(input_folder=str(source), output_folder=str(destination),
                         num_extract_workers=1, num_merge_workers=1,
                         keep_temp_folder=False, strict=True, inject_missing_state=True))
    if not (destination / "zero" / "optimizer_state.pt").is_file():
        raise RuntimeError("DeepSpeed Universal conversion did not write optimizer global state")
    if not list(destination.glob("*model_states.pt")):
        raise RuntimeError("DeepSpeed Universal conversion did not preserve model/client metadata")
    return destination


def resize_amp_scaler_state(state, world_size):
    """Replicate an agreed scaler state; never discard rank disagreements."""
    if state is None:
        return None
    if (not isinstance(state, dict) or state.get("schema_version") != 1
            or not isinstance(state.get("rank_states"), list)
            or len(state["rank_states"]) != state.get("world_size") or not state["rank_states"]):
        raise ValueError("Malformed portable DeepSpeed AMP scaler state")
    rows = state["rank_states"]
    if any(row.get("rank") != rank or row.get("state") != rows[0].get("state")
           for rank, row in enumerate(rows)):
        raise ValueError("Elastic DeepSpeed AMP scaler states disagree across saved ranks")
    return {"schema_version": 1, "world_size": world_size,
            "rank_states": [{"rank": rank, "state": rows[0]["state"]} for rank in range(world_size)]}
