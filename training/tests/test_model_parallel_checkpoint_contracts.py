"""Corrupt portable Adam/EMA states fail before silently restarting moments."""
import copy

import pytest
import torch

from training.parallel.checkpoint import restore_ema_state, restore_optimizer_state
from training.trainers.base_distill_trainer import EMAModel


def test_full_model_checkpoint_is_not_materialized_on_each_gpu(tmp_path, monkeypatch):
    from training.trainers.step_distill_trainer import StepDistillTrainer
    trainer = object.__new__(StepDistillTrainer)
    trainer.device = torch.device("cuda", 0)
    trainer.model_parallel = object()
    checkpoint = tmp_path / "trainer_state.pt"
    checkpoint.touch()
    seen = []
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: seen.append(kwargs["map_location"]) or {})
    trainer._load_checkpoint_file_all_ranks(str(checkpoint), description="test canonical snapshot")
    assert seen == ["cpu"]


def _fixture():
    model = torch.nn.Linear(2, 2)
    model._worlddistill_trainable_schema = {name: tuple(p.shape) for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    options = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
    names = list(dict(model.named_parameters()))
    state = {"schema": "worlddistill-canonical-adamw-v1", "parameter_names": names,
             "initialized_state_names": list(names),
             "options": {name: copy.deepcopy(options) for name in names},
             "state": {name: copy.deepcopy(optimizer.state[p]) for name, p in model.named_parameters()}}
    return model, optimizer, state


@pytest.mark.parametrize("damage", ["missing_state", "unknown_options", "missing_moment", "unknown_parameter"])
def test_optimizer_inventory_corruption_is_rejected(damage):
    model, optimizer, state = _fixture()
    if damage == "missing_state":
        state["state"].pop("weight")
    elif damage == "unknown_options":
        state["options"]["typo"] = state["options"]["weight"]
    elif damage == "missing_moment":
        state["state"]["weight"].pop("exp_avg")
    else:
        state["parameter_names"].append("typo")
    with pytest.raises(ValueError):
        restore_optimizer_state(model, optimizer, state)


def test_unused_parameter_has_legitimate_uninitialized_state():
    model, optimizer, state = _fixture()
    state["state"].pop("bias")
    state["initialized_state_names"].remove("bias")
    restore_optimizer_state(model, optimizer, state)
    assert model.bias not in optimizer.state
    assert "exp_avg" in optimizer.state[model.weight]


@pytest.mark.parametrize("damage", ["missing", "unexpected"])
def test_ema_inventory_corruption_is_rejected(damage):
    model, _, _ = _fixture()
    ema = EMAModel(model)
    state = copy.deepcopy(ema.state_dict())
    if damage == "missing":
        state["shadow"].pop("bias")
    else:
        state["shadow"]["typo"] = state["shadow"]["bias"]
    with pytest.raises(ValueError, match="EMA state mismatch"):
        restore_ema_state(model, ema, state)
