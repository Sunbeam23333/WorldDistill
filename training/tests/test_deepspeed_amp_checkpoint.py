"""Real CPU GradScaler plus mock engines: checkpoint contracts, not GPU evidence."""

import copy
from types import SimpleNamespace

import pytest
import torch

from training.trainers import base_distill_trainer as base


class _CheckpointTrainer(base.BaseDistillTrainer):
    def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
        return student_output.sum()

    def prepare_teacher_input(self, batch, noisy_latents, timesteps):
        return {"hidden_states": noisy_latents, "timestep": timesteps}


class _MemoryEngine:
    def __init__(self, scaler, stage):
        self.optimizer = SimpleNamespace()
        if stage == 0:
            self.torch_autocast_z0_gradscaler = scaler
        else:
            self.optimizer.torch_autocast_gradscaler = scaler
        self.saved_client_state = None

    def torch_autocast_enabled(self):
        return True

    def save_checkpoint(self, path, tag, client_state):
        self.saved_client_state = copy.deepcopy(client_state)
        return True

    def load_checkpoint(self, path):
        return "/mock/checkpoint", copy.deepcopy(self.saved_client_state)


def _bare_trainer(*, rank=0, world=1, stage=2, precision="fp16"):
    trainer = object.__new__(_CheckpointTrainer)
    trainer.args = SimpleNamespace(deepspeed_stage=stage, mixed_precision=precision)
    trainer.device = torch.device("cpu")
    trainer.world_size = world
    trainer.ema = None
    trainer.global_step, trainer.epoch, trainer.best_loss = 0, 4, 0.25
    # CPU GradScaler executes genuine scale/unscale/update logic without CUDA.
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=20)
    trainer._deepspeed_engine = _MemoryEngine(scaler, stage)
    trainer._checkpoint_rank = lambda: rank
    trainer._checkpoint_distributed = lambda: False
    trainer._collect_resume_state = lambda: {"cursor": 3}
    trainer._restore_resume_state = lambda state: setattr(trainer, "observed_resume_state", state)
    return trainer, scaler


def _table(world=2):
    return {
        "schema_version": 1,
        "world_size": world,
        "rank_states": [
            {"rank": rank, "state": {"scale": float(128 * (rank + 1)), "growth_factor": 2.0,
                                     "backoff_factor": 0.5, "growth_interval": 20,
                                     "_growth_tracker": rank + 2}}
            for rank in range(world)
        ],
    }


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_amp_scaler_getter_uses_correct_owner_not_native_zero_scaler(stage):
    # Stage zero is a helper-compatibility case, not a newly advertised CLI mode.
    trainer, scaler = _bare_trainer(stage=stage)
    trainer._deepspeed_engine.optimizer.loss_scaler = object()
    assert trainer._deepspeed_amp_scaler() is scaler


@pytest.mark.parametrize("rank", [0, 1])
def test_collect_saves_each_rank_state_after_collective_error_check(monkeypatch, rank):
    trainer, scaler = _bare_trainer(rank=rank, world=2)
    expected = _table()
    scaler.load_state_dict(expected["rank_states"][rank]["state"])
    trainer._checkpoint_distributed = lambda: True
    monkeypatch.setattr(base.dist, "get_world_size", lambda: 2)
    calls = []

    def gather(rows, local):
        calls.append(copy.deepcopy(local))
        if len(calls) == 1:
            assert local is None  # error propagation precedes payload gather
            rows[:] = [None, None]
        else:
            assert local == expected["rank_states"][rank]
            rows[:] = copy.deepcopy(expected["rank_states"])

    monkeypatch.setattr(base.dist, "all_gather_object", gather)
    assert trainer._collect_deepspeed_amp_scaler_state() == expected
    assert len(calls) == 2


def test_peer_scaler_capture_failure_stops_before_payload_gather(monkeypatch):
    trainer, _ = _bare_trainer(world=2)
    trainer._checkpoint_distributed = lambda: True
    monkeypatch.setattr(base.dist, "get_world_size", lambda: 2)
    calls = []

    def gather(rows, local):
        calls.append(local)
        assert local is None
        rows[:] = [None, "Capture DeepSpeed Torch AMP scaler failed on rank 1: missing scaler"]

    monkeypatch.setattr(base.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 1"):
        trainer._collect_deepspeed_amp_scaler_state()
    assert calls == [None]


@pytest.mark.parametrize("defect", ["missing_engine", "autocast_disabled", "missing_scaler", "disabled_scaler", "missing_api"])
def test_fp16_requires_an_enabled_engine_scaler(defect):
    trainer, scaler = _bare_trainer()
    if defect == "missing_engine":
        trainer._deepspeed_engine = None
    elif defect == "autocast_disabled":
        trainer._deepspeed_engine.torch_autocast_enabled = lambda: False
    elif defect == "missing_scaler":
        trainer._deepspeed_engine.optimizer.torch_autocast_gradscaler = None
    elif defect == "disabled_scaler":
        trainer._deepspeed_engine.optimizer.torch_autocast_gradscaler = torch.amp.GradScaler("cpu", enabled=False)
    else:
        trainer._deepspeed_engine.optimizer.torch_autocast_gradscaler = SimpleNamespace(is_enabled=lambda: True)
    with pytest.raises(RuntimeError, match="Torch AMP"):
        trainer._collect_deepspeed_amp_scaler_state()


@pytest.mark.parametrize("rank", [0, 1])
def test_restore_uses_local_rank_state_and_preserves_next_real_cpu_update(rank):
    trainer, scaler = _bare_trainer(rank=rank, world=2)
    state = _table()
    trainer._restore_deepspeed_amp_scaler_state(state)
    assert scaler.state_dict() == state["rank_states"][rank]["state"]
    reference = torch.amp.GradScaler("cpu")
    reference.load_state_dict(state["rank_states"][rank]["state"])

    def step(current_scaler):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        loss = parameter.square().sum()
        current_scaler.scale(loss).backward()
        current_scaler.step(optimizer)
        current_scaler.update()
        return parameter.detach(), current_scaler.state_dict()

    actual_parameter, actual_state = step(scaler)
    expected_parameter, expected_state = step(reference)
    torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
    assert actual_state == expected_state
    assert actual_state["_growth_tracker"] == rank + 3


@pytest.mark.parametrize("defect", ["missing", "wrong_schema", "bool_schema", "wrong_world", "missing_rank_states",
                                   "short_rank_states", "wrong_rank_id", "bool_rank_id", "missing_field", "extra_field"])
def test_missing_or_structurally_corrupt_amp_state_fails_before_restore(defect):
    trainer, scaler = _bare_trainer(world=2)
    before = scaler.state_dict()
    state = _table()
    if defect == "missing":
        state = None
    elif defect == "wrong_schema":
        state["schema_version"] = 2
    elif defect == "bool_schema":
        state["schema_version"] = True
    elif defect == "wrong_world":
        state["world_size"] = 1
    elif defect == "missing_rank_states":
        del state["rank_states"]
    elif defect == "short_rank_states":
        state["rank_states"].pop()
    elif defect == "wrong_rank_id":
        state["rank_states"][1]["rank"] = 0
    elif defect == "bool_rank_id":
        state["rank_states"][1]["rank"] = True
    elif defect == "missing_field":
        del state["rank_states"][1]["state"]["growth_interval"]
    else:
        state["rank_states"][1]["state"]["silent_extra"] = 1
    with pytest.raises(ValueError, match="Torch AMP scaler"):
        trainer._restore_deepspeed_amp_scaler_state(state)
    assert scaler.state_dict() == before


@pytest.mark.parametrize("field,value", [
    ("scale", 0.0), ("scale", -1.0), ("scale", float("inf")), ("scale", float("nan")),
    ("scale", "128"), ("scale", True), ("growth_factor", 1.0), ("growth_factor", 0.5),
    ("backoff_factor", 0.0), ("backoff_factor", 1.0), ("growth_interval", 0),
    ("growth_interval", 2.0), ("growth_interval", True), ("_growth_tracker", -1),
    ("_growth_tracker", 1.0), ("_growth_tracker", True),
])
def test_other_rank_numeric_corruption_is_rejected_before_local_state_changes(field, value):
    trainer, scaler = _bare_trainer(rank=0, world=2)
    before = scaler.state_dict()
    state = _table()
    state["rank_states"][1]["state"][field] = value
    with pytest.raises(ValueError, match="invalid numeric state"):
        trainer._restore_deepspeed_amp_scaler_state(state)
    assert scaler.state_dict() == before


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_non_fp16_legacy_checkpoint_accepts_none_but_rejects_fp16_scaler(precision):
    trainer, _ = _bare_trainer(precision=precision)
    assert trainer._collect_deepspeed_amp_scaler_state() is None
    trainer._restore_deepspeed_amp_scaler_state(None)
    with pytest.raises(ValueError, match="matching FP16 precision"):
        trainer._restore_deepspeed_amp_scaler_state(_table(world=1))


def test_save_load_client_state_preserves_amp_scaler_with_training_cursor():
    trainer, scaler = _bare_trainer()
    state = _table(world=1)
    scaler.load_state_dict(state["rank_states"][0]["state"])
    trainer._save_checkpoint_deepspeed(12, "/unused")
    client = trainer._deepspeed_engine.saved_client_state
    assert client["torch_amp_scaler_by_rank"] == state
    assert client["resume_state"] == {"cursor": 3}
    mutated = copy.deepcopy(scaler.state_dict())
    mutated["scale"], mutated["_growth_tracker"] = 1.0, 0
    scaler.load_state_dict(mutated)
    trainer.epoch, trainer.best_loss = 0, 99.0
    trainer._load_checkpoint_deepspeed("/unused")
    assert scaler.state_dict() == state["rank_states"][0]["state"]
    assert trainer.global_step == 12 and trainer.epoch == 4 and trainer.best_loss == 0.25
    assert trainer.observed_resume_state == {"cursor": 3}


@pytest.mark.parametrize("defect", ["missing", "bad_scale"])
def test_client_checkpoint_missing_or_damaged_amp_state_fails_collectively(defect):
    trainer, _ = _bare_trainer()
    trainer._save_checkpoint_deepspeed(12, "/unused")
    client = trainer._deepspeed_engine.saved_client_state
    if defect == "missing":
        del client["torch_amp_scaler_by_rank"]
    else:
        client["torch_amp_scaler_by_rank"]["rank_states"][0]["state"]["scale"] = 0
    with pytest.raises(RuntimeError, match="Torch AMP scaler"):
        trainer._load_checkpoint_deepspeed("/unused")
