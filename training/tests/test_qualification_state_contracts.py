"""CPU/mock guards for the qualification harness; not CUDA/ZeRO evidence."""

import contextlib
import copy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tools import check_distributed_training as qualification
from training.trainer_args import TrainerArgs
from training.trainers import base_distill_trainer as base
from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer


def _persistent_state():
    return {
        "optimizer": {"state": {0: {"step": torch.tensor(4.), "exp_avg": torch.tensor([0.01])}}},
        "lr_scheduler": {"last_epoch": 4, "_last_lr": [0.001]},
        "scaler": {"scale": 1024.0, "_growth_tracker": 4},
        "resume_state": {"rank_states": [
            {"rng": {"numpy": np.array([19, 23], dtype=np.uint32), "torch": torch.tensor([1, 2], dtype=torch.uint8)},
             "cursor": 1},
            {"rng": {"numpy": np.array([29, 31], dtype=np.uint32), "torch": torch.tensor([3, 4], dtype=torch.uint8)},
             "cursor": 1},
        ]},
        "epoch": 1,
        "step": 4,
        "ema": {"shadow": {"weight": torch.tensor([0.402])}, "step_count": 4, "decay": 0.99},
        "critic_state.pt": {"optimizer": {"step": 4}, "weight": torch.tensor([0.6])},
    }


def test_equal_independently_loaded_state_trees_are_accepted():
    expected = _persistent_state()
    # This sentinel is an intentional pre-evaluation scalar, not a model weight
    # or optimizer moment; finite tensor guards must not reject it wholesale.
    expected["best_loss"] = float("inf")
    qualification.assert_state_close(copy.deepcopy(expected), expected)


@pytest.mark.parametrize("field", ["optimizer", "scheduler", "scaler", "rank1_rng", "rank1_cursor", "ema", "critic", "missing"])
def test_non_student_state_corruption_is_not_hidden(field):
    expected = _persistent_state()
    actual = copy.deepcopy(expected)
    if field == "optimizer":
        actual["optimizer"]["state"][0]["exp_avg"].add_(0.001)
    elif field == "scheduler":
        actual["lr_scheduler"]["last_epoch"] = 3
    elif field == "scaler":
        actual["scaler"]["_growth_tracker"] = 3
    elif field == "rank1_rng":
        actual["resume_state"]["rank_states"][1]["rng"]["numpy"][0] += 1
    elif field == "rank1_cursor":
        actual["resume_state"]["rank_states"][1]["cursor"] = 0
    elif field == "ema":
        actual["ema"]["shadow"]["weight"].add_(0.01)
    elif field == "critic":
        actual["critic_state.pt"]["optimizer"]["step"] = 3
    else:
        del actual["scaler"]
    with pytest.raises(AssertionError):
        qualification.assert_state_close(actual, expected)


@pytest.mark.parametrize("actual,expected", [
    (torch.tensor([1.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float32)),
    (torch.tensor([[1.]]), torch.tensor([1.])),
    ([1, 2], (1, 2)),
    (float("nan"), float("nan")),
    (torch.tensor([float("nan")]), torch.tensor([float("nan")])),
    (torch.tensor([float("inf")]), torch.tensor([float("inf")])),
    (torch.tensor([-float("inf")]), torch.tensor([-float("inf")])),
    (torch.tensor([4], dtype=torch.int64), torch.tensor([5], dtype=torch.int64)),
])
def test_state_structure_dtype_nonfinite_and_integer_mismatches_fail(actual, expected):
    with pytest.raises(AssertionError):
        qualification.assert_state_close(actual, expected)


def _loss_scaler_class(name="DynamicLossScaler", module="deepspeed.runtime.fp16.loss_scaler"):
    # Real ZeRO serializes this object rather than scaler.state_dict(). Each
    # checkpoint load constructs an independent instance with identity equality.
    return type(name, (), {"__module__": module})


@pytest.mark.parametrize("class_name", ["LossScaler", "DynamicLossScaler", "LossScalerBase"])
def test_deepspeed_scaler_uses_persistent_fields_not_object_identity(class_name):
    scaler_type = _loss_scaler_class(class_name)
    expected, actual = scaler_type(), scaler_type()
    for instance in (actual, expected):
        instance.cur_scale = 65536.0
        instance.cur_iter = 4
        instance.last_overflow_iter = -1
    assert actual is not expected
    qualification.assert_state_close(actual, expected)
    actual.cur_iter = 3
    with pytest.raises(AssertionError, match="cur_iter"):
        qualification.assert_state_close(actual, expected)


def test_deepspeed_scaler_wrong_class_and_arbitrary_empty_objects_are_not_equal():
    dynamic, fixed = _loss_scaler_class(), _loss_scaler_class("LossScaler")
    with pytest.raises(AssertionError, match="types differ"):
        qualification.assert_state_close(fixed(), dynamic())
    unknown = _loss_scaler_class(module="some.untrusted.module")
    with pytest.raises(AssertionError):
        qualification.assert_state_close(unknown(), unknown())


@pytest.fixture
def mock_deepspeed(monkeypatch):
    modules = {}
    for name in ("deepspeed", "deepspeed.utils", "deepspeed.utils.zero_to_fp32"):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["deepspeed"].utils = modules["deepspeed.utils"]
    modules["deepspeed.utils"].zero_to_fp32 = modules["deepspeed.utils.zero_to_fp32"]
    modules["deepspeed.utils.zero_to_fp32"].get_fp32_state_dict_from_zero_checkpoint = (
        lambda _: {"weight": torch.tensor([0.404])}
    )
    return modules["deepspeed"]


def test_standard_checkpoint_collects_all_auxiliary_sidecars(tmp_path):
    state = _persistent_state()
    sidecar = state.pop("critic_state.pt")
    state["student_model"] = {"weight": torch.tensor([0.404])}
    torch.save(state, tmp_path / "trainer_state.pt")
    torch.save(sidecar, tmp_path / "critic_state.pt")
    weights, persistent = qualification.checkpoint_payload(tmp_path, "ddp")
    torch.testing.assert_close(weights["weight"], state["student_model"]["weight"])
    assert "student_model" not in persistent
    qualification.assert_state_close(persistent, _persistent_state())


def _write_engine_checkpoint(path, *, optimizer=True):
    tag = path / "global_step4"
    tag.mkdir(parents=True)
    (path / "latest").write_text(tag.name)
    state = _persistent_state()
    model = {key: state[key] for key in ("step", "epoch", "resume_state", "lr_scheduler", "ema")}
    model.update(global_steps=4, global_samples=8, skipped_steps=0, dp_world_size=2)
    model["torch_amp_scaler_by_rank"] = {
        "schema_version": 1, "world_size": 2,
        "rank_states": [{"rank": rank, "state": {"scale": 128. * (rank + 1), "growth_factor": 2.,
                                                   "backoff_factor": 0.5, "growth_interval": 20,
                                                   "_growth_tracker": rank + 2}} for rank in range(2)],
    }
    torch.save(model, tag / "mp_rank_00_model_states.pt")
    if optimizer:
        # Cover every optimizer partition, not only the rank-zero model file.
        for rank in range(2):
            torch.save({"optimizer_state_dict": {"step": 4, "partition": torch.tensor([rank + 0.1])}},
                       tag / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
    torch.save({"ema_model": {"weight": torch.tensor([0.402])}, "consistency_ema": state["ema"]},
               path / "ema_state.pt")
    return tag


def test_deepspeed_checkpoint_includes_client_ema_root_sidecars_and_all_shards(tmp_path, mock_deepspeed):
    _write_engine_checkpoint(tmp_path)
    _, persistent = qualification.checkpoint_payload(tmp_path, "deepspeed")
    assert "ema" in persistent["mp_rank_00_model_states.pt"]
    assert "torch_amp_scaler_by_rank" in persistent["mp_rank_00_model_states.pt"]
    assert "ema_state.pt" in persistent
    assert len([name for name in persistent if name.endswith("optim_states.pt")]) == 2
    actual = copy.deepcopy(persistent)
    actual["zero_pp_rank_1_mp_rank_00_optim_states.pt"]["optimizer_state_dict"]["partition"].add_(0.1)
    with pytest.raises(AssertionError):
        qualification.assert_state_close(actual, persistent)
    actual = copy.deepcopy(persistent)
    actual["mp_rank_00_model_states.pt"]["torch_amp_scaler_by_rank"]["rank_states"][1]["state"]["scale"] *= 2
    with pytest.raises(AssertionError):
        qualification.assert_state_close(actual, persistent)
    actual = copy.deepcopy(persistent)
    actual["ema_state.pt"]["consistency_ema"]["step_count"] -= 1
    with pytest.raises(AssertionError):
        qualification.assert_state_close(actual, persistent)


def test_deepspeed_weight_only_checkpoint_cannot_qualify_resume(tmp_path, mock_deepspeed):
    _write_engine_checkpoint(tmp_path, optimizer=False)
    with pytest.raises(ValueError, match="optimizer shards are missing"):
        qualification.checkpoint_payload(tmp_path, "deepspeed")


def _mock_resume_case(monkeypatch, tmp_path, *, parallel, reference=0.404, restored=0.404,
                      interrupted=0.4, corrupt_state=False):
    class NoTraining:
        def __init__(self, **kwargs):
            self.args = kwargs["args"]
            # FSDP rebuilds this scheduler with max_train_steps=2/4. A cosine
            # schedule would make interrupted and uninterrupted plans differ.
            assert self.args.lr_scheduler == "constant" and self.args.warmup_steps == 0

        def train(self):
            pass

    def payload(path, requested_parallel):
        assert requested_parallel == parallel
        name = path.parent.name
        state = _persistent_state()
        if name == "resumed" and corrupt_state:
            state["resume_state"]["rank_states"][1]["cursor"] = 0
        value = {"continuous": reference, "resumed": restored, "interrupted": interrupted}[name]
        return {"weight": torch.tensor([value]), "counter": torch.tensor(4)}, state

    monkeypatch.setitem(qualification.TRAINER_REGISTRY, "step_distill", NoTraining)
    monkeypatch.setattr(qualification, "checkpoint_payload", payload)
    monkeypatch.setattr(qualification, "check_replicas", lambda _: None)
    monkeypatch.setattr(qualification.gc, "collect", lambda: None)
    monkeypatch.setattr(qualification.dist, "is_initialized", lambda: False)
    options = SimpleNamespace(parallel=parallel, mixed_precision="no", fsdp_strategy="full",
                              zero_stage=2, cpu_offload=False, gradient_accumulation_steps=1)
    return qualification.train_resume_case("step_distill", tmp_path, torch.device("cpu"), options)


@pytest.mark.parametrize("parallel", ["ddp", "fsdp", "deepspeed"])
def test_mock_resume_with_nonzero_matching_update_passes(monkeypatch, tmp_path, parallel):
    result = _mock_resume_case(monkeypatch, tmp_path, parallel=parallel)
    assert result["status"] == "passed"
    assert result["max_reference_update"] > 0.003
    assert result["optimizer_rng_auxiliary_state_checked"]
    assert "synthetic" in result["scope"]


@pytest.mark.parametrize("parallel", ["ddp", "fsdp", "deepspeed"])
@pytest.mark.parametrize("scenario", ["both_frozen", "resume_frozen", "wrong_small_update", "rank1_state_corrupt"])
def test_frozen_or_inexact_restored_runs_never_pass(monkeypatch, tmp_path, parallel, scenario):
    values = {
        "both_frozen": {"reference": 0.4, "restored": 0.4},
        "resume_frozen": {"restored": 0.4},
        # Absolute weights are close at loose mixed-precision tolerances, but
        # the actual optimizer update is wrong by 50 percent.
        "wrong_small_update": {"restored": 0.402},
        "rank1_state_corrupt": {"corrupt_state": True},
    }[scenario]
    with pytest.raises(AssertionError):
        _mock_resume_case(monkeypatch, tmp_path, parallel=parallel, **values)


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_matching_nonfinite_model_weights_cannot_qualify(monkeypatch, tmp_path, value):
    with pytest.raises(AssertionError):
        _mock_resume_case(monkeypatch, tmp_path, parallel="ddp", reference=value, restored=value)


def test_finite_weights_with_overflowing_update_cannot_qualify(monkeypatch, tmp_path):
    with pytest.raises(AssertionError, match="Nonfinite student update"):
        _mock_resume_case(monkeypatch, tmp_path, parallel="ddp", reference=3e38, restored=3e38, interrupted=-3e38)


@pytest.mark.parametrize("parallel,zero_stage", [("ddp", 2), ("fsdp", 2), ("deepspeed", 2), ("deepspeed", 3)])
@pytest.mark.parametrize("mismatch", [False, True])
def test_replica_check_uses_full_parameters_and_rejects_local_mismatch(monkeypatch, mock_deepspeed, parallel, zero_stage, mismatch):
    student = torch.nn.Linear(2, 2)
    student.register_buffer("counter", torch.tensor(4))
    gathered, broadcasts, full_states = [], [], []
    active = [False]

    @contextlib.contextmanager
    def gathered_parameters(parameters, modifier_rank):
        assert modifier_rank is None
        assert parameters == list(student.parameters())
        gathered.append(True)
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    mock_deepspeed.zero = SimpleNamespace(GatheredParameters=gathered_parameters)
    monkeypatch.setattr(qualification.dist, "is_initialized", lambda: True)

    def broadcast(value, src):
        assert src == 0
        if parallel == "deepspeed" and zero_stage == 3:
            assert active[0]
        broadcasts.append(value)
        if mismatch:
            value.add_(1)

    monkeypatch.setattr(qualification.dist, "broadcast", broadcast)
    monkeypatch.setattr(qualification, "gather", lambda errors: [errors, []])
    if parallel == "fsdp":
        from torch.distributed.checkpoint import state_dict

        def full_state(model, options):
            assert model is student
            assert options.full_state_dict and not options.cpu_offload
            full_states.append(True)
            return model.state_dict()

        monkeypatch.setattr(state_dict, "get_model_state_dict", full_state)
    trainer = SimpleNamespace(student_model=student, parallel_mode=parallel,
                              args=SimpleNamespace(deepspeed_stage=zero_stage), _unwrap_model=lambda model: model)
    if mismatch:
        with pytest.raises(AssertionError, match="disagree across ranks"):
            qualification.check_replicas(trainer)
    else:
        qualification.check_replicas(trainer)
    assert len(broadcasts) == len(student.state_dict())
    assert bool(gathered) == (parallel == "deepspeed" and zero_stage == 3)
    assert bool(full_states) == (parallel == "fsdp")


def test_peer_only_replica_failure_is_visible_on_rank_zero(monkeypatch):
    monkeypatch.setattr(qualification.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qualification.dist, "broadcast", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "gather", lambda errors: [errors, ["rank1.weight"]])
    trainer = SimpleNamespace(student_model=torch.nn.Linear(1, 1), parallel_mode="ddp",
                              _unwrap_model=lambda model: model)
    with pytest.raises(AssertionError, match="rank1.weight"):
        qualification.check_replicas(trainer)


@pytest.mark.parametrize("enabled,dtype", [(True, torch.float16), (True, torch.bfloat16), (False, torch.float16)])
def test_deepspeed_microbatches_autocast_all_forwards_without_double_loss_scaling(monkeypatch, enabled, dtype):
    active, contexts, backward, steps = [False], [], [], []
    losses = iter([2.0, 4.0, 6.0])

    @contextlib.contextmanager
    def autocast(device_type, *, dtype, enabled):
        assert device_type == "cuda"
        contexts.append((dtype, enabled))
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    def forward(batch):
        assert active[0], "Teacher and EMA forwards must share the student's AMP scope"
        return torch.tensor(next(losses))

    def engine_backward(loss):
        assert not active[0]
        backward.append(loss.item())

    def engine_step():
        assert not active[0]
        steps.append(True)

    monkeypatch.setattr(torch.amp, "autocast", autocast)
    engine = SimpleNamespace(backward=engine_backward, step=engine_step,
                             get_global_grad_norm=lambda: 0.75, get_lr=lambda: [0.001])
    fake = SimpleNamespace(_deepspeed_engine=engine,
                           args=SimpleNamespace(gradient_accumulation_steps=3, learning_rate=0.001),
                           amp_dtype=dtype, use_amp=enabled, sp_group=None, _next_batch=lambda: {},
                           _move_batch_to_device=lambda value: value, _prepare_batch_for_model=lambda value: value,
                           _forward_and_loss=forward)
    result = base.BaseDistillTrainer._train_step_deepspeed(fake, {})
    assert contexts == [(dtype, enabled)] * 3
    assert backward == [2., 4., 6.]  # DeepSpeed, not this wrapper, scales losses.
    assert len(steps) == 3
    assert result["loss"] == 4. and result["grad_norm"] == 0.75


def test_consistency_cpu_staged_target_and_shadows_follow_execution_device(monkeypatch):
    student = torch.nn.Linear(2, 2)
    args = TrainerArgs(distill_method="consistency_distill", parallel_mode="deepspeed", deepspeed_stage=2)

    def minimal_base(self, **kwargs):
        self.args = kwargs["args"]
        self.student_model = kwargs["student_model"]
        self.device = kwargs["device"]
        self.prediction_type = "velocity"

    monkeypatch.setattr(base.BaseDistillTrainer, "__init__", minimal_base)
    # Meta is only a no-allocation device transition contract; no GPU executes.
    trainer = ConsistencyDistillTrainer(args=args, student_model=student, device=torch.device("meta"))
    assert all(value.device.type == "cpu" for value in student.parameters())
    assert all(value.device.type == "meta" for value in trainer.ema_model.parameters())
    assert all(value.device.type == "meta" for value in trainer.consistency_ema.shadow.values())
    assert trainer.ema_model is not student and not trainer.ema_model.training
    assert all(not value.requires_grad for value in trainer.ema_model.parameters())


@pytest.mark.parametrize("stage", [1, 2])
def test_deepspeed_base_ema_migrates_after_engine_placement_preserving_precision(monkeypatch, stage):
    student = torch.nn.Linear(2, 2, dtype=torch.float32)
    ema = base.EMAModel(student, decay=0.9)
    engine = SimpleNamespace(module=copy.deepcopy(student).to(device="meta", dtype=torch.bfloat16))
    args = TrainerArgs(parallel_mode="deepspeed", deepspeed_stage=stage, use_ema=True, mixed_precision="bf16")
    monkeypatch.setattr(base, "init_deepspeed", lambda **kwargs: (engine, "engine_optimizer", "engine_scheduler"))
    fake = SimpleNamespace(args=args, world_size=2, student_model=student, optimizer=object(), lr_scheduler=object(),
                           ema=ema, _refresh_distributed_state=lambda: None, _unwrap_model=lambda model: model.module)
    base.BaseDistillTrainer._init_deepspeed(fake)
    assert fake._deepspeed_engine is engine
    assert fake.optimizer == "engine_optimizer"
    assert all(value.device.type == "meta" for value in ema.shadow.values())
    assert all(value.dtype == torch.float32 for value in ema.shadow.values())
