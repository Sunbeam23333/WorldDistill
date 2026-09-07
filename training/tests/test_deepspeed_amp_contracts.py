"""CPU API/optimizer contracts only: no DeepSpeed extension or GPU is simulated as qualified."""

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from training.utils.distributed import get_deepspeed_config, init_deepspeed


class FakeCPUAdam(torch.optim.AdamW):
    def __init__(self, params, *, adamw_mode, fp32_optimizer_states, **kwargs):
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        super().__init__(params, **kwargs)


@pytest.fixture
def ds(monkeypatch):
    modules = {}
    for name in ("deepspeed", "deepspeed.runtime", "deepspeed.runtime.torch_autocast",
                 "deepspeed.ops", "deepspeed.ops.adam"):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    package = modules["deepspeed"]
    package.__version__ = "0.19.6"
    package.calls = []
    modules["deepspeed.ops.adam"].DeepSpeedCPUAdam = FakeCPUAdam
    amp = modules["deepspeed.runtime.torch_autocast"]
    amp.init_autocast_params = lambda *a: None
    amp.autocast_if_enabled = lambda *a: None

    class Engine:
        def __init__(self, model, optimizer, config):
            self.module, self.optimizer, self.config = model, optimizer, config
            if self.torch_autocast_enabled() and self.torch_autocast_dtype() == torch.float16:
                # CPU scaler exercises the real PyTorch API; not a GPU launch.
                scaler = torch.amp.GradScaler("cpu")
                if config["zero_optimization"]["stage"]:
                    self.optimizer.torch_autocast_gradscaler = scaler
                    self.optimizer.loss_scaler = SimpleNamespace(cur_scale=1.0)
                else:
                    self.torch_autocast_z0_gradscaler = scaler

        def torch_autocast_enabled(self):
            return self.config.get("torch_autocast", {}).get("enabled", False)

        def torch_autocast_dtype(self):
            return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
                self.config.get("torch_autocast", {}).get("dtype"))

        def fp16_enabled(self):
            return self.config.get("fp16", {}).get("enabled", False)

        def bfloat16_enabled(self):
            return self.config.get("bf16", {}).get("enabled", False)

        def scale(self, loss):
            return loss

    def initialize(**kwargs):
        package.calls.append(kwargs)
        engine = package.DeepSpeedEngine(kwargs["model"], kwargs["optimizer"], kwargs["config"])
        return engine, kwargs["optimizer"], None, kwargs["lr_scheduler"]

    package.DeepSpeedEngine = Engine
    package.initialize = initialize
    return package


def _model_optimizer():
    model = torch.nn.Linear(3, 2)
    return model, torch.optim.AdamW([
        {"params": [model.weight], "lr": 0.02, "weight_decay": 0.12, "name": "decay"},
        {"params": [model.bias], "lr": 0.003, "weight_decay": 0.0, "betas": (0.7, 0.98), "name": "bias"},
    ], lr=0.04, eps=1e-7)


@pytest.mark.parametrize("precision,dtype", [("fp16", "float16"), ("bf16", "bfloat16")])
@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_torch_amp_excludes_native_conversion_and_checks_correct_scaler_owner(ds, precision, dtype, stage):
    model, optimizer = _model_optimizer()
    config = get_deepspeed_config(stage=stage, mixed_precision=precision)
    assert config["fp16"] == config["bf16"] == {"enabled": False}
    assert config["torch_autocast"] == {"enabled": True, "dtype": dtype}
    engine, result, scheduler = init_deepspeed(model, optimizer, config)
    assert result is optimizer and scheduler is None
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    if precision == "fp16" and stage:
        assert optimizer.loss_scaler.cur_scale == 1.0  # Not mistaken for active torch scaler.
        assert optimizer.torch_autocast_gradscaler.is_enabled()


@pytest.mark.parametrize("version", ["0.14.0", "0.19.5", "0.19.6rc1", "unknown"])
def test_old_or_unparseable_deepspeed_never_silently_downgrades_amp(ds, version):
    ds.__version__ = version
    model, optimizer = _model_optimizer()
    with pytest.raises(RuntimeError, match=">=0.19.6"):
        init_deepspeed(model, optimizer, get_deepspeed_config(mixed_precision="bf16"))
    assert not ds.calls


def test_old_deepspeed_fp32_does_not_require_amp_api(ds):
    ds.__version__ = "0.14.0"
    del sys.modules["deepspeed.runtime.torch_autocast"]
    model, optimizer = _model_optimizer()
    init_deepspeed(model, optimizer, get_deepspeed_config(mixed_precision="no"))
    assert len(ds.calls) == 1


@pytest.mark.parametrize("missing", ["scale", "torch_autocast_dtype"])
def test_modern_version_string_without_real_features_is_rejected(ds, missing):
    setattr(ds.DeepSpeedEngine, missing, None)
    model, optimizer = _model_optimizer()
    with pytest.raises(RuntimeError, match="lacks required"):
        init_deepspeed(model, optimizer, get_deepspeed_config(mixed_precision="fp16"))
    assert not ds.calls


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_missing_or_disabled_torch_scaler_fails_after_engine_initialization(ds, stage):
    original = ds.initialize
    def initialize(**kwargs):
        engine, optimizer, _, scheduler = original(**kwargs)
        owner = optimizer if stage else engine
        name = "torch_autocast_gradscaler" if stage else "torch_autocast_z0_gradscaler"
        setattr(owner, name, torch.amp.GradScaler("cpu", enabled=False))
        return engine, optimizer, None, scheduler
    ds.initialize = initialize
    model, optimizer = _model_optimizer()
    with pytest.raises(RuntimeError, match="refusing unscaled"):
        init_deepspeed(model, optimizer, get_deepspeed_config(stage=stage, mixed_precision="fp16"))


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_native_amp_config_is_not_accepted_even_when_modern_amp_exists(ds, precision):
    config = get_deepspeed_config(mixed_precision=precision)
    config[precision]["enabled"] = True
    model, optimizer = _model_optimizer()
    with pytest.raises(ValueError, match="not DeepSpeed native"):
        init_deepspeed(model, optimizer, config)
    assert not ds.calls


def test_cpu_offload_preserves_groups_and_scheduler_without_advancing_it(ds):
    model, optimizer = _model_optimizer()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda n: .8 ** n, lambda n: .9 ** n])
    before = copy.deepcopy(scheduler.state_dict())
    groups = [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]
    engine, replacement, result_scheduler = init_deepspeed(
        model, optimizer, get_deepspeed_config(mixed_precision="no", cpu_offload=True), scheduler)
    assert isinstance(replacement, FakeCPUAdam) and replacement is not optimizer
    assert result_scheduler is scheduler and scheduler.optimizer is replacement
    assert scheduler.state_dict() == before
    for original, actual, metadata in zip(optimizer.param_groups, replacement.param_groups, groups):
        assert actual["params"][0] is original["params"][0]
        assert all(actual[key] == value for key, value in metadata.items())
    reference_lrs = [group["lr"] for group in optimizer.param_groups]
    replacement.step()
    scheduler.step()
    assert [group["lr"] for group in replacement.param_groups] == pytest.approx([
        reference_lrs[0] * .8, reference_lrs[1] * .9])
    assert [group["lr"] for group in optimizer.param_groups] == reference_lrs
    assert replacement.adam_w_mode and replacement.fp32_optimizer_states


def test_cpu_offload_preserves_nested_schedulers(ds):
    model, optimizer = _model_optimizer()
    first = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=2)
    second = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [first, second], milestones=[2])
    before = copy.deepcopy(scheduler.state_dict())
    _, replacement, _ = init_deepspeed(model, optimizer,
        get_deepspeed_config(mixed_precision="no", cpu_offload=True), scheduler)
    assert all(node.optimizer is replacement for node in (scheduler, first, second))
    assert scheduler.state_dict() == before


def test_existing_cpu_adam_is_never_replaced(ds):
    model = torch.nn.Linear(3, 2)
    optimizer = FakeCPUAdam(model.parameters(), adamw_mode=True, fp32_optimizer_states=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    _, result, _ = init_deepspeed(model, optimizer,
        get_deepspeed_config(mixed_precision="no", cpu_offload=True), scheduler)
    assert result is optimizer and scheduler.optimizer is optimizer


def test_cpu_offload_rejects_initialized_state_instead_of_losing_moments(ds):
    model, optimizer = _model_optimizer()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="losing state"):
        init_deepspeed(model, optimizer, get_deepspeed_config(mixed_precision="no", cpu_offload=True))
    assert not ds.calls
    assert optimizer.state_dict()["state"].keys() == before["state"].keys()
    for key in before["state"]:
        assert torch.equal(optimizer.state_dict()["state"][key]["exp_avg"], before["state"][key]["exp_avg"])


@pytest.mark.parametrize("flag", ["amsgrad", "maximize", "capturable", "differentiable"])
def test_cpu_adam_does_not_silently_change_requested_optimizer_semantics(ds, flag):
    model, optimizer = _model_optimizer()
    optimizer.param_groups[0][flag] = True
    with pytest.raises(ValueError, match="cannot preserve"):
        init_deepspeed(model, optimizer, get_deepspeed_config(mixed_precision="no", cpu_offload=True))
    assert not ds.calls


def test_failed_engine_initialization_restores_scheduler_binding(ds):
    model, optimizer = _model_optimizer()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    def fail(**kwargs):
        raise RuntimeError("CUDA extension failed")
    ds.initialize = fail
    with pytest.raises(RuntimeError, match="CUDA extension"):
        init_deepspeed(model, optimizer, get_deepspeed_config(mixed_precision="no", cpu_offload=True), scheduler)
    assert scheduler.optimizer is optimizer


def test_torch_amp_leaves_keyword_time_fp32_and_keeps_gradient_path(ds):
    class TimeAware(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 3)
        def forward(self, hidden_states, timestep):
            assert timestep.dtype == torch.float32
            # Intentional FP32 time feature computation is not blanket-cast.
            time_features = timestep.sin()
            return self.projection(hidden_states) * time_features[:, None]
    model = TimeAware()
    optimizer = torch.optim.AdamW(model.parameters())
    init_deepspeed(model, optimizer, get_deepspeed_config(mixed_precision="bf16"))
    hidden_states = torch.randn(2, 3, requires_grad=True)
    timestep = torch.tensor([.01001, .02003], requires_grad=True)
    # Mirrors the official engine's inner context with CPU AMP, not a GPU test.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(hidden_states=hidden_states, timestep=timestep)
        result.float().square().sum().backward()
    assert timestep.dtype == hidden_states.dtype == torch.float32
    assert torch.isfinite(timestep.grad).all() and timestep.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_cpu_offload_stage1_includes_optimizer_and_only_stage3_offloads_parameters(ds, stage):
    model, optimizer = _model_optimizer()
    config = get_deepspeed_config(stage=stage, mixed_precision="no", cpu_offload=True)
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert ("offload_param" in config["zero_optimization"]) == (stage == 3)
    _, replacement, _ = init_deepspeed(model, optimizer, config)
    assert isinstance(replacement, FakeCPUAdam)


def test_cpu_adam_preserves_coupled_adam_mode(ds):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.08)
    _, replacement, _ = init_deepspeed(model, optimizer, get_deepspeed_config(cpu_offload=True, mixed_precision="no"))
    assert not replacement.adam_w_mode


def test_cpu_adam_rejects_inconsistent_group_adam_modes(ds):
    model, optimizer = _model_optimizer()
    optimizer.param_groups[0]["decoupled_weight_decay"] = False
    with pytest.raises(ValueError, match="different Adam/AdamW"):
        init_deepspeed(model, optimizer, get_deepspeed_config(cpu_offload=True, mixed_precision="no"))
    assert not ds.calls


@pytest.mark.parametrize("kwargs", [
    {"mixed_precision": "auto"}, {"stage": 4}, {"stage": True}, {"stage": 0, "cpu_offload": True},
    {"train_batch_size": 0}, {"train_batch_size": 1.5}, {"gradient_accumulation_steps": -1},
    {"gradient_accumulation_steps": True}, {"train_batch_size": 3, "gradient_accumulation_steps": 2},
])
def test_invalid_or_noop_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        get_deepspeed_config(**kwargs)
