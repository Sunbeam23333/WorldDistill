"""CPU wrapping/API regressions; not multi-GPU collective qualification."""

from functools import partial
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.wrap import CustomPolicy, ModuleWrapPolicy, size_based_auto_wrap_policy

from training.model_adapter import NoiseRoutedDenoiser
from training.utils import distributed as subject


def routed():
    return NoiseRoutedDenoiser(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
                              nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), .5)


class Wrapped(nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.wrapped = module
        self.kwargs = kwargs


def apply_real_torch_auto_wrap(model, policy, mixed_precision):
    # This runs PyTorch's actual traversal, including AMP's internal OR policy,
    # but replaces CUDA FSDP allocation/collectives with a plain container.
    _auto_wrap(model, policy, set(), set(),
               {"mixed_precision": mixed_precision, "use_orig_params": True}, Wrapped)


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("kind", ["size", "always", "module", "custom"])
def test_real_torch_traversal_does_not_split_leaf_even_with_amp_or_custom_policy(amp, kind):
    leaf = routed()
    expert_modules = set(leaf.modules())
    model = nn.ModuleDict({"shared": nn.Linear(2, 2), "route": leaf})
    policy = {
        "size": partial(size_based_auto_wrap_policy, min_num_params=1),
        "always": lambda **kwargs: True,
        "module": ModuleWrapPolicy({nn.Linear, nn.Sequential, NoiseRoutedDenoiser}),
        "custom": CustomPolicy(lambda module: {"custom_tag": "preserved"}),
    }[kind]
    precision = MixedPrecision(param_dtype=torch.bfloat16) if amp else None
    guarded = subject._collective_leaf_wrap_policy(model, policy, precision)
    apply_real_torch_auto_wrap(model, guarded, precision)
    assert model["route"] is leaf
    assert set(leaf.modules()) == expert_modules
    assert not any(isinstance(module, Wrapped) for module in leaf.modules())
    assert isinstance(model["shared"], Wrapped)
    if kind == "custom":
        assert model["shared"].kwargs["custom_tag"] == "preserved"


@pytest.mark.parametrize("kind", ["callable", "object"])
def test_ordinary_model_policy_is_unchanged(kind):
    model = nn.Linear(2, 2)
    policy = (lambda **kwargs: True) if kind == "callable" else ModuleWrapPolicy({nn.Linear})
    assert subject._collective_leaf_wrap_policy(model, policy, MixedPrecision()) is policy


def test_root_routed_leaf_keeps_all_expert_parameters_for_root_fsdp():
    model = routed()
    before = set(model.parameters())
    guarded = subject._collective_leaf_wrap_policy(model, lambda **kwargs: True, None)
    assert guarded(model, True, 100) is False
    apply_real_torch_auto_wrap(model, guarded, None)
    assert set(model.parameters()) == before
    assert not any(isinstance(module, Wrapped) for module in model.modules())


def test_amp_batchnorm_override_cannot_bypass_leaf_protection():
    model = routed()
    model.high.append(nn.BatchNorm1d(2))
    with pytest.raises(ValueError, match="BatchNorm overrides would split"):
        subject._collective_leaf_wrap_policy(model, lambda **kwargs: True, MixedPrecision(param_dtype=torch.bfloat16))
    # FP32 has no implicit AMP override and can safely retain the whole group.
    guarded = subject._collective_leaf_wrap_policy(model, lambda **kwargs: True, None)
    apply_real_torch_auto_wrap(model, guarded, None)
    assert not any(isinstance(module, Wrapped) for module in model.modules())


def test_wrap_fsdp_default_size_policy_integrates_leaf_guard(monkeypatch):
    import torch.distributed.fsdp as fsdp
    import torch.distributed.fsdp.wrap as wrap
    monkeypatch.setattr(subject.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(subject.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(wrap, "size_based_auto_wrap_policy", partial(
        size_based_auto_wrap_policy, min_num_params=1,
        force_leaf_modules=size_based_auto_wrap_policy.FORCE_LEAF_MODULES,
        exclude_wrap_modules=size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES))
    captured = {}
    def fake_fsdp(model, **kwargs):
        captured.update(kwargs)
        apply_real_torch_auto_wrap(model, kwargs["auto_wrap_policy"], kwargs["mixed_precision"])
        return Wrapped(model)
    monkeypatch.setattr(fsdp, "FullyShardedDataParallel", fake_fsdp)
    leaf = routed()
    model = nn.ModuleDict({"shared": nn.Linear(2, 2), "route": leaf})
    result = subject.wrap_model_fsdp(model, mixed_precision="bf16")
    assert result.wrapped is model
    assert model["route"] is leaf and isinstance(model["shared"], Wrapped)
    assert not any(isinstance(module, Wrapped) for module in leaf.modules())
    assert captured["use_orig_params"] is True


@pytest.fixture
def fake_ds(monkeypatch):
    package = ModuleType("deepspeed")
    package.__version__ = "0.19.6"
    utils = ModuleType("deepspeed.utils")
    monkeypatch.setitem(sys.modules, "deepspeed", package)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils)
    package.utils = utils
    calls = []
    def set_leaves(model, types):
        calls.append(("leaf", model, types))
        found = [module for module in model.modules() if isinstance(module, tuple(types))]
        for module in found:
            module._z3_leaf = True
        return found
    utils.set_z3_leaf_modules = set_leaves
    def initialize(**kwargs):
        calls.append(("initialize", kwargs))
        if kwargs["config"]["zero_optimization"]["stage"] == 3:
            assert all(getattr(module, "_z3_leaf", False) for module in subject._collective_leaf_modules(kwargs["model"]))
        return SimpleNamespace(module=kwargs["model"]), kwargs["optimizer"], None, kwargs["lr_scheduler"]
    package.initialize = initialize
    return package, calls


def test_zero3_registers_exact_marked_instances_before_initializing(fake_ds):
    package, calls = fake_ds
    class MaybeRouted(nn.Sequential):
        pass
    marked = MaybeRouted(nn.Linear(2, 2))
    marked.requires_worlddistill_collective_leaf = True  # Instance marker, not class marker.
    unmarked = MaybeRouted(nn.Linear(2, 2))
    noise_routed = routed()
    model = nn.ModuleDict({"marked": marked, "unmarked": unmarked, "noise": noise_routed})
    optimizer = torch.optim.AdamW(model.parameters())
    subject.init_deepspeed(model, optimizer, subject.get_deepspeed_config(stage=3, mixed_precision="no"))
    assert calls[0] == ("leaf", marked, [MaybeRouted])
    assert calls[1] == ("leaf", noise_routed, [NoiseRoutedDenoiser])
    assert calls[2][0] == "initialize"
    assert not getattr(unmarked, "_z3_leaf", False)


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_non_zero3_never_requires_leaf_api(fake_ds, stage):
    package, calls = fake_ds
    del package.utils.set_z3_leaf_modules
    model = routed()
    subject.init_deepspeed(model, torch.optim.AdamW(model.parameters()),
                          subject.get_deepspeed_config(stage=stage, mixed_precision="no"))
    assert [call[0] for call in calls] == ["initialize"]


def test_ordinary_zero3_model_never_requires_leaf_api(fake_ds):
    package, calls = fake_ds
    del package.utils.set_z3_leaf_modules
    model = nn.Linear(2, 2)
    subject.init_deepspeed(model, torch.optim.AdamW(model.parameters()),
                          subject.get_deepspeed_config(stage=3, mixed_precision="no"))
    assert [call[0] for call in calls] == ["initialize"]


@pytest.mark.parametrize("failure", ["missing", "not_callable", "ineffective"])
def test_zero3_leaf_api_failure_is_fatal_before_engine_init(fake_ds, failure):
    package, calls = fake_ds
    if failure == "missing":
        del package.utils.set_z3_leaf_modules
    elif failure == "not_callable":
        package.utils.set_z3_leaf_modules = None
    else:
        package.utils.set_z3_leaf_modules = lambda module, classes: []
    model = routed()
    with pytest.raises(RuntimeError, match="collective leaf|leaf_modules|expert sharding"):
        subject.init_deepspeed(model, torch.optim.AdamW(model.parameters()),
                              subject.get_deepspeed_config(stage=3, mixed_precision="no"))
    assert not calls


def test_ddp_wrapper_does_not_add_zero_or_fsdp_leaf_hooks(monkeypatch):
    model = routed()
    monkeypatch.setattr(subject.dist, "is_initialized", lambda: True)
    captured = {}
    def ddp(module, **kwargs):
        captured.update(kwargs)
        return Wrapped(module)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", ddp)
    result = subject.wrap_model_ddp(model, device_ids=[0], find_unused_parameters=True)
    assert result.wrapped is model and not hasattr(model, "_z3_leaf")
    assert captured["find_unused_parameters"] is True
