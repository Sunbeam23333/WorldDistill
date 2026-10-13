from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.utils.zero3_context import suspend_zero3_parameter_hooks


class HookOffload:
    def __init__(self):
        self.module = nn.Sequential(nn.Linear(2, 2), nn.SiLU(), nn.Linear(2, 2))
        self.forward_hooks, self.backward_hooks = [], []
        self.events = []
        self.registrations = 0
        self.invalidations = 0
        self.param_coordinator = SimpleNamespace(is_invalid_trace=lambda: False, _invalidate_trace=self.invalidate)
        self.external = nn.Parameter(torch.ones(2))
        self.external.ds_active_sub_modules = {91}
        self.module[0].ds_external_parameters = lambda: [("external", self.external)]
        for parameter in self.module.parameters():
            parameter.ds_active_sub_modules = {3}
        self._register_deepspeed_module(self.module)

    def invalidate(self):
        self.invalidations += 1

    def _register_deepspeed_module(self, module):
        self.registrations += 1
        self.forward_hooks.append(module.register_forward_pre_hook(lambda *_: self.events.append("fetch")))
        # DeepSpeed's backward hook handles are themselves module forward
        # hooks that install autograd functions, so use that lifecycle here.
        self.backward_hooks.append(module.register_forward_hook(lambda *_: self.events.append("prepare_backward")))


@pytest.mark.parametrize("fail,wrapped", [(False, False), (False, True), (True, True)])
def test_detached_jvp_has_no_partition_hooks_and_restores_after_exception(fail, wrapped):
    offload = HookOffload()
    engine = SimpleNamespace(optimizer=SimpleNamespace(parameter_offload=offload) if wrapped else offload)
    parameters = list(offload.module.parameters())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    original_gradients = [parameter.grad.clone() for parameter in parameters]
    x = torch.randn(1, 2)
    try:
        with suspend_zero3_parameter_hooks(engine) as model:
            assert not offload.forward_hooks and not offload.backward_hooks
            assert not offload.external.ds_active_sub_modules
            assert all(not parameter.ds_active_sub_modules for parameter in parameters)
            _, derivative = torch.autograd.functional.jvp(model, x, torch.ones_like(x))
            assert derivative.shape == x.shape and not offload.events
            if fail:
                raise ValueError("JVP failed")
    except ValueError as error:
        assert fail and str(error) == "JVP failed"
    assert offload.registrations == 2 and offload.invalidations == 1
    assert len(offload.forward_hooks) == len(offload.backward_hooks) == 1
    offload.module(x)
    assert offload.events == ["fetch", "prepare_backward"]
    assert all(before is after for before, after in zip(parameters, offload.module.parameters()))
    for parameter, expected in zip(parameters, original_gradients):
        torch.testing.assert_close(parameter.grad, expected)
    assert not hasattr(offload, "_worlddistill_detached_hook_depth")


def test_nested_hook_suspension_restores_only_at_outer_exit():
    offload = HookOffload()
    engine = SimpleNamespace(optimizer=offload)
    with suspend_zero3_parameter_hooks(engine):
        with suspend_zero3_parameter_hooks(engine):
            offload.module(torch.zeros(1, 2))
        assert not offload.events and offload.registrations == 1
    assert offload.registrations == 2


def test_unknown_zero_hook_layout_fails_before_modifying_training_state():
    offload = HookOffload()
    original = list(offload.forward_hooks)
    offload._register_deepspeed_module = None
    with pytest.raises(RuntimeError, match="hook contract"):
        with suspend_zero3_parameter_hooks(SimpleNamespace(optimizer=offload)):
            pytest.fail("Unsupported hook layout must not run a detached forward")
    assert offload.forward_hooks == original
