import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "runtime" / "fused_supervision.py"
_SPEC = importlib.util.spec_from_file_location("fused_supervision_contract_subject", _PATH)
subject = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(subject)


def compare_loss_and_gradients(prediction, target, mask=None, scale=1.0):
    p, t = prediction.detach().requires_grad_(), target.detach().requires_grad_()
    ref_p, ref_t = p.detach().clone().requires_grad_(), t.detach().clone().requires_grad_()
    actual = subject.fused_masked_mse_loss(p, t, mask)
    reference = subject.fused_masked_mse_loss(ref_p, ref_t, mask, enabled=False)
    upstream = torch.tensor(scale, dtype=torch.float32, device=p.device)
    actual.backward(upstream)
    reference.backward(upstream)
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(p.grad, ref_p.grad, rtol=2e-3, atol=2e-6)
    torch.testing.assert_close(t.grad, ref_t.grad, rtol=2e-3, atol=2e-6)
    return actual, p.grad, t.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("masked", [False, True])
def test_fp32_arithmetic_matches_reference(dtype, masked):
    prediction = torch.linspace(-2, 3, 30).reshape(2, 3, 5).to(dtype)
    target = torch.linspace(1, -3, 30).reshape(2, 3, 5).to(dtype)
    mask = torch.tensor([[[0.0], [0.25], [1.0]]]) if masked else None
    compare_loss_and_gradients(prediction, target, mask, scale=1024)


def test_fp16_subtraction_does_not_overflow_before_reduction():
    _, grad, _ = compare_loss_and_gradients(torch.full((16,), 60000.0, dtype=torch.float16),
                                          torch.full((16,), -60000.0, dtype=torch.float16))
    assert torch.isfinite(grad).all()


def test_grad_scaler_upstream_is_not_cast_to_fp16():
    _, grad, _ = compare_loss_and_gradients(torch.full((4096,), 1e-4, dtype=torch.float16),
                                          torch.zeros(4096, dtype=torch.float16), scale=65536.0)
    assert torch.isfinite(grad).all() and (grad != 0).all()


def test_mixed_dtype_target_gradient_and_zero_mask():
    prediction = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    target = torch.full((3, 4), 0.234567, dtype=torch.float32)
    _, _, grad_target = compare_loss_and_gradients(prediction, target)
    assert grad_target.dtype == torch.float32
    loss, grad, grad_target = compare_loss_and_gradients(prediction, target, torch.zeros(3, 1))
    assert loss == 0 and torch.count_nonzero(grad) == 0 and torch.count_nonzero(grad_target) == 0


def test_masked_forward_does_not_read_a_device_scalar():
    p = torch.ones(2, 3, requires_grad=True)
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar synchronization")):
        loss = subject.fused_masked_mse_loss(p, torch.zeros_like(p), torch.ones(2, 1))
        loss.backward()
    torch.testing.assert_close(p.grad, torch.full_like(p, 1 / 3))


@pytest.mark.parametrize("enabled", [False, True])
def test_shape_contract_is_the_same_for_both_paths(enabled):
    with pytest.raises(ValueError, match="matching shape"):
        subject.fused_masked_mse_loss(torch.ones(2, 3), torch.ones(3), enabled=enabled)


def _cuda_metadata(device="cuda:2", dtype=torch.float16):
    return SimpleNamespace(is_cuda=True, device=torch.device(device), dtype=dtype, numel=lambda: 8)


@pytest.mark.parametrize("capability,eligible", [
    ((5, 2), False), ((6, 0), False), ((7, 0), False), ((7, 2), False), ((7, 5), False),
    ((8, 0), True), ((8, 6), True), ((8, 9), True), ((9, 0), True),
    ((10, 0), True), ((10, 3), True), ((12, 0), True), ((12, 1), True),
    ((8, 5), False), ((9, 9), False), ((10, 1), False), ((13, 0), False),
])
def test_architecture_gate_uses_actual_tensor_device(monkeypatch, capability, eligible):
    monkeypatch.setattr(subject, "_HAS_TRITON", True)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.version, "hip", None)
    tensor = _cuda_metadata()
    with patch.object(torch.cuda, "get_device_capability", return_value=capability) as query, \
            patch.object(torch.cuda, "current_device", side_effect=AssertionError("Do not use current device")):
        assert subject._can_use_fused_kernel(tensor, tensor, None) is eligible
    query.assert_called_once_with(torch.device("cuda:2"))


@pytest.mark.parametrize("cuda_version,hip_version", [(None, None), (None, "6.4"), ("12.8", "6.4")])
def test_cpu_and_rocm_builds_never_enter_nvidia_triton(monkeypatch, cuda_version, hip_version):
    monkeypatch.setattr(subject, "_HAS_TRITON", True)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    tensor = _cuda_metadata()
    with patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("Not NVIDIA CUDA")):
        assert not subject._can_use_fused_kernel(tensor, tensor, None)


def test_different_gpu_capabilities_do_not_share_current_device_decision(monkeypatch):
    monkeypatch.setattr(subject, "_HAS_TRITON", True)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.version, "hip", None)
    with patch.object(torch.cuda, "get_device_capability", side_effect=lambda device: (7, 5) if device.index == 0 else (8, 0)):
        old, new = _cuda_metadata("cuda:0"), _cuda_metadata("cuda:1")
        assert not subject._can_use_fused_kernel(old, old, None)
        assert subject._can_use_fused_kernel(new, new, None)
        assert not subject._can_use_fused_kernel(old, old, None)


def test_target_fallback_keeps_fp32_gradient_arithmetic(monkeypatch):
    # Force the unsupported-target decision in a real numerical CPU call;
    # architecture routing is independently tested above without CUDA hardware.
    monkeypatch.setattr(subject, "_can_use_fused_kernel", lambda *args: False)
    compare_loss_and_gradients(torch.full((4096,), 1e-4, dtype=torch.float16),
                              torch.zeros(4096, dtype=torch.float16), scale=65536.0)


def test_unsafe_kernel_errors_are_not_silently_swallowed(monkeypatch):
    class BrokenKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("CUDA illegal memory access")
            return launch
    monkeypatch.setattr(subject, "_can_use_fused_kernel", lambda *args: True)
    monkeypatch.setattr(subject, "triton", SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y))
    monkeypatch.setattr(subject, "_masked_mse_forward_kernel", BrokenKernel(), raising=False)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        subject.fused_masked_mse_loss(torch.ones(8, requires_grad=True), torch.zeros(8))


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.cuda, reason="real NVIDIA CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("masked", [False, True])
def test_real_cuda_forward_backward_tail(dtype, masked):
    if not subject.fused_supervision_available(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("Actual CUDA device is not eligible for the Triton fused path")
    prediction = torch.linspace(-3, 2, 2050, device="cuda").reshape(2, 1025).to(dtype)
    target = torch.randn_like(prediction)
    mask = torch.tensor([[0.25], [1.0]], device="cuda") if masked else None
    assert subject._can_use_fused_kernel(prediction, target, mask)
    compare_loss_and_gradients(prediction, target, mask, scale=65536.0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.cuda, reason="real NVIDIA CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_real_cuda_ineligible_target_uses_reference(dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    if subject.fused_supervision_available(device):
        pytest.skip("This device is eligible; the fused-path test covers it")
    prediction = torch.linspace(-2, 3, 30, device=device).reshape(2, 3, 5).to(dtype)
    target = torch.zeros_like(prediction)
    assert not subject._can_use_fused_kernel(prediction, target, None)
    compare_loss_and_gradients(prediction, target, scale=1024)
    torch.cuda.synchronize(device)
