import importlib.util
from pathlib import Path
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


@pytest.mark.skipif(not torch.cuda.is_available() or not subject.fused_supervision_available(), reason="real CUDA + Triton required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("masked", [False, True])
def test_real_cuda_forward_backward_tail(dtype, masked):
    prediction = torch.linspace(-3, 2, 2050, device="cuda").reshape(2, 1025).to(dtype)
    target = torch.randn_like(prediction)
    mask = torch.tensor([[0.25], [1.0]], device="cuda") if masked else None
    assert subject._can_use_fused_kernel(prediction, target, mask)
    compare_loss_and_gradients(prediction, target, mask, scale=65536.0)
    torch.cuda.synchronize()
