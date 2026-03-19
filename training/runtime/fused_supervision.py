"""Fused supervision kernels for continuous-space distillation losses.

This module provides a Triton-accelerated masked MSE reduction tailored for
teacher-student supervision in diffusion / world-model distillation. The fused
kernel avoids materializing the full squared-difference tensor during the
forward loss reduction path and falls back to vanilla PyTorch when Triton or
CUDA is unavailable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _masked_mse_forward_kernel(
        prediction_ptr,
        target_ptr,
        mask_ptr,
        partial_ptr,
        numel,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offsets < numel

        prediction = tl.load(prediction_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
        target = tl.load(target_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
        mask_value = tl.load(mask_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)

        diff = prediction - target
        partial = tl.sum(diff * diff * mask_value, axis=0)
        tl.store(partial_ptr + pid, partial)


class _FusedMaskedMSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must share the same shape for fused supervision: "
                f"got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )

        if mask is not None:
            expanded_mask = _prepare_mask(mask, prediction)
        else:
            expanded_mask = None

        if not _can_use_fused_kernel(prediction, target, expanded_mask):
            loss = _torch_masked_mse(prediction, target, expanded_mask)
            ctx.use_torch_backward = True
            ctx.has_mask = expanded_mask is not None
            if expanded_mask is None:
                ctx.save_for_backward(prediction, target)
                ctx.denominator = float(prediction.numel())
            else:
                ctx.save_for_backward(prediction, target, expanded_mask)
                ctx.denominator = float(expanded_mask.float().sum().item())
            return loss

        prediction_flat = prediction.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        if expanded_mask is None:
            mask_flat = torch.ones_like(prediction_flat, dtype=torch.float32)
        else:
            mask_flat = expanded_mask.contiguous().view(-1).to(device=prediction.device, dtype=torch.float32)

        numel = prediction_flat.numel()
        partial = torch.empty((triton.cdiv(numel, 1024),), device=prediction.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        _masked_mse_forward_kernel[grid](
            prediction_flat,
            target_flat,
            mask_flat,
            partial,
            numel,
            BLOCK_SIZE=1024,
        )

        numerator = partial.sum()
        denominator = mask_flat.sum().clamp(min=1.0)
        ctx.use_torch_backward = False
        ctx.has_mask = expanded_mask is not None
        if expanded_mask is None:
            ctx.save_for_backward(prediction, target)
        else:
            ctx.save_for_backward(prediction, target, expanded_mask)
        ctx.denominator = float(denominator.item())
        return numerator / denominator

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        if ctx.has_mask:
            prediction, target, mask = saved
            weight = mask.to(dtype=prediction.dtype)
        else:
            prediction, target = saved
            weight = None

        grad_scale = grad_output.to(dtype=prediction.dtype)
        denom = max(ctx.denominator, 1.0)
        grad_prediction = 2.0 * (prediction - target) / denom
        if weight is not None:
            grad_prediction = grad_prediction * weight
        grad_prediction = grad_prediction * grad_scale

        grad_target = None
        if ctx.needs_input_grad[1]:
            grad_target = -grad_prediction
        return grad_prediction, grad_target, None


def fused_masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    enabled: bool = True,
) -> torch.Tensor:
    """Compute a masked mean-squared supervision loss.

    Args:
        prediction: Student output tensor.
        target: Teacher target tensor.
        mask: Optional broadcastable mask. Non-zero entries contribute to loss.
        enabled: Whether to try the Triton fused path when available.

    Returns:
        Scalar mean-squared error over the masked region.
    """
    prepared_mask = _prepare_mask(mask, prediction) if mask is not None else None
    if not enabled:
        return _torch_masked_mse(prediction, target, prepared_mask)
    return _FusedMaskedMSEFunction.apply(prediction, target, prepared_mask)


def fused_supervision_available() -> bool:
    return _HAS_TRITON


def _torch_masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    diff = prediction.float() - target.float()
    loss = diff.square()
    if mask is None:
        return loss.mean()
    mask_float = mask.to(device=prediction.device, dtype=loss.dtype)
    return (loss * mask_float).sum() / mask_float.sum().clamp(min=1.0)


def _prepare_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.shape != reference.shape:
        mask = torch.broadcast_to(mask, reference.shape)
    return mask.to(device=reference.device)


def _can_use_fused_kernel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> bool:
    if not _HAS_TRITON:
        return False
    if not prediction.is_cuda or not target.is_cuda:
        return False
    if prediction.numel() == 0:
        return False
    if prediction.device != target.device:
        return False
    if prediction.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    if target.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    if mask is not None and not mask.is_cuda:
        return False
    return True


__all__ = [
    "fused_masked_mse_loss",
    "fused_supervision_available",
]
