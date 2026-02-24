"""Optimizers for distillation training.

Provides:
- Standard AdamW
- Muon optimizer (Newton-Schulz orthogonalization, from HY-WorldPlay)

The Muon optimizer applies Newton-Schulz orthogonalization to >=2D parameters
and falls back to AdamW for 1D parameters (biases, norms).

References:
- HY-WorldPlay trainer/training/muon.py
- https://github.com/KellerJordan/modded-nanogpt
"""

import math
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer


class Muon(Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalization.

    For parameters with >=2 dimensions, applies Newton-Schulz iteration
    to orthogonalize the update direction. For 1D parameters, uses AdamW.

    Args:
        muon_params: Parameters to optimize with Muon (>=2D).
        lr: Learning rate for Muon parameters.
        momentum: Momentum coefficient (default 0.95).
        nesterov: Use Nesterov momentum (default True).
        ns_steps: Number of Newton-Schulz iterations (default 5).
        adamw_params: Parameters to optimize with AdamW (1D).
        adamw_lr: Learning rate for AdamW parameters.
        adamw_betas: AdamW beta coefficients.
        adamw_eps: AdamW epsilon.
        adamw_wd: AdamW weight decay.
    """

    def __init__(
        self,
        muon_params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[Iterable] = None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        muon_params_list = list(muon_params)
        param_groups = [{"params": muon_params_list, "type": "muon"}]

        self.adamw_optimizer = None
        if adamw_params is not None:
            adamw_params_list = list(adamw_params)
            if adamw_params_list:
                self.adamw_optimizer = AdamW(
                    adamw_params_list,
                    lr=adamw_lr,
                    betas=adamw_betas,
                    eps=adamw_eps,
                    weight_decay=adamw_wd,
                )
                param_groups.append({"params": adamw_params_list, "type": "adamw"})

        super().__init__(param_groups, defaults)

    @staticmethod
    @torch.no_grad()
    def newton_schulz_(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """Apply Newton-Schulz iteration for approximate matrix orthogonalization.

        Iteratively computes: G_{k+1} = 1.5 * G_k - 0.5 * G_k @ G_k^T @ G_k
        which converges to the nearest orthogonal matrix.

        Args:
            G: Input matrix.
            steps: Number of iterations.
            eps: Stability epsilon.

        Returns:
            Approximately orthogonalized matrix.
        """
        assert G.ndim >= 2

        # Reshape to 2D for computation
        original_shape = G.shape
        if G.ndim > 2:
            G = G.reshape(G.shape[0], -1)

        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.float()
        X /= X.norm() + eps

        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True
        else:
            transposed = False

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        if transposed:
            X = X.T

        return X.to(G.dtype).reshape(original_shape)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("type") == "adamw":
                continue  # Handled by self.adamw_optimizer

            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Apply Newton-Schulz orthogonalization for >=2D params
                if g.ndim >= 2:
                    g = self.newton_schulz_(g, steps=ns_steps)

                p.add_(g, alpha=-lr)
                state["step"] += 1

        # Step AdamW for 1D params
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.step()

        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.zero_grad(set_to_none=set_to_none)


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    **kwargs,
) -> Optimizer:
    """Build optimizer from configuration.

    Args:
        model: Model to optimize.
        optimizer_type: "adamw" or "muon".
        lr: Learning rate.
        weight_decay: Weight decay (for AdamW).
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        Configured optimizer instance.
    """
    if optimizer_type == "adamw":
        # Separate weight decay and no-decay params
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(param_groups, lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_epsilon)

    elif optimizer_type == "muon":
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        return Muon(
            muon_params=muon_params,
            lr=lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            adamw_params=adamw_params,
            adamw_lr=lr * 0.1,
            adamw_betas=(adam_beta1, adam_beta2),
            adamw_eps=adam_epsilon,
            adamw_wd=weight_decay,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Use 'adamw' or 'muon'.")
