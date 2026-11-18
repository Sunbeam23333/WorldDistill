"""Optional generic sparse-MoE router losses.

The Wan2.2 noise router remains a model-native dual-expert path.  This module
adds a small, architecture-neutral auxiliary objective for custom models that
return ``router_logits`` or ``aux_loss`` alongside their prediction.  It does
not change expert dispatch or claim expert parallelism.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F


def _read_field(output: Any, name: str):
    if isinstance(output, dict):
        return output.get(name)
    return getattr(output, name, None)


def _iter_router_logits(output: Any) -> Iterable[torch.Tensor]:
    value = _read_field(output, "router_logits")
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        yield from (item for item in value if torch.is_tensor(item))


def _flatten_router_logits(logits: torch.Tensor, num_experts: int | None) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("router_logits must have at least two dimensions")
    if num_experts is not None and logits.shape[-1] != num_experts:
        raise ValueError(
            f"router_logits last dimension {logits.shape[-1]} disagrees with moe_num_experts={num_experts}"
        )
    return logits.float().reshape(-1, logits.shape[-1])


def router_load_balance_loss(
    router_logits: torch.Tensor,
    *,
    top_k: int = 2,
    num_experts: int | None = None,
    z_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Compute Switch-style importance/load balancing plus optional z-loss."""
    logits = _flatten_router_logits(router_logits, num_experts)
    experts = logits.shape[-1]
    if type(top_k) is not int or top_k < 1 or top_k > experts:
        raise ValueError("moe_top_k must be in [1, num_experts]")
    probabilities = logits.softmax(dim=-1)
    top_indices = logits.topk(top_k, dim=-1).indices
    assignments = F.one_hot(top_indices, num_classes=experts).float().sum(dim=1).clamp_max(1.0)
    importance = probabilities.mean(dim=0)
    load = assignments.mean(dim=0)
    loss = experts * torch.sum(importance * load)
    if z_loss_weight:
        loss = loss + float(z_loss_weight) * torch.logsumexp(logits, dim=-1).square().mean()
    return loss


def model_moe_auxiliary_loss(
    output: Any,
    *,
    top_k: int = 2,
    num_experts: int | None = None,
    load_balance_weight: float = 0.0,
    z_loss_weight: float = 0.0,
) -> torch.Tensor | None:
    """Extract model-provided router signals and form a differentiable penalty."""
    penalties: list[torch.Tensor] = []
    aux = _read_field(output, "aux_loss")
    if torch.is_tensor(aux):
        penalties.append(aux.float())
    if load_balance_weight or z_loss_weight:
        for logits in _iter_router_logits(output):
            penalties.append(
                float(load_balance_weight)
                * router_load_balance_loss(
                    logits,
                    top_k=top_k,
                    num_experts=num_experts,
                    z_loss_weight=z_loss_weight,
                )
            )
    if not penalties:
        return None
    return torch.stack([value if value.ndim == 0 else value.mean() for value in penalties]).sum()
