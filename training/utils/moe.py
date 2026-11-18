"""Generic sparse-MoE routing, capacity, and auxiliary-loss contracts.

The Wan2.2 noise router remains a model-native dual-expert path.  This module
adds architecture-neutral top-k token routing, capacity limiting, local
dispatch/combine, expert-parallel collectives, and router auxiliary losses.
Model adapters still own expert module execution and checkpoint layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenRoute:
    """Capacity-limited top-k assignment for a flattened token batch."""

    expert_indices: torch.Tensor
    combine_weights: torch.Tensor
    token_locations: torch.Tensor
    kept_mask: torch.Tensor
    capacity: int
    num_tokens: int
    num_experts: int
    top_k: int


def build_token_route(
    router_logits: torch.Tensor,
    *,
    top_k: int = 2,
    capacity_factor: float = 1.25,
    min_capacity: int = 4,
    capacity: Optional[int] = None,
) -> TokenRoute:
    """Build a deterministic, capacity-limited top-k token route.

    Router logits may have any leading shape and are flattened to ``N x E``.
    An explicit capacity lets expert-parallel callers use a global token count.
    """
    if router_logits.ndim < 2:
        raise ValueError("router_logits must have at least two dimensions")
    logits = router_logits.float().reshape(-1, router_logits.shape[-1])
    num_tokens, num_experts = logits.shape
    if type(top_k) is not int or top_k < 1 or top_k > num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if not torch.isfinite(logits).all():
        raise ValueError("router_logits must be finite")
    if capacity is None:
        if not torch.isfinite(torch.tensor(capacity_factor)) or capacity_factor <= 0:
            raise ValueError("capacity_factor must be finite and positive")
        if type(min_capacity) is not int or min_capacity < 1:
            raise ValueError("min_capacity must be a positive integer")
        capacity = max(min_capacity, int(torch.ceil(torch.tensor(
            capacity_factor * num_tokens * top_k / num_experts
        )).item()))
    if type(capacity) is not int or capacity < 1:
        raise ValueError("capacity must be a positive integer")

    probabilities = logits.softmax(dim=-1)
    top_probs, expert_indices = probabilities.topk(top_k, dim=-1)
    combine_weights = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    flat_experts = expert_indices.reshape(-1)
    token_locations = torch.zeros_like(flat_experts)
    for expert_id in range(num_experts):
        positions = torch.nonzero(flat_experts == expert_id, as_tuple=False).flatten()
        if positions.numel():
            token_locations[positions] = torch.arange(
                positions.numel(), device=router_logits.device, dtype=torch.long
            )
    token_locations = token_locations.reshape(num_tokens, top_k)
    kept_mask = token_locations < capacity
    combine_weights = combine_weights * kept_mask.to(combine_weights.dtype)
    combine_weights = combine_weights / combine_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return TokenRoute(
        expert_indices=expert_indices,
        combine_weights=combine_weights,
        token_locations=token_locations,
        kept_mask=kept_mask,
        capacity=capacity,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
    )


def build_expert_parallel_route(
    router_logits: torch.Tensor,
    *,
    top_k: int = 2,
    capacity_factor: float = 1.25,
    min_capacity: int = 4,
    expert_parallel_size: int = 1,
    group: Any = None,
) -> TokenRoute:
    """Build a route whose capacity uses the global EP token count."""
    _, world = _group_rank_world(group)
    if expert_parallel_size != world:
        if expert_parallel_size > 1 and world == 1 and (
            not torch.distributed.is_available() or not torch.distributed.is_initialized()
        ):
            raise RuntimeError("expert_parallel_size>1 requires an initialized torch.distributed process group")
        raise ValueError(f"expert_parallel_size={expert_parallel_size} disagrees with group world size {world}")
    global_tokens = router_logits.reshape(-1, router_logits.shape[-1]).shape[0]
    if expert_parallel_size > 1:
        count = torch.tensor([global_tokens], device=router_logits.device, dtype=torch.long)
        torch.distributed.all_reduce(count, group=group)
        global_tokens = int(count.item())
    num_experts = router_logits.shape[-1]
    if type(min_capacity) is not int or min_capacity < 1:
        raise ValueError("min_capacity must be a positive integer")
    if not torch.isfinite(torch.tensor(capacity_factor)) or capacity_factor <= 0:
        raise ValueError("capacity_factor must be finite and positive")
    capacity = max(min_capacity, int(torch.ceil(torch.tensor(
        capacity_factor * global_tokens * top_k / num_experts
    )).item()))
    return build_token_route(router_logits, top_k=top_k, min_capacity=min_capacity, capacity=capacity)


def dispatch_tokens(tokens: torch.Tensor, route: TokenRoute) -> torch.Tensor:
    """Pack tokens into a local ``[experts, capacity, hidden]`` buffer."""
    if tokens.ndim != 2 or tokens.shape[0] != route.num_tokens:
        raise ValueError("tokens must have shape [num_tokens, hidden]")
    output = tokens.new_zeros((route.num_experts, route.capacity, tokens.shape[-1]))
    for expert_id in range(route.num_experts):
        selected = (route.expert_indices == expert_id) & route.kept_mask
        token_idx, top_idx = torch.nonzero(selected, as_tuple=True)
        if token_idx.numel():
            output[expert_id, route.token_locations[token_idx, top_idx]] = tokens[token_idx]
    return output


def combine_tokens(expert_outputs: torch.Tensor, route: TokenRoute) -> torch.Tensor:
    """Combine a local ``[experts, capacity, hidden]`` buffer to tokens."""
    expected = (route.num_experts, route.capacity)
    if expert_outputs.ndim != 3 or tuple(expert_outputs.shape[:2]) != expected:
        raise ValueError("expert_outputs must have shape [num_experts, capacity, hidden]")
    result = expert_outputs.new_zeros((route.num_tokens, expert_outputs.shape[-1]))
    for expert_id in range(route.num_experts):
        selected = (route.expert_indices == expert_id) & route.kept_mask
        token_idx, top_idx = torch.nonzero(selected, as_tuple=True)
        if token_idx.numel():
            values = expert_outputs[expert_id, route.token_locations[token_idx, top_idx]]
            result.index_add_(0, token_idx, values * route.combine_weights[token_idx, top_idx].to(values.dtype).unsqueeze(-1))
    return result


@dataclass
class ExpertParallelDispatch:
    """Packed expert inputs and reverse-routing metadata."""

    expert_inputs: torch.Tensor
    local_expert_indices: torch.Tensor
    send_token_indices: torch.Tensor
    send_weights: torch.Tensor
    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    route: TokenRoute
    group: Any = None


def _group_rank_world(group: Any) -> tuple[int, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0, 1
    return torch.distributed.get_rank(group), torch.distributed.get_world_size(group)


def dispatch_expert_parallel(
    tokens: torch.Tensor,
    route: TokenRoute,
    *,
    expert_parallel_size: int = 1,
    group: Any = None,
) -> ExpertParallelDispatch:
    """Dispatch tokens with local fallback or ``all_to_all_single`` EP.

    Experts are evenly partitioned across the process group.  The returned
    payload is packed by destination rank, and ``local_expert_indices`` maps
    every row to the local expert that should process it.
    """
    if tokens.ndim != 2 or tokens.shape[0] != route.num_tokens:
        raise ValueError("tokens must have shape [num_tokens, hidden]")
    if type(expert_parallel_size) is not int or expert_parallel_size < 1:
        raise ValueError("expert_parallel_size must be a positive integer")
    rank, world = _group_rank_world(group)
    if expert_parallel_size > 1 and world == 1 and (
        not torch.distributed.is_available() or not torch.distributed.is_initialized()
    ):
        raise RuntimeError("expert_parallel_size>1 requires an initialized torch.distributed process group")
    if expert_parallel_size != world:
        raise ValueError(f"expert_parallel_size={expert_parallel_size} disagrees with group world size {world}")
    if route.num_experts % expert_parallel_size:
        raise ValueError("num_experts must be divisible by expert_parallel_size")
    experts_per_rank = route.num_experts // expert_parallel_size
    flat_experts = route.expert_indices.reshape(-1)
    flat_keep = route.kept_mask.reshape(-1)
    flat_tokens = torch.arange(route.num_tokens, device=tokens.device).view(-1, 1).expand(-1, route.top_k).reshape(-1)
    flat_weights = route.combine_weights.reshape(-1)
    destinations = flat_experts // experts_per_rank
    order = torch.argsort(destinations.masked_fill(~flat_keep, expert_parallel_size), stable=True)
    order = order[flat_keep[order]]
    send_token_indices = flat_tokens[order]
    send_weights = flat_weights[order]
    send_local_experts = (flat_experts[order] % experts_per_rank).to(torch.long)
    send_payload = tokens[send_token_indices]
    send_counts = torch.bincount(destinations[order], minlength=expert_parallel_size).to(torch.long)

    if expert_parallel_size == 1:
        return ExpertParallelDispatch(
            expert_inputs=send_payload,
            local_expert_indices=send_local_experts,
            send_token_indices=send_token_indices,
            send_weights=send_weights,
            send_counts=send_counts,
            recv_counts=send_counts.clone(),
            route=route,
            group=group,
        )
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("expert_parallel_size>1 requires an initialized torch.distributed process group")
    recv_counts = torch.empty_like(send_counts)
    torch.distributed.all_to_all_single(recv_counts, send_counts, group=group)
    recv_payload = tokens.new_empty((int(recv_counts.sum().item()), tokens.shape[-1]))
    torch.distributed.all_to_all_single(
        recv_payload,
        send_payload,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=group,
    )
    recv_experts = torch.empty((int(recv_counts.sum().item()),), device=tokens.device, dtype=torch.long)
    torch.distributed.all_to_all_single(
        recv_experts,
        send_local_experts,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=group,
    )
    return ExpertParallelDispatch(
        expert_inputs=recv_payload,
        local_expert_indices=recv_experts,
        send_token_indices=send_token_indices,
        send_weights=send_weights,
        send_counts=send_counts,
        recv_counts=recv_counts,
        route=route,
        group=group,
    )


def combine_expert_parallel(
    expert_outputs: torch.Tensor,
    dispatch: ExpertParallelDispatch,
) -> torch.Tensor:
    """Return packed outputs to source ranks and combine router weights."""
    if expert_outputs.ndim != 2 or expert_outputs.shape[0] != dispatch.expert_inputs.shape[0]:
        raise ValueError("expert_outputs must match dispatch.expert_inputs as [packed_tokens, hidden]")
    if dispatch.recv_counts.numel() == 1:
        returned = expert_outputs
    else:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("distributed process group is required for expert-parallel combine")
        returned = expert_outputs.new_empty((int(dispatch.send_counts.sum().item()), expert_outputs.shape[-1]))
        torch.distributed.all_to_all_single(
            returned,
            expert_outputs,
            output_split_sizes=dispatch.send_counts.tolist(),
            input_split_sizes=dispatch.recv_counts.tolist(),
            group=dispatch.group,
        )
    result = expert_outputs.new_zeros((dispatch.route.num_tokens, expert_outputs.shape[-1]))
    result.index_add_(
        0,
        dispatch.send_token_indices,
        returned * dispatch.send_weights.to(returned.dtype).unsqueeze(-1),
    )
    return result


@dataclass
class MoEBlockOutput:
    """Output contract understood by the trainer's prediction extractor."""

    sample: torch.Tensor
    router_logits: torch.Tensor
    aux_loss: torch.Tensor
    dropped_tokens: int


class TokenMoE(torch.nn.Module):
    """Small architecture-neutral sparse token MoE block.

    This block is intended for adapters and tests that need a real dispatch
    implementation.  Large model families can provide their own expert
    modules while reusing the route and expert-parallel contracts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        expert_factory: Any = None,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        min_capacity: int = 4,
        expert_parallel_size: int = 1,
        group: Any = None,
    ):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if type(num_experts) is not int or num_experts < 1:
            raise ValueError("num_experts must be a positive integer")
        rank, world = _group_rank_world(group)
        if expert_parallel_size != world:
            if expert_parallel_size > 1 and world == 1 and (
                not torch.distributed.is_available() or not torch.distributed.is_initialized()
            ):
                raise RuntimeError("expert_parallel_size>1 requires an initialized torch.distributed process group")
            raise ValueError(f"expert_parallel_size={expert_parallel_size} disagrees with group world size {world}")
        if num_experts % expert_parallel_size:
            raise ValueError("num_experts must be divisible by expert_parallel_size")
        if type(top_k) is not int or top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if capacity_factor <= 0 or not torch.isfinite(torch.tensor(capacity_factor)):
            raise ValueError("capacity_factor must be finite and positive")
        if type(min_capacity) is not int or min_capacity < 1:
            raise ValueError("min_capacity must be a positive integer")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = float(capacity_factor)
        self.min_capacity = min_capacity
        self.expert_parallel_size = expert_parallel_size
        self.group = group
        self.router = torch.nn.Linear(hidden_size, num_experts)
        local_experts = num_experts // expert_parallel_size
        if expert_factory is None:
            expert_factory = lambda width: torch.nn.Sequential(
                torch.nn.Linear(width, width * 4), torch.nn.GELU(), torch.nn.Linear(width * 4, width)
            )
        self.experts = torch.nn.ModuleList([expert_factory(hidden_size) for _ in range(local_experts)])

    def forward(self, hidden_states: torch.Tensor) -> MoEBlockOutput:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}")
        original_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.router(tokens).reshape(*original_shape[:-1], self.num_experts)
        route = build_expert_parallel_route(
            router_logits,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
            min_capacity=self.min_capacity,
            expert_parallel_size=self.expert_parallel_size,
            group=self.group,
        )
        if self.expert_parallel_size == 1:
            packed = dispatch_tokens(tokens, route)
            expert_outputs = packed.new_zeros(packed.shape)
            for expert_id, expert in enumerate(self.experts):
                expert_outputs[expert_id] = expert(packed[expert_id])
            output = combine_tokens(expert_outputs, route)
        else:
            dispatch = dispatch_expert_parallel(tokens, route, expert_parallel_size=self.expert_parallel_size, group=self.group)
            expert_outputs = dispatch.expert_inputs.new_zeros(dispatch.expert_inputs.shape)
            for local_id, expert in enumerate(self.experts):
                selected = dispatch.local_expert_indices == local_id
                if selected.any():
                    expert_outputs[selected] = expert(dispatch.expert_inputs[selected])
            output = combine_expert_parallel(expert_outputs, dispatch)
        dropped = int((~route.kept_mask).sum().item())
        aux_loss = router_load_balance_loss(router_logits, top_k=self.top_k, num_experts=self.num_experts)
        return MoEBlockOutput(
            sample=output.reshape(original_shape),
            router_logits=router_logits,
            aux_loss=aux_loss,
            dropped_tokens=dropped,
        )


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
