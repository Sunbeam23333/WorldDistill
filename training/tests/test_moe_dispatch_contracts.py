import pytest
import torch

from training.utils.moe import (
    build_expert_parallel_route,
    build_token_route,
    combine_expert_parallel,
    combine_tokens,
    dispatch_expert_parallel,
    dispatch_tokens,
    TokenMoE,
)
from training.trainer_args import TrainerArgs


def test_capacity_limited_local_route_roundtrip():
    torch.manual_seed(0)
    logits = torch.randn(8, 4, requires_grad=True)
    tokens = torch.randn(8, 6)
    route = build_token_route(logits, top_k=2, capacity_factor=2.0, min_capacity=1)
    packed = dispatch_tokens(tokens, route)
    restored = combine_tokens(packed, route)
    assert packed.shape == (4, route.capacity, 6)
    assert torch.allclose(restored, tokens, atol=1e-6)
    restored.square().mean().backward()
    assert logits.grad is not None


def test_capacity_drops_assignments_without_invalid_slots():
    route = build_token_route(torch.zeros(16, 4), top_k=2, capacity_factor=0.25, min_capacity=1)
    assert route.capacity == 2
    assert int(route.kept_mask.sum()) <= route.capacity * route.num_experts
    assert int(route.token_locations[route.kept_mask].max()) < route.capacity


def test_expert_parallel_single_rank_fallback_roundtrip():
    route = build_expert_parallel_route(torch.randn(6, 2), top_k=1, capacity_factor=2.0, min_capacity=1)
    tokens = torch.randn(6, 5)
    dispatch = dispatch_expert_parallel(tokens, route, expert_parallel_size=1)
    restored = combine_expert_parallel(dispatch.expert_inputs, dispatch)
    assert torch.allclose(restored, tokens, atol=1e-6)


def test_expert_parallel_requires_process_group_for_multiple_ranks():
    route = build_token_route(torch.randn(4, 2), top_k=1)
    with pytest.raises(RuntimeError, match="process group"):
        dispatch_expert_parallel(torch.randn(4, 3), route, expert_parallel_size=2)


def test_moe_capacity_and_expert_parallel_args_validate():
    args = TrainerArgs(
        report_to="none",
        moe_capacity_factor=1.5,
        moe_min_capacity=2,
        moe_expert_parallel_size=2,
    )
    assert args.moe_expert_parallel_size == 2
    with pytest.raises(ValueError, match="moe_capacity_factor"):
        TrainerArgs(report_to="none", moe_capacity_factor=0)


def test_token_moe_block_exposes_prediction_and_router_contract():
    block = TokenMoE(hidden_size=4, num_experts=2, top_k=1, capacity_factor=2.0, min_capacity=1)
    hidden = torch.randn(3, 5, 4, requires_grad=True)
    output = block(hidden)
    assert output.sample.shape == hidden.shape
    assert output.router_logits.shape == (3, 5, 2)
    assert output.aux_loss.ndim == 0
    (output.sample.square().mean() + output.aux_loss).backward()
    assert block.router.weight.grad is not None
