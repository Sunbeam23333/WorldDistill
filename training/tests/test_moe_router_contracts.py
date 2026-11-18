import pytest
import torch

from training.utils.moe import model_moe_auxiliary_loss, router_load_balance_loss


def test_router_load_balance_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 5, 4, requires_grad=True)
    loss = router_load_balance_loss(logits, top_k=2, num_experts=4, z_loss_weight=0.01)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_router_load_balance_validates_top_k_and_expert_count():
    logits = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="top_k"):
        router_load_balance_loss(logits, top_k=5)
    with pytest.raises(ValueError, match="num_experts"):
        router_load_balance_loss(logits, num_experts=3)


def test_model_auxiliary_loss_reads_router_logits_and_aux_loss():
    router_logits = torch.randn(6, 4, requires_grad=True)
    aux_loss = torch.tensor(0.25, requires_grad=True)
    output = {"sample": torch.randn(6, 2), "router_logits": router_logits, "aux_loss": aux_loss}
    loss = model_moe_auxiliary_loss(output, top_k=1, num_experts=4, load_balance_weight=0.5)
    assert loss is not None
    loss.backward()
    assert router_logits.grad is not None
    assert aux_loss.grad is not None
