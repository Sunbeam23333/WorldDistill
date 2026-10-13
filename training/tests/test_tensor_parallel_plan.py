"""Actual sharded Wan linear/attention/FFN gradients versus a full model."""
import copy
from contextlib import nullcontext
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from training.parallel.tensor_parallel import apply_wan_tensor_parallel, apply_tensor_parallel_plan
from training.parallel.state import collect_full_model_state, collect_full_named_tensors, load_full_model_state


def tiny_wan():
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=4,
        attention_head_dim=8, in_channels=2, out_channels=2, text_dim=8, freq_dim=8,
        ffn_dim=64, num_layers=2)


def add_lora(model):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=3, lora_alpha=5, use_rslora=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]))
    # Nonzero B exercises both factors' gradients, not only the initial B step.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    return model


def _worker(rank, world, path, kind, device_type="cpu"):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", init_method=f"file://{path}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=100))
    try:
        torch.manual_seed(83)
        if kind.startswith("wan"):
            reference = tiny_wan()
            if "lora" in kind:
                reference = add_lora(reference)
            reference.to(device)
            model = apply_wan_tensor_parallel(copy.deepcopy(reference), dist.group.WORLD)
            inputs = dict(hidden_states=torch.randn(1, 2, 2, 4, 4), timestep=torch.tensor([33.]),
                          encoder_hidden_states=torch.randn(1, 3, 8), return_dict=False)
            inputs = {name: value.to(device) if torch.is_tensor(value) else value for name, value in inputs.items()}
            with torch.autocast(device_type, dtype=torch.bfloat16) if "bf16" in kind else nullcontext():
                expected = reference(**inputs)[0]
                actual = model(**inputs)[0]
        else:
            reference = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
            reference.to(device)
            model = apply_tensor_parallel_plan(copy.deepcopy(reference), dist.group.WORLD, {"0": "column", "2": "row"})
            x = torch.randn(2, 3, 8, device=device)
            expected, actual = reference(x), model(x)
        low_precision = "bf16" in kind
        torch.testing.assert_close(actual, expected, atol=0.015 if low_precision else 3e-6,
                                   rtol=0.06 if low_precision else 3e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        grads = collect_full_named_tensors(model, {n: p.grad for n, p in model.named_parameters()}, dist.group.WORLD)
        for name, parameter in reference.named_parameters():
            assert (name in grads) == (parameter.grad is not None), name
            if parameter.grad is not None:
                torch.testing.assert_close(grads[name], parameter.grad.cpu(),
                                           atol=0.0015 if low_precision else 2e-6,
                                           rtol=0.06 if low_precision else 3e-4,
                                           msg=lambda msg: f"{name}: {msg}")
        assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in reference.parameters())
        # A finite epsilon prevents rounding-level near-zero gradient changes
        # from becoming a disproportionately large first Adam update.
        optimizer, baseline = ((torch.optim.SGD(model.parameters(), lr=0.01), torch.optim.SGD(reference.parameters(), lr=0.01))
            if low_precision else (torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-5), torch.optim.AdamW(reference.parameters(), lr=1e-3, eps=1e-5)))
        optimizer.step(), baseline.step()
        state = collect_full_model_state(model, dist.group.WORLD)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(state[name], value.cpu(), atol=2e-5 if low_precision else 1e-6,
                                       rtol=0.001 if low_precision else 1e-4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        load_full_model_state(model, state)
        recovered = collect_full_model_state(model, dist.group.WORLD)
        for name in state:
            torch.testing.assert_close(recovered[name], state[name], atol=0, rtol=0)
        with pytest.raises(KeyError, match="unexpected"):
            load_full_model_state(model, dict(state, not_a_parameter=torch.zeros(1)))
        with pytest.raises(KeyError, match="missing"):
            load_full_model_state(model, {name: value for name, value in state.items() if name != next(iter(state))})
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,kind", [(2, "linear"), (2, "wan"), (4, "wan"), (2, "wan_lora"), (2, "wan_bf16")])
def test_tensor_shards_forward_gradients_updates_and_full_reload(tmp_path, world, kind):
    pytest.importorskip("diffusers")
    mp.spawn(_worker, args=(world, str(tmp_path / "init"), kind), nprocs=world, join=True)
