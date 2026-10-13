"""Stage-local ownership, multi-forward autograd, TP×PP, and state reload."""
import copy
from datetime import timedelta
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from training.parallel.pipeline_parallel import apply_pipeline_parallel_plan, apply_wan_pipeline_parallel
from training.parallel.tensor_parallel import apply_wan_tensor_parallel
from training.parallel.state import collect_full_model_state, collect_full_named_tensors, load_full_model_state
from training.tests.test_tensor_parallel_plan import tiny_wan, add_lora


def _worker(rank, world, path, kind, device_type="cpu"):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", init_method=f"file://{path}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=100))
    try:
        tp_group, pp_group = None, dist.group.WORLD
        if kind.startswith("tp_pp"):
            for ranks in ([0, 1], [2, 3]):
                group = dist.new_group(ranks)
                if rank in ranks:
                    tp_group = group
            for ranks in ([0, 2], [1, 3]):
                group = dist.new_group(ranks)
                if rank in ranks:
                    pp_group = group
        torch.manual_seed(121)
        if kind == "linear":
            reference = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
            reference.to(device)
            model = apply_pipeline_parallel_plan(copy.deepcopy(reference), pp_group, {"0": 0, "2": 1})
            x = torch.randn(2, 3, 8, device=device, requires_grad=True)
            baseline_x = x.detach().clone().requires_grad_()
            expected = reference(reference(baseline_x) + baseline_x)
            actual = model(model(x) + x)
        else:
            reference = tiny_wan()
            if "lora" in kind:
                reference = add_lora(reference)
            if "checkpoint" in kind:
                reference.enable_gradient_checkpointing()
            reference.to(device)
            model = copy.deepcopy(reference)
            if tp_group is not None:
                apply_wan_tensor_parallel(model, tp_group)
            apply_wan_pipeline_parallel(model, pp_group)
            x = torch.randn(1, 2, 2, 4, 4, device=device, requires_grad=True)
            baseline_x = x.detach().clone().requires_grad_()
            inputs = dict(timestep=torch.tensor([33.], device=device), encoder_hidden_states=torch.randn(1, 3, 8, device=device), return_dict=False)
            expected = reference(hidden_states=reference(hidden_states=baseline_x, **inputs)[0] + baseline_x, **inputs)[0]
            actual = model(hidden_states=model(hidden_states=x, **inputs)[0] + x, **inputs)[0]
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(x.grad, baseline_x.grad, atol=3e-6, rtol=1e-4)
        grads = collect_full_named_tensors(model, {n: p.grad for n, p in model.named_parameters()}, dist.group.WORLD)
        for name, parameter in reference.named_parameters():
            assert (name in grads) == (parameter.grad is not None), name
            if parameter.grad is not None:
                torch.testing.assert_close(grads[name], parameter.grad.cpu(), atol=3e-6, rtol=5e-4,
                                           msg=lambda msg: f"{name}: {msg}")
        assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in reference.parameters())
        optimizer, baseline = torch.optim.SGD(model.parameters(), lr=0.01), torch.optim.SGD(reference.parameters(), lr=0.01)
        optimizer.step(), baseline.step()
        state = collect_full_model_state(model, dist.group.WORLD)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(state[name], value.cpu(), atol=2e-6, rtol=2e-4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        load_full_model_state(model, state)
        recovered = collect_full_model_state(model, dist.group.WORLD)
        for name in state:
            torch.testing.assert_close(state[name], recovered[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,kind", [(2, "linear"), (2, "wan"), (4, "tp_pp"),
    (2, "wan_checkpoint"), (4, "tp_pp_lora_checkpoint")])
def test_pipeline_owned_parameters_chained_rollout_and_combined_tp(tmp_path, world, kind):
    pytest.importorskip("diffusers")
    mp.spawn(_worker, args=(world, str(tmp_path / "init"), kind), nprocs=world, join=True)
