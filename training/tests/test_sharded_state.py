"""Shard-local EMA contracts and cross-rank FP16 overflow consensus."""

import copy
from datetime import timedelta
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from training.utils.sharded_state import ShardedEMA, build_training_grad_scaler


class LocalShard(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.empty = nn.Parameter(torch.empty(0))
        self.frozen = nn.Parameter(torch.tensor(99.0), requires_grad=False)


def test_sharded_ema_preserves_fp32_empty_shards_and_resume(tmp_path):
    model = LocalShard()
    ema = ShardedEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(2)
    ema.update(model)
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor([2.0, 3.0]))
    assert ema.shadow["empty"].numel() == 0
    assert "frozen" not in ema.shadow
    assert ema.step_count == 1
    path = ema.save(tmp_path)
    assert path.name == "ema_rank_00000.pt"
    restored = ShardedEMA(model, decay=0.9)
    restored.load(tmp_path)
    assert restored.decay == 0.5
    assert restored.step_count == 1
    restored.copy_to(model)
    torch.testing.assert_close(model.weight, torch.tensor([2.0, 3.0]))
    assert model.frozen == 99


def test_sharded_ema_decay_warmup_and_state_owns_its_tensors():
    model = LocalShard()
    ema = ShardedEMA(model, decay=0.9, warmup_steps=9)
    with torch.no_grad():
        model.weight.add_(10)
    ema.update(model)
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor([10.0, 11.0]))
    snapshot = ema.state_dict()
    snapshot["shadow"]["weight"].zero_()
    assert torch.count_nonzero(ema.shadow["weight"]) == 2


@pytest.mark.parametrize("mutation", ["rank", "world_size", "schema", "parallel_mode", "shape", "missing", "step_count", "decay"])
def test_sharded_ema_rejects_incompatible_state_atomically(mutation):
    ema = ShardedEMA(LocalShard(), decay=0.5)
    before = copy.deepcopy(ema.state_dict())
    state = copy.deepcopy(before)
    if mutation == "shape":
        state["shadow"]["weight"] = torch.ones(3)
    elif mutation == "missing":
        del state["shadow"]["weight"]
    elif mutation == "step_count":
        state[mutation] = -1
    elif mutation == "decay":
        state[mutation] = float("nan")
    elif mutation in {"rank", "world_size"}:
        state[mutation] += 1
    else:
        state[mutation] = "invalid"
    with pytest.raises(ValueError):
        ema.load_state_dict(state)
    assert ema.decay == before["decay"]
    assert ema.step_count == before["step_count"]
    torch.testing.assert_close(ema.shadow["weight"], before["shadow"]["weight"])


def test_sharded_ema_rejects_unsharded_views_during_update():
    model = LocalShard()
    ema = ShardedEMA(model)
    model.weight.data = torch.ones(4)
    with pytest.raises(ValueError, match="shape changed"):
        ema.update(model)
    assert ema.step_count == 0


def test_zero3_ema_reads_public_master_shards_not_empty_parameter_data(monkeypatch):
    utilities = ModuleType("deepspeed.utils")
    root = ModuleType("deepspeed")
    model = LocalShard()
    master = {id(model.weight): torch.tensor([4.0, 6.0]), id(model.empty): torch.empty(0)}
    model.weight.data = torch.empty(0)
    utilities.safe_get_local_fp32_param = lambda parameter: master[id(parameter)]
    utilities.safe_set_local_fp32_param = lambda parameter, value: master[id(parameter)].copy_(value)
    monkeypatch.setitem(sys.modules, "deepspeed", root)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utilities)
    ema = ShardedEMA(model, parallel_mode="deepspeed", decay=0.5)
    master[id(model.weight)].add_(2)
    ema.update(model)
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor([5.0, 7.0]))
    ema.copy_to(model)
    torch.testing.assert_close(master[id(model.weight)], torch.tensor([5.0, 7.0]))
    assert model.weight.numel() == 0


def test_scaler_factory_requires_fsdp_group_but_leaves_deepspeed_in_charge():
    with pytest.raises(RuntimeError, match="initialized process group"):
        build_training_grad_scaler("fsdp", "fp16", "cpu")
    assert not build_training_grad_scaler("deepspeed", "fp16", "cpu").is_enabled()
    assert not build_training_grad_scaler("fsdp", "bf16", "cpu").is_enabled()
    assert build_training_grad_scaler("ddp", "fp16", "cpu").is_enabled()


def _overflow_worker(rank, init_path, output_directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{init_path}", timeout=timedelta(seconds=30))
    try:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = build_training_grad_scaler("fsdp", "fp16", "cpu")
        scaler.scale(parameter.square()).backward()
        if rank == 1:
            parameter.grad.fill_(float("inf"))
        old_scale = scaler.get_scale()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        assert parameter.item() == 1.0
        assert scaler.get_scale() < old_scale
        # After a collective skip, all ranks must still execute a finite update.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(parameter.square()).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        torch.testing.assert_close(parameter, torch.tensor(0.8))
        torch.save({"value": parameter.detach(), "scale": scaler.get_scale()}, Path(output_directory) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo required")
def test_sharded_scaler_one_rank_overflow_skips_every_rank(tmp_path):
    init_path = tmp_path / "rendezvous"
    context = mp.spawn(_overflow_worker, args=(str(init_path), str(tmp_path)), nprocs=2, join=False)
    try:
        # Bounded wait: propagate child failures and do not leave failed workers.
        import time
        deadline = time.monotonic() + 50
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Sharded scaler Gloo test did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    states = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    assert states[0]["scale"] == states[1]["scale"]
    torch.testing.assert_close(states[0]["value"], states[1]["value"])
