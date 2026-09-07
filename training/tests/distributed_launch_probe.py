"""Tiny CPU worker used only by launch-contract subprocess tests."""
import json
import os

import torch
import torch.distributed as dist

from training.distributed_config import rank_environment, topology_rank_groups
from training.utils.distributed import cleanup_distributed, hybrid_process_groups, setup_distributed, wrap_model_ddp


def main():
    torch.set_num_threads(1)
    rank, world = setup_distributed()
    try:
        assert setup_distributed() == (rank, world)  # Idempotent only for matching env/backend.
        value = torch.tensor(float(rank + 1))
        dist.all_reduce(value)
        assert value.item() == world * (world + 1) / 2
        torch.manual_seed(123 + rank)
        model = wrap_model_ddp(torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            model(torch.tensor([[1., float(rank)]])).square().mean().backward()
            optimizer.step()
        for parameter in model.parameters():
            expected = parameter.detach().clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        info = rank_environment()
        workers = [None] * world
        dist.all_gather_object(workers, info.as_dict())
        shards, replicas = topology_rank_groups(workers)
        if os.environ.get("PROBE_HYBRID_GROUPS") == "1":
            for group, memberships in zip(hybrid_process_groups(), (shards, replicas)):
                ranks = next(members for members in memberships if rank in members)
                grouped = torch.tensor(float(rank + 1))
                dist.all_reduce(grouped, group=group)
                assert grouped.item() == sum(member + 1 for member in ranks)
        print("LAUNCH_PROBE=" + json.dumps({"rank": rank, "world_size": world,
                                             "local_world_size": info.local_world_size,
                                             "backend": dist.get_backend(), "shards": shards,
                                             "replicas": replicas}), flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
