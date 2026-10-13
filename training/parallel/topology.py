"""Deterministic DP x PP x TP coordinates and independent communication groups."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import torch.distributed as dist


def parallel_coordinates(rank: int, world_size: int, tp_size: int = 1, pp_size: int = 1):
    for name, value in (("world_size", world_size), ("tp_size", tp_size), ("pp_size", pp_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must identify a worker in world_size")
    model_size = tp_size * pp_size
    if world_size % model_size:
        raise ValueError("world_size must be divisible by tp_size * pp_size")
    return rank // model_size, (rank // tp_size) % pp_size, rank % tp_size


def parallel_rank_groups(world_size: int, tp_size: int = 1, pp_size: int = 1):
    """List groups in creation order; each axis partitions all global ranks."""
    parallel_coordinates(0, world_size, tp_size, pp_size)
    data_size = world_size // (pp_size * tp_size)

    def rank(d, p, t):
        return (d * pp_size + p) * tp_size + t

    return {
        "tp": [tuple(rank(d, p, t) for t in range(tp_size))
               for d, p in product(range(data_size), range(pp_size))],
        "pp": [tuple(rank(d, p, t) for p in range(pp_size))
               for d, t in product(range(data_size), range(tp_size))],
        "dp": [tuple(rank(d, p, t) for d in range(data_size))
               for p, t in product(range(pp_size), range(tp_size))],
        "mp": [tuple(rank(d, p, t) for p, t in product(range(pp_size), range(tp_size)))
               for d in range(data_size)],
    }


@dataclass(frozen=True)
class ParallelTopology:
    rank: int
    world_size: int
    dp_rank: int
    pp_rank: int
    tp_rank: int
    dp_size: int
    pp_size: int
    tp_size: int
    dp_group: Any
    pp_group: Any
    tp_group: Any
    mp_group: Any
    checkpoint_group: Any
    dp_ranks: tuple[int, ...]
    pp_ranks: tuple[int, ...]
    tp_ranks: tuple[int, ...]
    mp_ranks: tuple[int, ...]

    @property
    def mp_size(self):
        return self.pp_size * self.tp_size


def create_parallel_topology(tp_size: int = 1, pp_size: int = 1):
    """Collectively create groups, including size-one groups (never WORLD aliases).

    All ranks must call this in identical order before model plans are applied.
    The returned DP group contains only workers with matching TP/PP partitions.
    """
    if not dist.is_initialized():
        if tp_size != 1 or pp_size != 1:
            raise ValueError("Tensor/pipeline parallelism requires initialized distributed workers")
        return None
    rank, world = dist.get_rank(), dist.get_world_size()
    coordinates = parallel_coordinates(rank, world, tp_size, pp_size)
    selected, members = {}, {}
    for axis, groups in parallel_rank_groups(world, tp_size, pp_size).items():
        for ranks in groups:
            group = dist.new_group(ranks=list(ranks))
            if rank in ranks:
                selected[axis], members[axis] = group, ranks
    # Full CPU checkpoints must not pickle/gather a complete model through
    # NCCL's CUDA staging buffers. Create host groups collectively up front.
    if not dist.is_gloo_available():
        raise RuntimeError("TP/PP portable checkpoints require the Gloo CPU backend")
    checkpoint_group = None
    for ranks in parallel_rank_groups(world, tp_size, pp_size)["mp"]:
        group = dist.new_group(ranks=list(ranks), backend="gloo")
        if rank in ranks:
            checkpoint_group = group
    return ParallelTopology(
        rank=rank, world_size=world, dp_rank=coordinates[0], pp_rank=coordinates[1], tp_rank=coordinates[2],
        dp_size=world // (tp_size * pp_size), pp_size=pp_size, tp_size=tp_size,
        checkpoint_group=checkpoint_group,
        **{f"{axis}_group": group for axis, group in selected.items()},
        **{f"{axis}_ranks": ranks for axis, ranks in members.items()},
    )
