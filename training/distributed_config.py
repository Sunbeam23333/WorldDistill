"""Fixed-size torchrun environment contract (stdlib-only, no model imports)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import socket
from typing import Mapping


def positive_timeout(value: str | float) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Distributed timeout must be finite and positive")
    return seconds


def _integer(env: Mapping[str, str], name: str, default: int | None = None, minimum: int = 0) -> int:
    raw = env.get(name)
    if raw is None:
        if default is None:
            raise ValueError(f"Incomplete torchrun environment: {name} is required")
        return default
    if not raw.isdecimal() or int(raw) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {raw!r}")
    return int(raw)


@dataclass(frozen=True)
class RankEnvironment:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    node_id: str
    hostname: str

    def as_dict(self) -> dict:
        return asdict(self)


def rank_environment(env: Mapping[str, str] | None = None) -> RankEnvironment | None:
    env = os.environ if env is None else env
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not any(key in env for key in (*required, "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE")):
        if (_integer(env, "WORLD_DISTILL_EXPECTED_WORLD_SIZE", 1, 1) != 1
                or _integer(env, "WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE", 1, 1) != 1):
            raise ValueError("Expected distributed workers, but no torchrun RANK environment is present")
        return None
    rank = _integer(env, "RANK")
    world = _integer(env, "WORLD_SIZE", minimum=1)
    local_rank = _integer(env, "LOCAL_RANK")
    local_world = _integer(env, "LOCAL_WORLD_SIZE", 1 if world == 1 else None, minimum=1)
    if rank >= world or local_rank >= local_world or world % local_world:
        raise ValueError("Inconsistent RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE")
    for key, actual in (("WORLD_DISTILL_EXPECTED_WORLD_SIZE", world),
                        ("WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE", local_world),
                        ("GROUP_WORLD_SIZE", world // local_world)):
        if key in env and _integer(env, key, minimum=1) != actual:
            raise ValueError(f"{key} disagrees with the fixed-size worker topology; elastic world-size changes are unsupported")
    if env.get("WORLD_DISTILL_FIXED_WORLD_SIZE", "1") != "1":
        raise ValueError("Elastic world-size changes are unsupported")
    if _integer(env, "TORCHELASTIC_RESTART_COUNT", 0) and "WORLD_DISTILL_EXPECTED_WORLD_SIZE" not in env:
        raise ValueError("Restarted torchrun requires WORLD_DISTILL_EXPECTED_WORLD_SIZE to reject world-size changes")
    if "GROUP_RANK" in env:
        node_rank = _integer(env, "GROUP_RANK")
        if node_rank >= world // local_world:
            raise ValueError("GROUP_RANK exceeds the number of homogeneous worker nodes")
        node_id = f"agent:{node_rank}"
    else:
        # Direct env:// callers may omit GROUP_RANK. Host grouping is explicit,
        # and collective validation below still checks a complete local group.
        node_id = f"host:{socket.gethostname()}"
    if not env.get("MASTER_ADDR"):
        raise ValueError("MASTER_ADDR is required for torchrun env:// initialization")
    port = _integer(env, "MASTER_PORT", minimum=1)
    if port > 65535:
        raise ValueError("MASTER_PORT must be <= 65535")
    return RankEnvironment(rank, world, local_rank, local_world, node_id, socket.gethostname())


def topology_rank_groups(workers: list[dict]) -> tuple[list[list[int]], list[list[int]]]:
    """Validate homogeneous agent-local workers; return shard/replica ranks.

    Use the current agent/node identity and local rank, not a rank numbering
    assumption that may change after a fixed-size torchrun restart.
    """
    world = len(workers)
    if not workers or {row["rank"] for row in workers} != set(range(world)):
        raise ValueError("Worker topology must contain every global rank exactly once")
    sizes = {row["local_world_size"] for row in workers}
    if len(sizes) != 1 or any(row["world_size"] != world for row in workers):
        raise ValueError("Heterogeneous local worker counts/world sizes are unsupported")
    local_size = sizes.pop()
    if local_size < 1 or world % local_size:
        raise ValueError("World size must be divisible by the local worker count")
    nodes: dict[str, list[dict]] = {}
    for row in workers:
        nodes.setdefault(row["node_id"], []).append(row)
    ordered = [nodes[key] for key in sorted(nodes)]
    for node in ordered:
        if (len(node) != local_size or {row["local_rank"] for row in node} != set(range(local_size))
                or len({row["hostname"] for row in node}) != 1):
            raise ValueError("Each node agent must own exactly one complete, host-local worker group")
    shards = [[row["rank"] for row in sorted(node, key=lambda row: row["local_rank"])] for node in ordered]
    replicas = [[node[local] for node in shards] for local in range(local_size)]
    return shards, replicas
