#!/usr/bin/env python3
"""Report the launcher world size declared by a LightX2V JSON config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def declared_world_size(config: dict[str, Any]) -> int:
    parallel = config.get("parallel")
    if not parallel:
        return 1
    if not isinstance(parallel, dict):
        raise ValueError("'parallel' must be an object when enabled")

    tensor_parallel = int(parallel.get("tensor_p_size", 1))
    cfg_parallel = int(parallel.get("cfg_p_size", 1))
    sequence_parallel = int(parallel.get("seq_p_size", 1))
    for name, value in (
        ("tensor_p_size", tensor_parallel),
        ("cfg_p_size", cfg_parallel),
        ("seq_p_size", sequence_parallel),
    ):
        if value < 1:
            raise ValueError(f"parallel.{name} must be a positive integer")

    if tensor_parallel > 1:
        if cfg_parallel > 1 or sequence_parallel > 1:
            raise ValueError(
                "tensor_p_size cannot be combined with cfg_p_size or seq_p_size "
                "in the current LightX2V device-mesh implementation"
            )
        return tensor_parallel
    return cfg_parallel * sequence_parallel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_json", type=Path)
    args = parser.parse_args()
    with args.config_json.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    print(declared_world_size(config))


if __name__ == "__main__":
    main()
