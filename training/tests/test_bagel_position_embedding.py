from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "inference/lightx2v/models/networks/bagel/modeling_utils.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "worlddistill_bagel_modeling_utils",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_position_embedding_preserves_row_major_sincos_contract() -> None:
    module = _load_module()
    embedding = module.PositionEmbedding(2, 4)

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [math.sin(1.0), math.cos(1.0), 0.0, 1.0],
            [0.0, 1.0, math.sin(1.0), math.cos(1.0)],
            [math.sin(1.0), math.cos(1.0), math.sin(1.0), math.cos(1.0)],
        ],
        dtype=torch.float32,
    )

    assert embedding.pos_embed.requires_grad is False
    torch.testing.assert_close(embedding(torch.arange(4)), expected)


@pytest.mark.parametrize("side,hidden", [(0, 4), (2, 6), (2, 0)])
def test_position_embedding_rejects_invalid_dimensions(side: int, hidden: int) -> None:
    module = _load_module()
    with pytest.raises(ValueError):
        module.PositionEmbedding(side, hidden)
