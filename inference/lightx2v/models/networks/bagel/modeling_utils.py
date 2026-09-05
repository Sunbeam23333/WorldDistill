"""Small model-building utilities used by the vendored BAGEL adapter.

The position table is generated from the standard separable two-dimensional
sine/cosine encoding formula.  Keeping the implementation local avoids
shipping the earlier non-commercially licensed utility module.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _axis_sincos(positions: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Encode one coordinate axis as ``[sin, cos]`` frequency features."""

    if embedding_dim % 2:
        raise ValueError("Axis embedding dimension must be even")
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half_dim, dtype=torch.float64)
        / half_dim
    )
    angles = positions.reshape(-1, 1).to(torch.float64) * frequencies.reshape(1, -1)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def _build_2d_sincos_table(side: int, hidden_size: int) -> torch.Tensor:
    """Return a row-major ``[side * side, hidden_size]`` position table."""

    if side <= 0:
        raise ValueError("max_num_patch_per_side must be positive")
    if hidden_size <= 0 or hidden_size % 4:
        raise ValueError("hidden_size must be positive and divisible by four")

    coordinates = torch.arange(side, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    axis_dim = hidden_size // 2
    table = torch.cat(
        (
            _axis_sincos(grid_x.reshape(-1), axis_dim),
            _axis_sincos(grid_y.reshape(-1), axis_dim),
        ),
        dim=1,
    )
    return table.to(torch.float32)


class PositionEmbedding(nn.Module):
    """Frozen 2-D sine/cosine lookup table indexed by flattened patch IDs."""

    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        self.max_num_patch_per_side = int(max_num_patch_per_side)
        self.hidden_size = int(hidden_size)
        table = _build_2d_sincos_table(
            self.max_num_patch_per_side,
            self.hidden_size,
        )
        # Preserve the original adapter's state-dict and indexing contract.
        self.pos_embed = nn.Parameter(table, requires_grad=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]
