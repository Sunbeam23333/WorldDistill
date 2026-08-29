"""Strict model-state loading helpers shared by training entry points."""

from __future__ import annotations

import os
from typing import Any

import torch


def load_model_state_file(path: str, *, map_location: Any = "cpu") -> dict[str, torch.Tensor]:
    """Load one complete model state mapping from a checkpoint file.

    Directories are rejected deliberately: the generic trainer cannot infer a
    student architecture from every Diffusers/runner directory. Omit the path
    to clone the teacher, or provide a complete ``.pt/.pth/.safetensors`` state.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model state file not found: {path}")
    if not os.path.isfile(path):
        raise IsADirectoryError(
            f"Model state path must be a file, got directory: {path}. "
            "The generic student loader accepts a complete state dict file; "
            "omit the path to clone the teacher."
        )

    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - installation contract
            raise RuntimeError("Loading .safetensors requires the safetensors package") from exc
        state: Any = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location=map_location, weights_only=True)

    if isinstance(state, dict):
        for container_key in ("student_model", "model", "state_dict"):
            candidate = state.get(container_key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict) or not state:
        raise TypeError(f"Checkpoint does not contain a non-empty model state dict: {path}")
    invalid = [key for key, value in state.items() if not isinstance(key, str) or not torch.is_tensor(value)]
    if invalid:
        raise TypeError(
            f"Checkpoint is not a plain tensor state dict: {path}; invalid keys: {invalid[:3]}"
        )
    return state


__all__ = ["load_model_state_file"]
