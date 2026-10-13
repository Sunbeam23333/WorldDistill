"""Normalize model return types at the trainer/runtime boundary."""

from __future__ import annotations

from typing import Any

import torch


def extract_prediction_tensor(output: Any, *, tag: str = "model") -> torch.Tensor:
    """Extract a prediction tensor from Torch and Diffusers-style outputs."""

    if torch.is_tensor(output):
        return output
    audio = output.get("audio_sample") if isinstance(output, dict) else getattr(output, "audio_sample", None)
    if audio is not None:
        raise TypeError(f"{tag} returned audiovisual predictions; use joint_av_distill rather than discarding audio")
    if isinstance(output, (tuple, list)):
        if not output:
            raise TypeError(f"{tag} returned an empty tuple/list")
        return extract_prediction_tensor(output[0], tag=tag)

    sample = output.get("sample") if isinstance(output, dict) else getattr(output, "sample", None)
    if torch.is_tensor(sample):
        return sample

    raise TypeError(
        f"{tag} returned unsupported output type {type(output).__name__}; expected a Tensor, "
        "a tuple/list whose first item is a Tensor, or a Diffusers-style '.sample' Tensor."
    )


__all__ = ["extract_prediction_tensor"]
