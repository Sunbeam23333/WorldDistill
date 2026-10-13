"""Explicit time-axis contracts for native control tensors.

Pixel-rate action windows and Animate's reference prefix cannot be selected
with a latent-frame gather. Reject that composition before a denoiser runs.
"""
import torch


def slice_native_temporal_conditions(batch, frame_indices, total_frames):
    result = dict(batch)
    identity = torch.equal(frame_indices.detach().cpu().long(), torch.arange(total_frames))
    if not identity:
        for name in ("keyboard_actions", "mouse_actions", "pose_latents", "face_pixels"):
            if batch.get(name) is not None:
                raise ValueError(f"Chunked temporal selection of {name} requires a model-specific pixel-window/"
                                 "reference-prefix policy; use full-clip training for these controls")
        if batch.get("camera_intrinsics") is not None and batch.get("camera_poses") is not None:
            raise ValueError("Chunked LingBot camera training requires precomputed camera_features: compute relative "
                             "poses and translation normalization on the full clip before selecting frames")
    for name, axis in (("vace_context", 2), ("camera_features", 2), ("camera_intrinsics", 1),
                       ("worldplay_actions", 1), ("camera_view_matrices", 1), ("camera_intrinsics_normalized", 1)):
        value = batch.get(name)
        if value is None:
            continue
        if not torch.is_tensor(value) or value.ndim <= axis or value.shape[axis] != total_frames:
            raise ValueError(f"{name} must have {total_frames} latent-aligned frames on axis {axis}")
        result[name] = value.index_select(axis, frame_indices.to(value.device, dtype=torch.long))
    if batch.get("camera_intrinsics") is not None and batch.get("camera_poses") is not None:
        poses=batch["camera_poses"]
        if not torch.is_tensor(poses) or poses.ndim !=4 or poses.shape[1:]!=(total_frames,4,4):
            raise ValueError("Paired LingBot camera_poses must have explicit B,T,4,4 layout")
        result["camera_poses"]=poses.index_select(1,frame_indices.to(poses.device,dtype=torch.long))
    return result
