# Wan self-attention operations adapted from Hugging Face Diffusers:
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see repository LICENSE).
# WorldDistill modifications: local Ulysses backward compatibility, replicated
# input gradients, independent role meshes, and synchronized batch/RNG handling.
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
"""Training semantics around Diffusers' real Ulysses attention collectives.

The transformer splits tokens before its blocks, exchanges sequence/head axes
inside attention, and gathers predictions only after the final projection.
WORLD-averaged student gradients need an SP-size multiplier at backward; input
gradients need a CP sum when a differentiable rollout feeds another model call.
"""
from __future__ import annotations

import inspect
import random

import numpy as np
import torch
import torch.distributed as dist


def _native_sdpa_backward(ctx, grad_out, *args):
    """B,S,H,D derivative contract, including Diffusers 0.36 compatibility.

    0.36 transposed grad_out after already transposing the recomputed output
    back to B,S,H,D and recomputed without enable_grad. Keep the correction
    local to this processor instead of replacing global Diffusers functions.
    """
    with torch.enable_grad():
        q, k, v = [value.detach().transpose(1, 2).requires_grad_(True)
                   for value in ctx.saved_tensors]
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=ctx.attn_mask, dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal, scale=ctx.scale, enable_gqa=ctx.enable_gqa)
        gradients = torch.autograd.grad(output, (q, k, v), grad_out.transpose(1, 2))
    return tuple(gradient.transpose(1, 2) for gradient in gradients)


class _CompatibleUlysses(torch.autograd.Function):
    """Upstream collectives with 0.36's missing non-tensor gradient slot fixed."""
    @staticmethod
    def forward(ctx, *arguments):
        from diffusers.models.attention_dispatch import TemplatedUlyssesAttention
        return TemplatedUlyssesAttention.forward(ctx, *arguments)

    @staticmethod
    def backward(ctx, gradient, *unused):
        from diffusers.models.attention_dispatch import TemplatedUlyssesAttention
        result = TemplatedUlyssesAttention.backward(ctx, gradient, *unused)
        return (*result, *((None,) * (len(ctx.needs_input_grad) - len(result))))


class _WanUlyssesProcessor:
    """Wan self-attention with upstream Ulysses and a stable SDPA derivative.

    The QKV/norm/rotary/projection operations mirror WanAttnProcessor. Text
    and image cross-attention retain their original Diffusers processor.
    """
    _attention_backend = "native"

    def __init__(self, parallel_config):
        self._parallel_config = parallel_config

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        if encoder_hidden_states is not None or attn.add_k_proj is not None:
            raise ValueError("The Ulysses processor is installed only on Wan self-attention")
        from diffusers.models.transformers.transformer_wan import _get_qkv_projections
        from diffusers.models.attention_dispatch import _native_attention_forward_op
        query, key, value = _get_qkv_projections(attn, hidden_states, None)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            cos, sin = (component[..., 0::2] for component in rotary_emb)

            def rotate(tensor):
                first, second = tensor.unflatten(-1, (-1, 2)).unbind(-1)
                return torch.stack((first * cos - second * sin, first * sin + second * cos), -1).flatten(-2).to(tensor.dtype)

            query, key = rotate(query), rotate(key)
        result = _CompatibleUlysses.apply(
            query, key, value, attention_mask, 0.0, False, None, False, False,
            _native_attention_forward_op, _native_sdpa_backward, self._parallel_config)
        result = result.flatten(2, 3).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](result))


class _ReplicatedInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient):
        # Each rank's graph contains only its output-token contribution. A
        # preceding model call needs the complete replicated input derivative.
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


def replicated_input(value, group):
    if torch.is_tensor(value):
        return _ReplicatedInput.apply(value, group) if value.requires_grad else value
    if isinstance(value, dict):
        return {key: replicated_input(item, group) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(replicated_input(item, group) for item in value)
    if isinstance(value, list):
        return [replicated_input(item, group) for item in value]
    return value


def enable_wan_context_parallel(model, size: int, device: torch.device):
    """Install audited Wan token splits and upstream Ulysses forward/backward.

    Diffusers 0.36 has no public custom-mesh argument; its same config/setup,
    attention processors and hook installer are used directly for DP x CP.
    On newer CUDA releases the public enable_parallelism entry point is used.
    CPU uses the same attention/autograd implementation for Gloo regression.
    """
    if type(model).__name__ != "WanTransformer3DModel":
        raise ValueError("The built-in context-parallel training plan supports WanTransformer3DModel")
    if not dist.is_initialized() or size <= 1 or dist.get_world_size() % size:
        raise ValueError("Context parallelism requires initialized distributed ranks and world_size divisible by sp_size > 1")
    heads = int(model.config.num_attention_heads)
    if heads % size:
        raise ValueError(f"Wan Ulysses requires num_attention_heads={heads} divisible by sp_size={size}")
    if getattr(model, "_parallel_config", None) is not None:
        raise ValueError("A model with existing parallelism cannot be reconfigured by WorldDistill")
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("Wan context-parallel training requires zero dropout for replica-consistent RNG")
    from diffusers.models._modeling_parallel import (
        ContextParallelConfig, ContextParallelInput, ContextParallelOutput, ParallelConfig,
    )
    from diffusers.hooks.context_parallel import apply_context_parallel
    from torch.distributed.device_mesh import init_device_mesh

    # Every role gets independent process groups. All world ranks must invoke
    # this initializer in the same teacher/student/target order.
    mesh = init_device_mesh(device.type, (dist.get_world_size() // size, 1, size),
                            mesh_dim_names=("data", "ring", "ulysses"))["ring", "ulysses"]
    config = ContextParallelConfig(ring_degree=1, ulysses_degree=size)
    plan = {
        "rope": {0: ContextParallelInput(1, 4, True), 1: ContextParallelInput(1, 4, True)},
        "blocks.0": {"hidden_states": ContextParallelInput(1, 3)},
        "proj_out": ContextParallelOutput(1, 3),
        "": {"timestep": ContextParallelInput(1, 2)},
    }
    # Native SDPA selects CUDA flash/efficient/math implementations supported
    # by the installed torch build, and has upstream Ulysses backward support.
    model.set_attention_backend("native")
    if "mesh" in inspect.signature(ContextParallelConfig).parameters and device.type == "cuda":
        config.mesh = mesh
        model.enable_parallelism(config=config, cp_plan=plan)
    else:
        parallel = ParallelConfig(context_parallel_config=config)
        parallel.setup(dist.get_rank(), dist.get_world_size(), device, mesh=mesh)
        model._parallel_config = parallel
        for module in model.modules():
            processor = getattr(module, "processor", None)
            if processor is not None and hasattr(processor, "_parallel_config"):
                processor._parallel_config = parallel
        apply_context_parallel(model, config, plan)
    # Keep text/image KV replicated. This also fixes 0.36's cross-attention
    # processor, which otherwise applies CP to both Q and the unsharded KV.
    for block in model.blocks:
        block.attn2.processor._parallel_config = None
        block.attn1.set_processor(_WanUlyssesProcessor(model._parallel_config))
    return config._flattened_mesh.get_group()


def synchronize_context_batch(batch, group, device):
    """Broadcast one full encoded sample/condition tree within each CP group.

    Only shape/dtype and ordinary metadata use object collectives. Tensor data
    uses native device collectives (no full CUDA tensor pickle or CPU copy).
    """
    leader = dist.get_global_rank(group, 0)
    source = dist.get_rank() == leader
    tensors = []

    def describe(value):
        if torch.is_tensor(value):
            if value.requires_grad:
                raise ValueError("Context batch broadcast expects detached encoded inputs")
            tensors.append(value)
            return ("tensor", tuple(value.shape), value.dtype)
        if isinstance(value, dict):
            return ("dict", [(key, describe(item)) for key, item in value.items()])
        if isinstance(value, tuple):
            return ("tuple", [describe(item) for item in value])
        if isinstance(value, list):
            return ("list", [describe(item) for item in value])
        return ("value", value)

    metadata = [describe(batch) if source else None]
    dist.broadcast_object_list(metadata, src=leader, group=group, device=device)
    tensor_index = 0

    def materialize(description):
        nonlocal tensor_index
        kind = description[0]
        if kind == "tensor":
            value = tensors[tensor_index].to(device).contiguous() if source else torch.empty(
                description[1], dtype=description[2], device=device)
            tensor_index += 1
            dist.broadcast(value, src=leader, group=group)
            return value
        if kind == "dict":
            return {key: materialize(item) for key, item in description[1]}
        if kind == "tuple":
            return tuple(materialize(item) for item in description[1])
        if kind == "list":
            return [materialize(item) for item in description[1]]
        return description[1]

    return materialize(metadata[0])


def synchronize_context_rng(group, device):
    """Same noise, timesteps and branching per CP group; independent DP groups."""
    leader = dist.get_global_rank(group, 0)
    state = torch.get_rng_state().to(device)
    dist.broadcast(state, src=leader, group=group)
    torch.set_rng_state(state.cpu())
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device).to(device)
        dist.broadcast(cuda_state, src=leader, group=group)
        torch.cuda.set_rng_state(cuda_state.cpu(), device)
    objects = [(random.getstate(), np.random.get_state()) if dist.get_rank() == leader else None]
    dist.broadcast_object_list(objects, src=leader, group=group, device=device)
    random.setstate(objects[0][0])
    np.random.set_state(objects[0][1])
