"""Resolve a common, executable training precision before loading large models."""
from __future__ import annotations

import torch
import torch.distributed as dist
from loguru import logger


def training_precisions(args):
    """Hardware support cannot supply a missing optimizer/scaler algorithm."""
    allowed = {"no", "fp16", "bf16"}
    if args.parallel_mode == "fsdp" or args.distill_method in {"adversarial_distill", "dmd_distill"}:
        allowed.discard("fp16")
    return allowed


def resolve_training_precision(args, device):
    from cuda_compat import common_supported_precisions, probe_cuda_precision, select_mixed_precision, validate_tf32_request

    device = torch.device(device)
    requested = args.mixed_precision
    if device.type != "cuda":
        # Existing CPU numerical tests may explicitly request a CUDA autocast
        # setting (disabled by PyTorch on CPU); auto never advertises that as
        # successful GPU mixed precision.
        if requested == "auto":
            args.mixed_precision = "no"
        return {"requested": requested, "selected": args.mixed_precision, "scope": "CPU only", "gpu_qualified": False}
    controls = {name: getattr(args, name) for name in (
        "mixed_precision", "parallel_mode", "distill_method", "batch_size", "gradient_accumulation_steps",
        "sp_size", "seed", "enable_tf32", "float32_matmul_precision", "deepspeed_stage", "fsdp_shard_strategy",
    )}
    try:
        probe = probe_cuda_precision(torch, device=device)
        local = {"controls": controls, "probe": probe, "error": None}
    except Exception as exc:
        local = {"controls": controls, "probe": None, "error": repr(exc)}
    peers = [local]
    if dist.is_initialized():
        peers = [None] * dist.get_world_size()
        dist.all_gather_object(peers, local)
    if any(peer["controls"] != controls for peer in peers):
        raise ValueError("Ranks disagree on training/precision settings; use one identical training configuration")
    errors = [f"rank {rank}: {peer['error']}" for rank, peer in enumerate(peers) if peer["error"]]
    if errors:
        raise RuntimeError("GPU precision preflight failed: " + "; ".join(errors))
    reports = [peer["probe"] for peer in peers]
    common = set(common_supported_precisions(reports)) & training_precisions(args)
    selected = select_mixed_precision(requested, common)
    validate_tf32_request(bool(args.enable_tf32), reports)
    args.mixed_precision = selected
    result = {"requested": requested, "selected": selected, "common_training_precisions": sorted(common),
              "rank_probes": reports, "scope": "small local precision probes; not pretrained/GPU qualification"}
    # Probe kernels are memoized by cuda_compat, but the collective agreement
    # must run every time: a rank-local cache hit must never skip a collective
    # another rank enters after a setting changes.
    args._precision_resolution = result
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"Training precision: {requested} -> {selected}; all-rank executable intersection={sorted(common)}")
    return result
