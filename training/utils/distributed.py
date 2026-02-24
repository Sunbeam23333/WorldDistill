"""Distributed training utilities.

Supports:
- DDP (DistributedDataParallel) — default for multi-GPU
- FSDP (FullyShardedDataParallel) — for large models that don't fit on a single GPU
- DeepSpeed ZeRO (Stage 1/2/3) — for very large models with optimizer/gradient/param sharding
- Sequence Parallelism helpers — for long-sequence video DiT models

The base_distill_trainer calls wrap_model() which dispatches to the correct
strategy based on TrainerArgs.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger


def setup_distributed():
    """Initialize distributed training environment via torchrun."""
    if "RANK" not in os.environ:
        logger.info("Not in distributed mode, running single GPU.")
        return 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    logger.info(f"Distributed init: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and compute mean across ranks."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ==================== DDP ====================

def wrap_model_ddp(
    model: nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap model with DistributedDataParallel."""
    if not dist.is_initialized():
        return model
    if device_ids is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_ids = [local_rank]
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, find_unused_parameters=find_unused_parameters,
    )


# ==================== FSDP (PyTorch native) ====================

def wrap_model_fsdp(
    model: nn.Module,
    shard_strategy: str = "full",
    cpu_offload: bool = False,
    mixed_precision: str = "bf16",
    auto_wrap_policy: Optional[Any] = None,
) -> nn.Module:
    """Wrap model with FullyShardedDataParallel (PyTorch native FSDP).

    Args:
        model: Model to wrap.
        shard_strategy: 'full' for FULL_SHARD, 'hybrid' for HYBRID_SHARD.
        cpu_offload: Whether to offload parameters to CPU.
        mixed_precision: 'bf16', 'fp16', or 'no'.
        auto_wrap_policy: Custom auto-wrap policy (e.g., transformer_auto_wrap_policy).

    Returns:
        FSDP-wrapped model.
    """
    if not dist.is_initialized():
        logger.warning("FSDP requires distributed init. Returning unwrapped model.")
        return model

    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )
    except ImportError:
        logger.warning("FSDP not available (requires PyTorch >= 2.0). Using DDP fallback.")
        return wrap_model_ddp(model)

    # Sharding strategy
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map.get(shard_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision policy
    mp_policy = None
    if mixed_precision == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif mixed_precision == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # CPU offload
    offload = CPUOffload(offload_params=True) if cpu_offload else None

    # Auto-wrap policy: wrap individual transformer blocks for better sharding
    if auto_wrap_policy is None:
        # Use size-based policy as default: wrap modules > 100M parameters
        auto_wrap_policy = size_based_auto_wrap_policy

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        cpu_offload=offload,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # Required for compatibility with torch.compile and some optimizers
    )
    logger.info(f"FSDP wrapped | strategy={shard_strategy}, offload={cpu_offload}, mp={mixed_precision}")
    return model


# ==================== DeepSpeed ZeRO ====================

def get_deepspeed_config(
    stage: int = 2,
    train_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "bf16",
    learning_rate: float = 1e-5,
    max_grad_norm: float = 1.0,
    cpu_offload: bool = False,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    """Generate a DeepSpeed ZeRO configuration dict.

    This is used by deepspeed.initialize() instead of a JSON config file.

    Args:
        stage: ZeRO stage (0, 1, 2, or 3).
        train_batch_size: Total training batch size across all GPUs.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        mixed_precision: 'bf16', 'fp16', or 'no'.
        learning_rate: Not used directly (optimizer is external), but needed for DS config.
        max_grad_norm: Maximum gradient norm for clipping.
        cpu_offload: Whether to offload optimizer/param to CPU (for ZeRO-3).
        pin_memory: Pin CPU memory for offloading.

    Returns:
        DeepSpeed config dict.
    """
    ds_config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": max_grad_norm,
        "steps_per_print": 100,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
        "zero_allow_untested_optimizer": True,
    }

    # ZeRO-3 specific options
    if stage == 3:
        ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7
        ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e5
        ds_config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 1e9

    # CPU offloading
    if cpu_offload:
        if stage >= 2:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
            }
        if stage >= 3:
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
            }

    # Mixed precision
    if mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif mixed_precision == "fp16":
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        }

    return ds_config


def init_deepspeed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ds_config: Dict[str, Any],
    lr_scheduler: Any = None,
):
    """Initialize DeepSpeed engine.

    Args:
        model: Model to wrap.
        optimizer: Optimizer (DeepSpeed will wrap it with ZeRO).
        ds_config: DeepSpeed config dict from get_deepspeed_config().
        lr_scheduler: Optional LR scheduler.

    Returns:
        Tuple of (model_engine, optimizer, _, lr_scheduler).
        model_engine can be used like a regular model with .forward(), .backward(), .step().
    """
    try:
        import deepspeed
    except ImportError:
        raise ImportError(
            "DeepSpeed not installed. Install with: pip install deepspeed\n"
            "Or disable DeepSpeed by setting --parallel_mode ddp"
        )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        lr_scheduler=lr_scheduler,
    )
    logger.info(
        f"DeepSpeed initialized | ZeRO stage={ds_config['zero_optimization']['stage']}, "
        f"bf16={ds_config.get('bf16', {}).get('enabled', False)}, "
        f"fp16={ds_config.get('fp16', {}).get('enabled', False)}"
    )
    return model_engine, optimizer, lr_scheduler


# ==================== Sequence Parallelism ====================

def init_sequence_parallel(sp_size: int):
    """Initialize sequence parallel groups.

    For video DiT models, sequence parallelism splits the temporal/spatial
    sequence across GPUs within a group. Each sp_group handles a portion
    of the sequence, with all-gather for full-sequence attention.

    Args:
        sp_size: Number of GPUs per sequence parallel group.

    Returns:
        sp_group: ProcessGroup for sequence parallel communication.
    """
    if not dist.is_initialized():
        logger.warning("Sequence parallel requires distributed init.")
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if sp_size <= 1:
        return None

    if world_size % sp_size != 0:
        logger.warning(
            f"world_size={world_size} not divisible by sp_size={sp_size}. "
            "Falling back to no sequence parallelism."
        )
        return None

    # Create process groups
    num_groups = world_size // sp_size
    sp_group = None
    for i in range(num_groups):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            sp_group = group

    logger.info(f"Sequence parallel initialized | sp_size={sp_size}, num_groups={num_groups}")
    return sp_group


def scatter_sequence(tensor: torch.Tensor, sp_group, dim: int = 2):
    """Scatter a sequence tensor along the specified dimension across sp_group.

    For video latents (B, C, T, H, W), typically scatter along dim=2 (temporal).

    Args:
        tensor: Input tensor to scatter.
        sp_group: Sequence parallel process group.
        dim: Dimension to scatter along.

    Returns:
        Local chunk of the tensor for this rank.
    """
    if sp_group is None:
        return tensor

    sp_size = dist.get_world_size(group=sp_group)
    sp_rank = dist.get_rank(group=sp_group)

    seq_len = tensor.shape[dim]
    assert seq_len % sp_size == 0, (
        f"Sequence length {seq_len} must be divisible by sp_size {sp_size}"
    )

    chunk_size = seq_len // sp_size
    chunks = torch.chunk(tensor, sp_size, dim=dim)
    return chunks[sp_rank].contiguous()


def gather_sequence(tensor: torch.Tensor, sp_group, dim: int = 2):
    """Gather scattered sequence chunks back into full sequence.

    Inverse of scatter_sequence. Uses all_gather.

    Args:
        tensor: Local sequence chunk.
        sp_group: Sequence parallel process group.
        dim: Dimension that was scattered.

    Returns:
        Full gathered tensor.
    """
    if sp_group is None:
        return tensor

    sp_size = dist.get_world_size(group=sp_group)
    gather_list = [torch.zeros_like(tensor) for _ in range(sp_size)]
    dist.all_gather(gather_list, tensor, group=sp_group)
    return torch.cat(gather_list, dim=dim)


# ==================== Unified Model Wrapping ====================

def wrap_model(
    model: nn.Module,
    parallel_mode: str = "ddp",
    **kwargs,
) -> nn.Module:
    """Unified model wrapping function.

    Dispatches to the correct parallelism strategy based on parallel_mode.

    Args:
        model: Model to wrap.
        parallel_mode: One of 'ddp', 'fsdp', 'deepspeed', or 'none'.
        **kwargs: Strategy-specific arguments passed to the underlying wrapper.

    Returns:
        Wrapped model.
    """
    if parallel_mode == "ddp":
        return wrap_model_ddp(
            model,
            device_ids=kwargs.get("device_ids"),
            find_unused_parameters=kwargs.get("find_unused_parameters", False),
        )
    elif parallel_mode == "fsdp":
        return wrap_model_fsdp(
            model,
            shard_strategy=kwargs.get("shard_strategy", "full"),
            cpu_offload=kwargs.get("cpu_offload", False),
            mixed_precision=kwargs.get("mixed_precision", "bf16"),
            auto_wrap_policy=kwargs.get("auto_wrap_policy"),
        )
    elif parallel_mode == "none":
        return model
    else:
        logger.warning(f"Unknown parallel_mode '{parallel_mode}', using DDP.")
        return wrap_model_ddp(model)
