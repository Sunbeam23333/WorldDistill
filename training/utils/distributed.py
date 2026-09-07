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
from datetime import timedelta
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from training.distributed_config import positive_timeout, rank_environment, topology_rank_groups


_WORKER_TOPOLOGY = None
_PROCESS_GROUP_TIMEOUT = None


def setup_distributed(backend=None, timeout_seconds=None):
    """Initialize fixed-size torchrun workers, preserving scheduler visibility.

    The existing no-argument/tuple-return API is retained. Configuration may
    be passed explicitly or through WORLD_DISTILL_DIST_BACKEND/TIMEOUT.
    Gloo is the CPU validation path; NCCL requires visible CUDA devices.
    """
    global _WORKER_TOPOLOGY, _PROCESS_GROUP_TIMEOUT
    info = rank_environment()
    backend = backend or os.environ.get("WORLD_DISTILL_DIST_BACKEND", "auto")
    if backend not in {"auto", "gloo", "nccl"}:
        raise ValueError("Distributed backend must be auto, gloo or nccl")
    timeout = timedelta(seconds=positive_timeout(
        timeout_seconds if timeout_seconds is not None else os.environ.get("WORLD_DISTILL_DIST_TIMEOUT", "600")))
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and (not torch.version.cuda or getattr(torch.version, "hip", None)
                              or not dist.is_nccl_available() or not torch.cuda.is_available()):
        raise RuntimeError("NCCL requires an NVIDIA CUDA PyTorch build and visible NVIDIA GPUs; ROCm/RCCL is unsupported")
    if backend == "gloo" and not dist.is_gloo_available():
        raise RuntimeError("Gloo is unavailable in this PyTorch build")
    if backend == "gloo" and torch.cuda.is_available():
        raise ValueError("Gloo is the CPU validation path; explicitly set CUDA_VISIBLE_DEVICES='' to run on CPU")
    if info is None:
        if dist.is_initialized():
            raise ValueError("Process group exists without its worker RANK environment")
        logger.info("Not in distributed mode, running a single process.")
        return 0, 1
    if backend == "nccl":
        if info.local_world_size > torch.cuda.device_count():
            raise ValueError("Local worker count exceeds scheduler-visible CUDA devices")
        torch.cuda.set_device(info.local_rank)
    if dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size(), dist.get_backend()) != (info.rank, info.world_size, backend):
            raise ValueError("Initialized process group conflicts with the requested backend/rank environment")
        return info.rank, info.world_size
    dist.init_process_group(backend=backend, init_method="env://", rank=info.rank,
                            world_size=info.world_size, timeout=timeout)
    try:
        workers = [None] * info.world_size
        dist.all_gather_object(workers, info.as_dict())
        topology_rank_groups(workers)
        _WORKER_TOPOLOGY = workers
        _PROCESS_GROUP_TIMEOUT = timeout
    except Exception:
        dist.destroy_process_group()
        raise
    logger.info(f"Distributed init: backend={backend}, rank={info.rank}, world_size={info.world_size}, "
                f"local_rank={info.local_rank}, local_world_size={info.local_world_size}, timeout={timeout}")
    return info.rank, info.world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    global _WORKER_TOPOLOGY, _PROCESS_GROUP_TIMEOUT
    if dist.is_initialized():
        dist.destroy_process_group()
    _WORKER_TOPOLOGY = None
    _PROCESS_GROUP_TIMEOUT = None


def hybrid_process_groups():
    """Explicit intra-node sharding and inter-node same-local-rank replication."""
    if not dist.is_initialized():
        raise RuntimeError("Hybrid FSDP requires an initialized process group")
    workers = _WORKER_TOPOLOGY
    if workers is None:
        info = rank_environment()
        if info is None or (info.rank, info.world_size) != (dist.get_rank(), dist.get_world_size()):
            raise ValueError("Hybrid FSDP requires a matching torchrun rank environment")
        workers = [None] * dist.get_world_size()
        dist.all_gather_object(workers, info.as_dict())
    shards, replicas = topology_rank_groups(workers)
    if len(shards) < 2 or len(shards[0]) < 2:
        raise ValueError("Hybrid FSDP requires >=2 nodes and >=2 workers per node; use full sharding otherwise")
    timeout = _PROCESS_GROUP_TIMEOUT or timedelta(seconds=positive_timeout(os.environ.get("WORLD_DISTILL_DIST_TIMEOUT", "600")))
    shard_group = replica_group = None
    # Every rank creates every group in the same order, including nonmembers.
    for ranks in shards:
        group = dist.new_group(ranks=ranks, timeout=timeout)
        if dist.get_rank() in ranks:
            shard_group = group
    for ranks in replicas:
        group = dist.new_group(ranks=ranks, timeout=timeout)
        if dist.get_rank() in ranks:
            replica_group = group
    return shard_group, replica_group


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
    if device_ids is None and next(model.parameters()).is_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_ids = [local_rank]
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, find_unused_parameters=find_unused_parameters,
    )


# ==================== FSDP (PyTorch native) ====================

def _collective_leaf_modules(model):
    return [module for module in model.modules()
            if getattr(module, "requires_worlddistill_collective_leaf", False)]


def _collective_leaf_wrap_policy(model, policy, mixed_precision):
    """Keep dynamic expert subtrees inside their nearest ancestor FSDP unit.

    This is a collective-order correctness constraint, not a memory saving:
    the complete routed expert group is gathered together on every rank.
    """
    leaves = _collective_leaf_modules(model)
    if not leaves:
        return policy
    protected = {descendant for leaf in leaves for descendant in leaf.modules()}
    # FSDP composes its default BatchNorm mixed-precision override with the
    # supplied policy using OR, including an always-recurse branch. Do not
    # allow that implicit policy to split a protected expert subtree.
    ignored_types = tuple(getattr(mixed_precision, "_module_classes_to_ignore", ()))
    if ignored_types and any(isinstance(module, ignored_types) for module in protected):
        raise ValueError(
            "FSDP mixed-precision BatchNorm overrides would split a collective leaf; "
            "use mixed_precision=no or a routed model without those auto-wrapped module types"
        )
    from torch.distributed.fsdp.wrap import _Policy
    if isinstance(policy, _Policy):
        # Object policies preserve per-module kwargs. Filter their targets
        # rather than converting them to a bool policy and losing overrides.
        class CollectiveLeafPolicy(_Policy):
            def _run_policy(self, root_module, ignored_modules, root_kwargs):
                targets = policy._run_policy(root_module, ignored_modules, root_kwargs)
                return {module: kwargs for module, kwargs in targets.items() if module not in protected}
        guarded = CollectiveLeafPolicy()
    else:
        if not callable(policy):
            raise TypeError("FSDP collective leaf protection requires a callable or PyTorch policy object")
        def guarded(module, recurse, nonwrapped_numel):
            # Also veto all descendants when FSDP's internal AMP OR policy
            # recurses despite the leaf's recurse=False decision.
            return module not in protected and policy(
                module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel)
    logger.warning(
        f"FSDP collective leaf protection: {len(leaves)} routed group(s) gathered whole; "
        "peak memory includes every expert in the group, not only the active expert"
    )
    return guarded


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
        raise RuntimeError("FSDP requires distributed initialization; refusing an unsharded fallback")
    if (not torch.version.cuda or getattr(torch.version, "hip", None)
            or not torch.cuda.is_available() or dist.get_backend() != "nccl"):
        raise RuntimeError("FSDP training requires NVIDIA CUDA/NCCL, not ROCm/RCCL; CPU/Gloo only validates launch and rank grouping")

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
    except ImportError as error:
        raise ImportError("FSDP is unavailable; refusing a silent DDP fallback") from error

    # Sharding strategy
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    if shard_strategy not in strategy_map:
        raise ValueError(f"Unknown FSDP sharding strategy: {shard_strategy}")
    if mixed_precision not in {"bf16", "fp16", "no"}:
        raise ValueError("Resolve mixed_precision=auto before FSDP wrapping")
    sharding_strategy = strategy_map[shard_strategy]
    process_group = hybrid_process_groups() if shard_strategy == "hybrid" else None

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
    auto_wrap_policy = _collective_leaf_wrap_policy(model, auto_wrap_policy, mp_policy)

    model = FSDP(
        model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        cpu_offload=offload,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # Required for compatibility with torch.compile and some optimizers
        sync_module_states=True,
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
        cpu_offload: Offload optimizer for ZeRO-1/2/3 and parameters for ZeRO-3.
        pin_memory: Pin CPU memory for offloading.

    Returns:
        DeepSpeed config dict.
    """
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("Resolve auto precision before building the DeepSpeed configuration")
    if type(stage) is not int or stage not in {0, 1, 2, 3}:
        raise ValueError("DeepSpeed ZeRO stage must be 0, 1, 2 or 3")
    if cpu_offload and stage == 0:
        raise ValueError("WorldDistill CPU offload requires DeepSpeed ZeRO stage 1, 2 or 3")
    for name, value in (("train_batch_size", train_batch_size), ("gradient_accumulation_steps", gradient_accumulation_steps)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"DeepSpeed {name} must be a positive integer")
    if train_batch_size % gradient_accumulation_steps:
        raise ValueError("DeepSpeed train_batch_size must be divisible by gradient_accumulation_steps")
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
        if stage >= 1:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
            }
        if stage >= 3:
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
            }

    # Use the documented PyTorch AMP engine route, not native half-weight
    # conversion. The latter disables an enclosing autocast context and does
    # not cast our keyword inputs; blindly casting kwargs also corrupts FP32
    # time embeddings. DeepSpeed alone owns scaling in backward()/step().
    # Its AMP API/version is checked before initialize() below.
    ds_config["fp16"] = {"enabled": False}
    ds_config["bf16"] = {"enabled": False}
    if mixed_precision != "no":
        ds_config["torch_autocast"] = {
            "enabled": True,
            "dtype": "float16" if mixed_precision == "fp16" else "bfloat16",
        }

    return ds_config


def _require_deepspeed_torch_amp(deepspeed, ds_config):
    """Installation/API admission only; this does not qualify CUDA execution.

    The 0.19.6 training API documents nested autocast and ZeRO AMP scaling:
    https://deepspeed.readthedocs.io/en/latest/training.html#mixed-precision-training
    Older DeepSpeed remains usable for explicit FP32 training.
    """
    if ds_config.get("fp16", {}).get("enabled") or ds_config.get("bf16", {}).get("enabled"):
        raise ValueError("WorldDistill requires torch_autocast, not DeepSpeed native fp16/bf16; use get_deepspeed_config()")
    amp = ds_config.get("torch_autocast", {})
    if not amp.get("enabled", False):
        return None
    from importlib import import_module
    from packaging.version import InvalidVersion, Version

    version = str(getattr(deepspeed, "__version__", "unknown"))
    try:
        recent = Version(version) >= Version("0.19.6")
    except InvalidVersion:
        recent = False
    if not recent:
        raise RuntimeError(
            f"WorldDistill DeepSpeed FP16/BF16 torch AMP requires DeepSpeed >=0.19.6; found {version}. "
            "Upgrade DeepSpeed or explicitly select mixed_precision=no; no automatic downgrade is performed."
        )
    engine_type = getattr(deepspeed, "DeepSpeedEngine", None)
    required = ("torch_autocast_enabled", "torch_autocast_dtype", "fp16_enabled", "bfloat16_enabled", "scale")
    missing = [name for name in required if not callable(getattr(engine_type, name, None))]
    try:
        amp_module = import_module("deepspeed.runtime.torch_autocast")
        missing.extend(name for name in ("init_autocast_params", "autocast_if_enabled")
                       if not callable(getattr(amp_module, name, None)))
    except ImportError:
        missing.append("deepspeed.runtime.torch_autocast")
    if missing:
        raise RuntimeError("DeepSpeed build lacks required torch AMP APIs: " + ", ".join(missing))
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp.get("dtype"))
    if dtype is None:
        raise ValueError("WorldDistill DeepSpeed torch AMP dtype must be float16 or bfloat16")
    return dtype


def _validate_deepspeed_amp_engine(engine, dtype, stage):
    if dtype is None:
        return
    if (not engine.torch_autocast_enabled() or engine.torch_autocast_dtype() != dtype
            or engine.fp16_enabled() or engine.bfloat16_enabled()):
        raise RuntimeError("DeepSpeed did not initialize the requested exclusive torch AMP mode")
    if dtype == torch.float16:
        # ZeRO's native loss_scaler is not the torch AMP scaler! For FP32
        # parameters it can legitimately be a no-op even in AMP FP16 mode.
        owner, name = (engine, "torch_autocast_z0_gradscaler") if stage == 0 else (
            engine.optimizer, "torch_autocast_gradscaler")
        scaler = getattr(owner, name, None)
        required = ("is_enabled", "scale", "unscale_", "step", "update", "state_dict", "load_state_dict")
        if any(not callable(getattr(scaler, attr, None)) for attr in required) or not scaler.is_enabled():
            raise RuntimeError(f"DeepSpeed FP16 torch AMP requires an enabled {name}; refusing unscaled training")


def _prepare_deepspeed_cpu_optimizer(optimizer, ds_config, lr_scheduler):
    """Convert only a fresh standard Adam(W); never drop initialized state."""
    offload = ds_config.get("zero_optimization", {}).get("offload_optimizer", {})
    if offload.get("device") != "cpu":
        return optimizer
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
    except (ImportError, OSError) as exc:
        raise RuntimeError("ZeRO CPU offload requires a working DeepSpeedCPUAdam extension") from exc
    if isinstance(optimizer, DeepSpeedCPUAdam):
        return optimizer
    if type(optimizer) not in {torch.optim.Adam, torch.optim.AdamW}:
        raise ValueError("Automatic ZeRO CPU offload conversion supports only standard Adam/AdamW or existing DeepSpeedCPUAdam")
    if optimizer.state:
        raise ValueError("Cannot convert an initialized optimizer to CPUAdam without losing state; initialize ZeRO before restoring its checkpoint")
    adamw_mode = type(optimizer) is torch.optim.AdamW or bool(optimizer.defaults.get("decoupled_weight_decay", False))
    for group in optimizer.param_groups:
        if any(group.get(key, False) for key in ("amsgrad", "maximize", "capturable", "differentiable")):
            raise ValueError("CPUAdam cannot preserve amsgrad/maximize/capturable/differentiable optimizer semantics")
        if bool(group.get("decoupled_weight_decay", adamw_mode)) != adamw_mode:
            raise ValueError("CPUAdam cannot preserve different Adam/AdamW modes across parameter groups")
    # A scheduler factory will be called by DeepSpeed with its optimizer.
    # Stateful scheduler instances (including nested torch schedulers) must
    # refer to this optimizer; rebinding retains epoch, base_lrs and lambdas.
    if lr_scheduler is not None and not callable(lr_scheduler):
        _deepspeed_scheduler_nodes(lr_scheduler, optimizer)
    groups = [{**group, "params": list(group["params"])} for group in optimizer.param_groups]
    try:
        replacement = DeepSpeedCPUAdam(
            groups, lr=optimizer.defaults["lr"], betas=optimizer.defaults["betas"],
            eps=optimizer.defaults["eps"], weight_decay=optimizer.defaults["weight_decay"],
            adamw_mode=adamw_mode, fp32_optimizer_states=True,
        )
    except Exception as exc:
        raise RuntimeError("DeepSpeedCPUAdam could not initialize; install/build its CPU extension before using CPU offload") from exc
    logger.info("ZeRO CPU offload: preserving Adam(W) parameter groups with DeepSpeedCPUAdam")
    return replacement


def _deepspeed_scheduler_nodes(scheduler, expected_optimizer):
    nodes, seen = [], set()
    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if getattr(node, "optimizer", None) is not expected_optimizer:
            raise ValueError("CPU offload scheduler must reference the optimizer passed to init_deepspeed")
        nodes.append(node)
        for child in getattr(node, "_schedulers", ()):
            visit(child)
    visit(scheduler)
    return nodes


def _rebind_deepspeed_scheduler(scheduler, old_optimizer, new_optimizer):
    if scheduler is None or callable(scheduler) or old_optimizer is new_optimizer:
        return []
    nodes = _deepspeed_scheduler_nodes(scheduler, old_optimizer)
    # Mirror PyTorch LRScheduler's step tracking without reinitializing the
    # schedule (which would change the first LR or advance last_epoch).
    from functools import wraps
    from weakref import ref
    optimizer_ref = ref(new_optimizer)
    original_step = new_optimizer.step.__func__
    @wraps(original_step)
    def tracked_step(*args, **kwargs):
        target = optimizer_ref()
        target._opt_called = True
        return original_step(target, *args, **kwargs)
    tracked_step._wrapped_by_lr_sched = True
    new_optimizer.step = tracked_step
    for node in nodes:
        node.optimizer = new_optimizer
    return nodes


def _register_deepspeed_collective_leaves(model, ds_config):
    if ds_config.get("zero_optimization", {}).get("stage") != 3:
        return
    leaves = _collective_leaf_modules(model)
    if not leaves:
        return
    try:
        from deepspeed.utils import set_z3_leaf_modules
    except ImportError as exc:
        raise RuntimeError("Dynamic expert ZeRO-3 requires DeepSpeed set_z3_leaf_modules before initialization") from exc
    if not callable(set_z3_leaf_modules):
        raise RuntimeError("DeepSpeed set_z3_leaf_modules is unavailable; refusing unsafe dynamic expert sharding")
    for leaf in leaves:
        # Scope the class-based official API to the marked instance's subtree.
        # An unmarked sibling with the same Python class is not broadened into
        # a leaf. Descendants of this instance are gathered together anyway.
        registered = set_z3_leaf_modules(leaf, [type(leaf)])
        if leaf not in registered or not getattr(leaf, "_z3_leaf", False):
            raise RuntimeError("DeepSpeed failed to register a required collective leaf")
    logger.warning(
        f"ZeRO-3 collective leaf protection: {len(leaves)} routed group(s) gathered whole; "
        "peak memory includes every expert in the group, not only the active expert"
    )


def init_deepspeed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ds_config: Dict[str, Any],
    lr_scheduler: Any = None,
):
    """Initialize DeepSpeed engine.

    Args:
        model: Model to wrap.
        optimizer: Optimizer (DeepSpeed wraps it with ZeRO). For CPU offload,
            fresh standard Adam(W) is converted to CPUAdam without changing
            parameter groups. Already initialized state is never discarded.
        ds_config: DeepSpeed config dict from get_deepspeed_config().
        lr_scheduler: Optional LR scheduler.

    Returns:
        Tuple of (model_engine, optimizer, lr_scheduler).
        model_engine can be used like a regular model with .forward(), .backward(), .step().
        FP16/BF16 uses DeepSpeed >=0.19.6 torch AMP, not native half-weight
        conversion. Only engine.backward()/step() may scale/unscale losses.
    """
    try:
        import deepspeed
    except ImportError:
        raise ImportError(
            "DeepSpeed not installed. Install with: pip install deepspeed\n"
            "Or disable DeepSpeed by setting --parallel_mode ddp"
        )

    amp_dtype = _require_deepspeed_torch_amp(deepspeed, ds_config)
    _register_deepspeed_collective_leaves(model, ds_config)
    original_optimizer = optimizer
    optimizer = _prepare_deepspeed_cpu_optimizer(optimizer, ds_config, lr_scheduler)
    rebound = _rebind_deepspeed_scheduler(lr_scheduler, original_optimizer, optimizer)
    try:
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
            lr_scheduler=lr_scheduler,
        )
        _validate_deepspeed_amp_engine(model_engine, amp_dtype, ds_config["zero_optimization"]["stage"])
    except Exception:
        for scheduler in rebound:
            scheduler.optimizer = original_optimizer
        raise
    logger.info(
        f"DeepSpeed initialized | ZeRO stage={ds_config['zero_optimization']['stage']}, "
        f"precision={'FP32' if amp_dtype is None else f'torch AMP {amp_dtype}'}, "
        "loss_scaling_owner=DeepSpeed engine (installation checks, not GPU qualification)"
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
