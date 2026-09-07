"""WorldDistill Training Entry Point.

Usage:
    # Single GPU
    python -m training.train_distill \
        --distill_method step_distill \
        --teacher_model_path ./models/wan2.2-a14b \
        --model_cls wan2.2_moe \
        --distill_preset configs/distill_presets/step_distill_4step.json \
        --data_json data/train.json \
        --output_dir results/distill_step4

    # Multi-GPU with DDP (default)
    torchrun --nproc_per_node=8 -m training.train_distill \
        --distill_method step_distill \
        --parallel_mode ddp \
        --teacher_model_path ./models/wan2.2-a14b \
        --data_json data/train.json

    # Multi-GPU with post-load FSDP sharding (construction must fit first)
    torchrun --nproc_per_node=8 -m training.train_distill \
        --distill_method step_distill \
        --parallel_mode fsdp \
        --fsdp_shard_strategy full \
        --teacher_model_path ./models/wan2.2-a14b \
        --data_json data/train.json

    # Multi-GPU with DeepSpeed ZeRO-2
    torchrun --nproc_per_node=8 -m training.train_distill \
        --distill_method step_distill \
        --parallel_mode deepspeed \
        --deepspeed_stage 2 \
        --teacher_model_path ./models/wan2.2-a14b \
        --data_json data/train.json

Model loading contract:
1. Compatible Diffusers directory with a transformer/unet (supported path)
2. Architecture-specific directory construction when its config is recognized
3. Bare state dicts and inference Runner objects require a dedicated adapter

Sequence parallelism is intentionally not shown as a generic CLI recipe:
``--sp_size > 1`` requires teacher and student implementations with explicit
WorldDistill model-layer sequence-parallel adapters.
"""

import copy
import json
import os
import random
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from loguru import logger

if __package__ in {None, ""}:  # pragma: no cover - script entry fallback
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    validate_runtime_dependency_versions = import_module("training.env_compat").validate_runtime_dependency_versions
    parse_training_args = import_module("training.trainer_args").parse_training_args
    build_trainer = import_module("training.trainers").build_trainer
    runtime_module = import_module("training.runtime")
    build_distill_cache = runtime_module.build_distill_cache
    build_runtime = runtime_module.build_runtime
    DiffusersRawBatchEncoder = import_module("training.utils.batch_encoder").DiffusersRawBatchEncoder
    build_optimizer = import_module("training.utils.optimizers").build_optimizer
    build_lr_scheduler = import_module("training.utils.schedulers").build_lr_scheduler
    distributed_module = import_module("training.utils.distributed")
    setup_distributed = distributed_module.setup_distributed
    cleanup_distributed = distributed_module.cleanup_distributed
    is_main_process = distributed_module.is_main_process
    video_dataset_module = import_module("training.data.video_dataset")
    CachedLatentDataset = video_dataset_module.CachedLatentDataset
    VideoDataset = video_dataset_module.VideoDataset
    BucketSampler = import_module("training.data.bucket_sampler").BucketSampler
else:
    from .env_compat import validate_runtime_dependency_versions
    from .trainer_args import parse_training_args
    from .trainers import build_trainer
    from .runtime import build_distill_cache, build_runtime
    from .utils.batch_encoder import DiffusersRawBatchEncoder
    from .utils.optimizers import build_optimizer
    from .utils.schedulers import build_lr_scheduler
    from .utils.distributed import setup_distributed, cleanup_distributed, is_main_process
    from .data.video_dataset import CachedLatentDataset, VideoDataset
    from .data.bucket_sampler import BucketSampler


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def _configure_cuda_backend(args) -> None:
    """Configure low-level CUDA math knobs used by the training runtime."""
    if not torch.cuda.is_available():
        return

    enable_tf32 = bool(getattr(args, "enable_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32

    precision = str(getattr(args, "float32_matmul_precision", "high") or "high").lower()
    if hasattr(torch, "set_float32_matmul_precision") and precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(precision)

    if is_main_process():
        logger.info(
            f"CUDA backend | TF32={'on' if enable_tf32 else 'off'} | "
            f"float32 matmul precision={precision}"
        )



def _compile_model_if_requested(model: torch.nn.Module, args, role: str):
    """Optionally compile teacher/student models for DDP training."""
    if not getattr(args, "enable_torch_compile", False):
        return model

    scope = getattr(args, "torch_compile_scope", "student")
    if scope not in {role, "both"}:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        if is_main_process():
            logger.warning("当前 PyTorch 不支持 torch.compile，跳过编译优化。")
        return model

    parallel_mode = getattr(args, "parallel_mode", "ddp")
    if parallel_mode != "ddp":
        if is_main_process():
            logger.warning(
                f"当前 parallel_mode={parallel_mode}，为避免与封装器冲突，暂时只在 DDP 路径启用 torch.compile。"
            )
        return model

    fullgraph = bool(getattr(args, "torch_compile_fullgraph", False))
    if role == "student" and getattr(args, "gradient_checkpointing", False) and fullgraph:
        if is_main_process():
            logger.warning("student 同时开启 gradient checkpointing 与 fullgraph 编译较不稳定，自动回退为 fullgraph=False。")
        fullgraph = False

    compile_kwargs = {
        "backend": getattr(args, "torch_compile_backend", "inductor"),
        "mode": getattr(args, "torch_compile_mode", "reduce-overhead"),
        "fullgraph": fullgraph,
        "dynamic": bool(getattr(args, "torch_compile_dynamic", False)),
    }

    try:
        compiled_model = compile_fn(model, **compile_kwargs)
        if is_main_process():
            logger.info(f"torch.compile 已启用 | role={role} | kwargs={compile_kwargs}")
        return compiled_model
    except Exception as exc:
        if is_main_process():
            logger.warning(f"torch.compile 在 {role} 上启用失败，自动回退到 eager: {exc}")
        return model



def _resolve_data_mode(data_json: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    with open(data_json, "r") as f:
        manifest = json.load(f)

    if not manifest:
        raise ValueError(f"Dataset manifest is empty: {data_json}")

    sample = manifest[0]
    if "latent_path" in sample:
        return "cached"
    if "path" in sample or "video_path" in sample:
        return "raw"

    raise ValueError(
        f"Unable to infer data mode from manifest {data_json}. "
        "Please set --data_mode explicitly to `cached` or `raw`."
    )


def _build_dataset(
    data_json: str,
    data_mode: str,
    cache_dir: str,
    video_dir: str,
    resolution: str,
    num_frames: int,
):
    if data_mode == "cached":
        return CachedLatentDataset(data_json=data_json, cache_dir=cache_dir)
    if data_mode == "raw":
        return VideoDataset(
            data_json=data_json,
            video_dir=video_dir,
            resolution=resolution,
            num_frames=num_frames,
        )
    raise ValueError(f"Unsupported data_mode: {data_mode}")


def _bucket_sampler_drop_last(world_size: int) -> bool:
    """Distributed collectives require equal local batch shapes."""

    return world_size > 1


def _resolve_native_dual_args(args, student_model) -> None:
    from training.model_adapter import NoiseRoutedDenoiser

    if not isinstance(student_model, NoiseRoutedDenoiser):
        return
    if getattr(args, "student_low_model", None):
        raise ValueError("student_low_model cannot override a native dual-expert checkpoint; supply a complete matching student pipeline instead")
    if args.use_dual_model:
        logger.info("Checkpoint already supplies both noise experts; using its native boundary for teacher and student.")
        args.use_dual_model = False


def main():
    args = parse_training_args()
    validate_runtime_dependency_versions(strict=getattr(args, "strict_env_check", True),
                                         required_transformers_version=args.required_transformers_version)
    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Model/adaptor initialization must be identical before DDP/FSDP/ZeRO
    # synchronization. Per-rank stochastic training seeds are installed only
    # after every model and EMA target has been constructed.
    set_seed(args.seed)
    _configure_cuda_backend(args)

    if is_main_process():
        logger.info(f"WorldDistill Training | Method: {args.distill_method} | Model: {args.model_cls}")
        logger.info(
            f"Model metadata | architecture={args.model_architecture or 'unknown'} | "
            f"family={args.model_family or 'unknown'} | "
            f"checkpoint_format={args.checkpoint_format or 'unknown'} | "
            f"runner={args.resolved_runner_cls or args.model_cls}"
        )
        logger.info(f"Distributed: world_size={world_size}, device={device}")
        logger.info(f"Parallel mode: {args.parallel_mode} | SP size: {args.sp_size}")
        logger.info(f"Requested data mode: {args.data_mode}")
        if args.parallel_mode == "deepspeed":
            logger.info(f"DeepSpeed ZeRO stage: {args.deepspeed_stage}")
        elif args.parallel_mode == "fsdp":
            logger.info(f"FSDP shard strategy: {args.fsdp_shard_strategy}")
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Build Dataset & DataLoader ---
    train_data_mode = _resolve_data_mode(args.data_json, args.data_mode)
    dataset = _build_dataset(
        data_json=args.data_json,
        data_mode=train_data_mode,
        cache_dir=args.cache_dir,
        video_dir=args.video_dir,
        resolution=args.resolution,
        num_frames=args.num_frames,
    )

    if args.use_bucket_sampler:
        sampler = BucketSampler(
            dataset, batch_size=args.batch_size, shuffle=True,
            drop_last=_bucket_sampler_drop_last(world_size),
            seed=args.seed, rank=rank, world_size=world_size,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True,
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

    # --- Build Validation DataLoader (optional) ---
    val_dataloader = None
    val_data_mode = None
    if args.val_data_json:
        val_data_mode = _resolve_data_mode(args.val_data_json, args.data_mode)
        val_dataset = _build_dataset(
            data_json=args.val_data_json,
            data_mode=val_data_mode,
            cache_dir=args.val_cache_dir if args.val_cache_dir else args.cache_dir,
            video_dir=args.video_dir,
            resolution=args.resolution,
            num_frames=args.num_frames,
        )
        val_sampler = (
            torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            if world_size > 1
            else None
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    batch_encoder = None
    if train_data_mode == "raw" or val_data_mode == "raw":
        encoder_dtype = torch.float32
        if args.mixed_precision == "bf16":
            encoder_dtype = torch.bfloat16
        elif args.mixed_precision == "fp16":
            encoder_dtype = torch.float16
        batch_encoder = DiffusersRawBatchEncoder(
            model_path=args.teacher_model_path,
            device=device,
            dtype=encoder_dtype,
        )

    # --- Build Models ---
    teacher_dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.mixed_precision]
    teacher_model, student_model = _load_models(
        args.teacher_model_path,
        args.student_model_path,
        args.model_cls,
        args.config_json,
        device,
        teacher_dtype,
        student_device=torch.device("cpu") if world_size > 1 and args.parallel_mode in {"fsdp", "deepspeed"} else device,
    )
    _resolve_native_dual_args(args, student_model)

    # Apply LoRA if requested
    if args.use_lora:
        student_model = _apply_lora(student_model, args)

    # Gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        if hasattr(student_model, "gradient_checkpointing_enable"):
            student_model.gradient_checkpointing_enable()
        elif hasattr(student_model, "enable_gradient_checkpointing"):
            student_model.enable_gradient_checkpointing()

    teacher_model = _compile_model_if_requested(teacher_model, args, role="teacher")
    student_model = _compile_model_if_requested(student_model, args, role="student")

    # --- Build Optimizer & Scheduler ---
    # Note: For FSDP, optimizer will be re-created after wrapping in base_distill_trainer.
    # For DeepSpeed, optimizer will be wrapped by DS engine.
    optimizer = build_optimizer(
        student_model, optimizer_type=args.optimizer, lr=args.learning_rate,
        weight_decay=args.weight_decay, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer, scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps, total_steps=args.max_train_steps,
        min_lr_ratio=args.lr_min_ratio,
    )

    distill_cache = build_distill_cache(args)
    runtime = build_runtime(
        args=args,
        teacher_model=teacher_model,
        student_model=student_model,
        device=device,
        distill_cache=distill_cache,
    )

    # --- Build Trainer ---
    trainer = build_trainer(
        method=args.distill_method,
        args=args,
        teacher_model=teacher_model,
        student_model=student_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        device=device,
        batch_encoder=batch_encoder,
        runtime=runtime,
        distill_cache=distill_cache,
    )

    # --- Train ---
    set_seed(args.seed + rank)
    trainer.train()
    cleanup_distributed()


def _candidate_model_dirs(model_path: str, model_cls: str, config_json: str = "") -> list[str]:
    if not model_path:
        return []
    if os.path.isfile(model_path):
        return [os.path.dirname(model_path)]

    preferred_subdirs = [
        "",
        "transformer",
        "original",
        "distill_models",
        os.path.join("distill_models", "transformer"),
        "low_noise_model",
        "high_noise_model",
        os.path.join("distill_models", "low_noise_model"),
        os.path.join("distill_models", "high_noise_model"),
    ]
    if model_cls in {"hunyuan_video_1.5", "hunyuan_video_1.5_distill", "worldplay_distill", "worldplay_ar", "worldplay_bi"}:
        preferred_subdirs = ["transformer", "", *preferred_subdirs[2:]]

    config_overrides: dict[str, str] = {}
    if config_json and os.path.exists(config_json):
        try:
            with open(config_json, "r") as f:
                raw_config = json.load(f)
            if isinstance(raw_config, dict):
                config_overrides = {str(k): str(v) for k, v in raw_config.items() if isinstance(v, (str, os.PathLike))}
        except Exception as exc:
            logger.warning(f"Failed to read config_json {config_json}: {exc}")

    candidates: list[str] = []

    def _append_candidate(candidate: str) -> None:
        normalized = os.path.normpath(candidate)
        if os.path.isdir(normalized) and normalized not in candidates:
            candidates.append(normalized)

    for relative_dir in preferred_subdirs:
        candidate = os.path.join(model_path, relative_dir) if relative_dir else model_path
        _append_candidate(candidate)

    transformer_model_name = config_overrides.get("transformer_model_name", "")
    if transformer_model_name:
        _append_candidate(os.path.join(model_path, "transformer", transformer_model_name))
        _append_candidate(os.path.join(model_path, "distill_models", transformer_model_name))

    transformer_model_path = config_overrides.get("transformer_model_path", "")
    if transformer_model_path:
        if not os.path.isabs(transformer_model_path):
            transformer_model_path = os.path.join(model_path, transformer_model_path)
        _append_candidate(transformer_model_path)

    for checkpoint_key in ("dit_original_ckpt", "high_noise_original_ckpt", "low_noise_original_ckpt"):
        checkpoint_path = config_overrides.get(checkpoint_key, "")
        if not checkpoint_path:
            continue
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(model_path, checkpoint_path)
        _append_candidate(os.path.dirname(checkpoint_path))

    for nested_parent in ("transformer", "distill_models"):
        parent_dir = os.path.join(model_path, nested_parent)
        if not os.path.isdir(parent_dir):
            continue
        for child_name in sorted(os.listdir(parent_dir)):
            _append_candidate(os.path.join(parent_dir, child_name))

    return candidates


def _find_weight_files(model_dir: str) -> list[str]:
    import glob

    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")
    weight_files: list[str] = []
    for pattern in patterns:
        weight_files.extend(sorted(glob.glob(os.path.join(model_dir, pattern))))
        weight_files.extend(sorted(glob.glob(os.path.join(model_dir, "**", pattern), recursive=True)))

    deduped: list[str] = []
    for weight_file in weight_files:
        normalized = os.path.normpath(weight_file)
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _load_models(
    teacher_path: str,
    student_path: str,
    model_cls: str,
    config_json: str,
    device: torch.device,
    teacher_dtype: torch.dtype,
    student_device: torch.device | None = None,
):
    """Load teacher and student models.

    Strategy 1: diffusers DiffusionPipeline (recommended for HuggingFace models)
    Strategy 2: safetensors/state_dict with model architecture inference
    Strategy 3: Runner mechanism from inference engine (for custom models)

    Args:
        teacher_path: Path to teacher model weights.
        student_path: Path to student model weights (if empty, copies teacher).
        model_cls: Model architecture class name (e.g., 'wan2.2_moe').
        config_json: Optional config JSON path.
        device: Target device.

    Returns:
        Tuple of (teacher_model, student_model).
    """
    teacher_model = None

    if not teacher_path or not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher model path '{teacher_path}' does not exist. "
            "Please provide a valid --teacher_model_path."
        )

    candidate_dirs = _candidate_model_dirs(teacher_path, model_cls, config_json)

    if os.path.isfile(teacher_path):
        # An explicit file must not be replaced by a different, conventionally
        # named checkpoint in the same directory during from_pretrained().
        from training.model_adapter import load_diffusers_denoiser
        from training.utils.checkpoint_io import load_model_state_file

        parent = Path(teacher_path).parent
        if not (parent / "config.json").is_file():
            raise ValueError("An explicit teacher weight file requires its exact Diffusers config.json alongside it")
        teacher_model = load_diffusers_denoiser(parent, dtype=teacher_dtype, weights=False)
        state = load_model_state_file(teacher_path, map_location="cpu")
        if set(state) == set(teacher_model.state_dict()):
            teacher_model.load_state_dict(state, strict=True)
        else:
            teacher_model.model.load_state_dict(state, strict=True)
        candidate_dirs = []

    # Load the actual configured denoisers without also loading text/VAE weights.
    # This preserves BOTH Wan2.2 experts and avoids the old Transformer2D guess.
    from training.model_adapter import load_diffusers_training_model
    for candidate_dir in candidate_dirs:
        has_diffusers = any(os.path.exists(os.path.join(candidate_dir, name))
                            for name in ("model_index.json", "worlddistill_export.json"))
        if not has_diffusers:
            continue
        # A declared dual pipeline failing to load must not fall through to a
        # single-expert subdirectory and silently lose its low-noise teacher.
        teacher_model = load_diffusers_training_model(candidate_dir, dtype=teacher_dtype)
        break

    # --- Strategy 2: Direct state_dict loading ---
    if teacher_model is None:
        if os.path.isdir(teacher_path):
            for candidate_dir in candidate_dirs:
                weight_files = _find_weight_files(candidate_dir)
                if not weight_files:
                    continue
                logger.info(f"Found {len(weight_files)} weight files in {candidate_dir}")
                teacher_model = _construct_model_from_weights(
                    candidate_dir,
                    weight_files,
                    model_cls,
                    torch.device("cpu"),
                    teacher_dtype,
                )
                if teacher_model is not None:
                    break
        else:
            logger.info(f"Loading single weight file: {teacher_path}")
            teacher_model = _construct_model_from_weights(
                os.path.dirname(teacher_path),
                [teacher_path],
                model_cls,
                torch.device("cpu"),
                teacher_dtype,
            )

    if teacher_model is None:
        raise RuntimeError(
            f"Failed to load teacher model from '{teacher_path}'. "
            "Supported formats:\n"
            "  1. diffusers directory (with model_index.json or config.json)\n"
            "  2. Exact Diffusers architecture config plus its weights\n"
            "Inference-only tensor runners are not differentiable training adapters. "
            "Convert the original checkpoint with the model author's converter first."
        )

    # --- Build student model ---
    if student_path:
        from training.utils.checkpoint_io import load_model_state_file

        logger.info(f"Loading separate student model from {student_path}")
        if os.path.isdir(student_path):
            student_model = load_diffusers_training_model(student_path, dtype=torch.float32)
            if type(student_model) is not type(teacher_model):
                raise ValueError("Teacher/student expert topology differs; supply a matching student architecture")
        else:
            student_model = copy.deepcopy(teacher_model)
            student_state = load_model_state_file(student_path, map_location="cpu")
            student_model.load_state_dict(student_state, strict=True)
    else:
        logger.info("No student_model_path supplied; cloning teacher as the student model.")
        student_model = copy.deepcopy(teacher_model)

    # Native AdamW/FSDP/ZeRO expect FP32 master parameters. Autocast or the
    # distributed mixed-precision policy controls compute dtype; cloning the
    # reduced-precision teacher directly would otherwise update BF16/FP16
    # parameters in place.
    # FSDP/ZeRO consume CPU-staged parameters, avoiding a full FP32 student on
    # each GPU before wrapping/sharding. Teacher weights remain replicated.
    student_model = student_model.to(device=student_device or device, dtype=torch.float32)
    teacher_model = teacher_model.to(device)

    # Ensure student requires grad, teacher does not
    for p in teacher_model.parameters():
        p.requires_grad = False
    for p in student_model.parameters():
        p.requires_grad = True

    logger.info(
        f"Models loaded | Teacher params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.1f}M | "
        f"Student trainable: {sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6:.1f}M"
    )
    return teacher_model, student_model


def _construct_model_from_weights(
    model_dir: str,
    weight_files: list[str],
    model_cls: str,
    device: torch.device,
    teacher_dtype: torch.dtype,
):
    """Construct a model from weight files.

    Tries multiple strategies:
    1. diffusers model from config.json + weights
    2. Direct safetensors loading with auto model class detection
    """
    # Try loading from diffusers config
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)

            if config.get("_class_name"):
                from training.model_adapter import load_diffusers_training_model
                return load_diffusers_training_model(model_dir, dtype=teacher_dtype).to(device)
        except Exception as e:
            raise RuntimeError(f"Exact architecture loading failed from {model_dir}: {e}") from e

    # Try loading safetensors directly
    if any(f.endswith(".safetensors") for f in weight_files):
        try:
            from safetensors.torch import load_file
            state_dict = {}
            for wf in weight_files:
                if wf.endswith(".safetensors"):
                    state_dict.update(load_file(wf, device=str(device)))
            logger.info(f"Loaded safetensors state_dict: {len(state_dict)} keys")

            # Try to infer and construct model
            # For known architectures, attempt diffusers auto-detection
            try:
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    model_dir, torch_dtype=teacher_dtype, use_safetensors=True
                )
                if hasattr(pipe, "transformer") and pipe.transformer is not None:
                    return pipe.transformer.to(device)
                elif hasattr(pipe, "unet") and pipe.unet is not None:
                    return pipe.unet.to(device)
            except Exception:
                pass

            logger.warning(
                f"Loaded {len(state_dict)} keys from safetensors but cannot auto-construct model. "
                f"Sample keys: {list(state_dict.keys())[:5]}"
            )
        except ImportError:
            logger.warning("safetensors not installed. Install with: pip install safetensors")
        except Exception as e:
            logger.warning(f"safetensors loading failed: {e}")

    # Try loading .pt files
    pt_files = [f for f in weight_files if f.endswith(".pt") or f.endswith(".bin")]
    if pt_files:
        try:
            state_dict = torch.load(pt_files[0], map_location=device, weights_only=True)
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            logger.info(f"Loaded state_dict from {pt_files[0]}: {len(state_dict)} keys")
            logger.warning(
                "Cannot auto-construct model from bare state_dict. "
                "Please use diffusers format or specify --config_json."
            )
        except Exception as e:
            logger.warning(f"PT loading failed: {e}")

    return None


def _apply_lora(model, args):
    """Apply LoRA to the student model for parameter-efficient training."""
    try:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA applied | Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({trainable / total * 100:.2f}%)")
        return model
    except ImportError as exc:
        raise RuntimeError(
            "--use_lora requires PEFT. Install the training dependencies with "
            "`pip install -e '.[train]'` before starting LoRA training."
        ) from exc


if __name__ == "__main__":
    main()
