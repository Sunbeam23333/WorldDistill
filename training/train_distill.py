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

    # Multi-GPU with FSDP (for large models)
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

    # With Sequence Parallelism (sp_size=2 means 2 GPUs per SP group)
    torchrun --nproc_per_node=8 -m training.train_distill \
        --distill_method stream_distill \
        --parallel_mode ddp \
        --sp_size 2 \
        --teacher_model_path ./models/wan2.2-a14b \
        --data_json data/train.json

Model loading strategies (in priority order):
1. diffusers DiffusionPipeline.from_pretrained (directory with safetensors/bin)
2. Direct state_dict (.pt / .safetensors file) with auto model construction
3. Runner mechanism from inference engine (advanced)
"""

import copy
import os
import random
import sys

import numpy as np
import torch
from loguru import logger

from training.trainer_args import parse_training_args
from training.trainers import build_trainer
from training.utils.optimizers import build_optimizer
from training.utils.schedulers import build_lr_scheduler
from training.utils.distributed import setup_distributed, cleanup_distributed, is_main_process
from training.data.video_dataset import CachedLatentDataset
from training.data.bucket_sampler import BucketSampler


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    args = parse_training_args()
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    set_seed(args.seed + rank)  # Different seed per rank for data diversity

    if is_main_process():
        logger.info(f"WorldDistill Training | Method: {args.distill_method} | Model: {args.model_cls}")
        logger.info(f"Distributed: world_size={world_size}, device={device}")
        logger.info(f"Parallel mode: {args.parallel_mode} | SP size: {args.sp_size}")
        if args.parallel_mode == "deepspeed":
            logger.info(f"DeepSpeed ZeRO stage: {args.deepspeed_stage}")
        elif args.parallel_mode == "fsdp":
            logger.info(f"FSDP shard strategy: {args.fsdp_shard_strategy}")
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Build Dataset & DataLoader ---
    dataset = CachedLatentDataset(data_json=args.data_json, cache_dir=args.cache_dir)

    if args.use_bucket_sampler:
        sampler = BucketSampler(
            dataset, batch_size=args.batch_size, shuffle=True,
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
    if args.val_data_json:
        val_dataset = CachedLatentDataset(
            data_json=args.val_data_json,
            cache_dir=args.val_cache_dir if args.val_cache_dir else args.cache_dir,
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

    # --- Build Models ---
    teacher_model, student_model = _load_models(
        args.teacher_model_path,
        args.student_model_path,
        args.model_cls,
        args.config_json,
        device,
    )

    # Apply LoRA if requested
    if args.use_lora:
        student_model = _apply_lora(student_model, args)

    # Gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        if hasattr(student_model, "gradient_checkpointing_enable"):
            student_model.gradient_checkpointing_enable()
        elif hasattr(student_model, "enable_gradient_checkpointing"):
            student_model.enable_gradient_checkpointing()

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
    )

    # --- Train ---
    trainer.train()
    cleanup_distributed()


def _load_models(
    teacher_path: str,
    student_path: str,
    model_cls: str,
    config_json: str,
    device: torch.device,
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

    # --- Strategy 1: diffusers pipeline ---
    if os.path.isdir(teacher_path):
        # Check for model_index.json (diffusers format)
        has_diffusers = (
            os.path.exists(os.path.join(teacher_path, "model_index.json"))
            or os.path.exists(os.path.join(teacher_path, "config.json"))
        )
        if has_diffusers:
            try:
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    teacher_path, torch_dtype=torch.bfloat16
                )
                # Extract the core denoising model (transformer for DiT, unet for UNet)
                if hasattr(pipe, "transformer") and pipe.transformer is not None:
                    teacher_model = pipe.transformer.to(device)
                elif hasattr(pipe, "unet") and pipe.unet is not None:
                    teacher_model = pipe.unet.to(device)
                else:
                    logger.warning("diffusers pipeline loaded but no transformer/unet found.")
                if teacher_model is not None:
                    logger.info(f"Loaded teacher via diffusers: {type(teacher_model).__name__}")
                del pipe
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"diffusers loading failed: {e}")

    # --- Strategy 2: Direct state_dict loading ---
    if teacher_model is None:
        import glob
        if os.path.isdir(teacher_path):
            # Look for weight files
            weight_files = sorted(glob.glob(os.path.join(teacher_path, "*.safetensors")))
            if not weight_files:
                weight_files = sorted(glob.glob(os.path.join(teacher_path, "*.bin")))
            if not weight_files:
                weight_files = sorted(glob.glob(os.path.join(teacher_path, "*.pt")))

            if weight_files:
                logger.info(f"Found {len(weight_files)} weight files in {teacher_path}")
                # Try to load a transformer config and construct model
                teacher_model = _construct_model_from_weights(teacher_path, weight_files, model_cls, device)
        else:
            # Single file
            logger.info(f"Loading single weight file: {teacher_path}")
            teacher_model = _construct_model_from_weights(
                os.path.dirname(teacher_path), [teacher_path], model_cls, device
            )

    # --- Strategy 3: Runner mechanism (fallback) ---
    if teacher_model is None:
        try:
            from inference.lightx2v.utils.registry_factory import RUNNER_REGISTER
            runner_cls = RUNNER_REGISTER.get(model_cls)
            if runner_cls is not None:
                logger.info(f"Attempting Runner-based loading: {model_cls}")
                # NOTE: Runner integration requires inference engine setup.
                # This is a placeholder — in production, Runner handles full init.
                logger.warning("Runner integration not fully implemented for training. "
                               "Please use diffusers format or provide safetensors.")
        except ImportError:
            pass

    if teacher_model is None:
        raise RuntimeError(
            f"Failed to load teacher model from '{teacher_path}'. "
            "Supported formats:\n"
            "  1. diffusers directory (with model_index.json or config.json)\n"
            "  2. Directory with .safetensors / .bin / .pt files\n"
            "  3. Single .pt checkpoint file\n"
            "Hint: For diffusers models, use `huggingface-cli download <repo> --local-dir <path>`"
        )

    # --- Build student model ---
    if student_path and os.path.exists(student_path):
        logger.info(f"Loading separate student model from {student_path}")
        student_model = copy.deepcopy(teacher_model)
        student_state = torch.load(student_path, map_location=device, weights_only=True)
        if isinstance(student_state, dict) and "model" in student_state:
            student_state = student_state["model"]
        missing, unexpected = student_model.load_state_dict(student_state, strict=False)
        if missing:
            logger.warning(f"Student model missing keys: {len(missing)} (e.g., {missing[:3]})")
        if unexpected:
            logger.warning(f"Student model unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")
    else:
        logger.info("Cloning teacher as student model.")
        student_model = copy.deepcopy(teacher_model)

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
    model_dir: str, weight_files: list, model_cls: str, device: torch.device
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

            # Determine model class from config
            model_type = config.get("_class_name", "")
            if "Transformer" in model_type or "DiT" in model_type:
                from diffusers.models import Transformer2DModel
                model = Transformer2DModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
                return model.to(device)
        except Exception as e:
            logger.warning(f"Config-based model construction failed: {e}")

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
                    model_dir, torch_dtype=torch.bfloat16, use_safetensors=True
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
    except ImportError:
        logger.warning("peft not installed. Install with: pip install peft. Skipping LoRA.")
        return model


if __name__ == "__main__":
    main()
