"""Training arguments for distillation.

Defines all configurable hyperparameters for the distillation training pipeline.
References HY-WorldPlay's TrainingArgs and Open-Sora's config system.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, List, Optional

def _resolve_training_imports() -> tuple[str, Any]:
    if __package__ in {None, ""}:  # pragma: no cover - script entry fallback
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        return (
            import_module("training.env_compat").EXPECTED_TRANSFORMERS_VERSION,
            import_module("training.model_catalog").apply_model_metadata,
        )

    from .env_compat import EXPECTED_TRANSFORMERS_VERSION
    from .model_catalog import apply_model_metadata

    return EXPECTED_TRANSFORMERS_VERSION, apply_model_metadata


_expected_transformers_version, _apply_model_metadata = _resolve_training_imports()


@dataclass
class TrainerArgs:
    """Arguments for distillation training.

    Covers model paths, training hyperparams, distillation-specific settings,
    distributed training config, and logging.
    """

    # --- Model Paths ---
    teacher_model_path: str = ""
    student_model_path: str = ""
    model_cls: str = "wan2.1"
    model_architecture: str = ""
    model_family: str = ""
    model_status: str = ""
    checkpoint_format: str = ""
    resolved_runner_cls: str = ""
    supported_tasks: List[str] = field(default_factory=list)
    model_modalities: List[str] = field(default_factory=list)
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)
    primary_modality: str = ""
    distill_stage: str = ""
    teacher_model_cls: str = ""
    related_distill_models: List[str] = field(default_factory=list)
    distill_runtime_hints: List[str] = field(default_factory=list)
    strict_env_check: bool = True
    required_transformers_version: str = _expected_transformers_version
    config_json: str = ""
    output_dir: str = "./results/distill"

    # --- Distillation Method ---
    distill_method: str = "step_distill"  # step_distill | stream_distill | progressive_distill | consistency_distill | context_forcing | adversarial_distill | dmd_distill
    distill_preset: str = ""  # Path to distill preset JSON (e.g., configs/distill_presets/step_distill_4step.json)

    # --- Flow Matching / Noise ---
    prediction_type: str = "velocity"  # velocity | epsilon | x0
    logit_normal_mean: float = 0.0  # Logit-normal timestep sampling mean
    logit_normal_std: float = 1.0  # Logit-normal timestep sampling std
    sample_shift: float = 1.0  # Sigma shift for Wan2.x style schedule

    # --- Step Distillation ---
    denoising_step_list: List[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    boundary_step_index: int = 2
    use_dual_model: bool = False
    student_low_model: Optional[str] = None  # Path to low-noise student model (for dual-model)

    # --- Stream Distillation ---
    window_size: int = 16
    overlap_frames: int = 4
    noise_schedule: str = "monotonic_linear"  # monotonic_linear | cosine | sigmoid
    denoising_steps_per_frame: int = 4
    causal_attention: bool = True  # Use causal masking in temporal attention

    # --- Progressive Distillation ---
    progressive_stages: List[int] = field(default_factory=lambda: [64, 32, 16, 8, 4])
    progressive_stage_steps: int = 10000
    progressive_loss_space: str = "v"  # v | x0 (loss in v-space or x0-space)
    progressive_reset_optimizer: bool = True  # Reset optimizer when advancing stage

    # --- Consistency Distillation ---
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 0
    consistency_loss_type: str = "huber"  # huber | mse
    huber_c: float = 0.00054
    consistency_delta_schedule: str = "fixed"  # fixed | linear_decay
    consistency_delta_min: int = 5  # Minimum delta (for linear_decay schedule)

    # --- Context Forcing ---
    memory_frames: int = 20
    temporal_context_size: int = 12
    curriculum_training: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [17, 33, 49, 65])
    use_teacher_context: bool = True  # Use teacher-generated context (True) vs GT context (False)

    # --- Adversarial Distillation (ADD/LADD) ---
    adversarial_lambda_adv: float = 0.5  # Weight for adversarial (generator) loss
    adversarial_lambda_distill: float = 2.5  # Weight for score distillation loss
    adversarial_r1_weight: float = 1e-5  # R1 gradient penalty weight
    adversarial_disc_update_freq: int = 1  # Discriminator update frequency (every N steps)
    adversarial_disc_start_step: int = 0  # Step at which to start adversarial training
    adversarial_latent_channels: int = 16  # Latent channels for discriminator input
    adversarial_disc_hidden_dim: int = 256  # Discriminator hidden dimension
    adversarial_disc_num_blocks: int = 4  # Number of discriminator conv blocks

    # --- Distribution Matching Distillation (DMD/DMD2) ---
    dmd_variant: str = "dmd"  # dmd | dmd2
    dmd_lambda_distill: float = 1.0  # Weight for distribution-matching generator gradient
    dmd_lambda_reg: float = 1.0  # Paired teacher-trajectory regression, DMD only
    dmd_fake_score_lr_ratio: float = 1.0  # LR ratio for fake score network vs student
    dmd_fake_score_update_freq: int = 1  # Fake score update frequency
    dmd_fake_score_updates: int = 5  # DSM optimizer updates per generator iteration
    dmd_student_steps: int = 1  # Actual differentiable generator sampling steps
    dmd_teacher_steps: int = 32  # Online teacher sampler for DMD paired regression
    dmd_use_ema_fake_score: bool = False  # Use EMA for fake score network
    dmd_use_gan: bool = False  # Use GAN loss (DMD2-style)
    dmd_gan_weight: float = 0.1  # Weight for GAN generator loss
    dmd_r1_weight: float = 1e-5  # R1 gradient penalty weight
    dmd_disc_update_freq: int = 1  # Discriminator update frequency
    dmd_disc_start_step: int = 0  # Step to start GAN training
    dmd_latent_channels: int = 16  # Latent channels for discriminator input
    dmd_disc_hidden_dim: int = 256  # Discriminator hidden dim
    dmd_disc_num_blocks: int = 4  # Discriminator conv blocks
    dmd_disc_lr_ratio: float = 2.0  # Discriminator LR ratio vs student

    # --- EMA (general, for any trainer) ---
    use_ema: bool = False
    # ema_decay defined above in consistency section (shared)

    # --- Training Hyperparams ---
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_train_steps: int = 100000
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    grad_skip_threshold: float = float("inf")  # Native-only application-level step skip
    mixed_precision: str = "auto"  # auto | no | fp16 | bf16
    seed: int = 42

    # --- Loss ---
    loss_type: str = "mse"  # mse | huber

    # --- Optimizer ---
    optimizer: str = "adamw"  # adamw | muon
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    # --- LR Scheduler ---
    lr_scheduler: str = "cosine"  # cosine | linear | constant | constant_with_warmup | cosine_with_min_lr
    lr_min_ratio: float = 0.1

    # --- LoRA ---
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: float = 64.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])

    # --- Data ---
    data_json: str = ""
    data_mode: str = "auto"  # auto | cached | raw
    video_dir: str = ""
    cache_dir: str = ""  # Pre-computed latent/text embeddings
    resolution: str = "480p"  # 360p | 480p | 720p | 1080p
    num_frames: int = 49
    batch_size: int = 1
    num_workers: int = 4
    use_bucket_sampler: bool = True

    # --- Validation / Evaluation ---
    val_data_json: str = ""  # Validation dataset json (cached latents)
    val_cache_dir: str = ""  # Optional cache dir for validation
    eval_batches: int = 4  # Number of validation batches per eval

    # --- Distributed / Parallelism ---
    parallel_mode: str = "ddp"  # ddp | fsdp | deepspeed
    sp_size: int = 1  # Sequence parallel size (1 = disabled)
    use_fsdp: bool = False  # Legacy flag (use parallel_mode instead)
    fsdp_shard_strategy: str = "full"  # full | hybrid
    deepspeed_stage: int = 2  # ZeRO stage (1, 2, or 3) when parallel_mode=deepspeed
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    enable_tf32: bool = False
    float32_matmul_precision: str = "high"  # highest | high | medium
    enable_torch_compile: bool = False
    torch_compile_scope: str = "student"  # student | teacher | both
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_backend: str = "inductor"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False

    # --- Teacher-Student Runtime / DistillCache ---
    enable_runtime: bool = False
    runtime_name: str = "auto"  # auto | teacher_student | world_model | noop
    runtime_cache_backend: str = "none"  # none | memory | disk | hybrid
    runtime_cache_dir: str = ""
    runtime_cache_max_entries: int = 256
    runtime_cache_hot_entries: int = 64
    runtime_cache_pin_memory: bool = True
    runtime_freshness_steps: int = 0
    runtime_cache_identity: str = ""
    runtime_teacher_cache_mode: str = "disabled"  # disabled | teacher_output | teacher_context | hybrid
    runtime_prefetch_policy: str = "none"  # none | next_chunk | next_batch
    runtime_memory_policy: str = "dense_recent"  # dense_recent | strided_history | hybrid_sparse
    runtime_memory_budget_frames: int = 0
    runtime_memory_recent_ratio: float = 0.5
    runtime_enable_heterogeneous: bool = False
    runtime_teacher_offload: str = "none"  # none | cpu | nvme
    runtime_enable_dpp: bool = False
    runtime_teacher_stream_priority: int = 0
    enable_fused_supervision_kernel: bool = False
    fused_supervision_backend: str = "auto"  # auto | triton | none

    # --- Logging ---
    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 5000
    report_to: str = "console"  # console | tensorboard | wandb | all | none
    tensorboard_log_dir: str = ""
    use_wandb: bool = False
    wandb_project: str = "worlddistill"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_tags: str = ""
    resume_from: str = ""

    # --- Inference Steps (for script compatibility) ---
    num_inference_steps: int = 4
    profile_warmup_steps: int = 0
    profile_active_steps: int = 0

    def __post_init__(self):
        if self.distill_preset:
            self._load_preset(self.distill_preset)
        self._validate_and_normalize()

    def _validate_and_normalize(self) -> None:
        if self.mixed_precision not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError(
                f"mixed_precision must be 'auto', 'no', 'fp16', or 'bf16', got {self.mixed_precision!r}"
            )
        if self.parallel_mode not in {"ddp", "fsdp", "deepspeed"}:
            raise ValueError(
                f"parallel_mode must be ddp, fsdp, or deepspeed, got {self.parallel_mode!r}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.profile_warmup_steps < 0 or self.profile_active_steps < 0:
            raise ValueError("Profiler step counts must be non-negative")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        if self.grad_skip_threshold <= 0:
            raise ValueError("grad_skip_threshold must be positive or infinity")
        if self.loss_type not in {"mse", "huber"}:
            raise ValueError(
                f"Unsupported supervision loss_type={self.loss_type!r}; "
                "supported values are 'mse' and 'huber'."
            )
        if self.consistency_loss_type not in {"mse", "huber"}:
            raise ValueError(
                "Unsupported consistency_loss_type="
                f"{self.consistency_loss_type!r}; supported values are 'mse' and 'huber'."
            )
        if self.huber_c <= 0:
            raise ValueError(f"huber_c must be positive, got {self.huber_c}")
        # Legacy compat: if use_fsdp is True, set parallel_mode to fsdp
        if self.use_fsdp and self.parallel_mode == "ddp":
            self.parallel_mode = "fsdp"
        self.report_to = self._normalize_report_to(self.report_to, use_wandb=self.use_wandb)
        _ = _apply_model_metadata(self)

    @staticmethod
    def _normalize_report_to(report_to: str, use_wandb: bool = False) -> str:
        normalized: list[str] = []
        raw_items = [item.strip().lower() for item in str(report_to or "console").split(",") if item.strip()]
        if not raw_items:
            raw_items = ["console"]
        if "all" in raw_items:
            raw_items = ["console", "tensorboard", "wandb"]
        if "none" in raw_items:
            raw_items = ["none"]
        valid_items = {"none", "console", "tensorboard", "wandb"}
        for item in raw_items:
            if item in valid_items and item not in normalized:
                normalized.append(item)
        if use_wandb and "none" not in normalized and "wandb" not in normalized:
            normalized.append("wandb")
        if not normalized:
            normalized = ["console"]
        if normalized == ["none"]:
            return "none"
        ordered = [name for name in ("console", "tensorboard", "wandb") if name in normalized]
        return ",".join(ordered) if ordered else "console"

    @staticmethod
    def _iter_preset_paths(preset_spec: str) -> list[str]:
        return [path.strip() for path in preset_spec.split(",") if path.strip()]

    @staticmethod
    def _normalize_preset_item(key: str, value: Any) -> tuple[str, Any]:
        alias_map = {
            "method": "distill_method",
            "dual_model": "use_dual_model",
            "loss": "loss_type",
            "gradient_clip": "max_grad_norm",
            "huber_loss_c": "huber_c",
        }
        if key == "curriculum_schedule" and isinstance(value, dict):
            ordered = sorted((int(step), int(frames)) for step, frames in value.items())
            return "curriculum_stages", [frames for _, frames in ordered]
        return alias_map.get(key, key), value

    def _load_preset(self, preset_spec: str):
        """Load one or more distillation presets and override matching fields."""
        for preset_path in self._iter_preset_paths(preset_spec):
            if not os.path.exists(preset_path):
                raise FileNotFoundError(f"Distill preset not found: {preset_path}")
            with open(preset_path, "r") as f:
                preset = json.load(f)
            for key, value in preset.items():
                normalized_key, normalized_value = self._normalize_preset_item(key, value)
                if hasattr(self, normalized_key):
                    setattr(self, normalized_key, normalized_value)

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        explicit_fields: Optional[set[str]] = None,
    ) -> "TrainerArgs":
        """Create ``TrainerArgs`` with explicit CLI values above presets.

        ``argparse`` materializes defaults for every option, so callers may
        provide ``explicit_fields`` to distinguish user-supplied values from
        parser defaults. Presets override defaults; explicit CLI flags then
        override presets.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in vars(args).items() if k in known_fields and v is not None}
        resolved = cls(**kwargs)
        if explicit_fields:
            for field_name in explicit_fields:
                if field_name == "distill_preset" or field_name not in kwargs:
                    continue
                setattr(resolved, field_name, kwargs[field_name])
            resolved._validate_and_normalize()
        return resolved


def build_training_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for distillation training."""
    parser = argparse.ArgumentParser(description="WorldDistill Training")

    # Model
    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--student_model_path", type=str, default="")
    parser.add_argument("--model_cls", type=str, default="wan2.1")
    parser.add_argument("--strict_env_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--required_transformers_version", type=str, default=_expected_transformers_version)
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./results/distill")
    parser.add_argument("--profile_warmup_steps", type=int, default=0)
    parser.add_argument("--profile_active_steps", type=int, default=0)

    # Distillation
    parser.add_argument(
        "--distill_method",
        "--method",
        type=str,
        default="step_distill",
        choices=[
            "step_distill",
            "stream_distill",
            "progressive_distill",
            "consistency_distill",
            "context_forcing",
            "adversarial_distill",
            "dmd_distill",
        ],
    )
    parser.add_argument("--distill_preset", "--config", type=str, default="")
    parser.add_argument("--prediction_type", type=str, default="velocity", choices=["velocity", "epsilon", "x0"])
    parser.add_argument("--use_dual_model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--student_low_model", type=str, default=None)

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_skip_threshold", type=float, default=float("inf"))
    parser.add_argument("--mixed_precision", type=str, default="auto", choices=["auto", "no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=4)

    # Loss
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument(
        "--consistency_loss_type",
        type=str,
        default="huber",
        choices=["mse", "huber"],
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.00054,
        help="Positive Huber delta (and pseudo-Huber constant for consistency distillation)",
    )

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_min_ratio", type=float, default=0.1)

    # EMA
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument("--lora_target_modules", nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])

    # Data
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--data_mode", type=str, default="auto", choices=["auto", "cached", "raw"])
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--use_bucket_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val_data_json", type=str, default="")
    parser.add_argument("--val_cache_dir", type=str, default="")
    parser.add_argument("--eval_batches", type=int, default=4)

    # Distributed / Parallelism
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp", "deepspeed"],
        help="Parallelism strategy: ddp (default), fsdp, or deepspeed",
    )
    parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel group size (1 = disabled)")
    parser.add_argument("--fsdp_shard_strategy", type=str, default="full", choices=["full", "hybrid"], help="FSDP sharding strategy")
    parser.add_argument("--deepspeed_stage", type=int, default=2, choices=[1, 2, 3], help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--enable_tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--float32_matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--enable_torch_compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch_compile_scope", type=str, default="student", choices=["student", "teacher", "both"])
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument("--torch_compile_backend", type=str, default="inductor")
    parser.add_argument("--torch_compile_fullgraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch_compile_dynamic", action=argparse.BooleanOptionalAction, default=False)

    # Teacher-Student Runtime / DistillCache
    parser.add_argument("--enable_runtime", action="store_true")
    parser.add_argument("--runtime_name", type=str, default="auto", choices=["auto", "teacher_student", "world_model", "noop"])
    parser.add_argument("--runtime_cache_backend", type=str, default="none", choices=["none", "memory", "disk", "hybrid"])
    parser.add_argument("--runtime_cache_dir", type=str, default="")
    parser.add_argument("--runtime_cache_max_entries", type=int, default=256)
    parser.add_argument("--runtime_cache_hot_entries", type=int, default=64)
    parser.add_argument("--runtime_cache_pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime_freshness_steps", type=int, default=0)
    parser.add_argument(
        "--runtime_cache_identity",
        type=str,
        default="",
        help="Immutable teacher/config revision used to namespace persistent cache entries",
    )
    parser.add_argument("--runtime_teacher_cache_mode", type=str, default="disabled", choices=["disabled", "teacher_output", "teacher_context", "hybrid"])
    parser.add_argument("--runtime_prefetch_policy", type=str, default="none", choices=["none", "next_chunk", "next_batch"])
    parser.add_argument("--runtime_memory_policy", type=str, default="dense_recent", choices=["dense_recent", "strided_history", "hybrid_sparse"])
    parser.add_argument("--runtime_memory_budget_frames", type=int, default=0)
    parser.add_argument("--runtime_memory_recent_ratio", type=float, default=0.5)
    parser.add_argument("--runtime_enable_heterogeneous", action="store_true")
    parser.add_argument("--runtime_teacher_offload", type=str, default="none", choices=["none", "cpu", "nvme"])
    parser.add_argument("--runtime_enable_dpp", action="store_true")
    parser.add_argument("--runtime_teacher_stream_priority", type=int, default=0)
    parser.add_argument("--enable_fused_supervision_kernel", action="store_true")
    parser.add_argument("--fused_supervision_backend", type=str, default="auto", choices=["auto", "triton", "none"])

    # Adversarial Distillation
    parser.add_argument("--adversarial_lambda_adv", type=float, default=0.5)
    parser.add_argument("--adversarial_lambda_distill", type=float, default=2.5)
    parser.add_argument("--adversarial_r1_weight", type=float, default=1e-5)

    # DMD / DMD2
    parser.add_argument("--dmd_variant", type=str, default="dmd", choices=["dmd", "dmd2"])
    parser.add_argument("--dmd_lambda_distill", type=float, default=1.0)
    parser.add_argument("--dmd_lambda_reg", type=float, default=1.0)
    parser.add_argument("--dmd_fake_score_lr_ratio", type=float, default=1.0)
    parser.add_argument("--dmd_fake_score_update_freq", type=int, default=1)
    parser.add_argument("--dmd_fake_score_updates", type=int, default=5)
    parser.add_argument("--dmd_student_steps", type=int, default=1)
    parser.add_argument("--dmd_teacher_steps", type=int, default=32)
    parser.add_argument("--dmd_use_ema_fake_score", action="store_true")
    parser.add_argument("--dmd_use_gan", action="store_true")
    parser.add_argument("--dmd_gan_weight", type=float, default=0.1)
    parser.add_argument("--dmd_r1_weight", type=float, default=1e-5)
    parser.add_argument("--dmd_disc_update_freq", type=int, default=1)
    parser.add_argument("--dmd_disc_start_step", type=int, default=0)
    parser.add_argument("--dmd_latent_channels", type=int, default=16)
    parser.add_argument("--dmd_disc_hidden_dim", type=int, default=256)
    parser.add_argument("--dmd_disc_num_blocks", type=int, default=4)
    parser.add_argument("--dmd_disc_lr_ratio", type=float, default=2.0)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--report_to", type=str, default="console")
    parser.add_argument("--tensorboard_log_dir", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="worlddistill")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="")

    return parser


def parse_training_args() -> TrainerArgs:
    """Parse command-line arguments into TrainerArgs."""
    parser = build_training_arg_parser()
    args = parser.parse_args()
    explicit_fields = {
        parser._option_string_actions[token.split("=", 1)[0]].dest
        for token in sys.argv[1:]
        if token.split("=", 1)[0] in parser._option_string_actions
    }
    return TrainerArgs.from_args(args, explicit_fields=explicit_fields)
