"""Base Distillation Trainer.

Provides the common training loop infrastructure shared by all distillation methods.
Handles:
- Teacher/student model management
- Flow Matching forward process (noising)
- Gradient accumulation + distributed sync
- EMA (Exponential Moving Average) of student weights
- Checkpoint save/load (DDP/FSDP/DeepSpeed compatible)
- FSDP / DeepSpeed ZeRO / DDP parallel strategies
- Sequence Parallelism for long-sequence video DiT models
- Logging (console + optional WandB)

References:
- HY-WorldPlay TrainingPipeline: OOP training loop with train_one_step()
- Open-Sora train.py: Flow matching noising, bucket sampling, ColossalAI
- Progressive Distillation (Salimans & Ho, 2022): EMA teacher
- Consistency Models (Song et al., 2023): EMA target network
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from training.runtime.distill_cache import DistillCache
from training.runtime.fused_supervision import fused_masked_mse_loss, fused_supervision_available
from training.runtime.teacher_student_runtime import PendingTeacherForward, TeacherStudentRuntime
from training.trainer_args import TrainerArgs
from training.utils.distributed import (
    wrap_model,
    wrap_model_fsdp,
    init_deepspeed,
    get_deepspeed_config,
    init_sequence_parallel,
    scatter_sequence,
    gather_sequence,
)
from training.utils.experiment_tracking import ExperimentTracker
from training.utils.model_output import extract_prediction_tensor


@contextlib.contextmanager
def _nullcontext():
    """Backport of contextlib.nullcontext for compatibility."""
    yield


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters updated as:
        ema_param = decay * ema_param + (1 - decay) * param

    Supports warmup: decay ramps from 0 to target_decay over warmup_steps.

    For FSDP models: EMA must operate on the unwrapped (local shard) parameters.
    We handle this by accepting the raw model (not FSDP-wrapped).

    References:
    - Consistency Models: EMA for stable target network
    - Improved DDPM: EMA for sampling quality
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        # Store shadow parameters
        self.shadow = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def get_decay(self) -> float:
        """Get current EMA decay, with optional warmup ramp."""
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters with current model parameters."""
        decay = self.get_decay()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
        self.step_count += 1

    def apply_to(self, model: nn.Module):
        """Copy shadow parameters into model (for inference / target network)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "step_count": self.step_count, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]):
        if not isinstance(state, dict):
            raise TypeError(f"Expected EMA state dict, got {type(state).__name__}")
        required = {"shadow", "step_count", "decay"}
        missing = sorted(required.difference(state))
        if missing:
            raise KeyError(f"EMA state is missing required keys: {missing}")
        loaded_shadow = state["shadow"]
        if not isinstance(loaded_shadow, dict):
            raise TypeError("EMA shadow state must be a mapping")

        expected_names = set(self.shadow)
        loaded_names = set(loaded_shadow)
        missing_names = sorted(expected_names.difference(loaded_names))
        unexpected_names = sorted(loaded_names.difference(expected_names))
        if missing_names or unexpected_names:
            raise KeyError(
                "EMA parameter names do not match: "
                f"missing={missing_names}, unexpected={unexpected_names}"
            )

        restored_shadow = OrderedDict()
        for name, current in self.shadow.items():
            loaded = loaded_shadow[name]
            if not isinstance(loaded, torch.Tensor):
                raise TypeError(f"EMA shadow value for {name!r} must be a tensor")
            if loaded.shape != current.shape:
                raise ValueError(
                    f"EMA shadow shape mismatch for {name!r}: "
                    f"checkpoint={tuple(loaded.shape)}, expected={tuple(current.shape)}"
                )
            restored_shadow[name] = loaded.detach().to(
                device=current.device,
                dtype=current.dtype,
            ).clone()

        self.shadow = restored_shadow
        self.step_count = int(state["step_count"])
        self.decay = float(state["decay"])


class BaseDistillTrainer(ABC):
    """Abstract base trainer for all distillation methods.

    Subclasses must implement:
    - compute_distill_loss(): Define the distillation loss
    - prepare_teacher_input(): Prepare inputs for teacher forward pass

    Optionally override:
    - prepare_student_input(): If student input differs from teacher
    - on_train_step_end(): Custom per-step logic

    Supported parallelism:
    - DDP: Standard DistributedDataParallel (default for <~14B param models)
    - FSDP: FullyShardedDataParallel (for large models, shards params/grads/optim)
    - DeepSpeed: ZeRO Stage 1/2/3 (alternative to FSDP, with CPU offload)
    - Sequence Parallelism: Splits long sequences across GPUs within a group
    """

    def __init__(
        self,
        args: TrainerArgs,
        teacher_model: nn.Module,
        student_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: torch.device = None,
        batch_encoder: Optional[Any] = None,
        runtime: Optional[TeacherStudentRuntime] = None,
        distill_cache: Optional[DistillCache] = None,
    ):
        self.args = args
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_encoder = batch_encoder
        self.runtime = runtime
        self.distill_cache = distill_cache
        self.enable_fused_supervision = bool(getattr(args, "enable_fused_supervision_kernel", False))
        self.fused_supervision_backend = getattr(args, "fused_supervision_backend", "auto")
        self._fused_supervision_warned = False

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self._batches_in_epoch = 0
        self._pending_rng_state: Optional[Dict[str, Any]] = None

        # Parallel strategy
        self.parallel_mode = getattr(args, "parallel_mode", "ddp")  # ddp | fsdp | deepspeed
        self._deepspeed_engine = None  # Set during train() if using DeepSpeed
        self._validate_ema_parallel_contract()
        self._validate_optimizer_parallel_contract()
        self._validate_mixed_precision_contract()

        # Mixed precision
        self.use_amp = args.mixed_precision != "no"
        self.amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        # Note: When using DeepSpeed or FSDP, they handle their own mixed precision.
        # GradScaler is only needed for DDP + fp16.
        self._use_scaler = (args.mixed_precision == "fp16" and self.parallel_mode == "ddp")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_scaler)

        # Flow Matching parameters
        self.num_train_timesteps = 1000
        self.sigma_min = 0.0
        self.prediction_type = getattr(args, "prediction_type", "velocity")  # velocity | epsilon | x0

        # Distributed
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = self.rank == 0

        # Sequence Parallelism
        self.sp_size = getattr(args, "sp_size", 1)
        self.sp_group = None
        if self.sp_size > 1:
            unsupported_roles = [
                role
                for role, model in (
                    ("teacher", self.teacher_model),
                    ("student", self.student_model),
                )
                if not bool(
                    getattr(model, "supports_worlddistill_sequence_parallel", False)
                )
            ]
            if unsupported_roles:
                raise ValueError(
                    "sp_size>1 requires model-layer sequence-parallel adapters on both "
                    "teacher and student; missing capability declaration for "
                    f"{', '.join(unsupported_roles)}. Temporal tensor slicing alone is "
                    "not a correct sequence-parallel implementation."
                )
        if self.sp_size > 1 and self.is_distributed:
            self.sp_group = init_sequence_parallel(self.sp_size)
            if self.sp_group is not None:
                logger.info(f"Sequence parallelism enabled: sp_size={self.sp_size}")

        # Freeze teacher
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # EMA (optional, enabled by subclasses or args)
        self.ema: Optional[EMAModel] = None
        if getattr(args, "use_ema", False):
            self.ema = EMAModel(
                self._unwrap_model(self.student_model)
                if hasattr(self.student_model, "module")
                else self.student_model,
                decay=getattr(args, "ema_decay", 0.9999),
                warmup_steps=getattr(args, "ema_warmup_steps", 0),
            )

        self._tracker: Optional[ExperimentTracker] = None
        self._optimizer_skipped_steps = 0
        self._train_start_time = time.perf_counter()
        if self.is_main_process:
            self._tracker = ExperimentTracker(args)

    # ==================== Abstract Methods ====================

    @abstractmethod
    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the distillation loss."""
        ...

    @abstractmethod
    def prepare_teacher_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        """Prepare input kwargs for teacher model forward pass."""
        ...

    # ==================== Helpers ====================

    def _validate_ema_parallel_contract(self) -> None:
        """Reject EMA modes whose parameters would be sharded underneath it."""

        deepspeed_stage = int(getattr(self.args, "deepspeed_stage", 2))
        uses_parameter_sharding = self.parallel_mode == "fsdp" or (
            self.parallel_mode == "deepspeed" and deepspeed_stage == 3
        )
        if not uses_parameter_sharding:
            return

        if bool(getattr(self.args, "use_ema", False)):
            raise ValueError(
                "Base EMA is not supported with FSDP or DeepSpeed ZeRO-3. "
                "Use DDP/DeepSpeed ZeRO-1/2 or implement a sharded EMA state."
            )
        if getattr(self.args, "distill_method", "") == "consistency_distill":
            raise ValueError(
                "Consistency distillation requires a full EMA target and is not "
                "supported with FSDP or DeepSpeed ZeRO-3."
            )

    def _validate_optimizer_parallel_contract(self) -> None:
        """Reject optimizer/parallel combinations without correct shard semantics."""

        if getattr(self.args, "optimizer", "adamw") == "muon" and self.parallel_mode != "ddp":
            raise ValueError(
                "Muon currently supports serial/DDP training only. Its nested AdamW "
                "fallback is not yet compatible with FSDP or DeepSpeed optimizer sharding."
            )

    def _validate_mixed_precision_contract(self) -> None:
        if self.parallel_mode == "fsdp" and self.args.mixed_precision == "fp16":
            raise ValueError(
                "FSDP fp16 is disabled until a ShardedGradScaler path is validated; "
                "use bf16 or mixed_precision=no."
            )
        if self.parallel_mode == "deepspeed" and math.isfinite(self.args.grad_skip_threshold):
            raise ValueError(
                "grad_skip_threshold is a native PyTorch optimizer policy and is not "
                "supported by the DeepSpeed-managed step path. Leave it at infinity."
            )

    def _require_velocity_prediction(self, method_name: str) -> None:
        if self.prediction_type != "velocity":
            raise ValueError(
                f"{method_name} currently implements velocity-space trajectory math only; "
                f"got prediction_type={self.prediction_type!r}. Use 'velocity'."
            )

    @staticmethod
    def _checkpoint_distributed() -> bool:
        return bool(dist.is_available() and dist.is_initialized())

    def _checkpoint_rank(self) -> int:
        return dist.get_rank() if self._checkpoint_distributed() else 0

    def _checkpoint_is_main_process(self) -> bool:
        return self._checkpoint_rank() == 0

    @staticmethod
    def _format_checkpoint_error(description: str, exc: Exception, rank: int) -> str:
        return f"{description} failed on rank {rank}: {type(exc).__name__}: {exc}"

    def _run_rank0_or_raise(
        self,
        action: Callable[[], Any],
        description: str,
    ) -> Any:
        """Run rank-zero I/O and broadcast any error before raising everywhere."""

        result = None
        error = None
        if self._checkpoint_is_main_process():
            try:
                result = action()
            except Exception as exc:
                error = self._format_checkpoint_error(description, exc, rank=0)

        status = [error]
        if self._checkpoint_distributed():
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(status[0])
        return result

    def _run_all_ranks_or_raise(
        self,
        action: Callable[[], Any],
        description: str,
    ) -> Any:
        """Run an action everywhere and make local failures fatal on every rank."""

        result = None
        local_error = None
        rank = self._checkpoint_rank()
        try:
            result = action()
        except Exception as exc:
            local_error = self._format_checkpoint_error(description, exc, rank=rank)

        errors = [local_error]
        if self._checkpoint_distributed():
            errors = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(errors, local_error)
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError("\n".join(failures))
        return result

    def _broadcast_rank0_object(self, value: Any) -> Any:
        payload = [value if self._checkpoint_is_main_process() else None]
        if self._checkpoint_distributed():
            dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _atomic_save_rank0(
        self,
        path: str,
        state_factory: Callable[[], Any],
        description: str,
    ) -> None:
        """Atomically publish one rank-zero checkpoint file and synchronize errors."""

        def _save() -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            temporary_path = f"{path}.tmp-rank0-{os.getpid()}"
            try:
                torch.save(state_factory(), temporary_path)
                os.replace(temporary_path, path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

        self._run_rank0_or_raise(_save, description)

    def _load_checkpoint_file_all_ranks(
        self,
        path: str,
        *,
        description: str,
        weights_only: bool = False,
        map_location: Any = None,
    ) -> Any:
        """Preflight a checkpoint on rank zero, then load it strictly everywhere."""

        location = self.device if map_location is None else map_location

        def _load() -> Any:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            return torch.load(path, map_location=location, weights_only=weights_only)

        rank0_state = self._run_rank0_or_raise(_load, description)
        return self._run_all_ranks_or_raise(
            lambda: rank0_state if self._checkpoint_is_main_process() else _load(),
            description,
        )

    def _validate_standard_checkpoint_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(state).__name__}")
        required = {
            "step",
            "epoch",
            "student_model",
            "optimizer",
            "lr_scheduler",
            "scaler",
            "best_loss",
        }
        if self.ema is not None:
            required.add("ema")
        missing = sorted(required.difference(state))
        if missing:
            raise KeyError(f"Checkpoint is missing required keys: {missing}")

    def _capture_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _collect_resume_state(self) -> Dict[str, Any]:
        """Gather the exact next-batch/RNG state for every training rank."""

        local_state = {
            "rank": self._checkpoint_rank(),
            "batches_in_epoch": int(getattr(self, "_batches_in_epoch", 0)),
            "rng": self._capture_rng_state(),
        }
        rank_states = [local_state]
        if self._checkpoint_distributed():
            rank_states = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(rank_states, local_state)
        return {
            "schema_version": 1,
            "world_size": len(rank_states),
            "rank_states": rank_states,
        }

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        required = {"python", "numpy", "torch_cpu"}
        missing = sorted(required.difference(state))
        if missing:
            raise KeyError(f"Checkpoint RNG state is missing required keys: {missing}")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state(state["torch_cuda"], self.device)

    def _restore_resume_state(self, resume_state: Optional[Dict[str, Any]]) -> None:
        """Restore this rank's exact data cursor and defer RNG replay to train()."""

        if resume_state is None:
            logger.warning(
                "Legacy checkpoint has no RNG/dataloader cursor; model state is restored, "
                "but exact mid-epoch continuation is unavailable."
            )
            self._batches_in_epoch = 0
            return
        if not isinstance(resume_state, dict) or resume_state.get("schema_version") != 1:
            raise ValueError("Unsupported or malformed checkpoint resume_state")

        current_world_size = (
            dist.get_world_size()
            if self._checkpoint_distributed()
            else int(getattr(self, "world_size", 1))
        )
        current_rank = self._checkpoint_rank()
        saved_world_size = int(resume_state.get("world_size", -1))
        if saved_world_size != current_world_size:
            raise ValueError(
                "Exact resume requires the same world size: "
                f"checkpoint={saved_world_size}, current={current_world_size}"
            )
        rank_states = resume_state.get("rank_states")
        if not isinstance(rank_states, list) or len(rank_states) != saved_world_size:
            raise ValueError("Checkpoint resume_state has an invalid rank_states list")

        local_state = rank_states[current_rank]
        if not isinstance(local_state, dict) or int(local_state.get("rank", -1)) != current_rank:
            raise ValueError(f"Checkpoint resume state is missing rank {current_rank}")
        batches_in_epoch = int(local_state.get("batches_in_epoch", -1))
        if batches_in_epoch < 0:
            raise ValueError("Checkpoint batches_in_epoch must be non-negative")
        rng_state = local_state.get("rng")
        if not isinstance(rng_state, dict):
            raise ValueError("Checkpoint rank RNG state must be a dict")

        self._batches_in_epoch = batches_in_epoch
        self._pending_rng_state = rng_state
        # load_checkpoint() is also a public operation outside train(); restore
        # immediately, then replay once more after train() rebuilds the iterator.
        self._restore_rng_state(rng_state)

    def _refresh_distributed_state(self) -> None:
        self.is_distributed = self._checkpoint_distributed()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = self.rank == 0

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP/FSDP/DeepSpeed to get the raw module."""
        # DeepSpeed engine
        if hasattr(model, "module"):
            return model.module
        return model

    def _v_to_x0(self, v: torch.Tensor, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Convert velocity prediction to x0 prediction.

        In flow matching: x_t = (1-sigma)*x_0 + sigma*eps
        velocity v = eps - x_0 = dx/dsigma
        Therefore: x_0 = x_t - sigma * v
        """
        return x_t.float() - sigma * v.float()

    def _v_to_eps(self, v: torch.Tensor, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Convert velocity prediction to epsilon prediction.

        eps = x_t + (1-sigma) * v  (derived from v = eps - x_0)
        """
        return x_t.float() + (1 - sigma) * v.float()

    def _get_no_sync_context(self):
        """Get the appropriate no_sync context for the current parallel mode.

        - DDP: model.no_sync()
        - FSDP: model.no_sync()
        - DeepSpeed: handled internally, but we still use model.no_sync() if available
        """
        model = self.student_model
        if hasattr(model, "no_sync"):
            return model.no_sync()
        return _nullcontext()

    def _trainable_parameters(self):
        """Return every parameter owned by the optimizer-facing student path."""

        return self._unwrap_model(self.student_model).parameters()

    # ==================== Core Training Loop ====================

    def train(self):
        """Main training loop."""
        logger.info(
            f"Starting {self.args.distill_method} training | "
            f"Steps: {self.args.max_train_steps} | "
            f"LR: {self.args.learning_rate} | "
            f"Batch: {self.args.batch_size} | "
            f"Grad Accum: {self.args.gradient_accumulation_steps} | "
            f"Parallel: {self.parallel_mode}"
        )

        if self._tracker is not None:
            self._tracker.log_config(vars(self.args))

        # ---- Wrap model with chosen parallel strategy ----
        if self.parallel_mode == "deepspeed":
            self._init_deepspeed()
        elif self.parallel_mode == "fsdp":
            self._init_fsdp()
        else:
            # DDP (default)
            if self.is_distributed and not isinstance(
                self.student_model, torch.nn.parallel.DistributedDataParallel
            ):
                self.student_model = torch.nn.parallel.DistributedDataParallel(
                    self.student_model,
                    device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
                    find_unused_parameters=False,
                )

        # Resume only after the DeepSpeed/FSDP/DDP wrapper and its optimizer
        # representation exist.  Loading earlier silently discards FSDP optimizer
        # state and cannot discover DeepSpeed shard checkpoints.
        if self.args.resume_from:
            self.load_checkpoint(self.args.resume_from)

        self.student_model.train()
        self._initialize_data_iterator(resuming=bool(self.args.resume_from))

        try:
            while self.global_step < self.args.max_train_steps:
                # Get next batch
                batch = self._next_batch()

                # Move batch to device and encode raw inputs if needed
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)

                # Sequence parallel: scatter video latents along temporal dim
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

                step_start_time = time.perf_counter()
                step_metrics = self.train_step(batch)
                self.global_step += 1
                step_metrics = self._augment_step_metrics(step_metrics, batch, step_start_time)

                # EMA update
                if self.ema is not None:
                    self.ema.update(self._unwrap_model(self.student_model))

                # Stateful subclass hooks (for example, consistency EMA updates)
                # must run before evaluation/checkpointing so a resumed run sees
                # the exact state that the uninterrupted run would use next.
                self.on_train_step_end(step_metrics)

                # Logging
                if self.is_main_process and self.global_step % self.args.log_every == 0:
                    self._log_metrics(step_metrics)

                # Evaluation
                if (
                    self.val_dataloader is not None
                    and self.args.eval_every > 0
                    and self.global_step % self.args.eval_every == 0
                ):
                    eval_metrics = self.evaluate()
                    if self.is_main_process and eval_metrics:
                        self._log_metrics(eval_metrics)

                # Save checkpoint
                if self.global_step % self.args.save_every == 0:
                    self.save_checkpoint(self.global_step, self.args.output_dir)

            logger.info(f"Training completed at step {self.global_step}.")
            self.save_checkpoint(self.global_step, self.args.output_dir)
        finally:
            if self._tracker is not None:
                self._tracker.close()

    def _init_deepspeed(self):
        """Initialize DeepSpeed engine for training."""
        ds_stage = getattr(self.args, "deepspeed_stage", 2)
        ds_config = get_deepspeed_config(
            stage=ds_stage,
            train_batch_size=self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            learning_rate=self.args.learning_rate,
            max_grad_norm=self.args.max_grad_norm,
            cpu_offload=self.args.cpu_offload,
        )

        self.student_model, self.optimizer, self.lr_scheduler = init_deepspeed(
            model=self.student_model,
            optimizer=self.optimizer,
            ds_config=ds_config,
            lr_scheduler=self.lr_scheduler,
        )
        self._deepspeed_engine = self.student_model
        self._refresh_distributed_state()
        logger.info(f"DeepSpeed ZeRO-{ds_stage} initialized.")

    def _init_fsdp(self):
        """Initialize FSDP wrapping for training."""
        if self.is_distributed and not self._is_fsdp_wrapped(self.student_model):
            from training.utils.optimizers import partition_adamw_parameters

            decay_before_wrap, no_decay_before_wrap = partition_adamw_parameters(
                self.student_model
            )
            decay_ids = {id(parameter) for parameter in decay_before_wrap}
            no_decay_ids = {id(parameter) for parameter in no_decay_before_wrap}
            self.student_model = wrap_model_fsdp(
                self.student_model,
                shard_strategy=getattr(self.args, "fsdp_shard_strategy", "full"),
                cpu_offload=self.args.cpu_offload,
                mixed_precision=self.args.mixed_precision,
            )
            logger.info("Student model wrapped with FSDP.")

            # Re-create AdamW after wrapping, but retain decay identity from the
            # original parameter names/shapes. FSDP local views can be 1-D or
            # empty and must not be reclassified by their shard ndim.
            from training.utils.schedulers import build_lr_scheduler

            wrapped_parameters = [
                parameter
                for parameter in self.student_model.parameters()
                if parameter.requires_grad
            ]
            unknown = [
                parameter
                for parameter in wrapped_parameters
                if id(parameter) not in decay_ids and id(parameter) not in no_decay_ids
            ]
            if unknown:
                raise RuntimeError(
                    "FSDP did not preserve original parameter identities under use_orig_params=True; "
                    "cannot construct correct AdamW weight-decay groups."
                )
            param_groups = []
            decay_after_wrap = [p for p in wrapped_parameters if id(p) in decay_ids]
            no_decay_after_wrap = [p for p in wrapped_parameters if id(p) in no_decay_ids]
            if decay_after_wrap:
                param_groups.append(
                    {"params": decay_after_wrap, "weight_decay": self.args.weight_decay}
                )
            if no_decay_after_wrap:
                param_groups.append({"params": no_decay_after_wrap, "weight_decay": 0.0})
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            self.lr_scheduler = build_lr_scheduler(
                self.optimizer,
                scheduler_type=self.args.lr_scheduler,
                warmup_steps=self.args.warmup_steps,
                total_steps=self.args.max_train_steps,
                min_lr_ratio=self.args.lr_min_ratio,
            )
            logger.info("Optimizer/scheduler re-created for FSDP-wrapped model.")

    @staticmethod
    def _is_fsdp_wrapped(model: nn.Module) -> bool:
        """Check if model is already FSDP-wrapped."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return isinstance(model, FSDP)
        except ImportError:
            return False

    def _next_batch(self) -> Dict[str, Any]:
        """Get the next batch from dataloader, handling epoch transitions."""
        try:
            batch = next(self._data_iter)
            self._batches_in_epoch += 1
        except StopIteration:
            self.epoch += 1
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(self.epoch)
            if hasattr(self.train_dataloader, "batch_sampler") and hasattr(self.train_dataloader.batch_sampler, "set_epoch"):
                self.train_dataloader.batch_sampler.set_epoch(self.epoch)
            self._data_iter = iter(self.train_dataloader)
            batch = next(self._data_iter)
            self._batches_in_epoch = 1
        return batch

    def _initialize_data_iterator(self, *, resuming: bool) -> None:
        """Construct the epoch iterator and replay its deterministic cursor."""

        for sampler in (
            getattr(self.train_dataloader, "sampler", None),
            getattr(self.train_dataloader, "batch_sampler", None),
        ):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)

        self._data_iter = iter(self.train_dataloader)
        if resuming:
            for consumed in range(self._batches_in_epoch):
                try:
                    next(self._data_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Checkpoint dataloader cursor exceeds the configured epoch length: "
                        f"epoch={self.epoch}, batches_in_epoch={self._batches_in_epoch}, "
                        f"failed_after={consumed}. The dataset, sampler, batch size, or world "
                        "size changed since the checkpoint was written."
                    ) from exc

        # Iterator construction can consume the global torch RNG (for worker
        # base seeds), so restore the checkpoint RNG only after rebuilding and
        # advancing the iterator.
        if self._pending_rng_state is not None:
            self._restore_rng_state(self._pending_rng_state)
            self._pending_rng_state = None

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step with gradient accumulation.

        Handles DDP, FSDP, and DeepSpeed uniformly.
        """
        if self._deepspeed_engine is not None:
            return self._train_step_deepspeed(batch)
        else:
            return self._train_step_native(batch)

    def _train_step_native(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Native PyTorch training step (DDP / FSDP)."""
        self.optimizer.zero_grad()
        total_loss = 0.0

        for accum_idx in range(self.args.gradient_accumulation_steps):
            if accum_idx > 0:
                batch = self._next_batch()
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

            # Sync gradients only on last accumulation step
            is_last_accum = accum_idx == self.args.gradient_accumulation_steps - 1
            ctx = (
                self._get_no_sync_context()
                if self.is_distributed and not is_last_accum
                else _nullcontext()
            )

            with ctx:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self._forward_and_loss(batch)
                    loss = loss / self.args.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            total_loss += loss.item()

        # FP16 must always complete an unscale/inf-check cycle, even when
        # clipping is disabled. Disabled scalers do not need this call.
        if self._use_scaler:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        if self.args.max_grad_norm > 0:
            if self._is_fsdp_wrapped(self.student_model):
                # FSDP requires using its own clip_grad_norm_
                grad_norm = self.student_model.clip_grad_norm_(self.args.max_grad_norm)
            else:
                params = self._trainable_parameters()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.args.max_grad_norm)
        else:
            grad_norm = self._compute_grad_norm()

        # All-reduce loss for accurate logging
        if self.is_distributed:
            loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item() / self.world_size

        # Skip step if gradient is too large (following HY-WorldPlay)
        grad_skip_threshold = self.args.grad_skip_threshold
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        optimizer_skipped = 0.0
        if grad_norm_val < grad_skip_threshold:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
        else:
            optimizer_skipped = 1.0
            logger.warning(
                f"Step {self.global_step}: grad_norm={grad_norm_val:.2f} > {grad_skip_threshold}, skipping."
            )
            self.optimizer.zero_grad()
            # Every GradScaler unscale cycle must end in update(), including
            # an application-level grad-norm skip.
            self.scaler.update()

        return {
            "loss": total_loss,
            "grad_norm": grad_norm_val,
            "lr": self.optimizer.param_groups[0]["lr"],
            "optimizer_skipped": optimizer_skipped,
        }

    def _train_step_deepspeed(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """DeepSpeed training step.

        DeepSpeed handles gradient accumulation, mixed precision, and
        gradient clipping internally through its engine.
        """
        engine = self._deepspeed_engine
        assert engine is not None, "DeepSpeed engine should be initialized before calling _train_step_deepspeed"
        total_loss = 0.0

        accumulation_steps = self.args.gradient_accumulation_steps
        for accum_idx in range(accumulation_steps):
            if accum_idx > 0:
                batch = self._next_batch()
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

            loss = self._forward_and_loss(batch)

            # DeepSpeed owns loss scaling and the accumulation boundary. Its
            # step() is called for every microbatch and updates parameters only
            # when the configured boundary is reached.
            engine.backward(loss)
            total_loss += loss.item()
            engine.step()

        # Get grad norm from DeepSpeed
        grad_norm = 0.0
        if hasattr(engine, "get_global_grad_norm"):
            grad_norm = engine.get_global_grad_norm()

        return {
            "loss": total_loss / accumulation_steps,
            "grad_norm": grad_norm if isinstance(grad_norm, float) else 0.0,
            "lr": engine.get_lr()[0] if hasattr(engine, "get_lr") else self.args.learning_rate,
            "optimizer_skipped": 0.0,
        }

    def _augment_step_metrics(
        self,
        metrics: Dict[str, float],
        batch: Dict[str, Any],
        step_start_time: float,
    ) -> Dict[str, float]:
        enriched = dict(metrics)
        step_time = max(time.perf_counter() - step_start_time, 1e-6)
        effective_batch_size = (
            self._infer_batch_size(batch)
            * self.args.gradient_accumulation_steps
            * self.world_size
        )
        self._optimizer_skipped_steps += int(enriched.get("optimizer_skipped", 0.0))

        enriched["epoch"] = float(self.epoch)
        enriched["progress"] = self.global_step / max(1, self.args.max_train_steps)
        enriched["step_time_sec"] = step_time
        enriched["steps_per_sec"] = 1.0 / step_time
        enriched["elapsed_hours"] = max(time.perf_counter() - self._train_start_time, 0.0) / 3600.0
        enriched["eta_hours"] = max(self.args.max_train_steps - self.global_step, 0) * step_time / 3600.0
        enriched["optimizer_skipped_total"] = float(self._optimizer_skipped_steps)
        if effective_batch_size > 0:
            enriched["samples_per_sec"] = effective_batch_size / step_time

        if self.device.type == "cuda" and torch.cuda.is_available():
            enriched["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            enriched["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            enriched["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        if self.runtime is not None:
            runtime_stats = self.runtime.stats()
            for key, value in runtime_stats.items():
                enriched[f"runtime/{key}"] = float(value)
            hits = int(runtime_stats.get("hits", 0))
            misses = int(runtime_stats.get("misses", 0))
            if hits + misses > 0:
                enriched["runtime/cache_hit_rate"] = hits / (hits + misses)

        return enriched

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run teacher + student forward and compute distillation loss.

        Standard flow matching process:
        1. Sample timesteps t ~ logit-normal
        2. Create noisy latents: x_t = (1-sigma)*x_0 + sigma*noise
        3. Teacher forward (no grad) -> teacher prediction
        4. Student forward -> student prediction
        5. Compute distillation loss
        """
        latents = batch["latents"]  # (B, C, T, H, W) or (B, C, H, W)
        bs = latents.shape[0]

        # Sample timesteps
        timesteps = self._sample_timesteps(bs)
        sigmas = timesteps / self.num_train_timesteps

        # Create noisy latents via flow matching: x_t = (1-sigma)*x_0 + sigma*noise
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise

        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        teacher_output, student_output = self._run_teacher_student_pair(
            teacher_input=teacher_input,
            student_input=student_input,
            batch=batch,
            cache_extra={"timesteps": timesteps},
        )

        # Compute loss
        loss = self.compute_distill_loss(teacher_output, student_output, batch, timesteps)
        return loss

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for flow matching.

        Uses logit-normal distribution (following SD3/Open-Sora) for better
        coverage of the noise schedule. Mean/std are configurable.
        """
        logit_mean = getattr(self.args, "logit_normal_mean", 0.0)
        logit_std = getattr(self.args, "logit_normal_std", 1.0)
        u = torch.randn(batch_size, device=self.device) * logit_std + logit_mean
        t = torch.sigmoid(u)  # Maps to (0, 1)
        # Clamp to avoid numerical issues at boundaries
        t = t.clamp(min=0.001, max=0.999)
        timesteps = t * self.num_train_timesteps
        return timesteps.to(self.device)

    def prepare_student_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        """Prepare input for student model. Default: same as teacher."""
        return self.prepare_teacher_input(batch, noisy_latents, timesteps)

    def _augment_model_input(self, input_kwargs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("encoder_hidden_states", "image_cond", "camera_poses", "actions"):
            if key in batch and batch[key] is not None:
                input_kwargs[key] = batch[key]
        return input_kwargs

    @staticmethod
    def _slice_temporal_conditions(
        batch: Dict[str, Any],
        frame_indices: torch.Tensor,
        total_frames: int,
    ) -> Dict[str, Any]:
        """Align frame-wise camera/action conditions with selected latents.

        Global conditions are preserved. An ambiguous temporal layout is an
        error because guessing would silently associate controls with the
        wrong video frames.
        """

        selected = dict(batch)
        for key in ("camera_poses", "actions"):
            value = batch.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                continue

            temporal_dims = [
                dim for dim in range(1, value.ndim) if value.shape[dim] == total_frames
            ]
            if not temporal_dims:
                continue
            if len(temporal_dims) > 1:
                raise ValueError(
                    f"Cannot infer the temporal axis for {key}: shape={tuple(value.shape)} "
                    f"contains total_frames={total_frames} on axes {temporal_dims}."
                )

            index = frame_indices.to(device=value.device, dtype=torch.long)
            selected[key] = value.index_select(temporal_dims[0], index)
        return selected

    def run_teacher(
        self,
        input_kwargs: Dict[str, Any],
        batch: Dict[str, Any],
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[Dict[str, Any]] = None,
        allow_cache: bool = True,
    ) -> torch.Tensor:
        if self.runtime is not None:
            return self.runtime.run_teacher(
                input_kwargs=input_kwargs,
                batch=batch,
                global_step=self.global_step,
                cache_namespace=cache_namespace,
                cache_extra=cache_extra,
                allow_cache=allow_cache,
            )

        with torch.no_grad():
            output = self.teacher_model(**input_kwargs)
        return extract_prediction_tensor(output, tag="teacher")

    def launch_teacher(
        self,
        input_kwargs: Dict[str, Any],
        batch: Dict[str, Any],
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[Dict[str, Any]] = None,
        allow_cache: bool = True,
    ) -> Optional[PendingTeacherForward]:
        if self.runtime is None or not self.runtime.can_pipeline_teacher_student():
            return None
        return self.runtime.launch_teacher(
            input_kwargs=input_kwargs,
            batch=batch,
            global_step=self.global_step,
            cache_namespace=cache_namespace,
            cache_extra=cache_extra,
            allow_cache=allow_cache,
        )

    def wait_teacher(self, pending_teacher: Optional[PendingTeacherForward]) -> Optional[torch.Tensor]:
        if pending_teacher is None:
            return None
        assert self.runtime is not None
        return self.runtime.wait_teacher(pending_teacher)

    def run_student(
        self,
        input_kwargs: Dict[str, Any],
        batch: Dict[str, Any],
        model: Optional[nn.Module] = None,
        tag: str = "student",
    ) -> torch.Tensor:
        student_model = model if model is not None else self.student_model
        if self.runtime is not None:
            return self.runtime.run_student(
                model=student_model,
                input_kwargs=input_kwargs,
                batch=batch,
                global_step=self.global_step,
                tag=tag,
            )
        output = student_model(**input_kwargs)
        return extract_prediction_tensor(output, tag=tag)

    def select_runtime_memory_frames(
        self,
        all_frames: torch.Tensor,
        current_chunk_idx: int,
        chunk_size: int,
        memory_frames: int,
    ) -> Optional[torch.Tensor]:
        if self.runtime is not None:
            return self.runtime.select_memory_frames(
                all_frames=all_frames,
                current_chunk_idx=current_chunk_idx,
                chunk_size=chunk_size,
                memory_frames=memory_frames,
            )

        start_frame = current_chunk_idx * chunk_size
        mem_start = max(0, start_frame - memory_frames)
        mem_end = start_frame
        if mem_end <= mem_start:
            return None
        return all_frames[:, :, mem_start:mem_end]

    def select_runtime_memory_indices(
        self,
        total_frames: int,
        current_chunk_idx: int,
        chunk_size: int,
        memory_frames: int,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Select chronological context indices through the active runtime policy."""

        if self.runtime is not None and hasattr(self.runtime, "select_memory_indices"):
            return self.runtime.select_memory_indices(
                total_frames=total_frames,
                current_chunk_idx=current_chunk_idx,
                chunk_size=chunk_size,
                memory_frames=memory_frames,
                device=device,
            )

        history_end = min(max(0, current_chunk_idx * chunk_size), total_frames)
        if history_end == 0 or memory_frames <= 0:
            return None
        history_start = max(0, history_end - memory_frames)
        return torch.arange(history_start, history_end, device=device, dtype=torch.long)

    def _run_teacher_student_pair(
        self,
        teacher_input: Dict[str, Any],
        student_input: Dict[str, Any],
        batch: Dict[str, Any],
        cache_namespace: str = "teacher_output",
        cache_extra: Optional[Dict[str, Any]] = None,
        allow_cache: bool = True,
        student_model: Optional[nn.Module] = None,
        student_tag: str = "student",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pending_teacher = self.launch_teacher(
            teacher_input,
            batch,
            cache_namespace=cache_namespace,
            cache_extra=cache_extra,
            allow_cache=allow_cache,
        )
        student_output = self.run_student(
            student_input,
            batch,
            model=student_model,
            tag=student_tag,
        )
        if pending_teacher is not None:
            teacher_output = self.wait_teacher(pending_teacher)
        else:
            teacher_output = self.run_teacher(
                teacher_input,
                batch,
                cache_namespace=cache_namespace,
                cache_extra=cache_extra,
                allow_cache=allow_cache,
            )
        assert teacher_output is not None
        return teacher_output, student_output

    def compute_supervision_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must share the same shape for supervision: "
                f"got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )

        loss_type = self.args.loss_type
        if loss_type == "huber":
            element_loss = F.huber_loss(
                prediction.float(),
                target.float(),
                reduction="none",
                delta=float(self.args.huber_c),
            )
            if mask is None:
                return element_loss.mean()
            expanded_mask = torch.broadcast_to(mask, prediction.shape).to(
                device=prediction.device,
                dtype=element_loss.dtype,
            )
            return (element_loss * expanded_mask).sum() / expanded_mask.sum().clamp(min=1.0)

        if loss_type != "mse":
            raise ValueError(
                f"Unsupported supervision loss_type={loss_type!r}; "
                "supported values are 'mse' and 'huber'."
            )
        use_fused = self.enable_fused_supervision and self._should_use_fused_supervision(prediction, target)
        return fused_masked_mse_loss(
            prediction=prediction,
            target=target,
            mask=mask,
            enabled=use_fused,
        )

    def _should_use_fused_supervision(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> bool:
        if self.fused_supervision_backend == "none":
            return False
        if self.fused_supervision_backend not in {"auto", "triton"}:
            return False
        if not fused_supervision_available() and not self._fused_supervision_warned:
            logger.warning("请求启用 fused supervision kernel，但当前环境不可用，将自动回退到 PyTorch loss。")
            self._fused_supervision_warned = True
            return False
        if prediction.shape != target.shape:
            return False
        return fused_supervision_available()

    def _prepare_batch_for_model(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.batch_encoder is None or "latents" in batch:
            return batch
        if "pixel_values" not in batch:
            return batch

        prepared_batch = dict(batch)
        prepared_batch = self.batch_encoder.encode_batch(prepared_batch)
        return prepared_batch

    # ==================== Evaluation ====================

    def _forward_and_loss_eval(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Evaluation forward pass (no updates, no auxiliary losses)."""
        latents = batch["latents"]
        bs = latents.shape[0]

        timesteps = self._sample_timesteps(bs)
        sigmas = timesteps / self.num_train_timesteps

        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise

        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        teacher_output = self.run_teacher(
            teacher_input,
            batch,
            cache_extra={"timesteps": timesteps, "mode": "eval"},
        )

        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        student_output = self.run_student(student_input, batch)

        loss = self.compute_distill_loss(teacher_output, student_output, batch, timesteps)
        return loss

    def validation_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Validation step (override in subclasses if needed)."""
        return self._forward_and_loss_eval(batch)

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation dataloader and return metrics."""
        if self.val_dataloader is None:
            return {}

        self.student_model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_batch_to_device(batch)
                batch = self._prepare_batch_for_model(batch)
                if self.sp_group is not None and "latents" in batch:
                    batch["latents"] = scatter_sequence(batch["latents"], self.sp_group, dim=2)

                loss = self.validation_step(batch)
                total_loss += loss.item()
                num_batches += 1

                if num_batches >= self.args.eval_batches:
                    break

        if num_batches == 0:
            self.student_model.train()
            return {}

        avg_loss = total_loss / max(1, num_batches)
        if self.is_distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size

        self.student_model.train()
        return {"eval_loss": avg_loss}

    # ==================== Hooks ====================

    def on_train_step_end(self, metrics: Dict[str, float]):
        """Hook called after each training step. Override for custom logic."""
        pass

    # ==================== Checkpoint ====================

    def save_checkpoint(self, step: int, output_dir: str):
        """Save training checkpoint (DDP/FSDP/DeepSpeed compatible)."""
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        self._run_all_ranks_or_raise(
            lambda: os.makedirs(ckpt_dir, exist_ok=True),
            f"Create checkpoint directory {ckpt_dir}",
        )

        if self._deepspeed_engine is not None:
            self._save_checkpoint_deepspeed(step, ckpt_dir)
        elif self._is_fsdp_wrapped(self.student_model):
            self._save_checkpoint_fsdp(step, ckpt_dir)
        else:
            self._save_checkpoint_ddp(step, ckpt_dir)

        if self._checkpoint_distributed():
            dist.barrier()

    def _save_checkpoint_deepspeed(self, step: int, ckpt_dir: str) -> None:
        """Save a DeepSpeed checkpoint collectively with strict client state."""

        deepspeed_stage = int(getattr(self.args, "deepspeed_stage", 2))
        if self.ema is not None and deepspeed_stage >= 3:
            raise RuntimeError("Base EMA cannot be checkpointed from partitioned ZeRO-3 parameters.")

        resume_state = self._collect_resume_state()
        client_state = {
            "step": step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "resume_state": resume_state,
        }
        if self.ema is not None:
            client_state["ema"] = self.ema.state_dict()

        def _save() -> Any:
            result = self._deepspeed_engine.save_checkpoint(
                ckpt_dir,
                tag=f"step-{step}",
                client_state=client_state,
            )
            if result is False:
                raise RuntimeError("DeepSpeed save_checkpoint returned False")
            return result

        self._run_all_ranks_or_raise(_save, f"Save DeepSpeed checkpoint {ckpt_dir}")
        if self._checkpoint_is_main_process():
            logger.info(f"DeepSpeed checkpoint saved to {ckpt_dir}")

    def _save_checkpoint_fsdp(self, step: int, ckpt_dir: str):
        """Collect canonical full FSDP model/optimizer state on rank zero."""

        if self.ema is not None:
            raise RuntimeError("Base EMA is not supported with FSDP checkpoints.")

        resume_state = self._collect_resume_state()

        def _collect_state() -> tuple[dict[str, Any], dict[str, Any]]:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_state_dict,
            )

            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            return get_state_dict(self.student_model, self.optimizer, options=options)

        model_state, optimizer_state = self._run_all_ranks_or_raise(
            _collect_state,
            "Collect FSDP model and optimizer state",
        )
        checkpoint_path = os.path.join(ckpt_dir, "trainer_state.pt")
        self._atomic_save_rank0(
            checkpoint_path,
            lambda: {
                "step": step,
                "epoch": self.epoch,
                "student_model": model_state,
                "optimizer": optimizer_state,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_loss": self.best_loss,
                "resume_state": resume_state,
                "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
            },
            f"Write FSDP checkpoint {checkpoint_path}",
        )
        if self._checkpoint_is_main_process():
            logger.info(f"FSDP checkpoint saved to {ckpt_dir}")

    def _save_checkpoint_ddp(self, step: int, ckpt_dir: str):
        """Standard DDP checkpoint save."""
        checkpoint_path = os.path.join(ckpt_dir, "trainer_state.pt")
        resume_state = self._collect_resume_state()

        def _state_factory() -> dict[str, Any]:
            state = {
                "step": step,
                "epoch": self.epoch,
                "student_model": self._unwrap_model(self.student_model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_loss": self.best_loss,
                "resume_state": resume_state,
                "args": vars(self.args) if hasattr(self.args, "__dict__") else str(self.args),
            }
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()
            return state

        self._atomic_save_rank0(
            checkpoint_path,
            _state_factory,
            f"Write DDP checkpoint {checkpoint_path}",
        )
        if self._checkpoint_is_main_process():
            logger.info(f"Checkpoint saved to {ckpt_dir}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint (DDP/FSDP/DeepSpeed compatible)."""
        if self._deepspeed_engine is not None:
            self._load_checkpoint_deepspeed(path)
            return

        if self._is_fsdp_wrapped(self.student_model):
            self._load_checkpoint_fsdp(path)
            return

        ckpt_path = os.path.join(path, "trainer_state.pt") if os.path.isdir(path) else path
        state = self._load_checkpoint_file_all_ranks(
            ckpt_path,
            description=f"Load DDP checkpoint {ckpt_path}",
            weights_only=False,
        )
        self._run_all_ranks_or_raise(
            lambda: self._restore_standard_checkpoint_state(state),
            f"Restore DDP checkpoint {ckpt_path}",
        )
        logger.info(f"Resumed from step {self.global_step}")

    def _restore_standard_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Strictly restore a replicated model checkpoint on the local rank."""

        self._validate_standard_checkpoint_state(state)
        self._unwrap_model(self.student_model).load_state_dict(state["student_model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["step"]
        self.epoch = state["epoch"]
        self.best_loss = state["best_loss"]
        self._restore_resume_state(state.get("resume_state"))
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])

    def _load_checkpoint_deepspeed(self, path: str) -> None:
        """Collectively restore DeepSpeed state and validate client metadata."""

        deepspeed_stage = int(getattr(self.args, "deepspeed_stage", 2))
        if self.ema is not None and deepspeed_stage >= 3:
            raise RuntimeError("Base EMA cannot be restored from partitioned ZeRO-3 parameters.")

        def _load_and_restore() -> str:
            load_path, client_state = self._deepspeed_engine.load_checkpoint(path)
            if load_path is None:
                raise FileNotFoundError(f"DeepSpeed could not load a checkpoint from {path}")
            if not isinstance(client_state, dict):
                raise TypeError("DeepSpeed checkpoint client_state must be a dict")
            required = {"step", "epoch", "best_loss"}
            if self.ema is not None:
                required.add("ema")
            missing = sorted(required.difference(client_state))
            if missing:
                raise KeyError(f"DeepSpeed client_state is missing required keys: {missing}")

            self.global_step = client_state["step"]
            self.epoch = client_state["epoch"]
            self.best_loss = client_state["best_loss"]
            self._restore_resume_state(client_state.get("resume_state"))
            if self.ema is not None:
                self.ema.load_state_dict(client_state["ema"])
            return str(load_path)

        load_path = self._run_all_ranks_or_raise(
            _load_and_restore,
            f"Restore DeepSpeed checkpoint {path}",
        )
        if self._checkpoint_is_main_process():
            logger.info(f"DeepSpeed checkpoint resumed from {load_path}")

    def _load_checkpoint_fsdp(self, path: str):
        """Broadcast a rank-zero full checkpoint into FSDP model/optimizer shards."""

        if self.ema is not None:
            raise RuntimeError("Base EMA is not supported with FSDP checkpoints.")
        ckpt_path = os.path.join(path, "trainer_state.pt") if os.path.isdir(path) else path
        def _load_rank0_state() -> dict[str, Any]:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"FSDP checkpoint not found: {ckpt_path}")
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self._validate_standard_checkpoint_state(loaded)
            return loaded

        rank0_state = self._run_rank0_or_raise(
            _load_rank0_state,
            f"Load FSDP checkpoint {ckpt_path}",
        )
        metadata = self._broadcast_rank0_object(
            {
                "step": rank0_state["step"],
                "epoch": rank0_state["epoch"],
                "best_loss": rank0_state["best_loss"],
                "lr_scheduler": rank0_state["lr_scheduler"],
                "scaler": rank0_state["scaler"],
                "resume_state": rank0_state.get("resume_state"),
            }
            if self._checkpoint_is_main_process()
            else None
        )

        model_state = rank0_state["student_model"] if self._checkpoint_is_main_process() else {}
        optimizer_state = rank0_state["optimizer"] if self._checkpoint_is_main_process() else {}

        def _restore_distributed_state() -> None:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                set_state_dict,
            )

            options = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=True,
            )
            set_state_dict(
                self.student_model,
                self.optimizer,
                model_state_dict=model_state,
                optim_state_dict=optimizer_state,
                options=options,
            )

        self._run_all_ranks_or_raise(
            _restore_distributed_state,
            f"Distribute FSDP checkpoint {ckpt_path}",
        )

        def _restore_metadata() -> None:
            self.lr_scheduler.load_state_dict(metadata["lr_scheduler"])
            self.scaler.load_state_dict(metadata["scaler"])
            self.global_step = metadata["step"]
            self.epoch = metadata["epoch"]
            self.best_loss = metadata["best_loss"]
            self._restore_resume_state(metadata.get("resume_state"))

        self._run_all_ranks_or_raise(
            _restore_metadata,
            f"Restore FSDP checkpoint metadata {ckpt_path}",
        )
        logger.info(f"FSDP checkpoint resumed from {ckpt_path} at step {self.global_step}")

    # ==================== Utilities ====================

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor values in batch dict to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            else:
                result[k] = v
        return result

    @staticmethod
    def _infer_batch_size(batch: Dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
            if isinstance(value, list):
                return len(value)
        return 0

    def _compute_grad_norm(self) -> torch.Tensor:
        """Compute the global L2 gradient norm for replicated or sharded params."""
        total_squared = torch.tensor(0.0, device=self.device)
        for p in self._trainable_parameters():
            if p.grad is not None:
                total_squared += p.grad.detach().float().norm(2) ** 2
        if self._is_fsdp_wrapped(self.student_model) and self._checkpoint_distributed():
            dist.all_reduce(total_squared, op=dist.ReduceOp.SUM)
        return total_squared.sqrt()

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        msg_parts = [f"Step {self.global_step}/{self.args.max_train_steps}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                msg_parts.append(f"{k}={v:.6f}" if "loss" in k else f"{k}={v:.4f}")
            else:
                msg_parts.append(f"{k}={v}")
        logger.info(" | ".join(msg_parts))
        if self._tracker is not None:
            self._tracker.log_metrics(metrics, step=self.global_step)
