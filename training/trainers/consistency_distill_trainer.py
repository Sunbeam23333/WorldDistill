"""Consistency Distillation Trainer.

Implements Trajectory Consistency Distillation (TCD) where the student
learns a consistency function that maps any point on the ODE trajectory
to its origin (clean data).

Key ideas:
- Consistency function: f(x_t, t) -> x_0 for all t on the same ODE trajectory
- Self-consistency constraint: f(x_t, t) ≈ f(x_{t'}, t') for adjacent (t, t')
- EMA teacher provides stable targets
- Huber loss for training stability
- Delta (step size) can decrease over training (curriculum) for sharper targets

Mathematical formulation for flow matching:
- Model predicts velocity v = eps - x_0
- We recover x_0 via: x_0 = x_t - sigma * v
- Consistency function: f(x_t, t) = x_0_pred = x_t - sigma * F_theta(x_t, t)
  (i.e., c_skip=1, c_out=-sigma in semi-linear form)

References:
- Consistency Models: https://arxiv.org/abs/2303.01469
- Latent Consistency Models: https://arxiv.org/abs/2310.04378
- Trajectory Consistency Distillation: https://arxiv.org/abs/2402.19159
"""

import copy
import os
from typing import Any, Dict

import torch
import torch.nn.functional as F
from loguru import logger

from training.trainers.base_distill_trainer import BaseDistillTrainer, EMAModel
from training.utils.model_output import extract_prediction_tensor
from training.utils.sharded_state import ShardedEMA
from training.utils.distributed import wrap_model_fsdp
from training.runtime.teacher_offload import TeacherParameterOffloader


class ConsistencyDistillTrainer(BaseDistillTrainer):
    """Trainer for Trajectory Consistency Distillation (TCD/LCD).

    The core idea: for two points on the same ODE trajectory,
    the consistency function should map both to the same origin x_0.

    Flow matching formulation:
        x_t = (1 - sigma) * x_0 + sigma * eps
        v = eps - x_0  (model output)
        x_0 = x_t - sigma * v  (consistency mapping)

    Training:
        1. At t_n: student predicts x_0 from x_{t_n}
        2. Teacher does one ODE step t_n -> t_{n-1} to get x_{t_{n-1}}
        3. At t_{n-1}: EMA student predicts x_0 from x_{t_{n-1}}
        4. Loss: student's x_0 ≈ EMA's x_0
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._require_velocity_prediction("Consistency distillation")
        self.loss_type = self.args.consistency_loss_type
        self.huber_c = self.args.huber_c
        self.delta_schedule = getattr(self.args, "consistency_delta_schedule", "fixed")
        self.delta_min = getattr(self.args, "consistency_delta_min", 5)

        # Create dedicated EMA for consistency target (separate from base EMA)
        # This EMA tracks the student and provides stable targets.
        raw_student = self._unwrap_model(self.student_model)
        self.consistency_ema = None if self.parallel_mode == "fsdp" else EMAModel(
            raw_student,
            decay=self.args.ema_decay,
            warmup_steps=getattr(self.args, "ema_warmup_steps", 0),
        )
        # We also need a model instance to load EMA weights into for forward pass
        # ZeRO-1/2 may receive a CPU-staged student. The frozen full EMA target
        # is outside that engine and must live on its actual forward device.
        target_device = torch.device("cpu") if self._parameters_are_sharded() else self.device
        self.ema_model = copy.deepcopy(raw_student).to(target_device)
        if self.consistency_ema is not None:
            self.consistency_ema.shadow = {
                name: value.to(target_device) for name, value in self.consistency_ema.shadow.items()
            }
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self._ema_offloader = None

    def _initialize_parallel_targets(self):
        if self.parallel_mode == "fsdp" and not self._is_fsdp_wrapped(self.ema_model):
            if not self._is_fsdp_wrapped(self.student_model):
                raise RuntimeError("Initialize the FSDP student before its consistency target")
            self.ema_model = wrap_model_fsdp(
                self.ema_model,
                shard_strategy=self.args.fsdp_shard_strategy,
                cpu_offload=self.args.cpu_offload,
                mixed_precision=self.args.mixed_precision,
            )
            self.consistency_ema = ShardedEMA(
                self._unwrap_model(self.student_model), parallel_mode="fsdp",
                decay=self.args.ema_decay, warmup_steps=self.args.ema_warmup_steps,
            )
        elif self.parallel_mode == "deepspeed" and self.args.deepspeed_stage == 3 and self._ema_offloader is None:
            from deepspeed.utils import safe_get_full_fp32_param

            for name, parameter in self._unwrap_model(self.student_model).named_parameters():
                if name in self.consistency_ema.shadow:
                    value = safe_get_full_fp32_param(parameter)
                    if value is None or value.shape != self.consistency_ema.shadow[name].shape:
                        raise RuntimeError(f"ZeRO-3 consistency initialization cannot read {name!r}")
                    self.consistency_ema.shadow[name].copy_(value.detach().cpu())
            self.consistency_ema.apply_to(self.ema_model)
            self._ema_offloader = TeacherParameterOffloader(self.ema_model, self.device, target="cpu")
        elif self.consistency_ema is not None and self.consistency_ema.step_count == 0:
            parameters = dict(self._unwrap_model(self.student_model).named_parameters())
            self.consistency_ema.shadow = {
                name: parameters[name].detach().to(device=value.device, dtype=value.dtype).clone()
                for name, value in self.consistency_ema.shadow.items()
            }

    def _close_parallel_targets(self):
        if self._ema_offloader is not None:
            self._ema_offloader.close()

    def _get_delta(self) -> float:
        """Get current ODE step size delta, optionally decayed over training."""
        base_delta = self.num_train_timesteps / 50  # = 20
        if self.delta_schedule == "linear_decay":
            # Linearly decay delta from base_delta to delta_min
            progress = self.global_step / max(1, self.args.max_train_steps)
            delta = base_delta - progress * (base_delta - self.delta_min)
            return max(self.delta_min, delta)
        return base_delta

    def _consistency_function(
        self,
        model_output: torch.Tensor,
        noisy_latents: torch.Tensor,
        sigma_expanded: torch.Tensor,
    ) -> torch.Tensor:
        """Apply consistency function: map (x_t, t) -> x_0.

        For flow matching with velocity prediction:
            x_0 = x_t - sigma * v

        This is the correct formulation because:
            x_t = (1-sigma)*x_0 + sigma*eps
            v = eps - x_0
            => x_t = x_0 + sigma*v
            => x_0 = x_t - sigma*v
        """
        return noisy_latents.float() - sigma_expanded * model_output.float()

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """TCD forward: student and EMA-teacher at adjacent timesteps."""
        latents = batch["latents"]
        bs = latents.shape[0]

        # Sample timestep pair: (t_n, t_{n-1}) where t_n > t_{n-1}
        t_n = torch.rand(bs, device=self.device) * 0.98 + 0.02  # [0.02, 1.0]
        t_n = t_n * self.num_train_timesteps

        # Adjacent timestep (slightly less noisy)
        delta = self._get_delta()
        t_n1 = (t_n - delta).clamp(min=1.0)  # Avoid sigma=0 exactly

        sigma_n = t_n / self.num_train_timesteps
        sigma_n1 = t_n1 / self.num_train_timesteps

        # Create noisy latents at t_n
        noise = torch.randn_like(latents)
        sigma_n_expanded = sigma_n.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents_n = (1 - sigma_n_expanded) * latents + sigma_n_expanded * noise

        teacher_input = self.prepare_teacher_input(batch, noisy_latents_n, t_n)
        pending_teacher = self.launch_teacher(
            teacher_input,
            batch,
            cache_extra={"mode": "consistency_teacher", "timesteps": t_n},
        )

        # --- Student prediction at t_n ---
        student_input = self.prepare_student_input(batch, noisy_latents_n, t_n)
        student_output = self.run_student(student_input, batch)

        # Consistency function: student's x_0 prediction
        student_x0 = self._consistency_function(student_output, noisy_latents_n, sigma_n_expanded)

        # --- One-step ODE solve: x_{t_n} -> x_{t_{n-1}} using teacher ---
        teacher_v = self.wait_teacher(pending_teacher)
        if teacher_v is None:
            teacher_v = self.run_teacher(
                teacher_input,
                batch,
                cache_extra={"mode": "consistency_teacher", "timesteps": t_n},
            )

        # Euler step: dx = v * dsigma, so x_{t-dt} = x_t - dt * v
        dt = sigma_n - sigma_n1
        dt_expanded = dt.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents_n1 = noisy_latents_n.float() - dt_expanded * teacher_v.float()
        noisy_latents_n1 = noisy_latents_n1.to(noisy_latents_n.dtype)

        # --- EMA model prediction at t_{n-1} (target) ---
        # Sync EMA weights into ema_model
        if self.consistency_ema is None:
            raise RuntimeError("Initialize distributed consistency targets before forward")
        if self._ema_offloader is not None:
            self._ema_offloader.synchronize()
        target = self._unwrap_model(self.ema_model) if isinstance(self.consistency_ema, ShardedEMA) else self.ema_model
        self.consistency_ema.apply_to(target)

        ema_input = self.prepare_teacher_input(batch, noisy_latents_n1, t_n1)
        with torch.no_grad():
            if self._ema_offloader is not None:
                ema_output, _ = self._ema_offloader.forward(ema_input)
            else:
                ema_output = self.ema_model(**ema_input)
            ema_output = extract_prediction_tensor(ema_output, tag="consistency EMA")

        sigma_n1_expanded = sigma_n1.view(bs, *([1] * (latents.dim() - 1)))
        ema_x0 = self._consistency_function(ema_output, noisy_latents_n1.float(), sigma_n1_expanded)

        # --- Compute consistency loss ---
        loss = self._compute_consistency_loss(student_x0, ema_x0.detach())
        return loss

    def _compute_consistency_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency loss (Huber or MSE)."""
        if self.loss_type == "huber":
            diff = prediction - target
            loss = torch.sqrt(diff ** 2 + self.huber_c ** 2) - self.huber_c
            return loss.mean()
        if self.loss_type == "mse":
            return F.mse_loss(prediction, target)
        raise ValueError(
            f"Unsupported consistency_loss_type={self.loss_type!r}; "
            "supported values are 'mse' and 'huber'."
        )

    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Not used directly; _forward_and_loss handles full logic."""
        return F.mse_loss(student_output.float(), teacher_output.float())

    def prepare_teacher_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        input_kwargs = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,
        }
        return self._augment_model_input(input_kwargs, batch)

    def on_train_step_end(self, metrics):
        """Update EMA after each step."""
        if metrics.get("optimizer_skipped", 0):
            return
        raw_student = self._unwrap_model(self.student_model)
        if self.parallel_mode == "deepspeed" and self.args.deepspeed_stage == 3:
            # Gather one original parameter at a time using the public ZeRO-3
            # API. Its FP32 EMA lives on CPU; the target is offloaded between
            # forwards instead of keeping another full model on every GPU.
            from deepspeed.utils import safe_get_full_fp32_param

            decay = self.consistency_ema.get_decay()
            with torch.no_grad():
                for name, parameter in raw_student.named_parameters():
                    if name not in self.consistency_ema.shadow:
                        continue
                    value = safe_get_full_fp32_param(parameter)
                    if value is None or value.shape != self.consistency_ema.shadow[name].shape:
                        raise RuntimeError(f"ZeRO-3 consistency target cannot read full parameter {name!r}")
                    self.consistency_ema.shadow[name].mul_(decay).add_(value.detach().cpu(), alpha=1 - decay)
            self.consistency_ema.step_count += 1
        else:
            self.consistency_ema.update(raw_student)

    def _local_target_state(self):
        model = self._unwrap_model(self.ema_model)
        return {
            "parameters": {name: value.detach().cpu().clone() for name, value in model.named_parameters()},
            "buffers": {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
        }

    @torch.no_grad()
    def _restore_local_target_state(self, state):
        model = self._unwrap_model(self.ema_model)
        for kind, current in (("parameters", dict(model.named_parameters())), ("buffers", dict(model.named_buffers()))):
            values = state.get(kind)
            if not isinstance(values, dict) or values.keys() != current.keys():
                raise ValueError(f"Consistency target {kind} names differ")
            for name, value in values.items():
                if not isinstance(value, torch.Tensor) or value.shape != current[name].shape:
                    raise ValueError(f"Consistency target {name} shard shape differs")
            for name, value in values.items():
                current[name].copy_(value.to(current[name].device, dtype=current[name].dtype))

    def save_checkpoint(self, step: int, output_dir: str):
        """Save checkpoint including EMA model."""
        super().save_checkpoint(step, output_dir)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        if self._ema_offloader is not None:
            self._ema_offloader.synchronize()
        if isinstance(self.consistency_ema, ShardedEMA):
            ema_path = os.path.join(ckpt_dir, f"consistency_ema_rank_{self.rank:05d}.pt")
            def _save_local():
                temporary = f"{ema_path}.tmp-{os.getpid()}"
                try:
                    torch.save({"consistency_ema": self.consistency_ema.state_dict(), "target": self._local_target_state()}, temporary)
                    os.replace(temporary, ema_path)
                finally:
                    if os.path.exists(temporary):
                        os.remove(temporary)
            self._run_all_ranks_or_raise(_save_local, "Save sharded consistency target")
            return
        ema_path = os.path.join(ckpt_dir, "ema_state.pt")
        self._atomic_save_rank0(
            ema_path,
            lambda: {
                "ema_model": self.ema_model.state_dict(),
                "consistency_ema": self.consistency_ema.state_dict(),
            },
            f"Write consistency EMA checkpoint {ema_path}",
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint including EMA model."""
        super().load_checkpoint(path)
        ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
        if isinstance(self.consistency_ema, ShardedEMA):
            ema_path = os.path.join(ckpt_dir, f"consistency_ema_rank_{self.rank:05d}.pt")
            def _restore_local():
                state = torch.load(ema_path, map_location="cpu", weights_only=True)
                self.consistency_ema.load_state_dict(state["consistency_ema"])
                self._restore_local_target_state(state["target"])
            self._run_all_ranks_or_raise(_restore_local, "Restore sharded consistency target")
            return
        ema_path = os.path.join(ckpt_dir, "ema_state.pt")
        ema_state = self._load_checkpoint_file_all_ranks(
            ema_path,
            description=f"Load consistency EMA checkpoint {ema_path}",
            weights_only=False,
            map_location="cpu",
        )

        def _restore_consistency_ema():
            if self._ema_offloader is not None:
                self._ema_offloader.synchronize()
            if not isinstance(ema_state, dict):
                raise TypeError(
                    f"Expected consistency EMA checkpoint dict, got {type(ema_state).__name__}"
                )
            missing = sorted({"ema_model", "consistency_ema"}.difference(ema_state))
            if missing:
                raise KeyError(f"Consistency EMA checkpoint is missing required keys: {missing}")
            self.ema_model.load_state_dict(ema_state["ema_model"], strict=True)
            self.consistency_ema.load_state_dict(ema_state["consistency_ema"])

        self._run_all_ranks_or_raise(
            _restore_consistency_ema,
            f"Restore consistency EMA checkpoint {ema_path}",
        )
        logger.info("Consistency EMA restored from checkpoint.")
