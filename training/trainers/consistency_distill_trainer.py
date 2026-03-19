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
        self.loss_type = self.args.consistency_loss_type
        self.huber_c = self.args.huber_c
        self.delta_schedule = getattr(self.args, "consistency_delta_schedule", "fixed")
        self.delta_min = getattr(self.args, "consistency_delta_min", 5)

        # Create dedicated EMA for consistency target (separate from base EMA)
        # This EMA tracks the student and provides stable targets.
        raw_student = self._unwrap_model(self.student_model)
        self.consistency_ema = EMAModel(
            raw_student,
            decay=self.args.ema_decay,
            warmup_steps=getattr(self.args, "ema_warmup_steps", 0),
        )
        # We also need a model instance to load EMA weights into for forward pass
        self.ema_model = copy.deepcopy(raw_student)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

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
        self.consistency_ema.apply_to(self.ema_model)

        ema_input = self.prepare_teacher_input(batch, noisy_latents_n1, t_n1)
        with torch.no_grad():
            ema_output = self.ema_model(**ema_input)
            if isinstance(ema_output, (tuple, list)):
                ema_output = ema_output[0]

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
        elif self.loss_type == "mse":
            return F.mse_loss(prediction, target)
        else:
            return F.mse_loss(prediction, target)

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
        raw_student = self._unwrap_model(self.student_model)
        self.consistency_ema.update(raw_student)

    def save_checkpoint(self, step: int, output_dir: str):
        """Save checkpoint including EMA model."""
        super().save_checkpoint(step, output_dir)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        torch.save(
            {
                "ema_model": self.ema_model.state_dict(),
                "consistency_ema": self.consistency_ema.state_dict(),
            },
            os.path.join(ckpt_dir, "ema_state.pt"),
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint including EMA model."""
        super().load_checkpoint(path)
        ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
        ema_path = os.path.join(ckpt_dir, "ema_state.pt")
        if os.path.exists(ema_path):
            ema_state = torch.load(ema_path, map_location=self.device, weights_only=False)
            self.ema_model.load_state_dict(ema_state["ema_model"])
            self.consistency_ema.load_state_dict(ema_state["consistency_ema"])
            logger.info("Consistency EMA restored from checkpoint.")
