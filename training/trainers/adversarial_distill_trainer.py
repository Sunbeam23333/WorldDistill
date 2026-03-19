"""Adversarial Diffusion Distillation (ADD / LADD) Trainer.

Implements the ADD method (Sauer et al., ECCV 2024) where the student is trained
with a combination of adversarial loss (from a discriminator) and score distillation
loss (from a pretrained teacher diffusion model).

Key ideas:
- A discriminator D_phi is trained to distinguish between:
  (a) Real images/frames x_0 from the dataset, and
  (b) Student-generated samples: x_0_hat = denoise(x_t) using the student.
- The student is trained with:
  L_student = lambda_adv * L_adversarial + lambda_distill * L_distill
  where L_adversarial = -E[D_phi(G_theta(x_t))]  (generator loss)
  and   L_distill = E[||v_student - v_teacher||^2]  (score distillation)

LADD (Latent ADD) extends ADD to operate entirely in the latent space,
which is more efficient for high-resolution video generation.

Training dynamics:
- Student and discriminator are trained alternately.
- Discriminator uses R1 gradient penalty for stability.
- Feature matching loss is optional for additional stability.
- The discriminator operates on features from a pretrained encoder (e.g., DINOv2)
  or directly on latent representations.

References:
- ADD: https://arxiv.org/abs/2311.17042 (ECCV 2024)
- LADD: https://arxiv.org/abs/2403.12015
- SDXL-Turbo: Real-time text-to-image via ADD
"""

import copy
import math
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from training.trainers.base_distill_trainer import BaseDistillTrainer


class ProjectionDiscriminator(nn.Module):
    """Lightweight discriminator for adversarial distillation.

    Operates on latent representations (LADD-style) or extracted features.
    Uses spectral normalization for training stability.

    Architecture:
    - Input projection -> 4 conv blocks with spectral norm -> output head
    - Each block: Conv -> GroupNorm -> LeakyReLU -> Conv -> skip connection
    - Supports both 2D (image) and 3D (video) inputs.

    For video: operates per-frame and aggregates with temporal attention.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        is_video: bool = True,
    ):
        super().__init__()
        self.is_video = is_video

        # Input projection
        self.input_proj = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dim = hidden_dim * (2 ** min(i, 2))  # Cap at 4x hidden_dim
            next_dim = hidden_dim * (2 ** min(i + 1, 2))
            self.blocks.append(self._make_block(dim, next_dim))

        final_dim = hidden_dim * (2 ** min(num_blocks, 2))

        # Temporal aggregation for video (simple attention pooling)
        if is_video:
            self.temporal_pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
            )

        # Output head: spatial pool -> linear -> scalar
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(final_dim, 1)),
        )

    def _make_block(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False)),
            nn.GroupNorm(min(32, out_dim), out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_dim, out_dim, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W) for images or (B, C, T, H, W) for videos.

        Returns:
            Discriminator logits (B, 1).
        """
        if x.dim() == 5:
            # Video: process per-frame, then aggregate
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
            features = self._forward_2d(x)  # (B*T, final_dim)
            features = features.view(B, T, -1).permute(0, 2, 1)  # (B, D, T)
            features = self.temporal_pool(features).squeeze(-1)  # (B, D)
            return self.head[2](features).unsqueeze(-1)  # Use only the linear layer
        else:
            features = self._forward_2d(x)
            return self.head(features.unsqueeze(-1).unsqueeze(-1))

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Process 2D input through conv blocks."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x


class AdversarialDistillTrainer(BaseDistillTrainer):
    """Trainer for Adversarial Diffusion Distillation (ADD/LADD).

    Training alternates between:
    1. Discriminator update: classify real vs student-generated samples
    2. Student (generator) update: fool discriminator + match teacher score

    Loss formulation:
    - L_D = L_D_real + L_D_fake + lambda_r1 * R1_penalty
    - L_G = lambda_adv * L_adv + lambda_distill * L_distill + lambda_feat * L_feat
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Hyperparameters
        self.lambda_adv = getattr(self.args, "adversarial_lambda_adv", 0.5)
        self.lambda_distill = getattr(self.args, "adversarial_lambda_distill", 2.5)
        self.lambda_feat = getattr(self.args, "adversarial_lambda_feat", 0.0)
        self.r1_penalty_weight = getattr(self.args, "adversarial_r1_weight", 1e-5)
        self.disc_update_freq = getattr(self.args, "adversarial_disc_update_freq", 1)
        self.disc_start_step = getattr(self.args, "adversarial_disc_start_step", 0)

        # Build discriminator
        is_video = True  # Assume video DiT
        latent_channels = getattr(self.args, "adversarial_latent_channels", 16)
        self.discriminator = ProjectionDiscriminator(
            in_channels=latent_channels,
            hidden_dim=getattr(self.args, "adversarial_disc_hidden_dim", 256),
            num_blocks=getattr(self.args, "adversarial_disc_num_blocks", 4),
            is_video=is_video,
        ).to(self.device)

        # Discriminator optimizer (separate from student optimizer)
        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.args.learning_rate * 2.0,  # Discriminator typically uses higher LR
            betas=(0.0, 0.99),  # Standard GAN optimizer settings
            weight_decay=0.0,
        )

        # Track discriminator loss for logging
        self._disc_loss_ema = 0.0
        self._gen_loss_ema = 0.0

    def train(self):
        """Override train to also DDP-wrap discriminator."""
        if self.is_distributed and not isinstance(
            self.discriminator, nn.parallel.DistributedDataParallel
        ):
            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator,
                device_ids=[self.rank],
                find_unused_parameters=False,
            )
        super().train()

    def _student_one_step_generate(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a clean sample from the student in one step.

        Student predicts velocity v, then:
            x_0 = x_t - sigma * v
        """
        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        student_v = self.run_student(student_input, batch)

        sigmas = timesteps / self.num_train_timesteps
        bs = noisy_latents.shape[0]
        sigmas_expanded = sigmas.view(bs, *([1] * (noisy_latents.dim() - 1)))
        x0_pred = self._v_to_x0(student_v, noisy_latents, sigmas_expanded)
        return x0_pred

    def _compute_r1_penalty(
        self, real_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute R1 gradient penalty on real samples.

        R1 = E[||grad D(x_real)||^2]
        This regularizes the discriminator to prevent mode collapse.
        """
        real_samples = real_samples.detach().requires_grad_(True)
        real_logits = self.discriminator(real_samples)
        grad_outputs = torch.ones_like(real_logits)

        gradients = torch.autograd.grad(
            outputs=real_logits,
            inputs=real_samples,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        r1_penalty = gradients.view(gradients.size(0), -1).norm(2, dim=1).pow(2).mean()
        return r1_penalty

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Alternating training: discriminator then generator (student).

        This method handles both updates in a single call.
        The discriminator is updated every disc_update_freq steps.
        """
        latents = batch["latents"]
        bs = latents.shape[0]

        # Sample timesteps and create noisy latents
        timesteps = self._sample_timesteps(bs)
        sigmas = timesteps / self.num_train_timesteps
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise

        # ---- Step 1: Discriminator update ----
        disc_loss = torch.tensor(0.0, device=self.device)
        if (
            self.global_step >= self.disc_start_step
            and self.global_step % self.disc_update_freq == 0
        ):
            self.discriminator.train()
            self.disc_optimizer.zero_grad()

            with torch.no_grad():
                # Student generates fake samples
                fake_x0 = self._student_one_step_generate(
                    batch, noisy_latents, timesteps
                )

            # Discriminator on real
            real_logits = self.discriminator(latents)
            d_loss_real = F.relu(1.0 - real_logits).mean()  # Hinge loss

            # Discriminator on fake
            fake_logits = self.discriminator(fake_x0.detach())
            d_loss_fake = F.relu(1.0 + fake_logits).mean()  # Hinge loss

            # R1 gradient penalty
            r1_penalty = self._compute_r1_penalty(latents)

            disc_loss = d_loss_real + d_loss_fake + self.r1_penalty_weight * r1_penalty
            disc_loss.backward()

            # Gradient clipping for discriminator
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=1.0
            )
            self.disc_optimizer.step()
            self._disc_loss_ema = 0.9 * self._disc_loss_ema + 0.1 * disc_loss.item()

        # ---- Step 2: Generator (student) update ----
        # Score distillation loss
        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        teacher_output, student_output = self._run_teacher_student_pair(
            teacher_input=teacher_input,
            student_input=student_input,
            batch=batch,
            cache_extra={"mode": "adversarial_teacher", "timesteps": timesteps},
        )

        distill_loss = self.compute_supervision_loss(student_output, teacher_output.detach())

        # Adversarial loss (generator wants discriminator to output high scores)
        gen_adv_loss = torch.tensor(0.0, device=self.device)
        if self.global_step >= self.disc_start_step:
            # Generate from student (with grad this time)
            fake_x0 = self._v_to_x0(student_output, noisy_latents, sigmas_expanded)
            fake_logits = self.discriminator(fake_x0)
            gen_adv_loss = -fake_logits.mean()  # Non-saturating GAN loss

        # Combined generator loss
        total_gen_loss = (
            self.lambda_distill * distill_loss
            + self.lambda_adv * gen_adv_loss
        )

        self._gen_loss_ema = 0.9 * self._gen_loss_ema + 0.1 * total_gen_loss.item()
        return total_gen_loss

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

    def on_train_step_end(self, metrics: Dict[str, float]):
        """Log discriminator and generator losses."""
        metrics["disc_loss"] = self._disc_loss_ema
        metrics["gen_adv_loss"] = self._gen_loss_ema

    def save_checkpoint(self, step: int, output_dir: str):
        """Save checkpoint including discriminator."""
        import os

        super().save_checkpoint(step, output_dir)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        disc_state = (
            self.discriminator.module.state_dict()
            if hasattr(self.discriminator, "module")
            else self.discriminator.state_dict()
        )
        torch.save(
            {
                "discriminator": disc_state,
                "disc_optimizer": self.disc_optimizer.state_dict(),
            },
            os.path.join(ckpt_dir, "discriminator_state.pt"),
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint including discriminator."""
        import os

        super().load_checkpoint(path)
        ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
        disc_path = os.path.join(ckpt_dir, "discriminator_state.pt")
        if os.path.exists(disc_path):
            state = torch.load(disc_path, map_location=self.device, weights_only=False)
            raw_disc = (
                self.discriminator.module
                if hasattr(self.discriminator, "module")
                else self.discriminator
            )
            raw_disc.load_state_dict(state["discriminator"])
            self.disc_optimizer.load_state_dict(state["disc_optimizer"])
            logger.info("Discriminator state restored from checkpoint.")
