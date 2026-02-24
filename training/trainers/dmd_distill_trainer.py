"""Distribution Matching Distillation (DMD / DMD2) Trainer.

DMD (CVPR 2024):
- Uses a learned fake score network to estimate the student's distribution.
- Two-time-scale update: fake score update + student update.
- Student loss combines teacher distillation + distribution regularization.

DMD2 (arXiv 2024):
- Removes regression dataset construction and adds GAN loss on real data.
- Student loss combines teacher distillation + GAN adversarial loss.

This trainer supports both variants via `dmd_variant`:
- dmd: fake score network + distribution matching regularization
- dmd2: GAN loss on real data (no fake score by default)

References:
- DMD: https://arxiv.org/abs/2311.18828
- DMD2: https://arxiv.org/abs/2405.14867
"""

import copy
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from training.trainers.adversarial_distill_trainer import ProjectionDiscriminator
from training.trainers.base_distill_trainer import BaseDistillTrainer, EMAModel


class DMDDistillTrainer(BaseDistillTrainer):
    """Trainer for Distribution Matching Distillation (DMD/DMD2)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Variant selection
        self.dmd_variant = getattr(self.args, "dmd_variant", "dmd")  # dmd | dmd2

        # DMD hyperparameters
        self.lambda_distill = getattr(self.args, "dmd_lambda_distill", 1.0)
        self.lambda_reg = getattr(self.args, "dmd_lambda_reg", 1.0)
        self.fake_score_lr_ratio = getattr(self.args, "dmd_fake_score_lr_ratio", 1.0)
        self.fake_score_update_freq = getattr(self.args, "dmd_fake_score_update_freq", 1)
        self.use_ema_fake_score = getattr(self.args, "dmd_use_ema_fake_score", True)

        # DMD2 / GAN hyperparameters
        self.use_gan = getattr(self.args, "dmd_use_gan", self.dmd_variant == "dmd2")
        self.gan_weight = getattr(self.args, "dmd_gan_weight", 0.1)
        self.disc_update_freq = getattr(self.args, "dmd_disc_update_freq", 1)
        self.disc_start_step = getattr(self.args, "dmd_disc_start_step", 0)
        self.r1_penalty_weight = getattr(self.args, "dmd_r1_weight", 1e-5)
        self.disc_lr_ratio = getattr(self.args, "dmd_disc_lr_ratio", 2.0)

        # Fake score network is only used in DMD
        self.enable_fake_score = self.dmd_variant == "dmd" and self.lambda_reg > 0

        # Build fake score network (DMD only)
        self.fake_score_model = None
        self.fake_score_optimizer = None
        self.fake_score_ema = None
        self.fake_score_ema_model = None
        if self.enable_fake_score:
            raw_teacher = self._unwrap_model(self.teacher_model)
            self.fake_score_model = copy.deepcopy(raw_teacher)
            self.fake_score_model.train()
            for p in self.fake_score_model.parameters():
                p.requires_grad = True
            self.fake_score_model.to(self.device)

            self.fake_score_optimizer = torch.optim.AdamW(
                self.fake_score_model.parameters(),
                lr=self.args.learning_rate * self.fake_score_lr_ratio,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )

            if self.use_ema_fake_score:
                self.fake_score_ema = EMAModel(
                    self.fake_score_model,
                    decay=self.args.ema_decay,
                    warmup_steps=getattr(self.args, "ema_warmup_steps", 0),
                )
                self.fake_score_ema_model = copy.deepcopy(raw_teacher)
                self.fake_score_ema_model.eval()
                for p in self.fake_score_ema_model.parameters():
                    p.requires_grad = False

        # Build discriminator (DMD2 / GAN)
        self.discriminator = None
        self.disc_optimizer = None
        if self.use_gan:
            latent_channels = getattr(self.args, "dmd_latent_channels", 16)
            self.discriminator = ProjectionDiscriminator(
                in_channels=latent_channels,
                hidden_dim=getattr(self.args, "dmd_disc_hidden_dim", 256),
                num_blocks=getattr(self.args, "dmd_disc_num_blocks", 4),
                is_video=True,
            ).to(self.device)

            self.disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.args.learning_rate * self.disc_lr_ratio,
                betas=(0.0, 0.99),
                weight_decay=0.0,
            )

        # Logging
        self._distill_loss_ema = 0.0
        self._reg_loss_ema = 0.0
        self._fake_score_loss_ema = 0.0
        self._gan_loss_ema = 0.0
        self._disc_loss_ema = 0.0

        logger.info(
            f"DMD initialized | variant={self.dmd_variant}, "
            f"lambda_distill={self.lambda_distill}, lambda_reg={self.lambda_reg}, "
            f"use_gan={self.use_gan}"
        )

    def train(self):
        """Override train to also DDP-wrap fake_score_model/discriminator."""
        if self.is_distributed:
            if self.enable_fake_score and not isinstance(self.fake_score_model, nn.parallel.DistributedDataParallel):
                self.fake_score_model = nn.parallel.DistributedDataParallel(
                    self.fake_score_model,
                    device_ids=[self.rank],
                    find_unused_parameters=False,
                )
            if self.use_gan and not isinstance(self.discriminator, nn.parallel.DistributedDataParallel):
                self.discriminator = nn.parallel.DistributedDataParallel(
                    self.discriminator,
                    device_ids=[self.rank],
                    find_unused_parameters=False,
                )
        super().train()

    def _generate_student_samples(self, batch: Dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
        """Generate samples from the student in one step (no grad)."""
        bs = noise.shape[0]
        timesteps = torch.full((bs,), self.num_train_timesteps * 0.999, device=self.device)

        student_input = self.prepare_student_input(batch, noise, timesteps)
        with torch.no_grad():
            student_v = self.student_model(**student_input)
            if isinstance(student_v, (tuple, list)):
                student_v = student_v[0]

        sigma = 0.999
        x0_pred = noise.float() - sigma * student_v.float()
        return x0_pred.detach()

    def _generate_student_samples_with_grad(self, batch: Dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
        """Generate samples from the student in one step (with grad)."""
        bs = noise.shape[0]
        timesteps = torch.full((bs,), self.num_train_timesteps * 0.999, device=self.device)

        student_input = self.prepare_student_input(batch, noise, timesteps)
        student_v = self.student_model(**student_input)
        if isinstance(student_v, (tuple, list)):
            student_v = student_v[0]

        sigma = 0.999
        x0_pred = noise.float() - sigma * student_v.float()
        return x0_pred

    def _compute_r1_penalty(self, real_samples: torch.Tensor) -> torch.Tensor:
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
        """DMD/DMD2 forward with optional fake score update and GAN loss."""
        latents = batch["latents"]
        bs = latents.shape[0]

        # ---- GAN Discriminator Update (DMD2) ----
        if self.use_gan and self.global_step >= self.disc_start_step:
            if self.global_step % self.disc_update_freq == 0:
                self.discriminator.train()
                self.disc_optimizer.zero_grad()

                with torch.no_grad():
                    gen_noise = torch.randn_like(latents)
                    fake_x0 = self._generate_student_samples(batch, gen_noise)

                real_logits = self.discriminator(latents)
                d_loss_real = F.relu(1.0 - real_logits).mean()

                fake_logits = self.discriminator(fake_x0.detach())
                d_loss_fake = F.relu(1.0 + fake_logits).mean()

                r1_penalty = (
                    self._compute_r1_penalty(latents) if self.r1_penalty_weight > 0 else 0.0
                )
                disc_loss = d_loss_real + d_loss_fake + self.r1_penalty_weight * r1_penalty
                disc_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.disc_optimizer.step()
                self._disc_loss_ema = 0.9 * self._disc_loss_ema + 0.1 * disc_loss.item()

        # ---- Fake Score Update (DMD only) ----
        fake_score_loss = torch.tensor(0.0, device=self.device)
        if self.enable_fake_score and self.global_step % self.fake_score_update_freq == 0:
            self.fake_score_optimizer.zero_grad()

            gen_noise = torch.randn_like(latents)
            with torch.no_grad():
                student_x0 = self._generate_student_samples(batch, gen_noise)

            t_fake = self._sample_timesteps(bs)
            sigma_fake = t_fake / self.num_train_timesteps
            noise_fake = torch.randn_like(student_x0)
            sigma_fake_exp = sigma_fake.view(bs, *([1] * (student_x0.dim() - 1)))
            student_xt = (1 - sigma_fake_exp) * student_x0 + sigma_fake_exp * noise_fake

            teacher_input = self.prepare_teacher_input(batch, student_xt.to(latents.dtype), t_fake)
            with torch.no_grad():
                teacher_v_on_student = self.teacher_model(**teacher_input)
                if isinstance(teacher_v_on_student, (tuple, list)):
                    teacher_v_on_student = teacher_v_on_student[0]

            fake_input = self.prepare_teacher_input(batch, student_xt.to(latents.dtype), t_fake)
            fake_score_v = self.fake_score_model(**fake_input)
            if isinstance(fake_score_v, (tuple, list)):
                fake_score_v = fake_score_v[0]

            fake_score_loss = F.mse_loss(fake_score_v.float(), teacher_v_on_student.float().detach())
            fake_score_loss.backward()

            if self.args.max_grad_norm > 0:
                raw_fake = self._unwrap_model(self.fake_score_model)
                torch.nn.utils.clip_grad_norm_(raw_fake.parameters(), self.args.max_grad_norm)
            self.fake_score_optimizer.step()

            if self.fake_score_ema is not None:
                self.fake_score_ema.update(self._unwrap_model(self.fake_score_model))

            self._fake_score_loss_ema = 0.9 * self._fake_score_loss_ema + 0.1 * fake_score_loss.item()

        # ---- Student Update ----
        timesteps = self._sample_timesteps(bs)
        sigmas = timesteps / self.num_train_timesteps
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise

        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        with torch.no_grad():
            teacher_output = self.teacher_model(**teacher_input)
            if isinstance(teacher_output, (tuple, list)):
                teacher_output = teacher_output[0]

        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        student_output = self.student_model(**student_input)
        if isinstance(student_output, (tuple, list)):
            student_output = student_output[0]

        distill_loss = F.mse_loss(student_output.float(), teacher_output.float())

        reg_loss = torch.tensor(0.0, device=self.device)
        if self.enable_fake_score and self.lambda_reg > 0:
            if self.fake_score_ema is not None and self.fake_score_ema_model is not None:
                self.fake_score_ema.apply_to(self.fake_score_ema_model)
                fake_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
                with torch.no_grad():
                    fake_v = self.fake_score_ema_model(**fake_input)
                    if isinstance(fake_v, (tuple, list)):
                        fake_v = fake_v[0]
            else:
                fake_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
                with torch.no_grad():
                    fake_v = self.fake_score_model(**fake_input)
                    if isinstance(fake_v, (tuple, list)):
                        fake_v = fake_v[0]
            reg_loss = F.mse_loss(student_output.float(), fake_v.float().detach())

        gan_loss = torch.tensor(0.0, device=self.device)
        if self.use_gan and self.global_step >= self.disc_start_step:
            gen_noise = torch.randn_like(latents)
            fake_x0 = self._generate_student_samples_with_grad(batch, gen_noise)
            fake_logits = self.discriminator(fake_x0)
            gan_loss = -fake_logits.mean()

        total_loss = (
            self.lambda_distill * distill_loss
            + self.lambda_reg * reg_loss
            + self.gan_weight * gan_loss
        )

        self._distill_loss_ema = 0.9 * self._distill_loss_ema + 0.1 * distill_loss.item()
        self._reg_loss_ema = 0.9 * self._reg_loss_ema + 0.1 * reg_loss.item()
        self._gan_loss_ema = 0.9 * self._gan_loss_ema + 0.1 * gan_loss.item()

        return total_loss

    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
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
        if "encoder_hidden_states" in batch:
            input_kwargs["encoder_hidden_states"] = batch["encoder_hidden_states"]
        if "image_cond" in batch:
            input_kwargs["image_cond"] = batch["image_cond"]
        return input_kwargs

    def on_train_step_end(self, metrics: Dict[str, float]):
        metrics["distill_loss"] = self._distill_loss_ema
        metrics["reg_loss"] = self._reg_loss_ema
        if self.enable_fake_score:
            metrics["fake_score_loss"] = self._fake_score_loss_ema
        if self.use_gan:
            metrics["gan_loss"] = self._gan_loss_ema
            metrics["disc_loss"] = self._disc_loss_ema

    def save_checkpoint(self, step: int, output_dir: str):
        import os

        super().save_checkpoint(step, output_dir)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")

        save_dict = {}
        if self.enable_fake_score:
            save_dict.update(
                {
                    "fake_score_model": self._unwrap_model(self.fake_score_model).state_dict(),
                    "fake_score_optimizer": self.fake_score_optimizer.state_dict(),
                }
            )
            if self.fake_score_ema is not None:
                save_dict["fake_score_ema"] = self.fake_score_ema.state_dict()

        if self.use_gan:
            disc_state = (
                self.discriminator.module.state_dict()
                if hasattr(self.discriminator, "module")
                else self.discriminator.state_dict()
            )
            save_dict["discriminator"] = disc_state
            save_dict["disc_optimizer"] = self.disc_optimizer.state_dict()

        if save_dict:
            torch.save(save_dict, os.path.join(ckpt_dir, "dmd_state.pt"))
            logger.info(f"DMD state saved to {ckpt_dir}")

    def load_checkpoint(self, path: str):
        import os

        super().load_checkpoint(path)
        ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
        dmd_path = os.path.join(ckpt_dir, "dmd_state.pt")
        if os.path.exists(dmd_path):
            state = torch.load(dmd_path, map_location=self.device, weights_only=False)

            if self.enable_fake_score and "fake_score_model" in state:
                raw_fake = self._unwrap_model(self.fake_score_model)
                raw_fake.load_state_dict(state["fake_score_model"])
                self.fake_score_optimizer.load_state_dict(state["fake_score_optimizer"])
                if self.fake_score_ema is not None and "fake_score_ema" in state:
                    self.fake_score_ema.load_state_dict(state["fake_score_ema"])

            if self.use_gan and "discriminator" in state:
                raw_disc = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
                raw_disc.load_state_dict(state["discriminator"])
                self.disc_optimizer.load_state_dict(state["disc_optimizer"])

            logger.info("DMD state restored from checkpoint.")
