"""Small real-autograd checks for the algorithm/runtime completion work.

These are CPU correctness tests, not GPU-throughput or checkpoint-quality claims.
"""

import copy
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.runtime.distill_cache import HybridDistillCache
from training.runtime.world_model_runtime import WorldModelTeacherStudentRuntime
from training.trainer_args import TrainerArgs
from training.trainers import TRAINER_REGISTRY
from training.trainers.dmd_distill_trainer import DMDDistillTrainer


class TinyFlow(nn.Module):
    def __init__(self, scale=0.4):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.calls = []

    def forward(self, hidden_states, timestep, **kwargs):
        self.calls.append((hidden_states.detach().clone(), timestep.detach().clone()))
        return hidden_states * self.scale


def make_trainer(method, output_dir, **overrides):
    values = dict(
        distill_method=method, output_dir=output_dir, mixed_precision="no",
        report_to="none", max_train_steps=2, save_every=100, log_every=100,
        gradient_checkpointing=False, learning_rate=0.01, num_frames=6,
        window_size=4, overlap_frames=2, denoising_steps_per_frame=2,
        temporal_context_size=2, memory_frames=2, curriculum_training=False,
        progressive_stages=[8, 4, 2], progressive_stage_steps=1,
        dmd_fake_score_updates=2, dmd_student_steps=2, dmd_teacher_steps=2,
        dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1, dmd_latent_channels=2,
        adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1,
        adversarial_latent_channels=2,
    )
    values.update(overrides)
    args = TrainerArgs(**values)
    teacher, student = TinyFlow(0.8), TinyFlow(0.4)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loader = DataLoader([{"latents": torch.randn(2, 6, 8, 8)}], batch_size=1)
    return TRAINER_REGISTRY[method](
        args=args, teacher_model=teacher, student_model=student,
        optimizer=optimizer, lr_scheduler=scheduler, train_dataloader=loader,
        device=torch.device("cpu"),
    )


class CacheCorrectnessTests(unittest.TestCase):
    def test_promotion_preserves_original_supervision_age(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = HybridDistillCache(directory, hot_max_entries=1, freshness_steps=10, pin_memory=False)
            cache.put("old", torch.tensor([7.0]), current_step=0)
            original_time = cache.hot_cache._storage["old"].created_time
            cache.put("other", torch.tensor([8.0]), current_step=1)
            self.assertEqual(cache.get("old", current_step=9).item(), 7.0)
            self.assertEqual(cache.hot_cache._storage["old"].created_step, 0)
            # Cold persistence has its own wall-time, but promotion never resets it.
            self.assertLessEqual(original_time, cache.hot_cache._storage["old"].created_time)
            self.assertIsNone(cache.get("old", current_step=11))

    def test_context_prefetch_executes_disk_io_on_worker_and_consumes_result(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = HybridDistillCache(directory, hot_max_entries=1, pin_memory=False)
            args = TrainerArgs(runtime_cache_backend="hybrid", runtime_teacher_cache_mode="teacher_context",
                               runtime_prefetch_policy="next_chunk", runtime_cache_identity="test")
            runtime = WorldModelTeacherStudentRuntime(args, TinyFlow(), TinyFlow(), torch.device("cpu"), cache)
            batch, indices = {"sample_id": ["clip"]}, [0, 3]
            runtime.get_or_create_teacher_context(batch, 1, 0, 4, indices, lambda: torch.tensor([11.0]))
            cache.put("other", torch.tensor([0.0]), current_step=1)
            worker_names = []
            original_get = cache.cold_cache.get_entry
            def record_worker(*args, **kwargs):
                worker_names.append(threading.current_thread().name)
                return original_get(*args, **kwargs)
            try:
                with patch.object(cache.cold_cache, "get_entry", side_effect=record_worker):
                    self.assertTrue(runtime.prefetch_teacher_context(batch, 2, indices))
                    value = runtime.get_or_create_teacher_context(
                        batch, 2, 0, 4, indices,
                        lambda: self.fail("cached context must not recompute teacher"),
                    )
                self.assertEqual(value.item(), 11.0)
                self.assertTrue(any(name.startswith("worlddistill-cache") for name in worker_names))
                self.assertEqual(runtime.stats()["prefetch_consumed"], 1)
            finally:
                runtime.close()
            self.assertIsNone(runtime._prefetch_executor)

    def test_prefetch_cannot_bypass_freshness_at_later_step(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = HybridDistillCache(directory, freshness_steps=2, pin_memory=False)
            args = TrainerArgs(runtime_cache_backend="hybrid", runtime_teacher_cache_mode="teacher_context",
                               runtime_prefetch_policy="next_chunk", runtime_cache_identity="test")
            runtime = WorldModelTeacherStudentRuntime(args, TinyFlow(), TinyFlow(), torch.device("cpu"), cache)
            batch = {"sample_id": ["clip"]}
            try:
                runtime.get_or_create_teacher_context(batch, 0, 0, 1, [0], lambda: torch.tensor([1.0]))
                runtime.prefetch_teacher_context(batch, 1, [0])
                value = runtime.get_or_create_teacher_context(batch, 4, 0, 1, [0], lambda: torch.tensor([4.0]))
                self.assertEqual(value.item(), 4.0)
            finally:
                runtime.close()


class AlgorithmCorrectnessTests(unittest.TestCase):
    def test_native_noise_router_is_not_duplicated_by_legacy_dual_flag(self):
        from training.model_adapter import NoiseRoutedDenoiser
        teacher = NoiseRoutedDenoiser(TinyFlow(0.8), TinyFlow(0.7), 0.5)
        student = NoiseRoutedDenoiser(TinyFlow(0.4), TinyFlow(0.3), 0.5)
        optimizer = torch.optim.AdamW(student.parameters())
        trainer = TRAINER_REGISTRY["step_distill"](
            args=TrainerArgs(use_dual_model=True, mixed_precision="no", report_to="none"),
            teacher_model=teacher, student_model=student, optimizer=optimizer,
            lr_scheduler=None, train_dataloader=[], device=torch.device("cpu"),
        )
        self.assertFalse(trainer.use_dual_model)
        self.assertIsNone(trainer.student_low)
        self.assertIs(trainer.student_model, student)
        with self.assertRaisesRegex(ValueError, "cannot override"):
            TRAINER_REGISTRY["step_distill"](
                args=TrainerArgs(use_dual_model=True, student_low_model="conflicting.pt", mixed_precision="no", report_to="none"),
                teacher_model=teacher, student_model=student, optimizer=optimizer,
                lr_scheduler=None, train_dataloader=[], device=torch.device("cpu"),
            )

    def test_chunked_image_condition_is_packed_on_temporal_axis(self):
        from training.trainers.base_distill_trainer import BaseDistillTrainer
        image = torch.arange(7.).reshape(1, 1, 7, 1, 1)
        embeds = torch.randn(1, 7, 16)
        packed = BaseDistillTrainer._slice_temporal_conditions(
            {"image_cond": image, "image_embeds": embeds}, torch.tensor([0, 3, 5, 6]), 7
        )
        torch.testing.assert_close(packed["image_cond"].flatten(), torch.tensor([0., 3., 5., 6.]))
        self.assertIs(packed["image_embeds"], embeds)
        with self.assertRaisesRegex(ValueError, "frame-aligned"):
            BaseDistillTrainer._slice_temporal_conditions({"image_cond": image[:, :, :1]}, torch.tensor([5, 6]), 7)

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "optional diffusers package is absent")
    def test_real_tiny_wan_trainers_forward_backward(self):
        from diffusers import WanTransformer3DModel
        from training.model_adapter import DiffusersTrainingAdapter
        variants = [(name, {}) for name in ("step_distill", "progressive_distill", "consistency_distill",
                                             "context_forcing", "adversarial_distill")]
        variants += [("dmd_distill", {"dmd_variant": variant}) for variant in ("dmd", "dmd2")]
        for method, overrides in variants:
            with self.subTest(method=method, **overrides), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(19)
                teacher = DiffusersTrainingAdapter(WanTransformer3DModel(
                    num_attention_heads=2, attention_head_dim=8, in_channels=2, out_channels=2,
                    text_dim=16, freq_dim=8, ffn_dim=32, num_layers=1,
                ))
                student = copy.deepcopy(teacher)
                with torch.no_grad():
                    for parameter in student.parameters():
                        parameter.add_(0.01 * torch.randn_like(parameter))
                args = TrainerArgs(
                    distill_method=method, output_dir=directory, mixed_precision="no", report_to="none",
                    gradient_checkpointing=False, learning_rate=0.001, num_frames=3,
                    curriculum_training=False, temporal_context_size=2, memory_frames=1,
                    dmd_fake_score_updates=1, dmd_student_steps=2, dmd_teacher_steps=2,
                    dmd_latent_channels=2, dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1,
                    adversarial_latent_channels=2, adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1,
                    **overrides,
                )
                optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                trainer = TRAINER_REGISTRY[method](
                    args=args, teacher_model=teacher, student_model=student, optimizer=optimizer,
                    lr_scheduler=scheduler, train_dataloader=[], device=torch.device("cpu"),
                )
                before = [parameter.detach().clone() for parameter in student.parameters()]
                metrics = trainer.train_step({"latents": torch.randn(1, 2, 3, 4, 4),
                                              "encoder_hidden_states": torch.randn(1, 4, 16)})
                self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
                self.assertGreater(metrics["grad_norm"], 0.0)
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, student.parameters())))

    def test_progressive_samples_final_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer("progressive_distill", directory)
            observed = []
            original = trainer._two_step_teacher_prediction
            def record(batch, noisy, start, middle, end):
                observed.append((start, end))
                return original(batch, noisy, start, middle, end)
            with patch.object(trainer, "_two_step_teacher_prediction", side_effect=record):
                with patch("torch.randint", side_effect=lambda low, high, shape: torch.tensor([high - 1])):
                    loss = trainer._forward_and_loss(next(iter(trainer.train_dataloader)))
            loss.backward()
            self.assertEqual(observed, [(250.0, 0.0)])
            self.assertTrue(torch.isfinite(trainer.student_model.scale.grad))

    def test_stream_unroll_uses_generated_overlap_and_all_denoising_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer("stream_distill", directory)
            batch = next(iter(trainer.train_dataloader))
            loss = trainer._forward_and_loss(batch)
            loss.backward()
            calls = trainer.student_model.calls
            self.assertEqual(len(calls), 4)  # two windows times two solver steps
            initial, times = calls[0]
            sigma = times[:, None, :, None, None] / trainer.num_train_timesteps
            first_generated = initial * (1 - sigma * trainer.student_model.scale.detach() / 2).square()
            self.assertTrue(torch.allclose(calls[2][0][:, :, :2], first_generated[:, :, -2:], atol=1e-6))
            self.assertTrue(torch.equal(calls[2][1][:, :2], torch.zeros(1, 2)))
            self.assertTrue(torch.isfinite(trainer.student_model.scale.grad))
            self.assertNotEqual(trainer.student_model.scale.grad.item(), 0.0)

    def test_distribution_surrogate_has_score_difference_gradient(self):
        generated = torch.tensor([[2.0, 4.0]], requires_grad=True)
        real = torch.tensor([[1.0, 2.0]], requires_grad=True)
        fake = torch.tensor([[3.0, 1.0]], requires_grad=True)
        loss = DMDDistillTrainer.distribution_matching_loss(generated, real, fake)
        loss.backward()
        expected = (fake.detach() - real.detach()) / 1.5 / generated.numel()
        self.assertTrue(torch.allclose(generated.grad, expected))
        self.assertIsNone(real.grad)
        self.assertIsNone(fake.grad)

    def test_dmd2_keeps_fake_score_and_dsm_never_queries_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer("dmd_distill", directory, dmd_variant="dmd2")
            self.assertIsNotNone(trainer.fake_score_model)
            batch = next(iter(trainer.train_dataloader))
            generated = torch.randn_like(batch["latents"], requires_grad=True)
            before = trainer.fake_score_model.scale.detach().clone()
            with patch.object(trainer, "run_teacher", side_effect=AssertionError("DSM must use known noise target")):
                loss = trainer._update_fake_score(batch, generated)
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNone(generated.grad)
            self.assertFalse(torch.equal(before, trainer.fake_score_model.scale))

    def test_all_seven_trainers_run_real_cpu_updates_and_save(self):
        for method in TRAINER_REGISTRY:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(17)
                trainer = make_trainer(method, directory)
                original = trainer.student_model.scale.detach().clone()
                trainer.train()
                self.assertEqual(trainer.global_step, 2)
                self.assertTrue(torch.isfinite(trainer.student_model.scale))
                self.assertFalse(torch.equal(original, trainer.student_model.scale))
                self.assertTrue((Path(directory) / "checkpoint-2" / "trainer_state.pt").exists())

    def test_dmd2_two_time_scale_and_validation_does_not_update_auxiliaries(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer("dmd_distill", directory, dmd_variant="dmd2")
            batch = next(iter(trainer.train_dataloader))
            with patch.object(trainer.fake_score_optimizer, "step", wraps=trainer.fake_score_optimizer.step) as step:
                metrics = trainer.train_step(batch)
                self.assertEqual(step.call_count, 2)
                self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
                before = {name: tensor.clone() for name, tensor in trainer.discriminator.state_dict().items()}
                trainer.validation_step(batch)
                self.assertEqual(step.call_count, 2)
                for name, tensor in trainer.discriminator.state_dict().items():
                    self.assertTrue(torch.equal(tensor, before[name]))


if __name__ == "__main__":
    unittest.main()
