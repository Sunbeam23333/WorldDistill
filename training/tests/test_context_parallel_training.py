"""Real two-rank Ulysses forward/backward against an unpartitioned tiny Wan."""
import copy
import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training.model_adapter import DiffusersTrainingAdapter
from training.utils.context_parallel import synchronize_context_batch, synchronize_context_rng


def _tiny_wan(image=False):
    from diffusers import WanTransformer3DModel
    return DiffusersTrainingAdapter(WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
        in_channels=4 if image else 2, out_channels=2, text_dim=8, freq_dim=8,
        ffn_dim=32, num_layers=2, image_dim=8 if image else None,
        added_kv_proj_dim=16 if image else None,
    ))


def _wan_worker(rank, world, rendezvous, checkpointing):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        torch.manual_seed(41)
        reference = _tiny_wan()
        model = copy.deepcopy(reference)
        model.enable_worlddistill_sequence_parallel(2, torch.device("cpu"))
        if checkpointing:
            model.enable_gradient_checkpointing()
            reference.enable_gradient_checkpointing()
        observed_tokens = []
        handle = model.model.blocks[-1].register_forward_hook(
            lambda module, inputs, output: observed_tokens.append(output.shape[1]))
        torch.manual_seed(50 + rank // 2)
        full = torch.randn(1, 2, 4, 2, 2)
        conditions = {"encoder_hidden_states": torch.randn(1, 3, 8)}
        times = torch.tensor([31.])
        # A two-call differentiable rollout exercises the replicated-input
        # gradient boundary, not just first-order parameter gradients.
        expected_first = reference(full, times, **conditions)
        expected = reference(expected_first + full, times, **conditions)
        expected.square().mean().backward()
        actual_first = model(full, times, **conditions)
        actual = model(actual_first + full, times, **conditions)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        (actual.square().mean() * 2).backward()
        for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            assert (parameter.grad is None) == (expected_parameter.grad is None), name
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
                dist.all_reduce(expected_parameter.grad)
                expected_parameter.grad.div_(world)
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=6e-6, rtol=3e-4,
                                           msg=lambda message: f"{name}: {message}")
        assert observed_tokens and set(observed_tokens) == {2}
        handle.remove()

        batch = {"latents": full + rank, "nested": {"tag": f"rank{rank}"},
                 "conditions": [conditions["encoder_hidden_states"], None]}
        batch = synchronize_context_batch(batch, model._context_parallel_group, torch.device("cpu"))
        leader = (rank // 2) * 2
        torch.testing.assert_close(batch["latents"], full + leader)
        assert batch["nested"]["tag"] == f"rank{leader}"
        torch.manual_seed(42 + rank)
        synchronize_context_rng(model._context_parallel_group, torch.device("cpu"))
        sample = torch.randn(3)
        torch.manual_seed(42 + leader)
        torch.testing.assert_close(sample, torch.randn(3))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,checkpointing", [(2, False), (2, True), (4, False)])
def test_real_wan_ulysses_matches_full_forward_and_rollout_gradients(tmp_path, world, checkpointing):
    pytest.importorskip("diffusers")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Two-rank CPU Gloo is unavailable")
    mp.spawn(_wan_worker, args=(world, str(tmp_path / "rendezvous"), checkpointing), nprocs=world, join=True)


def test_context_parallel_capability_is_architecture_specific():
    class OtherVideo(torch.nn.Module):
        def forward(self, hidden_states, timestep):
            return hidden_states
    assert not DiffusersTrainingAdapter(OtherVideo()).supports_worlddistill_sequence_parallel
    pytest.importorskip("diffusers")
    model = _tiny_wan()
    assert model.supports_worlddistill_sequence_parallel
    with pytest.raises(ValueError, match="initialized distributed"):
        model.enable_worlddistill_sequence_parallel(2, torch.device("cpu"))


def _training_worker(rank, rendezvous, output_dir, method, offload="none", parallel="ddp", use_cuda=False, use_lora=False):
    from training.trainer_args import TrainerArgs
    from training.trainers import TRAINER_REGISTRY
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if use_cuda else "gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=180))
    try:
        trained = []
        for sp_size in (1, 2):
            torch.manual_seed(110)
            student = _tiny_wan().to(device)
            if use_lora:
                from peft import get_peft_model, LoraConfig
                student = get_peft_model(student, LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_v"]))
            teacher = _tiny_wan().to(device if offload == "none" else "cpu")
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            data = [{"latents": torch.randn(2, 4, 4, 4),
                     "encoder_hidden_states": torch.randn(3, 8)} for _ in range(4)]
            args = TrainerArgs(
                distill_method=method, output_dir=str(Path(output_dir) / f"sp{sp_size}"),
                max_train_steps=2, mixed_precision="no", num_workers=0,
                gradient_accumulation_steps=2, sp_size=sp_size,
                save_every=100, log_every=100, report_to="none", max_grad_norm=0.,
                learning_rate=1e-4, grad_skip_threshold=1e9,
                denoising_step_list=[1000, 500], progressive_stages=[4, 2, 1],
                progressive_stage_steps=1, window_size=3, overlap_frames=1,
                denoising_steps_per_frame=1, causal_attention=False,
                num_frames=4, memory_frames=1, temporal_context_size=2,
                curriculum_training=False, adversarial_latent_channels=2,
                adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1,
                adversarial_r1_weight=0., dmd_latent_channels=2,
                dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1,
                dmd_fake_score_updates=1, dmd_teacher_steps=2, dmd_student_steps=2,
                runtime_teacher_offload=offload,
                parallel_mode=parallel, deepspeed_stage=3,
                runtime_enable_dpp=use_cuda, use_ema=use_cuda,
                use_lora=use_lora,
            )
            runtime = None
            if offload != "none":
                from training.runtime.teacher_student_runtime import TeacherStudentRuntime
                runtime = TeacherStudentRuntime(args, teacher, student, device)
            trainer = TRAINER_REGISTRY[method](
                args=args, teacher_model=teacher, student_model=student,
                optimizer=optimizer, lr_scheduler=scheduler,
                train_dataloader=torch.utils.data.DataLoader(data, batch_size=1),
                device=device, runtime=runtime)
            trainer.train()
            assert trainer.global_step == 2
            if method == "dmd_distill":
                fake = trainer._unwrap_model(trainer.fake_score_model)
                assert fake._context_parallel_group is None
            trained.append({name: value.detach().clone() for name, value in
                            trainer._unwrap_model(trainer.student_model).named_parameters()})
        for name in trained[0]:
            torch.testing.assert_close(trained[1][name], trained[0][name], atol=5e-6, rtol=2e-4,
                                       msg=lambda message: f"{method}: {name}: {message}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method", ["step_distill", "consistency_distill", "progressive_distill",
                                   "stream_distill", "context_forcing", "dmd_distill", "adversarial_distill"])
def test_wan_training_cp_updates_match_full_replica_training(tmp_path, method):
    pytest.importorskip("diffusers")
    mp.spawn(_training_worker, args=(str(tmp_path / "init"), str(tmp_path), method), nprocs=2, join=True)


@pytest.mark.parametrize("method,offload", [("step_distill", "cpu"), ("progressive_distill", "nvme")])
def test_wan_cp_parameter_offload_and_revision(tmp_path, method, offload):
    pytest.importorskip("diffusers")
    mp.spawn(_training_worker, args=(str(tmp_path / "init"), str(tmp_path), method, offload), nprocs=2, join=True)


def test_wan_cp_progressive_lora_nvme_stage_handoff_matches_full_replica(tmp_path):
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    mp.spawn(_training_worker,
             args=(str(tmp_path / "init"), str(tmp_path), "progressive_distill", "nvme", "ddp", False, True),
             nprocs=2, join=True)


@pytest.mark.parametrize("parallel,method,offload", [("ddp", "step_distill", "cpu"),
    ("fsdp", "consistency_distill", "nvme"), ("deepspeed", "consistency_distill", "cpu")])
def test_cuda_wan_context_parallel_dpp_offload_and_sharded_training(tmp_path, parallel, method, offload):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    pytest.importorskip("diffusers")
    if parallel == "deepspeed" and importlib.util.find_spec("deepspeed") is None:
        pytest.skip("Requires DeepSpeed")
    mp.spawn(_training_worker, args=(str(tmp_path / "init"), str(tmp_path), method, offload, parallel, True),
             nprocs=2, join=True)


def _conditional_worker(rank, rendezvous, variant):
    from training.model_adapter import NoiseRoutedDenoiser
    from training.runtime.teacher_student_runtime import TeacherStudentRuntime
    from training.runtime.distill_cache import MemoryDistillCache
    from training.trainer_args import TrainerArgs
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(120)
        reference = (NoiseRoutedDenoiser(_tiny_wan(), _tiny_wan(), .5) if variant == "routed"
                     else _tiny_wan(image=True))
        model = copy.deepcopy(reference)
        model.enable_worlddistill_sequence_parallel(2, torch.device("cpu"))
        full = torch.randn(2, 2, 4, 2, 2)
        times = torch.tensor([800., 300.])
        conditions = {"encoder_hidden_states": torch.randn(2, 512 if variant == "image" else 3, 8)}
        if variant == "image":
            conditions.update(image_cond=torch.randn_like(full), encoder_hidden_states_image=torch.randn(2, 257, 8))
        expected = reference(full, times, **conditions)
        actual = model(full, times, **conditions)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=1e-4)
        expected.square().mean().backward()
        (actual.square().mean() * 2).backward()
        for parameter, baseline in zip(model.parameters(), reference.parameters()):
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(2)
                torch.testing.assert_close(parameter.grad, baseline.grad, atol=6e-6, rtol=1e-3)
        runtime = TeacherStudentRuntime(TrainerArgs(runtime_teacher_cache_mode="teacher_output"),
                                        model, model, torch.device("cpu"), MemoryDistillCache())
        runtime.context_parallel_group = model._context_parallel_group
        kwargs = dict(hidden_states=full, timestep=times, **conditions)
        batch = {"latents": full}
        first = runtime.launch_teacher(kwargs, batch, 0)
        assert not first.from_cache
        runtime.wait_teacher(first)
        key = runtime.build_cache_key(namespace="teacher_output", batch=batch, input_kwargs=kwargs, extra=None)
        if rank:
            runtime.distill_cache.invalidate(key)
        second = runtime.launch_teacher(kwargs, batch, 1)
        assert not second.from_cache  # A rank-local miss makes every rank execute.
        runtime.wait_teacher(second)
        third = runtime.launch_teacher(kwargs, batch, 2)
        assert third.from_cache
        runtime.wait_teacher(third)
        runtime.close()
        if variant == "routed":
            with pytest.raises(ValueError, match="identical routing"):
                model(full, times + rank * 1000, **conditions)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", ["routed", "image"])
def test_wan_routed_and_image_conditioning_and_cache_consensus(tmp_path, variant):
    pytest.importorskip("diffusers")
    mp.spawn(_conditional_worker, args=(str(tmp_path / "init"), variant), nprocs=2, join=True)
