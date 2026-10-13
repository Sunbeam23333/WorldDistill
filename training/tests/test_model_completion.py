import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from training.model_adapter import (DiffusersTrainingAdapter, LTX2TrainingAdapter,
    attach_training_scheduler, load_diffusers_denoiser, load_diffusers_training_model, save_training_component)
from training.native_model_adapter import NativeTrainingAdapter
from training.student_export import export_student, sample_student_latents, read_student_manifest
from training.utils.batch_encoder import DiffusersRawBatchEncoder


class TinyActionDenoiser(nn.Module):
    config = {"out_channels": 2}

    def __init__(self, channels=2):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))
        self.action_proj = nn.Linear(3, channels, bias=False)

    def forward(self, video, noise_time, control, camera=None):
        # Native layout is B,T,C,H,W; actions materially change the prediction.
        return video * self.gain[None, None, :, None, None] + self.action_proj(control)[:, :, :, None, None]


@pytest.fixture
def native_dir(tmp_path, monkeypatch):
    save_file = pytest.importorskip("safetensors.torch").save_file
    module = ModuleType("worlddistill_test_native")
    module.build = TinyActionDenoiser
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = {"schema_version": 1, "factory": "worlddistill_test_native:build",
              "constructor_kwargs": {"channels": 2}, "weights": "model.safetensors",
              "latent_layout": "BTCHW", "input_map": {"hidden_states": "video", "timestep": "noise_time",
                                                       "actions": "control", "camera_poses": "camera"},
              "required_conditions": ["actions"], "model_config": {"out_channels": 2}}
    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "worlddistill_model.json").write_text(json.dumps(config))
    save_file(TinyActionDenoiser().state_dict(), str(directory / "model.safetensors"))
    return directory


def test_native_factory_requires_explicit_trust_before_import(native_dir):
    with pytest.raises(ValueError, match="allow_native_model_code"):
        load_diffusers_training_model(native_dir)


def test_native_flow_prediction_metadata_controls_step_target_and_trajectory_guard(native_dir):
    from training.trainer_args import TrainerArgs
    from training.trainers.step_distill_trainer import StepDistillTrainer
    from training.trainers.progressive_distill_trainer import ProgressiveDistillTrainer
    path = native_dir / "worlddistill_model.json"
    config = json.loads(path.read_text())
    config["prediction_type"] = "epsilon"
    path.write_text(json.dumps(config))
    teacher = load_diffusers_training_model(native_dir, allow_native_code=True)
    config["prediction_type"] = "velocity"
    path.write_text(json.dumps(config))
    student = load_diffusers_training_model(native_dir, allow_native_code=True)
    kwargs = dict(args=TrainerArgs(report_to="none", mixed_precision="no"), teacher_model=teacher,
        student_model=student, optimizer=torch.optim.AdamW(student.parameters()), lr_scheduler=None,
        train_dataloader=[], device=torch.device("cpu"))
    trainer = StepDistillTrainer(**kwargs)
    assert trainer.teacher_noise_objective.prediction_type == "epsilon"
    assert trainer.noise_objective.prediction_type == "velocity"
    noisy, epsilon, t = torch.randn(1, 2, 3, 2, 2), torch.randn(1, 2, 3, 2, 2), torch.tensor([500.])
    torch.testing.assert_close(trainer._teacher_target(epsilon, noisy, t), 2 * epsilon - 2 * noisy)
    with pytest.raises(ValueError, match="velocity"):
        ProgressiveDistillTrainer(**kwargs)
    for prediction in ("epsilon", "x0", "velocity"):
        config["prediction_type"] = prediction
        path.write_text(json.dumps(config))
        assert load_diffusers_training_model(native_dir, allow_native_code=True).training_prediction_type == prediction
    config["prediction_type"] = "v_prediction"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="flow prediction_type"):
        load_diffusers_training_model(native_dir, allow_native_code=True)


def test_native_real_weights_actions_gradient_and_roundtrip(native_dir, tmp_path):
    adapter = load_diffusers_training_model(native_dir, allow_native_code=True)
    x, actions, t = torch.randn(2, 2, 3, 2, 2), torch.randn(2, 3, 3), torch.ones(2)
    actual = adapter(x, t, actions=actions)
    expected = adapter.model(x.permute(0, 2, 1, 3, 4), t, actions).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, adapter(x, t, actions=torch.zeros_like(actions)))
    actual.square().mean().backward()
    assert adapter.model.action_proj.weight.grad.abs().sum() > 0
    saved = tmp_path / "copy"
    save_training_component(adapter, saved)
    restored = load_diffusers_denoiser(saved, allow_native_code=True)
    torch.testing.assert_close(actual, restored(x, t, actions=actions))


@pytest.mark.parametrize("bad", ["missing_action", "bad_time_axis", "ignored_key", "meanflow"])
def test_native_condition_contract_rejects_silent_drop(native_dir, bad):
    adapter = load_diffusers_training_model(native_dir, allow_native_code=True)
    inputs = {"actions": torch.randn(1, 3, 3)}
    if bad == "missing_action":
        inputs.clear()
    elif bad == "bad_time_axis":
        inputs["actions"] = torch.randn(1, 2, 3)
    elif bad == "ignored_key":
        inputs["actionss"] = torch.zeros(1)
    else:
        inputs["timestep_r"] = torch.zeros(1)
    with pytest.raises(ValueError):
        adapter(torch.randn(1, 2, 3, 2, 2), torch.ones(1), **inputs)


def test_native_weight_path_cannot_escape_descriptor(native_dir):
    path = native_dir / "worlddistill_model.json"
    config = json.loads(path.read_text())
    config["weights"] = "../outside.safetensors"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="in-directory"):
        load_diffusers_training_model(native_dir, allow_native_code=True)


def test_native_student_bundle_exact_state(native_dir, tmp_path):
    model = load_diffusers_training_model(native_dir, allow_native_code=True)
    with torch.no_grad():
        model.model.action_proj.weight.add_(0.5)
    checkpoint = tmp_path / "state.pt"
    torch.save({"student_model": model.state_dict(), "step": 4, "args": {"distill_method": "step_distill",
               "denoising_step_list": [900, 500]}}, checkpoint)
    bundle = tmp_path / "bundle"
    export_student(str(native_dir), str(checkpoint), str(bundle), allow_native_model_code=True)
    read_student_manifest(bundle)
    restored = load_diffusers_training_model(bundle, allow_native_code=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def raw_encoder(family):
    encoder = object.__new__(DiffusersRawBatchEncoder)
    encoder.device, encoder.dtype = torch.device("cpu"), torch.float32
    encoder.pipe = type(family, (), {})()
    encoder.vae_scaling_factor = 2.0
    encoder.pipe.vision_num_semantic_tokens, encoder.pipe.vision_states_dim = 3, 4
    return encoder


def test_hunyuan15_t2v_builds_native_zero_branches():
    encoder = raw_encoder("HunyuanVideo15Pipeline")
    x = torch.randn(2, 2, 3, 2, 2)
    out = encoder.encode_batch({"latents": x})
    assert out["image_cond"].shape == (2, 3, 3, 2, 2)
    assert out["image_embeds"].shape == (2, 3, 4)
    assert torch.count_nonzero(out["image_cond"]) == torch.count_nonzero(out["image_embeds"]) == 0


def test_hunyuan15_i2v_keeps_each_images_mode_and_no_future_leakage():
    pytest.importorskip("PIL")
    encoder = raw_encoder("HunyuanVideo15ImageToVideoPipeline")
    seen = []
    def encode(pixels):
        seen.append(pixels.clone())
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: pixels[:, :2],
            sample=lambda: pytest.fail("I2V reference should use VAE mode")))
    encoder.vae = SimpleNamespace(config=SimpleNamespace(), encode=encode)
    encoder.pipe.encode_image = lambda image, batch_size, device, dtype: torch.full((1, 3, 4), float(image.getpixel((0, 0))[0]))
    pixels = torch.ones(2, 3, 3, 2, 2)
    pixels[0, :, 0] = -1
    pixels[1, :, 0] = 0
    result = encoder.encode_batch({"latents": torch.zeros(2, 2, 3, 2, 2), "pixel_values": pixels})
    assert seen[0].shape[2] == 1
    assert result["image_cond"][0, 0, 0, 0, 0] == -2
    assert torch.count_nonzero(result["image_cond"][:, :, 1:]) == 0
    assert torch.all(result["image_cond"][:, -1, 0] == 1)
    assert torch.all(result["image_embeds"][0] == 0)
    assert torch.all(result["image_embeds"][1] == 128)


def test_ltx_uses_registered_normalization_buffers_and_scale():
    encoder = raw_encoder("LTXPipeline")
    vae = type("AutoencoderKLLTXVideo", (), {})()
    vae.config = SimpleNamespace()
    vae.latents_mean, vae.latents_std = torch.tensor([1., 3.]), torch.tensor([2., 4.])
    encoder.vae = vae
    x = torch.ones(1, 2, 2, 1, 1) * 5
    expected = torch.tensor([4., 1.]).reshape(1, 2, 1, 1, 1).expand_as(x)
    torch.testing.assert_close(encoder._normalize_latents(x), expected)


def test_ltx2_prompt_uses_connector_outputs_for_both_modalities():
    encoder = raw_encoder("LTX2Pipeline")
    encoder.tokenizer = SimpleNamespace(padding_side="left")
    encoder.encode_prompt = lambda prompt, device: (torch.ones(1, 3, 4), torch.ones(1, 3), None, None)
    encoder.pipe.connectors = lambda embeds, mask, padding_side: (embeds + 1, embeds + 2, mask)
    result = encoder._encode_text_via_pipeline(["hello"])
    assert torch.all(result["encoder_hidden_states"] == 2)
    assert torch.all(result["audio_encoder_hidden_states"] == 3)


def test_cached_audio_manifest_preserves_latents_and_fps(tmp_path):
    from training.data.video_dataset import CachedLatentDataset
    video, audio = torch.randn(2, 3, 2, 2), torch.randn(5, 4)
    torch.save(video, tmp_path / "video.pt")
    torch.save(audio, tmp_path / "audio.pt")
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps([{"latent_path": "video.pt", "audio_latent_path": "audio.pt", "fps": 30}]))
    result = CachedLatentDataset(str(manifest), cache_dir=str(tmp_path))[0]
    torch.testing.assert_close(result["audio_latents"], audio)
    assert result["fps"] == 30


def test_raw_audio_features_require_explicit_sampled_timeline(tmp_path, monkeypatch):
    from training.data.video_dataset import VideoDataset
    torch.save(torch.ones(1, 12, 4), tmp_path / "audio.pt")
    manifest = tmp_path / "raw.json"
    manifest.write_text(json.dumps([{"path": "video.mp4", "audio_feature_path": "audio.pt", "fps": 24}]))
    dataset = VideoDataset(str(manifest), video_dir=str(tmp_path))
    monkeypatch.setattr(dataset, "_load_video", lambda *a, **kw: (torch.zeros(3, 3, 2, 2), torch.tensor([0, 4, 8]), 9))
    with pytest.raises(ValueError, match="audio_alignment"):
        dataset[0]
    dataset.data[0]["audio_alignment"] = "sampled_video"
    assert dataset[0]["audio_pixel_values"].shape == (1, 12, 4)


def test_ltx2_audio_encoder_packs_and_normalizes_all_channels():
    encoder = raw_encoder("LTX2Pipeline")
    audio = torch.arange(24.).reshape(1, 2, 3, 4)
    encoder.pipe.audio_vae = SimpleNamespace(encode=lambda value: SimpleNamespace(latents=audio),
        latents_mean=torch.zeros(8), latents_std=torch.ones(8) * 2)
    batch = {"latents": torch.randn(1, 2, 3, 1, 1), "audio_pixel_values": torch.zeros(1, 1, 12, 8)}
    result = encoder.encode_batch(batch)
    torch.testing.assert_close(result["audio_latents"], audio.transpose(1, 2).flatten(2, 3) / 2)


def test_meanflow_component_and_full_export_roundtrip(native_dir, tmp_path):
    from training.utils.meanflow import MeanFlowTimeAdapter
    native = load_diffusers_training_model(native_dir, allow_native_code=True)
    model = MeanFlowTimeAdapter(native, latent_channels=2, hidden_dim=12)
    with torch.no_grad():
        model.time_conditioner[-1].weight.normal_()
    inputs = dict(hidden_states=torch.randn(1, 2, 3, 2, 2), timestep=torch.tensor([900.]),
                  timestep_r=torch.tensor([400.]), actions=torch.randn(1, 3, 3))
    directory = tmp_path / "meanflow"
    save_training_component(model, directory)
    restored = load_diffusers_denoiser(directory, allow_native_code=True)
    torch.testing.assert_close(restored(**inputs), model(**inputs))
    checkpoint = tmp_path / "meanflow.pt"
    torch.save({"student_model": model.state_dict(), "step": 4, "args": {"distill_method": "mean_flow_distill",
        "meanflow_latent_channels": 2, "meanflow_hidden_dim": 12}}, checkpoint)
    bundle = tmp_path / "meanflow_bundle"
    export_student(str(native_dir), str(checkpoint), str(bundle), num_steps=2, allow_native_model_code=True)
    restored = load_diffusers_training_model(bundle, allow_native_code=True)
    torch.testing.assert_close(restored(**inputs), model(**inputs))


def test_vp_schedule_uses_explicit_checkpoint_config():
    diffusers = pytest.importorskip("diffusers")
    class VP(nn.Module):
        training_noise_process = "vp"
    model = VP()
    config = dict(diffusers.CogVideoXDDIMScheduler(num_train_timesteps=20, prediction_type="v_prediction").config)
    config["_class_name"] = "CogVideoXDDIMScheduler"
    attach_training_scheduler(model, config)
    expected = diffusers.CogVideoXDDIMScheduler.from_config(config)
    torch.testing.assert_close(model.training_alphas_cumprod, expected.alphas_cumprod.float())
    assert model.training_prediction_type == "v_prediction" and model.num_train_timesteps == 20
    with pytest.raises(ValueError, match="explicit"):
        attach_training_scheduler(VP(), {})


def test_vp_routed_sampler_uses_saved_alphas_and_clean_endpoint():
    from training.model_adapter import NoiseRoutedDenoiser
    class KnownClean(nn.Module):
        training_noise_process = "vp"
        training_prediction_type = "sample"
        num_train_timesteps = 4
        config = {}
        def __init__(self, value):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(value))
            self.register_buffer("training_alphas_cumprod", torch.tensor([.9, .7, .4, .1]))
            self.times = []
        def forward(self, hidden_states, timestep):
            self.times.extend(timestep.tolist())
            return torch.ones_like(hidden_states) * self.value
    high, low = KnownClean(3.), KnownClean(2.)
    routed = NoiseRoutedDenoiser(high, low, .5, 4)
    actual = sample_student_latents(routed, torch.randn(1, 2, 2, 2, 2), num_steps=3, timesteps=[3, 2, 0])
    torch.testing.assert_close(actual, torch.full_like(actual, 2.))
    assert high.times == [3, 2] and low.times == [0]
    assert routed.training and high.training and low.training
    with pytest.raises(ValueError, match="shift=1"):
        sample_student_latents(routed, actual, num_steps=3, shift=3)
    with pytest.raises(ValueError, match="integer"):
        sample_student_latents(routed, actual, num_steps=2, timesteps=[2.5, 0])
    low.training_alphas_cumprod[0] = .8
    with pytest.raises(ValueError, match="identical"):
        NoiseRoutedDenoiser(high, low, .5, 4)


def test_flow_sampler_replays_exact_exported_time_points():
    class TimeVelocity(nn.Module):
        num_train_timesteps = 20
        def forward(self, hidden_states, timestep):
            return torch.ones_like(hidden_states) * timestep[:, None, None, None, None]
    latent = torch.zeros(1, 1, 1, 1, 1)
    actual = sample_student_latents(TimeVelocity(), latent, num_steps=4, timesteps=[20, 18, 15, 10])
    # dt*v at 20->18->15->10->0 on the normalized flow time scale.
    torch.testing.assert_close(actual, torch.full_like(latent, -13.45))


def test_native_vp_dual_export_keeps_full_schedule_and_both_predictions(native_dir, tmp_path):
    from training.model_adapter import NoiseRoutedDenoiser
    config_path = native_dir / "worlddistill_model.json"
    config = json.loads(config_path.read_text())
    config.update(noise_process="vp", prediction_type="v_prediction", num_train_timesteps=4,
                  alphas_cumprod=[.9, .7, .4, .1])
    config_path.write_text(json.dumps(config))
    high = load_diffusers_training_model(native_dir, allow_native_code=True)
    low = copy.deepcopy(high)
    with torch.no_grad():
        low.model.gain.add_(.3)
    inputs = {"hidden_states": torch.randn(2, 2, 3, 2, 2), "timestep": torch.tensor([3., 1.]),
              "actions": torch.randn(2, 3, 3)}
    expected = NoiseRoutedDenoiser(high, low, .5, 4)(**inputs)
    checkpoint = tmp_path / "trainer_state.pt"
    torch.save({"student_model": high.state_dict(), "step": 3, "args": {"distill_method": "step_distill",
        "use_dual_model": True, "denoising_step_list": [4, 3, 2, 1], "sample_shift": 1,
        "boundary_step_index": 2}}, checkpoint)
    torch.save(low.state_dict(), tmp_path / "student_low.pt")
    output = tmp_path / "dual_vp"
    manifest = export_student(str(native_dir), str(checkpoint), str(output), allow_native_model_code=True)
    restored = load_diffusers_training_model(output, allow_native_code=True)
    torch.testing.assert_close(restored(**inputs), expected)
    torch.testing.assert_close(restored.training_alphas_cumprod, high.training_alphas_cumprod)
    assert manifest["trained_sampling_timesteps"] == [3, 3, 2, 1]
    assert restored.training_noise_process == "vp" and restored.training_prediction_type == "v_prediction"
    sampled = sample_student_latents(restored, inputs["hidden_states"], num_steps=4,
        conditions={"actions": inputs["actions"]}, timesteps=manifest["trained_sampling_timesteps"])
    assert torch.isfinite(sampled).all() and sampled.shape == inputs["hidden_states"].shape


def tiny_ltx2():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "LTX2VideoTransformer3DModel", None)
    if cls is None:
        pytest.skip("LTX2 requires a Diffusers version providing its architecture")
    return cls(in_channels=2, out_channels=2, num_attention_heads=2, attention_head_dim=12,
        cross_attention_dim=24, audio_in_channels=4, audio_out_channels=4,
        audio_num_attention_heads=2, audio_attention_head_dim=12, audio_cross_attention_dim=24,
        caption_channels=8, num_layers=1, rope_double_precision=False)


def av_inputs():
    return dict(hidden_states=torch.randn(1, 2, 2, 2, 2), audio_hidden_states=torch.randn(1, 3, 4),
                timestep=torch.tensor([500.]), encoder_hidden_states=torch.randn(1, 3, 8),
                audio_encoder_hidden_states=torch.randn(1, 3, 8))


def test_real_ltx2_forward_both_heads_backward_export(tmp_path):
    adapter = LTX2TrainingAdapter(tiny_ltx2())
    inputs = av_inputs()
    output = adapter(**inputs)
    assert output["sample"].shape == inputs["hidden_states"].shape
    assert output["audio_sample"].shape == inputs["audio_hidden_states"].shape
    (output["sample"].square().mean() + output["audio_sample"].square().mean()).backward()
    assert adapter.model.proj_out.weight.grad.abs().sum() > 0
    assert adapter.model.audio_proj_out.weight.grad.abs().sum() > 0
    save_training_component(adapter, tmp_path)
    restored = load_diffusers_denoiser(tmp_path)
    after = restored(**inputs)
    for key in output:
        torch.testing.assert_close(output[key], after[key])


def test_joint_trainer_both_modalities_update_and_runtime_retains_audio(tmp_path):
    from training.trainer_args import TrainerArgs
    from training.trainers.joint_av_distill_trainer import JointAVDistillTrainer
    from training.runtime.teacher_student_runtime import TeacherStudentRuntime
    teacher = LTX2TrainingAdapter(tiny_ltx2())
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.model.proj_out.bias.add_(0.1)
        student.model.audio_proj_out.bias.add_(0.2)
    inputs = av_inputs()
    batch = {"latents": inputs.pop("hidden_states"), "audio_latents": inputs.pop("audio_hidden_states"),
             "encoder_hidden_states": inputs["encoder_hidden_states"],
             "audio_encoder_hidden_states": inputs["audio_encoder_hidden_states"]}
    args = TrainerArgs(distill_method="joint_av_distill", mixed_precision="no", report_to="none", output_dir=str(tmp_path))
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    runtime = TeacherStudentRuntime(args=args, teacher_model=teacher, student_model=student, device=torch.device("cpu"))
    trainer = JointAVDistillTrainer(args=args, teacher_model=teacher, student_model=student,
        optimizer=optimizer, lr_scheduler=None, train_dataloader=[], device=torch.device("cpu"), runtime=runtime)
    video_before, audio_before = student.model.proj_out.bias.detach().clone(), student.model.audio_proj_out.bias.detach().clone()
    loss = trainer._forward_and_loss(batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    optimizer.step()
    assert not torch.equal(video_before, student.model.proj_out.bias)
    assert not torch.equal(audio_before, student.model.audio_proj_out.bias)
    assert all(p.grad is None for p in teacher.parameters())


def test_joint_audio_output_cannot_be_silently_discarded():
    from training.utils.model_output import extract_prediction_tensor
    with pytest.raises(TypeError, match="joint_av_distill"):
        extract_prediction_tensor({"sample": torch.ones(1), "audio_sample": torch.ones(1)})


def test_exported_latent_sampler_tracks_both_modalities_and_restores_mode():
    class Velocity(nn.Module):
        supports_joint_av = True
        def forward(self, hidden_states, timestep, audio_hidden_states, audio_timestep):
            return {"sample": torch.ones_like(hidden_states), "audio_sample": torch.ones_like(audio_hidden_states) * 2}
    model = Velocity()
    model.train()
    output = sample_student_latents(model, torch.ones(1, 2, 3, 1, 1), audio_latents=torch.ones(1, 5, 4), num_steps=4)
    torch.testing.assert_close(output["latents"], torch.zeros(1, 2, 3, 1, 1))
    torch.testing.assert_close(output["audio_latents"], -torch.ones(1, 5, 4))
    assert model.training
