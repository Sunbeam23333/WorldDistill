from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from training.model_adapter import DiffusersTrainingAdapter, NoiseRoutedDenoiser, load_diffusers_training_model
from training.utils.batch_encoder import DiffusersRawBatchEncoder


def tiny_wan():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=2,
                                         out_channels=2, text_dim=16, freq_dim=8, ffn_dim=32, num_layers=1)


def test_real_wan_forward_backward_and_exact_architecture_loading(tmp_path):
    torch.manual_seed(5)
    model = tiny_wan()
    model.save_pretrained(tmp_path)
    adapter = load_diffusers_training_model(tmp_path)
    assert type(adapter.model) is type(model)
    x = torch.randn(2, 2, 3, 4, 4)
    conditions = torch.randn(2, 4, 16)
    t = torch.tensor([800.0, 200.0])
    expected = model(x, t, conditions).sample
    actual = adapter(hidden_states=x, timestep=t, encoder_hidden_states=conditions)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    with pytest.raises(ValueError, match="actions"):
        adapter(hidden_states=x, timestep=t, encoder_hidden_states=conditions, actions=torch.ones(2, 3, 2))


def test_real_wan_per_frame_timesteps(tmp_path):
    adapter = DiffusersTrainingAdapter(tiny_wan())
    out = adapter(hidden_states=torch.randn(1, 2, 3, 4, 4), timestep=torch.tensor([[0.0, 500.0, 900.0]]),
                  encoder_hidden_states=torch.randn(1, 4, 16))
    assert out.shape == (1, 2, 3, 4, 4)
    out.sum().backward()


def test_both_wan_experts_survive_load_train_export_reload(tmp_path):
    from training.train_distill import _load_models
    from training.student_export import export_student
    base = tmp_path / "base"
    high, low = tiny_wan(), tiny_wan()
    high.save_pretrained(base / "transformer")
    low.save_pretrained(base / "transformer_2")
    (base / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"],
                                                     "transformer_2": ["diffusers", "WanTransformer3DModel"], "boundary_ratio": 0.5}))
    teacher, student = _load_models(str(base), "", "wan2.2_moe", "", torch.device("cpu"), torch.float32)
    assert isinstance(teacher, NoiseRoutedDenoiser) and isinstance(student, NoiseRoutedDenoiser)
    x, t, text = torch.randn(2, 2, 3, 4, 4), torch.tensor([800., 200.]), torch.randn(2, 4, 16)
    expected = torch.cat((high(x[:1], t[:1], text[:1]).sample, low(x[1:], t[1:], text[1:]).sample))
    torch.testing.assert_close(teacher(x, t, encoder_hidden_states=text), expected)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    # Perturb first so teacher/student do not start with a zero distillation gradient.
    with torch.no_grad():
        next(student.parameters()).add_(0.1)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (student(x, t, encoder_hidden_states=text) - expected.detach()).square().mean()
        loss.backward()
        assert any(p.grad is not None for p in student.high.parameters())
        assert any(p.grad is not None for p in student.low.parameters())
        optimizer.step()
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    torch.save({"student_model": student.state_dict(), "step": 2, "args": {"distill_method": "step_distill"}}, checkpoint / "trainer_state.pt")
    manifest = export_student(str(base), str(checkpoint), str(tmp_path / "export"), num_steps=4)
    assert manifest["global_step"] == 2 and set(manifest["components"]) == {"transformer", "transformer_2"}
    reloaded = load_diffusers_training_model(tmp_path / "export")
    torch.testing.assert_close(reloaded(x, t, encoder_hidden_states=text), student(x, t, encoder_hidden_states=text))


def test_exact_class_loader_rejects_unknown_architecture(tmp_path):
    pytest.importorskip("diffusers")
    (tmp_path / "config.json").write_text(json.dumps({"_class_name": "NotARealTransformer"}))
    with pytest.raises(ValueError, match="Unsupported"):
        load_diffusers_training_model(tmp_path)


def test_raw_vae_channel_normalization_and_temporal_controls():
    encoder = object.__new__(DiffusersRawBatchEncoder)
    encoder.device, encoder.dtype = torch.device("cpu"), torch.float32
    encoder.vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[1., 3.], latents_std=[2., 4.]))
    encoder.vae_scaling_factor = 1.0
    z = torch.tensor([5., 11.]).reshape(1, 2, 1, 1, 1)
    torch.testing.assert_close(encoder._normalize_latents(z), torch.full_like(z, 2.))
    encoder.temporal_compression_ratio = 4
    encoder._encode_pixels = lambda x: torch.zeros(1, 2, 3, 1, 1)
    batch = {"pixel_values": torch.zeros(1, 3, 9, 4, 4), "actions": torch.arange(9.).reshape(1, 9, 1),
             "camera_poses": torch.arange(9.).reshape(1, 9, 1)}
    result = encoder.encode_batch(batch)
    torch.testing.assert_close(result["actions"].flatten(), torch.tensor([0., 4., 8.]))


def test_multi_encoder_conditions_are_preserved():
    encoder = object.__new__(DiffusersRawBatchEncoder)
    encoder.device, encoder.dtype = torch.device("cpu"), torch.float32
    encoder.pipe = type("HunyuanVideoPipeline", (), {})()
    outputs = (torch.ones(1, 3, 4), torch.ones(1, 8), torch.ones(1, 3))
    encoder.encode_prompt = lambda prompt, device: outputs
    result = encoder._encode_text_via_pipeline(["test"])
    assert set(result) == {"encoder_hidden_states", "pooled_projections", "encoder_attention_mask"}
