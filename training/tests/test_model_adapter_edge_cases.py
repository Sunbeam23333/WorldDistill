import json
from types import SimpleNamespace

import pytest
import torch

from training.model_adapter import DiffusersTrainingAdapter, NoiseRoutedDenoiser, load_diffusers_training_model
from training.utils.batch_encoder import DiffusersRawBatchEncoder


def test_i2v_latent_channels_match_direct_wan_forward():
    diffusers = pytest.importorskip("diffusers")
    model = diffusers.WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=5,
                                           out_channels=2, text_dim=16, freq_dim=8, ffn_dim=32, num_layers=1)
    adapter = DiffusersTrainingAdapter(model)
    x, condition = torch.randn(1, 2, 3, 4, 4), torch.randn(1, 3, 3, 4, 4)
    timestep, text = torch.tensor([700.]), torch.randn(1, 3, 16)
    expected = model(torch.cat([x, condition], dim=1), timestep, text).sample
    actual = adapter(x, timestep, image_cond=condition, encoder_hidden_states=text)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    with pytest.raises(ValueError, match="input channels"):
        adapter(x, timestep, encoder_hidden_states=text)
    with pytest.raises(ValueError, match="shape B,T"):
        adapter(x, torch.ones(1, 2), image_cond=condition, encoder_hidden_states=text)


def test_real_hunyuan15_requires_all_encoder_and_image_conditions():
    diffusers = pytest.importorskip("diffusers")
    model = diffusers.HunyuanVideo15Transformer3DModel(in_channels=5, out_channels=2, num_attention_heads=2,
        attention_head_dim=8, num_layers=1, num_refiner_layers=1, text_embed_dim=8,
        text_embed_2_dim=6, image_embed_dim=4, rope_axes_dim=(2, 2, 4))
    adapter = DiffusersTrainingAdapter(model)
    x, timestep = torch.randn(1, 2, 2, 2, 2), torch.tensor([300.])
    # Actual T2V condition contract: a zero latent/mask branch and zero vision
    # tokens, not a smaller in_channels architecture bypassing those inputs.
    image_cond = torch.zeros(1, 3, 2, 2, 2)
    conditions = {"encoder_hidden_states": torch.randn(1, 3, 8), "encoder_attention_mask": torch.ones(1, 3),
                  "encoder_hidden_states_2": torch.randn(1, 2, 6), "encoder_attention_mask_2": torch.ones(1, 2),
                  "image_embeds": torch.zeros(1, 2, 4)}
    expected = model(torch.cat([x, image_cond], dim=1), timestep, **conditions).sample
    actual = adapter(x, timestep, image_cond=image_cond, **conditions)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    conditions.pop("image_embeds")
    with pytest.raises(ValueError, match="image_embeds"):
        adapter(x, timestep, image_cond=image_cond, **conditions)


def test_raw_time_preserving_vae_uses_identity_control_mapping():
    encoder = object.__new__(DiffusersRawBatchEncoder)
    encoder.device = torch.device("cpu")
    encoder.temporal_compression_ratio = 4
    encoder._encode_pixels = lambda x: torch.zeros(1, 2, 5, 1, 1)
    actions = torch.arange(5.).reshape(1, 5, 1)
    result = encoder.encode_batch({"pixel_values": torch.zeros(1, 3, 5, 4, 4), "actions": actions.clone()})
    torch.testing.assert_close(result["actions"], actions)


@pytest.mark.parametrize("mean,std", [([0., 0.], [1., float("nan")]), ([0., float("inf")], [1., 1.]), ([0., 0.], [1.])])
def test_invalid_channel_normalization_is_rejected(mean, std):
    encoder = object.__new__(DiffusersRawBatchEncoder)
    encoder.vae = SimpleNamespace(config=SimpleNamespace(latents_mean=mean, latents_std=std))
    with pytest.raises(ValueError, match="normalization"):
        encoder._normalize_latents(torch.ones(1, 2, 3, 1, 1))


def test_multi_encoder_tuple_must_not_be_silently_truncated():
    encoder = object.__new__(DiffusersRawBatchEncoder)
    encoder.device, encoder.dtype = torch.device("cpu"), torch.float32
    encoder.pipe = type("HunyuanVideo15Pipeline", (), {})()
    encoder.encode_prompt = lambda prompt, device: (torch.ones(1, 2, 3), torch.ones(1, 2))
    with pytest.raises(TypeError, match="invalid conditioning tuple"):
        encoder._encode_text_via_pipeline(["test"])


def test_dual_routing_uses_scheduler_timestep_scale(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    for name in ("transformer", "transformer_2"):
        model = diffusers.WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=2,
            out_channels=2, text_dim=16, freq_dim=8, ffn_dim=32, num_layers=1)
        model.save_pretrained(tmp_path / name)
    (tmp_path / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "WanTransformer3DModel"],
        "transformer_2": ["diffusers", "WanTransformer3DModel"], "boundary_ratio": 0.5}))
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler/scheduler_config.json").write_text(json.dumps({"num_train_timesteps": 2000}))
    model = load_diffusers_training_model(tmp_path)
    assert isinstance(model, NoiseRoutedDenoiser) and model.num_train_timesteps == 2000


def test_ltx_temporal_patch_noise_must_be_consistent():
    class IdentityTokens(torch.nn.Module):
        config = {"patch_size": 2, "patch_size_t": 2, "in_channels": 8}
        def forward(self, hidden_states, timestep, **kwargs):
            self.timestep = timestep
            return hidden_states
    IdentityTokens.__name__ = "LTXVideoTransformer3DModel"
    model = IdentityTokens()
    adapter = DiffusersTrainingAdapter(model)
    x = torch.randn(1, 1, 4, 4, 4)
    timestep = torch.tensor([[300., 300., 700., 700.]])
    torch.testing.assert_close(adapter(x, timestep), x)
    torch.testing.assert_close(model.timestep, torch.tensor([[300., 300., 300., 300., 700., 700., 700., 700.]]))
    with pytest.raises(ValueError, match="patches cannot mix"):
        adapter(x, torch.tensor([[300., 500., 700., 700.]]))


def test_native_dual_must_not_ignore_explicit_legacy_low_student():
    from training.train_distill import _resolve_native_dual_args
    high, low = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    native = NoiseRoutedDenoiser(high, low, 0.5)
    args = SimpleNamespace(use_dual_model=True, student_low_model="chosen-low.pt")
    with pytest.raises(ValueError, match="student_low_model"):
        _resolve_native_dual_args(args, native)
    assert args.use_dual_model
    args.student_low_model = None
    _resolve_native_dual_args(args, native)
    assert not args.use_dual_model
