import json
from types import SimpleNamespace

import pytest
import torch

from tools.evaluate_generated import (action_accuracy, evaluate_fvd, frechet_feature_distance, trajectory_metrics)
from tools.sample_student import denormalize_video_latents, parse_latent_shape, sample_adapter, write_generated_media
from training.tests.test_model_completion import native_dir
from training.model_adapter import load_diffusers_training_model
from training.student_export import export_student


def test_frechet_distance_matches_analytic_mean_and_covariance():
    features = torch.tensor([[-1., 0], [0, 1], [1, -1]])
    assert frechet_feature_distance(features, features) == pytest.approx(0, abs=1e-7)
    assert frechet_feature_distance(features, features + 1) == pytest.approx(2, abs=1e-7)
    assert frechet_feature_distance(torch.tensor([[-1.], [1.]]), torch.tensor([[-2.], [2.]])) == pytest.approx(2)


@pytest.mark.parametrize("features", [torch.ones(1, 4), torch.full((2, 4), float("nan")), torch.ones(4)])
def test_invalid_fvd_features_rejected(features):
    with pytest.raises(ValueError):
        frechet_feature_distance(features, torch.ones(2, 4))


def test_feature_metadata_prevents_cross_extractor_comparisons(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    for name in ("a", "b"):
        save_file({"features": torch.tensor([[-1.], [1.]])}, str(tmp_path / f"{name}.safetensors"))
        videos = []
        for index in range(2):
            path = tmp_path / f"{name}{index}.mp4"
            path.write_bytes(b"video fixture")
            videos.append(path.name)
        (tmp_path / f"{name}.json").write_text(json.dumps({"features": f"{name}.safetensors", "videos": videos,
            "extractor": {"name": "i3d", "weights_sha256": name * 64, "preprocessing": {"frames": 16}}}))
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_fvd(tmp_path / "a.json", tmp_path / "b.json")
    record = json.loads((tmp_path / "b.json").read_text())
    record["extractor"]["weights_sha256"] = "a" * 64
    (tmp_path / "b.json").write_text(json.dumps(record))
    assert evaluate_fvd(tmp_path / "a.json", tmp_path / "b.json")["fvd"] == pytest.approx(0)


def test_trajectory_mask_final_valid_frame_and_coordinate_units():
    target = torch.zeros(2, 3, 2)
    prediction = torch.tensor([[[3., 4], [0, 0], [9, 9]], [[0., 2], [0, 4], [0, 6]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    report = trajectory_metrics(prediction, target, valid)
    assert report["ade"] == pytest.approx(17 / 5)
    assert report["fde"] == pytest.approx(3)
    assert report["valid_points"] == 5
    with pytest.raises(ValueError, match="at least one"):
        trajectory_metrics(prediction, target, torch.zeros_like(valid))


def test_action_accuracy_requires_valid_labels():
    logits = torch.tensor([[1., 0], [1., 2], [2., 1]])
    assert action_accuracy(logits, torch.tensor([0, 1, 1]))["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="vocabulary"):
        action_accuracy(logits, torch.tensor([0, 1, 2]))


def test_adapter_sampling_cli_saves_generated_latents(native_dir, tmp_path):
    from safetensors.torch import load_file, save_file
    model = load_diffusers_training_model(native_dir, allow_native_code=True)
    checkpoint = tmp_path / "trainer.pt"
    torch.save({"student_model": model.state_dict(), "step": 2, "args": {
        "distill_method": "step_distill", "denoising_step_list": [900, 400]}}, checkpoint)
    bundle = tmp_path / "bundle"
    manifest = export_student(str(native_dir), str(checkpoint), str(bundle), allow_native_model_code=True)
    conditions = tmp_path / "controls.safetensors"
    save_file({"actions": torch.ones(1, 3, 3)}, str(conditions))
    args = SimpleNamespace(bundle=str(bundle), latent_shape="1,2,3,2,2", dtype="float32", device="cpu",
        allow_native_model_code=True, conditions=None, conditioning_tensors=str(conditions), prompt=None,
        image_path=None, seed=42, audio_latent_shape=None, num_steps=2, shift=1., fps=16,
        decoder_factory=None, decode_timestep=None, decode_noise_scale=None)
    output = tmp_path / "sample.safetensors"
    report = sample_adapter(args, manifest, output)
    tensor = load_file(str(output))["latents"]
    assert tensor.shape == (1, 2, 3, 2, 2) and torch.isfinite(tensor).all()
    assert report["artifact_type"] == "normalized_generated_latents"
    assert report["latency_scope"] == "denoising_only"
    assert report["conditioning_sha256"] and report["output_sha256"]


def test_latent_shape_and_inverse_vae_normalization():
    with pytest.raises(ValueError, match="batch size 1"):
        parse_latent_shape("2,2,3,4,4", 5)
    vae = type("AutoencoderKLLTXVideo", (), {})()
    vae.config = SimpleNamespace(scaling_factor=2.)
    vae.latents_mean, vae.latents_std = torch.tensor([1., 3.]), torch.tensor([2., 4.])
    torch.testing.assert_close(denormalize_video_latents(torch.ones(1, 2, 1, 1, 1), vae).flatten(), torch.tensor([2., 5.]))


def test_audiovisual_writer_passes_audio_to_encoder(monkeypatch, tmp_path):
    pytest.importorskip("diffusers")
    import diffusers.utils
    if not hasattr(diffusers.utils, "encode_video"):
        pytest.skip("Audiovisual encode_video requires a recent Diffusers release")
    calls = []
    monkeypatch.setattr(diffusers.utils, "encode_video", lambda *args, **kwargs: calls.append((args, kwargs)))
    audio = torch.ones(1, 8)
    write_generated_media(["frame"], tmp_path / "out.mp4", fps=24, audio=audio, sample_rate=24000)
    assert calls[0][1]["audio_sample_rate"] == 24000
    torch.testing.assert_close(calls[0][1]["audio"], audio)
    with pytest.raises(ValueError, match="sample rate"):
        write_generated_media(["frame"], tmp_path / "out.mp4", fps=24, audio=audio)
