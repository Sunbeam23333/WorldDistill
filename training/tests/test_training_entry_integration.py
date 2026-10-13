"""Offline real-architecture CLI training integration."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("method,offload", [
    ("step_distill", "none"), ("step_distill", "cpu"),
    ("step_distill", "nvme"), ("mean_flow_distill", "cpu"),
])
def test_real_wan_cli_train_resume_and_export(tmp_path, method, offload):
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from training.model_adapter import DiffusersTrainingAdapter
    from training.student_export import export_student
    root = Path(__file__).resolve().parents[2]
    torch.manual_seed(37)
    model = diffusers.WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=2,
        out_channels=2, text_dim=16, freq_dim=8, ffn_dim=32, num_layers=1)
    base = tmp_path / "teacher"
    model.save_pretrained(base)
    student = DiffusersTrainingAdapter(model)
    with torch.no_grad():
        next(student.parameters()).add_(0.1)
    torch.save(student.state_dict(), tmp_path / "student.pt")
    torch.save(torch.randn(2, 3, 4, 4), tmp_path / "latent.pt")
    torch.save(torch.randn(4, 16), tmp_path / "text.pt")
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps([{"latent_path": "latent.pt", "text_embed_path": "text.pt",
                                     "num_frames": 3, "resolution": [4, 4]}]))
    output = tmp_path / "run"
    command = [sys.executable, "-m", "training.train_distill", "--teacher_model_path", str(base),
        "--distill_method", method, "--runtime_teacher_offload", offload,
        "--student_model_path", str(tmp_path / "student.pt"), "--model_cls", "wan2.1", "--data_mode", "cached",
        "--data_json", str(manifest), "--cache_dir", str(tmp_path), "--output_dir", str(output),
        "--mixed_precision", "no", "--num_workers", "0", "--batch_size", "1", "--report_to", "none",
        "--log_every", "1", "--save_every", "100", "--required_transformers_version", transformers.__version__]
    first = subprocess.run(command + ["--max_train_steps", "2"], cwd=root, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint = output / "checkpoint-2" / "trainer_state.pt"
    assert checkpoint.is_file()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 2
    trained_base = {key.removeprefix("base_model."): value for key, value in state["student_model"].items()
                    if key.removeprefix("base_model.") in student.state_dict()}
    assert trained_base.keys() == student.state_dict().keys()
    assert any(not torch.equal(tensor, student.state_dict()[key]) for key, tensor in trained_base.items())
    resumed = subprocess.run(command + ["--max_train_steps", "3", "--resume_from", str(checkpoint.parent)],
                             cwd=root, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = export_student(str(base), str(output), str(tmp_path / "export"),
                            num_steps=4 if method == "mean_flow_distill" else None)
    assert result["global_step"] == 3
    assert (output / "host_manifest.json").is_file()
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 3


def test_real_ltx2_cli_joint_train_resume_and_export(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    cls = getattr(diffusers, "LTX2VideoTransformer3DModel", None)
    if cls is None:
        pytest.skip("Installed Diffusers has no LTX2 architecture")
    from training.model_adapter import LTX2TrainingAdapter, load_diffusers_training_model
    from training.student_export import export_student
    root = Path(__file__).resolve().parents[2]
    torch.manual_seed(17)
    model = cls(in_channels=2, out_channels=2, num_attention_heads=2, attention_head_dim=12,
        cross_attention_dim=24, audio_in_channels=4, audio_out_channels=4,
        audio_num_attention_heads=2, audio_attention_head_dim=12, audio_cross_attention_dim=24,
        caption_channels=8, num_layers=1, rope_double_precision=False)
    base = tmp_path / "teacher"
    model.save_pretrained(base)
    student = LTX2TrainingAdapter(model)
    with torch.no_grad():
        model.proj_out.bias.add_(0.1)
        model.audio_proj_out.bias.add_(0.2)
    initial = {key: value.clone() for key, value in student.state_dict().items()}
    torch.save(initial, tmp_path / "student.pt")
    torch.save(torch.randn(2, 2, 2, 2), tmp_path / "video.pt")
    torch.save(torch.randn(3, 4), tmp_path / "audio.pt")
    torch.save({"encoder_hidden_states": torch.randn(3, 8),
                "audio_encoder_hidden_states": torch.randn(3, 8)}, tmp_path / "conditions.pt")
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps([{"latent_path": "video.pt", "audio_latent_path": "audio.pt",
        "conditioning_path": "conditions.pt", "fps": 24, "num_frames": 2, "resolution": [2, 2]}]))
    output = tmp_path / "run"
    command = [sys.executable, "-m", "training.train_distill", "--teacher_model_path", str(base),
        "--distill_method", "joint_av_distill", "--runtime_teacher_offload", "cpu",
        "--student_model_path", str(tmp_path / "student.pt"), "--model_cls", "ltx2", "--data_mode", "cached",
        "--data_json", str(manifest), "--cache_dir", str(tmp_path), "--output_dir", str(output),
        "--mixed_precision", "no", "--num_workers", "0", "--batch_size", "1", "--report_to", "none",
        "--log_every", "1", "--save_every", "100", "--required_transformers_version", transformers.__version__]
    first = subprocess.run(command + ["--max_train_steps", "2"], cwd=root, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    resumed = subprocess.run(command + ["--max_train_steps", "3", "--resume_from", str(output / "checkpoint-2")],
                             cwd=root, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = torch.load(output / "checkpoint-3" / "trainer_state.pt", map_location="cpu", weights_only=False)
    for key in ("model.proj_out.bias", "model.audio_proj_out.bias"):
        assert not torch.equal(state["student_model"][key], initial[key])
    bundle = tmp_path / "export"
    export_student(str(base), str(output), str(bundle), num_steps=2)
    restored = load_diffusers_training_model(bundle)
    for key, value in state["student_model"].items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
