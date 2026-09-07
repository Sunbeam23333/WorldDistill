"""Offline real-architecture CLI training; no checkpoint download or GPU claim."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_real_wan_cli_train_resume_and_export(tmp_path):
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
    assert any(not torch.equal(tensor, student.state_dict()[key]) for key, tensor in state["student_model"].items())
    resumed = subprocess.run(command + ["--max_train_steps", "3", "--resume_from", str(checkpoint.parent)],
                             cwd=root, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = export_student(str(base), str(output), str(tmp_path / "export"))
    assert result["global_step"] == 3
    assert (output / "host_manifest.json").is_file()
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 3
