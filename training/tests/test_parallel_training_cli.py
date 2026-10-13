"""Real torchrun entry, CPU staging, auto resume, and exported student identity."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_torchrun_model_parallel_elastic_auto_resume_and_export(tmp_path, tp, pp):
    pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from training.tests.test_context_parallel_training import _tiny_wan
    from training.model_adapter import load_diffusers_training_model
    from training.student_export import export_student
    root = Path(__file__).resolve().parents[2]
    teacher = _tiny_wan()
    base = tmp_path / "teacher"
    teacher.model.save_pretrained(base)
    with torch.no_grad():
        next(teacher.parameters()).add_(.1)
    torch.save(teacher.state_dict(), tmp_path / "student.pt")
    torch.save(torch.randn(2, 3, 2, 2), tmp_path / "latent.pt")
    torch.save(torch.randn(3, 8), tmp_path / "text.pt")
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"latent_path": "latent.pt", "text_embed_path": "text.pt",
                                 "num_frames": 3, "resolution": [2, 2]}] * 4))
    output = tmp_path / "run"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # A parent test runner may itself be launched by torchrun; this subprocess
    # is an independent local agent with a fresh rendezvous.
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE"):
        environment.pop(key, None)
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
        "--local_addr=127.0.0.1", "--rdzv_backend=c10d", "--rdzv_endpoint=127.0.0.1:0",
        "-m", "training.train_distill", "--teacher_model_path", str(base),
        "--student_model_path", str(tmp_path / "student.pt"), "--model_cls", "wan2.1",
        "--data_mode", "cached", "--data_json", str(data), "--cache_dir", str(tmp_path),
        "--output_dir", str(output), "--tp_size", str(tp), "--pp_size", str(pp),
        "--elastic_resume", "--elastic_auto_resume", "--runtime_teacher_offload", "nvme",
        "--use_ema", "--mixed_precision", "no", "--num_workers", "0", "--batch_size", "1",
        "--report_to", "none", "--log_every", "1", "--save_every", "100",
        "--required_transformers_version", transformers.__version__]
    for step in (1, 2):
        result = subprocess.run(command + ["--max_train_steps", str(step)], cwd=root,
            env=environment, capture_output=True, text=True, timeout=150)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (output / f"checkpoint-{step}/elastic_complete.json").is_file()
    state = torch.load(output / "checkpoint-2/trainer_state.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 2
    assert state["resume_state"]["elastic"]["sampler"]["global_cursor"] == 2
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    destination = tmp_path / "export"
    export_student(str(base), str(output), str(destination))
    exported = load_diffusers_training_model(destination)
    for name, value in exported.state_dict().items():
        torch.testing.assert_close(value, state["student_model"][name], atol=0, rtol=0)
