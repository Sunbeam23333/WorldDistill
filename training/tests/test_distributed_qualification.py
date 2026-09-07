import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.check_distributed_training import collective_cases, shared_output_directory, unsupported_combination


def test_collective_single_process_cpu_is_only_a_control():
    result = collective_cases(torch.device("cpu"))
    assert result["status"] == "passed" and len(result["latency_seconds"]) == 3


def test_evidence_directory_is_never_overwritten(tmp_path):
    shared_output_directory(tmp_path)
    with pytest.raises(RuntimeError, match="never overwritten"):
        shared_output_directory(tmp_path)


def test_unsupported_parallel_cells_are_not_qualified():
    fsdp = SimpleNamespace(parallel="fsdp", zero_stage=3, gradient_accumulation_steps=1, mixed_precision="auto")
    assert unsupported_combination("step_distill", fsdp) is None
    assert unsupported_combination("dmd_distill", fsdp)
    assert unsupported_combination("consistency_distill", fsdp)


@pytest.mark.skipif(torch.cuda.is_available(), reason="This is the explicit CPU/Gloo control job")
def test_actual_torchrun_two_rank_training_qualification(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except PermissionError:
        pytest.skip("Local loopback IPC forbidden; rerun with local socket permission")
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
               "--rdzv-backend=static", "--master-addr=127.0.0.1", f"--master-port={port}",
               "-m", "tools.check_distributed_training", "--device", "cpu", "--methods", "all",
               "--output-dir", str(tmp_path / "evidence")]
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "evidence/summary.json").read_text())
    assert report["status"] == "passed" and report["qualified"] is False
    assert len(report["ranks"]) == 2 and len(report["cases"]) == 7
    assert all(case["status"] == "passed" for case in report["cases"])
