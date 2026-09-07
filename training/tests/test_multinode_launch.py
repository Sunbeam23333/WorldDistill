"""Launch contract and real logical multi-node CPU/Gloo tests, not GPU certification."""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from training.distributed_config import positive_timeout, rank_environment, topology_rank_groups
from training.env_compat import validate_local_distributed_devices
from training.utils import distributed

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_train.sh"


def worker_env(**overrides):
    return {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500", **overrides}


@pytest.mark.parametrize("overrides", [
    {"LOCAL_RANK": "2"}, {"WORLD_SIZE": "3"}, {"RANK": "2"}, {"RANK": "-1"},
    {"MASTER_PORT": "0"}, {"MASTER_PORT": "70000"}, {"GROUP_RANK": "1"},
    {"WORLD_DISTILL_EXPECTED_WORLD_SIZE": "4"}, {"WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE": "1"},
    {"WORLD_DISTILL_FIXED_WORLD_SIZE": "0"}, {"TORCHELASTIC_RESTART_COUNT": "1"},
    {"GROUP_WORLD_SIZE": "2"},
])
def test_inconsistent_rank_environments_fail(overrides):
    with pytest.raises(ValueError):
        rank_environment(worker_env(**overrides))


def test_partial_rank_environment_and_missing_launch_fail():
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        rank_environment({"RANK": "0"})
    with pytest.raises(ValueError, match="Expected distributed"):
        rank_environment({"WORLD_DISTILL_EXPECTED_WORLD_SIZE": "2"})
    assert rank_environment({}) is None
    env = worker_env(TORCHELASTIC_RESTART_COUNT="3", WORLD_DISTILL_EXPECTED_WORLD_SIZE="2")
    assert rank_environment(env).world_size == 2


def topology():
    # Deliberately noncontiguous global rank assignment; agent/local identity
    # determines the shard groups, not rank // local_world_size.
    return [{"rank": rank, "world_size": 4, "local_world_size": 2,
             "local_rank": rank // 2, "node_id": f"agent:{rank % 2}", "hostname": f"host-{rank % 2}"}
            for rank in range(4)]


def test_node_aware_group_plan_handles_permuted_global_ranks():
    assert topology_rank_groups(topology()) == ([[0, 2], [1, 3]], [[0, 1], [2, 3]])


@pytest.mark.parametrize("mutation", ["heterogeneous", "duplicate", "hostname", "local_rank"])
def test_invalid_group_topologies_fail(mutation):
    workers = topology()
    if mutation == "heterogeneous": workers[0]["local_world_size"] = 1
    if mutation == "duplicate": workers[0]["rank"] = 1
    if mutation == "hostname": workers[0]["hostname"] = "wrong-node"
    if mutation == "local_rank": workers[0]["local_rank"] = 1
    with pytest.raises(ValueError):
        topology_rank_groups(workers)


@pytest.mark.parametrize("timeout", [0, -1, "nan", "inf"])
def test_invalid_timeout_fails(timeout):
    with pytest.raises(ValueError):
        positive_timeout(timeout)


def test_nccl_and_visible_device_checks(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NCCL"):
        validate_local_distributed_devices("nccl", 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="scheduler-visible"):
        validate_local_distributed_devices("nccl", 3)
    with pytest.raises(ValueError, match="CPU-only"):
        validate_local_distributed_devices("gloo", 2)
    assert validate_local_distributed_devices("auto", 2)["backend"] == "nccl"


@pytest.mark.parametrize("backend", ["auto", "nccl"])
@pytest.mark.parametrize("cuda_version,hip_version", [(None, "6.4"), (None, None), ("13.0", "6.4")])
def test_rocm_cuda_compatibility_names_do_not_imply_nvidia_support(monkeypatch, backend, cuda_version, hip_version):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(distributed, "rank_environment", lambda: None)
    with pytest.raises(RuntimeError, match="NVIDIA.*ROCm/RCCL"):
        validate_local_distributed_devices(backend, 2)
    with pytest.raises(RuntimeError, match="NVIDIA.*ROCm/RCCL"):
        distributed.setup_distributed(backend)
    # Direct FSDP callers cannot bypass the guard by initializing RCCL first.
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    with pytest.raises(RuntimeError, match="NVIDIA.*ROCm/RCCL"):
        distributed.wrap_model_fsdp(torch.nn.Linear(2, 2))


def test_explicit_gloo_on_gpu_build_with_devices_hidden_remains_supported(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_gloo_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed, "rank_environment", lambda: None)
    assert validate_local_distributed_devices("gloo", 2)["backend"] == "gloo"
    assert distributed.setup_distributed("gloo") == (0, 1)


def test_setup_passes_timeout_and_checks_initialized_state(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("WORLD_DISTILL_", "TORCHELASTIC_", "GROUP_")):
            monkeypatch.delenv(key)
    for key, value in worker_env().items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    calls = []
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    def gather(target, source):
        target[:] = [source, {**source, "rank": 1, "local_rank": 1}]
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(distributed, "_WORKER_TOPOLOGY", None)
    monkeypatch.setattr(distributed, "_PROCESS_GROUP_TIMEOUT", None)
    assert distributed.setup_distributed("gloo", 17) == (0, 2)
    assert calls[0]["timeout"] == datetime.timedelta(seconds=17)
    assert calls[0]["init_method"] == "env://"
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    with pytest.raises(ValueError, match="conflicts"):
        distributed.setup_distributed("gloo")


def test_fsdp_refuses_uninitialized_fallback(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="unsharded fallback"):
        distributed.wrap_model_fsdp(torch.nn.Linear(2, 2))


def test_hybrid_fsdp_rejects_single_node_topology(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    workers = [{"rank": rank, "world_size": 2, "local_world_size": 2, "local_rank": rank,
                "node_id": "agent:0", "hostname": "host0"} for rank in range(2)]
    monkeypatch.setattr(distributed, "_WORKER_TOPOLOGY", workers)
    with pytest.raises(ValueError, match=">=2 nodes"):
        distributed.hybrid_process_groups()


def dry_launch(*extra, env=None):
    child_env = {k: v for k, v in os.environ.items() if k not in {"RANK", "NODE_RANK", "NNODES", "RDZV_ENDPOINT", "RDZV_ID"}}
    child_env.update(env or {})
    return subprocess.run(["bash", str(SCRIPT), "--teacher_model", "/nonexistent/teacher",
                           "--data_json", "/nonexistent/data", "--dry_run", *extra],
                          cwd=ROOT, env=child_env, capture_output=True, text=True)


def test_dry_run_preserves_scheduler_mask_and_does_not_import_models(tmp_path):
    output = tmp_path / "not-created"
    result = dry_launch("--nnodes", "2", "--node_rank", "1", "--gpus", "4", "--rdzv_endpoint", "node0:29500",
                        "--rdzv_id", "job7", "--max_restarts", "3", "--dist_timeout", "41",
                        "--rdzv_timeout", "59", "--mixed_precision", "fp16", "--required_transformers_version", "5.16.0",
                        "--resume_from", "/saved/checkpoint-9", "--output_dir", str(output),
                        env={"CUDA_VISIBLE_DEVICES": "GPU-abcd,GPU-defg"})
    assert result.returncode == 0, result.stderr
    assert "GPU-abcd,GPU-defg" in result.stdout
    for part in ("--nnodes=2", "--nproc_per_node=4", "--node_rank=1", "--max_restarts=3", "--rdzv_conf=timeout=59",
                 "WORLD_DISTILL_DIST_TIMEOUT=41", "WORLD_DISTILL_EXPECTED_WORLD_SIZE=8",
                 "--mixed_precision fp16", "--required_transformers_version 5.16.0", "--resume_from /saved/checkpoint-9"):
        assert part in result.stdout
    assert not output.exists()
    assert "export CUDA_VISIBLE_DEVICES" not in SCRIPT.read_text()


@pytest.mark.parametrize("arguments", [
    ("--nnodes", "1:2"), ("--nnodes", "2"), ("--node_rank", "1"),
    ("--gpus", "0"), ("--dist_timeout", "nan"), ("--max_restarts", "-1"),
    ("--rdzv_endpoint", "node:0"), ("--rdzv_backend", "etcd"),
    ("--parallel", "fsdp", "--fsdp_strategy", "hybrid"),
    ("--parallel", "tp"), ("--parallel", "pp"), ("--sp_size", "0"),
])
def test_invalid_launch_arguments_fail(arguments):
    assert dry_launch(*arguments).returncode != 0


def test_slurm_one_agent_per_node_adapter():
    env = {"SLURM_NNODES": "2", "SLURM_NTASKS": "2", "SLURM_PROCID": "1", "SLURM_LOCALID": "0",
           "SLURM_JOB_ID": "1234", "CUDA_VISIBLE_DEVICES": "3,7"}
    result = dry_launch("--launcher", "slurm", "--gpus", "2", "--rdzv_endpoint", "node0:29500", env=env)
    assert result.returncode == 0, result.stderr
    assert "--nnodes=2" in result.stdout and "--node_rank=1" in result.stdout and "--rdzv_id=1234" in result.stdout
    result = dry_launch("--launcher", "slurm", "--nnodes", "3", "--rdzv_endpoint", "node0:29500", env=env)
    assert result.returncode != 0 and "conflicts" in result.stderr
    env.update(SLURM_NTASKS="4", SLURM_LOCALID="1")
    result = dry_launch("--launcher", "slurm", "--rdzv_endpoint", "node0:29500", env=env)
    assert result.returncode != 0 and "one srun task per GPU" in result.stderr


@pytest.mark.parametrize("backend,local_workers", [("static", 1), ("c10d", 2)])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_real_two_agent_cpu_launch_and_training(backend, local_workers):
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except PermissionError:
        pytest.skip("Loopback sockets blocked; rerun with IPC permission")
    env = os.environ.copy()
    for key in list(env):
        if key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE"} or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", WORLD_DISTILL_DIST_BACKEND="gloo",
               WORLD_DISTILL_DIST_TIMEOUT="30", WORLD_DISTILL_EXPECTED_WORLD_SIZE=str(2 * local_workers),
               WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE=str(local_workers), WORLD_DISTILL_FIXED_WORLD_SIZE="1",
               GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo", PROBE_HYBRID_GROUPS=str(int(local_workers > 1)))
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=2", f"--nproc_per_node={local_workers}",
               "--local_addr=127.0.0.1",
               f"--rdzv_backend={backend}", f"--rdzv_endpoint=127.0.0.1:{port}", f"--rdzv_id=probe-{port}",
               "--max_restarts=0", "--rdzv_conf=" + ("timeout=30" if backend == "static" else "join_timeout=30,last_call_timeout=1,read_timeout=30")]
    processes = []
    try:
        for node in range(2):
            processes.append(subprocess.Popen([*command, f"--node_rank={node}", "-m", "training.tests.distributed_launch_probe"],
                                              cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        outputs = [process.communicate(timeout=90)[0] for process in processes]
        assert [process.returncode for process in processes] == [0, 0], "\n".join(outputs)
        records = [json.loads(line.split("LAUNCH_PROBE=", 1)[1]) for output in outputs for line in output.splitlines()
                   if "LAUNCH_PROBE=" in line]
        assert {record["rank"] for record in records} == set(range(2 * local_workers))
        assert all(record["backend"] == "gloo" and record["local_world_size"] == local_workers for record in records)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
