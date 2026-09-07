"""Real per-rank collectives, training and checkpoint/restart smoke.

Run with torchrun on every allocated node, e.g. --nnodes=2 --nproc-per-node=8
--node-rank=... --master-addr=... --master-port=... -m tools.check_distributed_training
--output-dir /shared/unique-run --device cuda. No SSH, job allocation, model
download or fabricated scaling result is performed. Tiny synthetic denoisers
test execution semantics; they are NOT pretrained video/world-model benchmarks.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from training.trainer_args import TrainerArgs
from training.trainers import TRAINER_REGISTRY
from training.utils.distributed import setup_distributed, cleanup_distributed


class TinyDenoiser(torch.nn.Module):
    """Deliberately synthetic, but uses real AMP-eligible CUDA convolutions."""
    def __init__(self, scale):
        super().__init__()
        self.projection = torch.nn.Conv3d(2, 2, 1)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[0, 0] = scale
            self.projection.weight[1, 1] = scale
            self.projection.bias.fill_(0.01)

    def forward(self, hidden_states, timestep, **conditions):
        return self.projection(hidden_states)


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def shared_output_directory(root: Path):
    rank = dist.get_rank() if dist.is_initialized() else 0
    token, error = None, None
    if rank == 0:
        try:
            if root.exists() and any(root.iterdir()):
                raise FileExistsError("Use a new empty --output-dir; existing evidence is never overwritten")
            root.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            (root / ".shared-storage-probe").write_text(token)
        except Exception as exc:
            error = repr(exc)
    leader = gather({"token": token, "error": error})[0]
    if leader["error"]:
        raise RuntimeError(leader["error"])
    try:
        observed = (root / ".shared-storage-probe").read_text()
        if observed != leader["token"]:
            raise ValueError("Output path does not refer to the same shared storage")
        local = None
    except Exception as exc:
        local = repr(exc)
    errors = gather(local)
    if any(errors):
        raise RuntimeError(f"Every rank must read the rank-zero checkpoint directory: {errors}")


def collective_cases(device, iterations=3):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    value = torch.tensor(float(rank + 1), device=device)
    if dist.is_initialized():
        dist.all_reduce(value)
    expected = world * (world + 1) / 2
    torch.testing.assert_close(value, torch.tensor(float(expected), device=device), rtol=0, atol=0)
    samples = []
    # Deliberately a small correctness/latency case, not a fabric-bandwidth test.
    for _ in range(iterations):
        if dist.is_initialized():
            dist.barrier()
        buffer = torch.full((4096,), float(rank), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if dist.is_initialized():
            dist.all_reduce(buffer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
        torch.testing.assert_close(buffer, torch.full_like(buffer, world * (world - 1) / 2), rtol=0, atol=0)
    return {"status": "passed", "bytes_per_rank": 4096 * 4, "latency_seconds": samples,
            "interpretation": "small all-reduce correctness; not NCCL-tests bus bandwidth or scaling efficiency"}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unsupported_combination(method, options):
    if options.parallel != "ddp" and method in {"progressive_distill", "adversarial_distill", "dmd_distill"}:
        return "This trainer's auxiliary/stage transitions currently require replicated DDP"
    if method == "consistency_distill" and (options.parallel == "fsdp" or
            (options.parallel == "deepspeed" and options.zero_stage == 3)):
        return "A sharded EMA target is not implemented for this combination"
    if method in {"adversarial_distill", "dmd_distill"} and options.gradient_accumulation_steps != 1:
        return "Auxiliary optimizer update frequency currently requires accumulation=1"
    if options.mixed_precision == "fp16" and (options.parallel == "fsdp" or
            method in {"adversarial_distill", "dmd_distill"}):
        return "The required sharded/auxiliary GradScaler contract is not implemented"
    return None


def check_replicas(trainer):
    if not dist.is_initialized():
        return
    errors = []
    for model in (trainer.student_model, getattr(trainer, "fake_score_model", None),
                  getattr(trainer, "discriminator", None)):
        if model is None:
            continue
        context = contextlib.nullcontext()
        if trainer.parallel_mode == "deepspeed":
            import deepspeed
            if trainer.args.deepspeed_stage == 3:
                context = deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None)
        with context:
            if trainer.parallel_mode == "fsdp":
                from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
                state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=False))
            else:
                state = trainer._unwrap_model(model).state_dict()
            for name, value in state.items():
                reference = value.detach().clone()
                dist.broadcast(reference, src=0)
                try:
                    torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
                except AssertionError:
                    errors.append(name)
    failures = gather(errors)
    if any(failures):
        raise AssertionError(f"Replicated parameters/buffers disagree across ranks: {failures}")


def assert_state_close(actual, expected, path="state"):
    """Compare optimizer/EMA/RNG/cursor trees, not just consolidated weights."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise AssertionError(f"{path}: expected a tensor")
        if expected.is_floating_point() and not (bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())):
            raise AssertionError(f"{path}: nonfinite tensor state")
        torch.testing.assert_close(actual, expected, atol=1e-8 if expected.is_floating_point() else 0,
                                   rtol=1e-4 if expected.is_floating_point() else 0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: state keys differ")
        for key in expected:
            assert_state_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError(f"{path}: state sequence differs")
        for index, value in enumerate(expected):
            assert_state_close(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=1e-4, atol=1e-8, equal_nan=False):
            raise AssertionError(f"{path}: scalar differs")
    elif (type(expected).__module__ == "deepspeed.runtime.fp16.loss_scaler" and
          type(expected).__name__ in {"LossScaler", "DynamicLossScaler", "LossScalerBase"}):
        # ZeRO-1/2 stores a scaler object, whose default equality is identity.
        # Compare its persistent fields, without treating arbitrary objects as
        # equal merely because they expose an empty __dict__.
        if type(actual) is not type(expected):
            raise AssertionError(f"{path}: loss-scaler types differ")
        assert_state_close(vars(actual), vars(expected), f"{path}.loss_scaler")
    elif actual != expected:
        raise AssertionError(f"{path}: value differs")


def checkpoint_payload(path, parallel):
    if parallel != "deepspeed":
        state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
        persistent = {key: state[key] for key in ("optimizer", "lr_scheduler", "scaler", "resume_state", "epoch", "step")}
        if "ema" in state:
            persistent["ema"] = state["ema"]
        for sidecar in path.glob("*.pt"):
            if sidecar.name != "trainer_state.pt":
                persistent[sidecar.name] = torch.load(sidecar, map_location="cpu", weights_only=False)
        return state["student_model"], persistent
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    weights = get_fp32_state_dict_from_zero_checkpoint(str(path))
    tag = (path / "latest").read_text().strip()
    persistent = {}
    for artifact in (path / tag).glob("*.pt"):
        value = torch.load(artifact, map_location="cpu", weights_only=False)
        if artifact.name.endswith("optim_states.pt"):
            persistent[artifact.name] = value
        elif artifact.name.endswith("model_states.pt"):
            persistent[artifact.name] = {key: value[key] for key in ("step", "epoch", "resume_state", "lr_scheduler")}
            for key in ("ema", "torch_amp_scaler_by_rank", "best_loss", "skipped_steps", "global_steps", "global_samples", "dp_world_size"):
                if key in value:
                    persistent[artifact.name][key] = value[key]
    # Method-specific EMA/critic checkpoints live alongside the engine tag,
    # not inside it. Consolidated student weights alone cannot cover them.
    for sidecar in path.glob("*.pt"):
        persistent[sidecar.name] = torch.load(sidecar, map_location="cpu", weights_only=False)
    if not persistent or not any(name.endswith("optim_states.pt") for name in persistent):
        raise ValueError("DeepSpeed optimizer shards are missing; weight-only equality cannot qualify resume")
    return weights, persistent


def train_resume_case(method, output, device, options):
    rank = dist.get_rank() if dist.is_initialized() else 0
    def construct(name, steps, resume=""):
        # Initialization identical across ranks, data/noise stream rank-specific.
        _seed(431)
        teacher = TinyDenoiser(0.8).to(device)
        student = TinyDenoiser(0.4)
        if options.parallel == "ddp":
            student.to(device)
        args = TrainerArgs(
            distill_method=method, output_dir=str(output / name), parallel_mode=options.parallel,
            mixed_precision=options.mixed_precision, fsdp_shard_strategy=options.fsdp_strategy,
            deepspeed_stage=options.zero_stage, cpu_offload=options.cpu_offload,
            max_train_steps=steps, learning_rate=0.001, warmup_steps=0,
            lr_scheduler="constant", resume_from=resume,
            report_to="none", log_every=1, save_every=100, gradient_checkpointing=False,
            gradient_accumulation_steps=options.gradient_accumulation_steps,
            batch_size=1, num_workers=0, window_size=4, overlap_frames=2,
            denoising_steps_per_frame=2, curriculum_training=False,
            temporal_context_size=2, memory_frames=2,
            progressive_stages=[8, 4, 2], progressive_stage_steps=2,
            dmd_variant="dmd2", dmd_student_steps=2, dmd_fake_score_updates=1, dmd_teacher_steps=2,
            dmd_latent_channels=2, dmd_disc_hidden_dim=8, dmd_disc_num_blocks=1,
            adversarial_latent_channels=2, adversarial_disc_hidden_dim=8, adversarial_disc_num_blocks=1,
        )
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
        generator = torch.Generator().manual_seed(601 + rank)
        loader = [{"latents": torch.randn(1, 2, 6, 8, 8, generator=generator)} for _ in range(3)]
        trainer = TRAINER_REGISTRY[method](args=args, teacher_model=teacher, student_model=student,
            optimizer=optimizer, lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            train_dataloader=loader, device=device)
        _seed(731 + rank)
        return trainer

    def run(name, steps, resume=""):
        trainer = construct(name, steps, resume)
        trainer.train()
        check_replicas(trainer)
        checkpoint = output / name / f"checkpoint-{steps}"
        selected = trainer.args.mixed_precision
        del trainer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return checkpoint, selected

    reference, precision = run("continuous", 4)
    partial, _ = run("interrupted", 2)
    restored, _ = run("resumed", 4, str(partial))
    if dist.is_initialized():
        dist.barrier()
    # FSDP consolidates to trainer_state; DeepSpeed needs its public checkpoint
    # reader for fp32 parameters, not a misleading rank-local shard comparison.
    expected, expected_state = checkpoint_payload(reference, options.parallel)
    actual, actual_state = checkpoint_payload(restored, options.parallel)
    baseline, _ = checkpoint_payload(partial, options.parallel)
    assert_state_close(actual_state, expected_state)
    if set(expected) != set(actual):
        raise AssertionError("Continuous/resumed checkpoint parameter names differ")
    # Compare the actual update, not wide relative tolerances on 0.4-valued
    # initial weights that could hide the entire 0.001-sized optimizer update.
    atol, rtol = 1e-8, 1e-3
    error = 0.0
    largest_update = 0.0
    for key in expected:
        if actual[key].is_floating_point():
            if not all(bool(torch.isfinite(state[key]).all()) for state in (expected, actual, baseline)):
                raise AssertionError(f"Nonfinite student parameter: {key}")
            expected_delta = expected[key].float() - baseline[key].float()
            actual_delta = actual[key].float() - baseline[key].float()
            if not (bool(torch.isfinite(expected_delta).all()) and bool(torch.isfinite(actual_delta).all())):
                raise AssertionError(f"Nonfinite student update: {key}")
            torch.testing.assert_close(actual_delta, expected_delta, atol=atol, rtol=rtol)
            largest_update = max(largest_update, expected_delta.abs().max().item())
            error = max(error, (actual[key].float() - expected[key].float()).abs().max().item())
        else:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    if largest_update <= 1e-8 or all(torch.equal(actual[key], baseline[key]) for key in actual):
        raise AssertionError("No student parameter changed after resume")
    return {"status": "passed", "method": method, "precision": precision, "atol": atol, "rtol": rtol,
            "max_parameter_error": error, "max_reference_update": largest_update,
            "optimizer_rng_auxiliary_state_checked": True, "continuous_steps": 4, "interruption_step": 2,
            "scope": "synthetic denoiser optimizer/save/resume, not pretrained model quality"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--parallel", choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    parser.add_argument("--methods", default="all", help="Comma-separated registered trainers, or all")
    parser.add_argument("--mixed-precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--fsdp-strategy", choices=["full", "hybrid"], default="full")
    parser.add_argument("--zero-stage", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120, help="Process-group timeout in seconds")
    options = parser.parse_args()
    methods = list(TRAINER_REGISTRY) if options.methods == "all" else options.methods.split(",")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(TRAINER_REGISTRY):
        parser.error("--methods must select distinct registered trainers")
    if options.gradient_accumulation_steps < 1 or options.timeout < 1:
        parser.error("gradient accumulation and timeout must be positive")
    if options.device == "cpu" and (options.parallel != "ddp" or torch.cuda.is_available()):
        parser.error("CPU validation supports DDP/Gloo only; on a GPU host explicitly hide GPUs for the CPU control run")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = {"schema_version": 1, "scope": "distributed synthetic training smoke", "qualified": False,
              "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cases": []}
    rank = int(os.environ.get("RANK", "0"))
    output_ready = False
    try:
        if options.device == "cuda" and (not torch.cuda.is_available() or not torch.version.cuda):
            raise RuntimeError("NVIDIA CUDA unavailable; no GPU training cases ran")
        rank, world = setup_distributed(backend="nccl" if options.device == "cuda" else "gloo", timeout_seconds=options.timeout)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank) if options.device == "cuda" else torch.device("cpu")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        shared_output_directory(options.output_dir)
        output_ready = True
        metadata = {"rank": rank, "local_rank": local_rank, "world_size": world, "hostname": platform.node(),
                    "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device),
                    "python": platform.python_version()}
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            metadata.update(gpu=properties.name, capability=list(torch.cuda.get_device_capability(device)),
                            total_memory_bytes=properties.total_memory, uuid=str(getattr(properties, "uuid", "unavailable")))
        report["ranks"] = gather(metadata)
        report["launch"] = vars(options).copy()
        report["launch"]["output_dir"] = str(options.output_dir)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        report["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        report["collectives"] = gather(collective_cases(device))
        for method in methods:
            # Unsupported method/strategy combinations are retained as failures,
            # not quietly omitted from an `all` request or labelled qualified.
            unsupported = unsupported_combination(method, options)
            if unsupported:
                report["cases"].append({"method": method, "status": "unsupported", "reason": unsupported})
                continue
            try:
                local_result = train_resume_case(method, options.output_dir / method, device, options)
            except Exception as exc:
                local_result = {"status": "failed", "error": repr(exc)}
            results = gather(local_result)
            status = "passed" if all(result["status"] == "passed" for result in results) else "failed"
            report["cases"].append({"method": method, "status": status, "ranks": results})
            if status == "failed":
                break  # A device failure may poison the process group.
        statuses = {case["status"] for case in report["cases"]}
        report["status"] = "failed" if "failed" in statuses else "partial" if "unsupported" in statuses else "passed"
        report["qualified"] = options.device == "cuda" and world >= 2 and report["status"] == "passed"
        report["qualification_scope"] = {"world_size": world, "physical_hosts_observed": len({row["hostname"] for row in report["ranks"]}),
                                          "single_rank_control": world == 1,
                                          "note": "qualifies only selected synthetic multi-rank checks, never all features/cards"}
        report["not_qualified"] = ["pretrained model output/quality", "bandwidth/scaling efficiency", "unrequested methods/strategies", "other devices"]
    except Exception as exc:
        report.update(status="failed" if report["cases"] or output_ready else "unavailable", error=repr(exc))
    finally:
        # Per-rank failure reports survive even if a peer failed inside a
        # collective. Process-group timeout bounds the remaining rank wait.
        if output_ready:
            (options.output_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            if rank == 0:
                (options.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0:
            print(json.dumps(report, indent=2))
        cleanup_distributed()
    return 0 if report.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
