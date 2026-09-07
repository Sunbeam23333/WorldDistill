"""Real-device attention qualification. Missing/skipped kernels never pass.

Example: python tools/check_attention_kernels.py --backends torch_sdpa,flash_attn2
         --mode both --output results/a100_attention.json
Exit 0: every requested backend passed the requested mode; 1: partial/failure;
2: no NVIDIA CUDA device. This is an operator smoke, not a model benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cuda_compat import import_attention_callable, inspect_torch_cuda, profile_cuda_device


BACKENDS = ("torch_sdpa", "flash_attn2", "flash_attn3", "flash_attn4", "sage_attn2", "sage_attn3")


def _versions():
    versions = {}
    for package in ("torch", "triton", "flash-attn", "flash-attn-3", "flash-attn-4", "sageattention", "sageattn3"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _reference():
    spec = importlib.util.spec_from_file_location("worlddistill_attention_reference", ROOT / "inference/lightx2v/utils/attention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernel(backend, profile):
    if backend == "torch_sdpa":
        return None
    if backend == "sage_attn2":
        module = importlib.import_module("sageattention")
        names = ("sageattn_qk_int8_pv_fp16_triton", "sageattn") if profile.capability == (8, 9) else ("sageattn", "sageattn_qk_int8_pv_fp16_triton")
        return next((getattr(module, name) for name in names if callable(getattr(module, name, None))), None)
    return import_attention_callable(backend)


def _run_candidate(backend, fn, ref, q, k, v, cq, ck, causal):
    import torch
    if backend == "torch_sdpa":
        return ref.attention(q, k, v, backend="torch_sdpa", cu_seqlens_q=cq,
                             cu_seqlens_k=ck, causal=causal)
    shape = (*q.shape[:-1], v.shape[-1])
    qp, kp, vp, cq, ck, mq, mk = ref.pack_qkv(q, k, v, cq, ck)
    if backend.startswith("flash_"):
        options = {"causal": causal}
        if backend == "flash_attn2":
            options["dropout_p"] = 0.0
        if backend == "flash_attn4":
            out = fn(qp, kp, vp, cu_seqlens_q=cq, cu_seqlens_k=ck,
                     max_seqlen_q=mq, max_seqlen_k=mk, **options)
        else:
            out = fn(qp, kp, vp, cq, ck, mq, mk, **options)
        return out.reshape(shape)
    qs, ks = cq.cpu().tolist(), ck.cpu().tolist()
    outputs = []
    for qa, qb, ka, kb in zip(qs, qs[1:], ks, ks[1:]):
        query, key, value = qp[qa:qb].unsqueeze(0), kp[ka:kb].unsqueeze(0), vp[ka:kb].unsqueeze(0)
        repeats = query.shape[2] // key.shape[2]
        key, value = (x.repeat_interleave(repeats, dim=2) for x in (key, value))
        if backend == "sage_attn2":
            out = fn(query, key, value, tensor_layout="NHD", is_causal=causal)
        else:
            out = fn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=causal).transpose(1, 2)
        outputs.append(out.squeeze(0))
    return torch.cat(outputs).reshape(shape)


def _error(actual, expected):
    import torch
    if not torch.isfinite(actual).all():
        return {"finite": False, "max_abs": None, "relative_l2": None}
    delta = (actual.float() - expected.float())
    return {"finite": True, "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-12))}


def _case(backend, fn, ref, device, dtype, name, q_lengths, k_lengths, causal, packed, backward):
    import torch
    generator = torch.Generator(device=device).manual_seed(23333)
    qshape = (sum(q_lengths), 4, 64) if packed else (len(q_lengths), q_lengths[0], 4, 64)
    kshape = (sum(k_lengths), 2, 64) if packed else (len(k_lengths), k_lengths[0], 2, 64)
    q, k, v = (torch.randn(shape, device=device, dtype=dtype, generator=generator).requires_grad_(backward)
               for shape in (qshape, kshape, kshape))
    cq = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32) if packed else None
    ck = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32) if packed else None
    # Force the math backend for a reference independent of fused dispatch.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    qr, kr, vr = (x.detach().float().requires_grad_(backward) for x in (q, k, v))
    with sdpa_kernel(SDPBackend.MATH):
        expected = ref.attention(qr, kr, vr, backend="torch_sdpa", cu_seqlens_q=cq, cu_seqlens_k=ck, causal=causal)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = _run_candidate(backend, fn, ref, q, k, v, cq, ck, causal)
    torch.cuda.synchronize(device)
    row = {"case": name, "dtype": str(dtype), "q_shape": list(q.shape), "k_shape": list(k.shape),
           "q_lengths": q_lengths, "k_lengths": k_lengths, "causal": causal,
           "forward_ms": (time.perf_counter() - started) * 1000, "forward": _error(actual, expected)}
    tolerance = .08 if backend.startswith("sage_") else .025
    row["relative_l2_tolerance"] = tolerance
    row["forward"]["passed"] = row["forward"]["finite"] and row["forward"]["relative_l2"] <= tolerance
    if backward:
        upstream = torch.randn(actual.shape, device=device, dtype=dtype, generator=generator)
        actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
        expected_grads = torch.autograd.grad(expected, (qr, kr, vr), upstream.float())
        torch.cuda.synchronize(device)
        row["backward"] = {name: _error(a, b) for name, a, b in zip(("q", "k", "v"), actual_grads, expected_grads)}
        row["backward"]["passed"] = all(e["finite"] and e["relative_l2"] <= .05 for e in row["backward"].values())
    else:
        row["backward"] = {"passed": False, "status": "not_requested_or_unsupported"}
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="torch_sdpa", help="Comma-separated names, or all; unavailable entries are failures, never passes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("inference", "training", "both"), default="both")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    requested = list(BACKENDS) if args.backends == "all" else list(dict.fromkeys(args.backends.split(",")))
    if any(name not in BACKENDS for name in requested):
        parser.error(f"Backends must belong to {BACKENDS}")
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "actual CUDA operator smoke; not full-model or distributed qualification",
              "requested_mode": args.mode, "requested_backends": requested, "packages": _versions(), "results": []}
    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        report["working_tree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        report["commit"] = None
    try:
        import torch
    except ImportError as exc:
        report.update(status="unavailable", reason=str(exc), qualified=False)
        return _emit(report, args.output, 2)
    report["cuda"] = inspect_torch_cuda(torch)
    if not torch.cuda.is_available() or not torch.version.cuda:
        report.update(status="unavailable", reason="NVIDIA CUDA is unavailable; no device test executed", qualified=False)
        return _emit(report, args.output, 2)
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("Requested CUDA device index is out of range")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    profile = profile_cuda_device(torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
    report["device"] = profile.to_dict()
    report["device"]["index"] = args.device
    try:
        report["driver_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        report["driver_inventory"] = None
    ref = _reference()
    dtypes = [torch.float16, torch.bfloat16] if args.dtype == "both" else [torch.float16 if args.dtype == "fp16" else torch.bfloat16]
    for backend in requested:
        row = {"backend": backend, "qualified_inference": False, "qualified_training": False, "cases": []}
        report["results"].append(row)
        compatible = backend in profile.attention_preference
        compatible |= backend == "flash_attn4" and profile.capability in {(9, 0), (10, 0), (10, 3)}
        compatible |= backend == "sage_attn3" and profile.capability in {(12, 0), (12, 1)}
        if not compatible:
            row.update(status="unsupported_architecture", reason="No architecture policy for this optional backend")
            continue
        try:
            fn = _kernel(backend, profile)
            if backend != "torch_sdpa" and not callable(fn):
                row.update(status="unavailable", reason="Optional kernel callable is not installed")
                continue
            row["callable_module"] = getattr(fn, "__module__", "torch.nn.functional")
            backward = args.mode in ("training", "both") and not backend.startswith("sage_")
            cases = [("dense_self", [17, 17], [17, 17], False, False),
                     ("dense_cross_gqa", [17, 17], [23, 23], False, False),
                     ("packed_causal", [17, 9], [17, 9], True, True)]
            if not backend.startswith("sage_"):
                cases.append(("packed_cached_causal", [7, 3], [17, 9], True, True))
            else:
                row["unsupported_features"] = ["backward", "dropout", "unequal-length causal (runtime uses SDPA)"]
            for dtype in dtypes:
                for name, ql, kl, causal, packed in cases:
                    row["cases"].append(_case(backend, fn, ref, device, dtype, name, ql, kl, causal, packed, backward))
            row["qualified_inference"] = bool(row["cases"]) and all(case["forward"]["passed"] for case in row["cases"])
            row["qualified_training"] = backward and row["qualified_inference"] and all(case["backward"]["passed"] for case in row["cases"])
            row["status"] = "passed" if row["qualified_inference"] and (args.mode == "inference" or row["qualified_training"]) else "failed_or_unsupported_requested_mode"
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    report["qualified"] = bool(report["results"]) and all(row["status"] == "passed" for row in report["results"])
    report["status"] = "passed" if report["qualified"] else "failed_or_incomplete"
    return _emit(report, args.output, 0 if report["qualified"] else 1)


def _emit(report, output, status):
    payload = json.dumps(report, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")
    print(payload)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
