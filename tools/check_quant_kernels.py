#!/usr/bin/env python3
"""Execute real CUDA numerical gates; never count a CPU fallback as a pass.

    python tools/check_quant_kernels.py --output results/kernels.json
    compute-sanitizer --tool memcheck --error-exitcode=99 \
        python tools/check_quant_kernels.py --output results/kernels-memcheck.json

The sanitizer's exit code/log is separate evidence: this script cannot attest
that an external sanitizer ran. These small tests establish neither model
quality nor throughput, and do not validate third-party GEMM extensions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualification_status(cases):
    """A partial selection is not evidence that every requested kernel ran."""
    if any(case["status"] == "failed" for case in cases):
        return "failed"
    if not cases or any(case["status"] != "passed" for case in cases):
        return "partial"
    return "passed"


def _fused_case(torch, fused, dtype, masked):
    p = torch.linspace(-3, 4, 2050, device="cuda", dtype=dtype).reshape(2, 1025).requires_grad_()
    t = torch.ones_like(p).requires_grad_()
    rp, rt = p.detach().clone().requires_grad_(), t.detach().clone().requires_grad_()
    mask = torch.tensor([[0.25], [1]], device="cuda") if masked else None
    if not fused._can_use_fused_kernel(p, t, mask):
        raise RuntimeError("fused kernel is unavailable; a fallback is not a GPU validation")
    actual = fused.fused_masked_mse_loss(p, t, mask)
    reference = fused.fused_masked_mse_loss(rp, rt, mask, enabled=False)
    upstream = torch.tensor(65536.0, device="cuda")
    actual.backward(upstream)
    reference.backward(upstream)
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(p.grad, rp.grad, atol=2e-4, rtol=3e-3)
    torch.testing.assert_close(t.grad, rt.grad, atol=2e-4, rtol=3e-3)
    torch.cuda.synchronize()
    return {"loss": actual.item(), "max_gradient_error": (p.grad.float() - rp.grad.float()).abs().max().item()}


def _gemm_case(torch, kernels, int8, shape, bias_enabled, fuse_gelu):
    m, n, k = shape
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(n, k, device="cuda")
    a[0].zero_()
    quantize = kernels.int8_quantize_triton if int8 else kernels.fp8_quantize_triton
    qa, sa = quantize(a)
    qb, sb = quantize(b)
    if not torch.isfinite(sa).all() or not torch.isfinite(sb).all() or torch.count_nonzero(qa[0].float()):
        raise AssertionError("zero-row quantization produced invalid scales or nonzero values")
    bias = torch.randn(n, device="cuda") if bias_enabled else None
    reference = (qa.double() @ qb.double().T) * sa.double()[:, None] * sb.double()[None, :]
    if bias is not None:
        reference += bias.double()
    if fuse_gelu:
        reference = reference * torch.sigmoid(1.702 * reference)
    # Exercise non-contiguous matrix and per-row scale inputs as well as tails.
    qa = qa.T.contiguous().T
    qb = qb.T.contiguous().T
    strided_sa = torch.empty(m * 2, device="cuda")
    strided_sa[::2] = sa
    actual = kernels._gemm(qa, qb, strided_sa[::2], sb, bias, int8=int8,
                           fuse_gelu=fuse_gelu, output_dtype=torch.float32)
    torch.testing.assert_close(actual, reference.float(), atol=2e-4, rtol=3e-4)
    torch.cuda.synchronize()
    return {"max_absolute_error": (actual - reference.float()).abs().max().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="Write a machine-readable evidence record")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--require-fp8", action="store_true", help="Fail rather than skip FP8 on pre-SM89 devices")
    args = parser.parse_args()
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "host": platform.node(), "scope": "local Triton GEMM and fused supervision numerical smoke only",
              "sanitizer_attested": False, "cases": []}
    paths = ["quant_compat.py", "training/runtime/fused_supervision.py", "inference/lightx2v/common/ops/mm/triton_kernels.py"]
    report["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    report["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    report["versions"] = {}
    for package in ("torch", "triton"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None
    try:
        import torch
        if not torch.cuda.is_available() or not torch.version.cuda:
            raise RuntimeError("No NVIDIA CUDA runtime/device is available; no GPU cases were run")
        torch.cuda.set_device(args.device)
        torch.manual_seed(23)
        capability = tuple(torch.cuda.get_device_capability(args.device))
        report["device"] = {"index": args.device, "name": torch.cuda.get_device_name(args.device),
                            "capability": capability, "torch_cuda": torch.version.cuda,
                            "torch_arch_list": torch.cuda.get_arch_list()}
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10)
            report["host_gpu_drivers"] = driver.stdout.strip().splitlines() if driver.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            report["host_gpu_drivers"] = None
        kernels = _load("worlddistill_smoke_quant", paths[2])
        fused = _load("worlddistill_smoke_fused", paths[1])
        if kernels.triton is None or not fused.fused_supervision_available():
            raise RuntimeError("Triton must be installed; no GPU fallback is accepted")

        def run_case(name, function):
            try:
                result = function()
                report["cases"].append({"name": name, "status": "passed", **result})
            except Exception as exc:
                report["cases"].append({"name": name, "status": "failed", "error": repr(exc)})

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for masked in (False, True):
                run_case(f"fused-{dtype}-masked-{masked}", lambda: _fused_case(torch, fused, dtype, masked))
        for int8 in (True, False):
            if not int8 and capability < (8, 9):
                report["cases"].append({"name": "fp8", "status": "failed" if args.require_fp8 else "skipped",
                                        "reason": "native FP8 requires SM89+"})
                continue
            for shape in ((1, 1, 1), (3, 5, 63), (37, 71, 129), (129, 131, 127), (128, 128, 128)):
                for bias in (False, True):
                    for gelu in (False, True):
                        name = f"{'int8' if int8 else 'fp8'}-{shape}-bias-{bias}-gelu-{gelu}"
                        run_case(name, lambda: _gemm_case(torch, kernels, int8, shape, bias, gelu))
        report["status"] = qualification_status(report["cases"])
    except Exception as exc:
        report["status"] = "unavailable" if not report["cases"] else "failed"
        report["error"] = repr(exc)
    report["qualified"] = report["status"] == "passed"
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
