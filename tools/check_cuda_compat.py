from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_compat import (
    common_supported_precisions,
    inspect_torch_cuda,
    probe_cuda_precision,
    select_mixed_precision,
    validate_tf32_request,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect WorldDistill CUDA runtime compatibility.")
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    parser.add_argument("--strict", action="store_true", help="Fail on runtime issues or unverified architecture; does not require native SASS.")
    parser.add_argument("--strict-native-arch", action="store_true", help="Separately require an exact sm_* in Torch's architecture list, even if PTX/cubin runs.")
    parser.add_argument("--probe-precision", action="store_true", help="Execute small GEMM/math-SDPA forward/backward tests, not a full-model qualification.")
    parser.add_argument("--precision", choices=("auto", "no", "fp16", "bf16"), default="auto", help="Required common precision after probing; explicit requests never downgrade.")
    parser.add_argument("--enable-tf32", action="store_true", help="Require actual-device TF32 eligibility; does not mutate Torch settings.")
    parser.add_argument("--device", type=int, help="Probe one CUDA index; by default inspect/probe every visible device.")
    parser.add_argument("--output", type=Path, help="Write the full JSON report.")
    args = parser.parse_args(argv)
    if (args.precision != "auto" or args.enable_tf32) and not args.probe_precision:
        parser.error("--precision / --enable-tf32 requires --probe-precision")

    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        report = {"status": "unavailable", "qualified": False, "reason": str(exc), "devices": [], "issues": ["PyTorch could not load"]}
        return _emit(report, args, 2)

    report = inspect_torch_cuda(torch)
    report.update(schema_version=2, timestamp_utc=datetime.now(timezone.utc).isoformat(), status="metadata_only")
    if not report["cuda_available"]:
        report["status"] = "unavailable"
        return _emit(report, args, 2)
    if args.device is not None:
        if not 0 <= args.device < torch.cuda.device_count():
            parser.error("Requested CUDA device index is out of range")
        # Metadata still inventories all devices. Selected-device probing is
        # explicitly scoped and does not certify the other visible GPUs.
        indices = [args.device]
    else:
        indices = [device["index"] for device in report["devices"]]
    exit_code = int(bool(args.strict and (report["issues"] or any(not d["known_architecture"] for d in report["devices"]))))
    if args.strict_native_arch and (report["native_arch_issues"] or not report["compiled_arches"]):
        exit_code = 1
    if args.probe_precision:
        probes = [probe_cuda_precision(torch, index) for index in indices]
        report["precision_probes"] = probes
        report["probed_device_indices"] = indices
        report["scope"] = "selected devices: small GEMM/math-SDPA precision smoke, not model/optional-kernel qualification"
        report["common_supported_precisions"] = list(common_supported_precisions(probes))
        try:
            report["selected_precision"] = select_mixed_precision(args.precision, report["common_supported_precisions"])
            report["tf32_requested_and_eligible"] = validate_tf32_request(args.enable_tf32, probes)
        except RuntimeError as exc:
            report["issues"].append(str(exc))
            exit_code = 1
        # A successful kernel launch cannot override an explicitly requested
        # strict metadata/native-SASS requirement. Keep the JSON verdict and
        # process exit status consistent for automated qualification readers.
        report["qualified"] = exit_code == 0 and bool(probes) and all(p["qualified"] for p in probes) and not report["issues"]
        report["status"] = "passed" if report["qualified"] else "partial_or_failed"
        if not report["qualified"]:
            exit_code = 1
    return _emit(report, args, exit_code)


def _emit(report, args, exit_code):
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"PyTorch {report.get('torch_version')} | CUDA runtime {report.get('cuda_runtime')} | "
            f"CUDA available={report.get('cuda_available', False)} | status={report['status']} | "
            f"qualified={report['qualified']}"
        )
        for device in report["devices"]:
            major, minor = device["capability"]
            print(
                f"[{device['index']}] {device['name']} | cc={major}.{minor} | "
                f"{device['architecture']} | minimum CUDA={device['minimum_cuda']} | "
                f"attention={' > '.join(device['attention_preference'])}"
            )
        for issue in report["issues"]:
            print(f"ISSUE: {issue}")
        for warning in report.get("warnings", []) + report.get("native_arch_issues", []):
            print(f"NOTE: {warning}")
        for probe in report.get("precision_probes", []):
            print(f"[{probe.get('device_index')}] precision smoke={probe['status']} | "
                  f"passed={probe['supported_precisions']} | TF32 eligible={probe['tf32_available']}")
        if "selected_precision" in report:
            print(f"Common precision: {report['selected_precision']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
