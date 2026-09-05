from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_compat import inspect_torch_cuda


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect WorldDistill CUDA runtime compatibility.")
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    parser.add_argument("--strict", action="store_true", help="Fail when compatibility issues are detected.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is not installed in the active environment.", file=sys.stderr)
        return 1

    report = inspect_torch_cuda(torch)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"PyTorch {report['torch_version']} | CUDA runtime {report['cuda_runtime']} | "
            f"CUDA available={report['cuda_available']}"
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

    return int(bool(args.strict and report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
