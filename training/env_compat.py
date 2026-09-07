from __future__ import annotations

import argparse
from importlib import metadata
from typing import Any

try:
    from packaging.version import Version
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    Version = None  # type: ignore[assignment]

EXPECTED_TRANSFORMERS_VERSION = "4.57.1"
_COMPATIBLE_RANGES: dict[str, tuple[str | None, str | None]] = {
    "diffusers": ("0.33.0", None),
    "accelerate": ("0.34.2", None),
    "peft": ("0.17.0", None),
    "huggingface_hub": ("0.25.0", None),
}


def _installed_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _check_range(package_name: str, version_str: str, minimum: str | None, maximum: str | None) -> str | None:
    if Version is None:
        return None
    current = Version(version_str)
    if minimum is not None and current < Version(minimum):
        return f"{package_name}=={version_str} 低于建议下界 {minimum}"
    if maximum is not None and current >= Version(maximum):
        return f"{package_name}=={version_str} 超出建议上界 {maximum}"
    return None


def validate_runtime_dependency_versions(strict: bool = True, required_transformers_version: str = EXPECTED_TRANSFORMERS_VERSION) -> dict[str, Any]:
    issues: list[str] = []
    versions: dict[str, str | None] = {
        "transformers": _installed_version("transformers"),
    }
    if versions["transformers"] != required_transformers_version:
        issues.append(
            "transformers 版本不匹配: "
            f"检测到 {versions['transformers']!r}, 需要严格使用 {required_transformers_version}."
        )

    for package_name, bounds in _COMPATIBLE_RANGES.items():
        version_str = _installed_version(package_name)
        versions[package_name] = version_str
        if version_str is None:
            issues.append(f"缺少依赖: {package_name}")
            continue
        range_issue = _check_range(package_name, version_str, bounds[0], bounds[1])
        if range_issue is not None:
            issues.append(range_issue)

    result = {"ok": not issues, "versions": versions, "issues": issues}
    if strict and issues:
        install_hint = (
            "pip install "
            f"'transformers=={required_transformers_version}' "
            "'diffusers>=0.33.0,<1' 'accelerate>=0.34.2,<2' 'peft>=0.17.0,<1'"
        )
        raise RuntimeError("\n".join([*issues, f"建议执行: {install_hint}"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldDistill environment compatibility checker")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--required_transformers_version", default=EXPECTED_TRANSFORMERS_VERSION)
    parser.add_argument("--dist_backend", choices=["auto", "nccl", "gloo"])
    parser.add_argument("--nproc_per_node", type=int)
    args = parser.parse_args()

    result = validate_runtime_dependency_versions(strict=not args.no_strict,
                                                  required_transformers_version=args.required_transformers_version)
    if args.dist_backend is not None or args.nproc_per_node is not None:
        validate_local_distributed_devices(args.dist_backend or "auto",
                                           args.nproc_per_node if args.nproc_per_node is not None else 1)
    print(f"[worlddistill:{args.mode}] dependency check ok={result['ok']}")
    for package_name, version_str in result["versions"].items():
        print(f"  - {package_name}: {version_str}")
    if result["issues"]:
        print("  issues:")
        for issue in result["issues"]:
            print(f"    * {issue}")


def validate_local_distributed_devices(backend: str = "auto", nproc_per_node: int = 1) -> dict[str, Any]:
    """Preflight local workers without importing any model or changing visibility."""
    import torch
    import torch.distributed as dist

    if type(nproc_per_node) is not int or nproc_per_node < 1:
        raise ValueError("nproc_per_node must be a positive integer")
    if backend not in {"auto", "nccl", "gloo"}:
        raise ValueError("Distributed backend must be auto, nccl or gloo")
    cuda_available = torch.cuda.is_available()
    selected = ("nccl" if cuda_available else "gloo") if backend == "auto" else backend
    if selected == "nccl":
        if (not torch.version.cuda or getattr(torch.version, "hip", None)
                or not cuda_available or not dist.is_nccl_available()):
            raise RuntimeError("NCCL requires an NVIDIA CUDA PyTorch build and visible NVIDIA GPUs; ROCm/RCCL is unsupported")
        if nproc_per_node > torch.cuda.device_count():
            raise ValueError("Worker count exceeds scheduler-visible GPUs; reduce --nproc_per_node, do not replace the GPU mask")
    elif cuda_available:
        raise ValueError("Gloo is CPU-only validation; set CUDA_VISIBLE_DEVICES='' explicitly to hide CUDA")
    elif not dist.is_gloo_available():
        raise RuntimeError("Gloo is unavailable in this PyTorch build")
    return {"backend": selected, "nproc_per_node": nproc_per_node,
            "visible_cuda_devices": torch.cuda.device_count() if cuda_available else 0}


if __name__ == "__main__":
    main()
