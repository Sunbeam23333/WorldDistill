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


def validate_runtime_dependency_versions(strict: bool = True) -> dict[str, Any]:
    issues: list[str] = []
    versions: dict[str, str | None] = {
        "transformers": _installed_version("transformers"),
    }
    if versions["transformers"] != EXPECTED_TRANSFORMERS_VERSION:
        issues.append(
            "transformers 版本不匹配: "
            f"检测到 {versions['transformers']!r}, 需要严格使用 {EXPECTED_TRANSFORMERS_VERSION}."
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
            f"'transformers=={EXPECTED_TRANSFORMERS_VERSION}' "
            "'diffusers>=0.33.0,<1' 'accelerate>=0.34.2,<2' 'peft>=0.17.0,<1'"
        )
        raise RuntimeError("\n".join([*issues, f"建议执行: {install_hint}"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LightX2V environment compatibility checker")
    parser.add_argument("--mode", type=str, default="infer", choices=["infer"])
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    result = validate_runtime_dependency_versions(strict=not args.no_strict)
    print(f"[lightx2v:{args.mode}] dependency check ok={result['ok']}")
    for package_name, version_str in result["versions"].items():
        print(f"  - {package_name}: {version_str}")
    if result["issues"]:
        print("  issues:")
        for issue in result["issues"]:
            print(f"    * {issue}")


if __name__ == "__main__":
    main()
