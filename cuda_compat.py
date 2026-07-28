from __future__ import annotations

"""Dependency-light CUDA compatibility policy shared by train and inference.

The policy separates hardware capability from optional-kernel availability.
Callers can therefore select a safe backend before importing FlashAttention,
SageAttention, or project-specific CUDA extensions.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable


Capability = tuple[int, int]


@dataclass(frozen=True)
class CudaDeviceProfile:
    name: str
    capability: Capability
    architecture: str
    minimum_cuda: str
    supports_bf16: bool
    supports_tf32: bool
    supports_fp8: bool
    supports_fp4: bool
    attention_preference: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_cuda_device(name: str, capability: Capability) -> CudaDeviceProfile:
    major, minor = capability
    normalized_name = str(name or "Unknown NVIDIA GPU")

    if (major, minor) == (8, 0):
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="ampere",
            minimum_cuda="11.0",
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=False,
            supports_fp4=False,
            attention_preference=("flash_attn2", "sage_attn2", "torch_sdpa"),
        )
    if major == 9:
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="hopper",
            minimum_cuda="11.8",
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=True,
            supports_fp4=False,
            attention_preference=("flash_attn3", "flash_attn2", "sage_attn2", "torch_sdpa"),
        )
    if (major, minor) in {(10, 0), (10, 3)}:
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="blackwell-datacenter",
            minimum_cuda="12.9" if minor >= 3 else "12.8",
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=True,
            supports_fp4=True,
            attention_preference=("sage_attn3", "flash_attn3", "flash_attn2", "torch_sdpa"),
        )
    if major >= 12:
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="blackwell",
            minimum_cuda="12.8",
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=True,
            supports_fp4=True,
            attention_preference=("sage_attn3", "flash_attn3", "flash_attn2", "torch_sdpa"),
        )

    return CudaDeviceProfile(
        name=normalized_name,
        capability=capability,
        architecture="generic-cuda",
        minimum_cuda="11.0",
        supports_bf16=major >= 8,
        supports_tf32=major >= 8,
        supports_fp8=major > 8 or (major == 8 and minor == 9),
        supports_fp4=major >= 10,
        attention_preference=("flash_attn2", "torch_sdpa"),
    )


def select_attention_backend(
    requested: str,
    profile: CudaDeviceProfile,
    available_backends: Iterable[str],
    *,
    strict: bool = False,
) -> str:
    """Resolve an attention backend with a deterministic safe fallback."""
    available = {str(name).strip() for name in available_backends if str(name).strip()}
    requested = str(requested or "auto").strip()

    if requested != "auto" and requested in available:
        return requested
    if requested != "auto" and strict:
        raise RuntimeError(
            f"Attention backend '{requested}' is unavailable on {profile.name} "
            f"(compute capability {profile.capability[0]}.{profile.capability[1]}). "
            f"Available backends: {sorted(available)}"
        )

    for candidate in profile.attention_preference:
        if candidate in available:
            return candidate
    raise RuntimeError(
        f"No compatible attention backend is available for {profile.name}. "
        "Install an architecture-compatible optimized backend or enable torch_sdpa."
    )


def minimum_cuda_issue(profile: CudaDeviceProfile, cuda_version: str | None) -> str | None:
    if not cuda_version:
        return "PyTorch does not report a CUDA runtime version."

    def _pair(value: str) -> tuple[int, int]:
        parts = value.split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

    try:
        current = _pair(cuda_version)
        required = _pair(profile.minimum_cuda)
    except (TypeError, ValueError):
        return f"Could not parse CUDA runtime version {cuda_version!r}."

    if current < required:
        return (
            f"{profile.name} requires CUDA >= {profile.minimum_cuda} for native "
            f"{profile.architecture} support; PyTorch reports CUDA {cuda_version}."
        )
    return None


def inspect_torch_cuda(torch_module: Any) -> dict[str, Any]:
    """Return a JSON-serializable report without importing optional kernels."""
    cuda = torch_module.cuda
    report: dict[str, Any] = {
        "cuda_available": bool(cuda.is_available()),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_runtime": getattr(getattr(torch_module, "version", None), "cuda", None),
        "compiled_arches": list(cuda.get_arch_list()) if hasattr(cuda, "get_arch_list") else [],
        "devices": [],
        "issues": [],
    }
    if not report["cuda_available"]:
        report["issues"].append("CUDA is not available in this PyTorch runtime.")
        return report

    for index in range(cuda.device_count()):
        profile = profile_cuda_device(cuda.get_device_name(index), tuple(cuda.get_device_capability(index)))
        issue = minimum_cuda_issue(profile, report["cuda_runtime"])
        device = profile.to_dict()
        device["index"] = index
        report["devices"].append(device)
        if issue is not None:
            report["issues"].append(issue)

    return report
