from __future__ import annotations

"""Dependency-light CUDA compatibility policy shared by train and inference.

The policy separates hardware capability from optional-kernel availability.
Callers can therefore select a safe backend before importing FlashAttention,
SageAttention, or project-specific CUDA extensions.
"""

import importlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, MutableMapping


Capability = tuple[int, int]

DENSE_ATTENTION_BACKENDS = frozenset(
    {"auto", "flash_attn2", "flash_attn3", "flash_attn4", "sage_attn2", "sage_attn3", "torch_sdpa"}
)
DENSE_ATTENTION_CONFIG_KEYS = (
    "attn_type",
    "self_attn_1_type",
    "cross_attn_1_type",
    "cross_attn_2_type",
    "adapter_attn_type",
)

LIGHTX2V_SM120_QUANT_SCHEMES = frozenset(
    {"nvfp4", "mxfp4", "mxfp6-mxfp8", "mxfp8"}
)


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
    if (major, minor) == (8, 9):
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="ada",
            minimum_cuda="11.8",
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=True,
            supports_fp4=False,
            attention_preference=("sage_attn2", "flash_attn2", "torch_sdpa"),
        )
    if (major, minor) == (9, 0):
        # H20 first appears in NVIDIA's CUDA 12.2 / R535 support matrix.  Keep
        # the broader SM90 policy usable for H100/H200 builds based on 11.8,
        # but do not infer that older toolkit support for those products also
        # covers H20 merely because the compute capability is identical.
        is_h20 = re.search(r"\bH20(?:\b|-)", normalized_name, re.IGNORECASE) is not None
        # The public FlashAttention-3 beta names H100/H800 as its supported
        # hardware.  H20 is also SM90, but matching compute capability alone
        # is not sufficient evidence that this product is accepted by the
        # released kernel.  Prefer architecture-generic Hopper backends there.
        attention_preference = (
            ("flash_attn2", "sage_attn2", "torch_sdpa")
            if is_h20
            else ("flash_attn3", "flash_attn2", "sage_attn2", "torch_sdpa")
        )
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="hopper",
            minimum_cuda="12.2" if is_h20 else "11.8",
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=True,
            supports_fp4=False,
            attention_preference=attention_preference,
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
            # The currently detected Sage3 callable targets SM120/121, FA2 is
            # documented through Hopper, and the imported FA3 beta targets
            # H100/H800.  Importability is therefore not compatibility evidence
            # for SM100/103. Keep data-center Blackwell on native Torch SDPA
            # until an architecture-specific backend has a real device probe.
            attention_preference=("torch_sdpa",),
        )
    if (major, minor) in {(12, 0), (12, 1)}:
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="blackwell",
            minimum_cuda="12.8",
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=True,
            supports_fp4=True,
            # Importing a SageAttention3 callable does not prove that its wheel
            # contains a runnable kernel for this exact SM target (especially
            # SM121). Keep automatic dispatch on native SDPA until a real
            # device-side smoke probe succeeds; operators may then opt into an
            # optimized backend in a hardware-qualified build.
            attention_preference=("torch_sdpa",),
        )

    # Any capability without an explicit, evidence-backed row is unverified.
    # Importability of an optional extension is not device compatibility: a
    # wheel can omit SASS/PTX for the active SM and fail only at kernel launch.
    # Fail closed to native Torch SDPA for old, intermediate, and future CCs.
    return CudaDeviceProfile(
        name=normalized_name,
        capability=capability,
        architecture="unverified-cuda",
        minimum_cuda="0.0",
        supports_bf16=False,
        supports_tf32=False,
        supports_fp8=False,
        supports_fp4=False,
        attention_preference=("torch_sdpa",),
    )


def select_attention_backend(
    requested: str,
    profile: CudaDeviceProfile,
    available_backends: Iterable[str],
    *,
    strict: bool = False,
    validated_backends: Iterable[str] = (),
) -> str:
    """Resolve an attention backend with a deterministic safe fallback."""
    available = {str(name).strip() for name in available_backends if str(name).strip()}
    requested = str(requested or "auto").strip()

    compatible = available.intersection(profile.attention_preference)
    # FA4 is opt-in after a real kernel probe on this exact host, never merely
    # because its Python package imports. Auto stays on established defaults.
    if profile.capability in {(9, 0), (10, 0), (10, 3)}:
        compatible.update(available.intersection(validated_backends).intersection({"flash_attn4"}))

    if requested != "auto" and requested in compatible:
        return requested
    if requested != "auto" and strict:
        raise RuntimeError(
            f"Attention backend '{requested}' is unavailable or incompatible on {profile.name} "
            f"(compute capability {profile.capability[0]}.{profile.capability[1]}). "
            f"Compatible installed backends: {sorted(compatible)}"
        )

    for candidate in profile.attention_preference:
        if candidate in available:
            return candidate
    raise RuntimeError(
        f"No compatible attention backend is available for {profile.name}. "
        "Install an architecture-compatible optimized backend or enable torch_sdpa."
    )


def validate_lightx2v_quant_backend(
    scheme: str,
    capability: Capability,
    *,
    callables_available: bool,
) -> None:
    """Fail early for the vendored CUTLASS kernels' exact build target.

    The current ``lightx2v_kernel`` sources and CMake flags are SM120a-only.
    They are not a generic Blackwell binary and must not be selected on B200
    (SM100), B300 (SM103), Hopper, or Ampere.
    """

    if scheme not in LIGHTX2V_SM120_QUANT_SCHEMES:
        return
    if tuple(capability) != (12, 0):
        raise RuntimeError(
            f"Quantization scheme {scheme!r} requires the vendored sm_120a "
            f"lightx2v_kernel build, but this device reports sm_{capability[0]}{capability[1]}. "
            "Choose an architecture-compatible quantization backend or unquantized weights."
        )
    if not callables_available:
        raise RuntimeError(
            f"Quantization scheme {scheme!r} requires callable lightx2v_kernel "
            "SM120a operators. Build the optional extension against this PyTorch/CUDA "
            "environment or choose another quantization scheme."
        )


def import_attention_callable(backend: str, *, dense: bool = False):
    """Support current packaged FA3 and its legacy source-install namespace."""
    probes = {
        "flash_attn2": (("flash_attn.flash_attn_interface", "flash_attn_varlen_func"),),
        "flash_attn3": (("flash_attn_3.flash_attn_interface", "flash_attn_varlen_func"),
                        ("flash_attn_interface", "flash_attn_varlen_func")),
        "flash_attn4": (("flash_attn.cute", "flash_attn_varlen_func"),),
        "sage_attn3": (("sageattn3", "sageattn3_blackwell"),),
    }
    for module_name, name in probes.get(backend, ()):
        if dense and backend.startswith("flash_attn"):
            name = "flash_attn_func"
        try:
            fn = getattr(importlib.import_module(module_name), name, None)
        except (ImportError, OSError, RuntimeError):
            continue
        if callable(fn):
            return fn
    return None


_FA4_PROBE_RESULTS: dict[tuple, dict[str, Any]] = {}


def probe_flash_attention4(torch_module: Any, profile: CudaDeviceProfile) -> dict[str, Any]:
    """Small forward-only host qualification; NOT an end-to-end benchmark.

    Test FP16/BF16, packed batches, GQA and unequal-length causal attention
    against native SDPA before permitting explicit FA4 dispatch. A successful
    probe does not validate all model shapes or backward execution.
    """
    cuda = torch_module.cuda
    fn = import_attention_callable("flash_attn4")
    if not cuda.is_available() or not callable(fn) or profile.capability not in {(9, 0), (10, 0), (10, 3)}:
        return {"passed": False, "reason": "FA4 requires a supported CUDA device and callable"}
    index = cuda.current_device()
    key = (str(torch_module.__version__), str(torch_module.version.cuda), index, profile.capability, id(fn))
    if key in _FA4_PROBE_RESULTS:
        return dict(_FA4_PROBE_RESULTS[key])
    report = {"passed": False, "device_index": index, "capability": profile.capability,
              "scope": "forward-only fp16/bf16 packed GQA + causal smoke"}
    try:
        device = torch_module.device("cuda", index)
        generator = torch_module.Generator(device=device).manual_seed(1701)
        with torch_module.no_grad(), cuda.device(index):
            for dtype in (torch_module.float16, torch_module.bfloat16):
                q = torch_module.randn(2, 16, 4, 64, device=device, dtype=dtype, generator=generator)
                k = torch_module.randn(2, 24, 2, 64, device=device, dtype=dtype, generator=generator)
                v = torch_module.randn(2, 24, 2, 64, device=device, dtype=dtype, generator=generator)
                cq = torch_module.tensor([0, 16, 32], device=device, dtype=torch_module.int32)
                ck = torch_module.tensor([0, 24, 48], device=device, dtype=torch_module.int32)
                for causal in (False, True):
                    mask = None
                    if causal:
                        mask = torch_module.arange(24, device=device)[None, :] <= torch_module.arange(16, device=device)[:, None] + 8
                    expected = torch_module.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.repeat_interleave(2, dim=2).transpose(1, 2),
                        v.repeat_interleave(2, dim=2).transpose(1, 2), attn_mask=mask).transpose(1, 2)
                    out = fn(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
                        cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=16, max_seqlen_k=24,
                        causal=causal).reshape_as(expected)
                    cuda.synchronize(index)
                    if not torch_module.allclose(out.float(), expected.float(), atol=0.04, rtol=0.04):
                        raise RuntimeError(f"FA4 numerical probe failed for {dtype}, causal={causal}")
        report["passed"] = True
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    _FA4_PROBE_RESULTS[key] = report
    return dict(report)


def detect_attention_backends(torch_module: Any | None = None) -> set[str]:
    """Detect callable dense-attention implementations, not import specs alone."""

    available: set[str] = set()
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except (ImportError, OSError, RuntimeError):
            torch_module = None

    if torch_module is not None:
        functional = getattr(getattr(torch_module, "nn", None), "functional", None)
        if callable(getattr(functional, "scaled_dot_product_attention", None)):
            available.add("torch_sdpa")

    for backend in ("flash_attn2", "flash_attn3", "flash_attn4", "sage_attn3"):
        if callable(import_attention_callable(backend)):
            available.add(backend)

    try:
        sage_module = importlib.import_module("sageattention")
    except (ImportError, OSError, RuntimeError):
        sage_module = None
    if sage_module is not None and any(
        callable(getattr(sage_module, name, None))
        for name in ("sageattn", "sageattn_qk_int8_pv_fp16_triton")
    ):
        available.add("sage_attn2")

    return available


def resolve_attention_config(
    config: MutableMapping[str, Any],
    *,
    torch_module: Any | None = None,
    profile: CudaDeviceProfile | None = None,
    available_backends: Iterable[str] | None = None,
    strict: bool | None = None,
) -> dict[str, tuple[str, str]]:
    """Resolve dense attention fields after all model config overlays.

    Sparse and distributed algorithms (for example ``ulysses``, ``ring`` or
    ``svg``) are deliberately left untouched.  The returned mapping records
    only fields whose value changed.
    """

    if torch_module is None and profile is None:
        torch_module = importlib.import_module("torch")
    if profile is None:
        cuda = torch_module.cuda
        if cuda.is_available():
            device_index = cuda.current_device() if hasattr(cuda, "current_device") else 0
            profile = profile_cuda_device(
                cuda.get_device_name(device_index),
                tuple(cuda.get_device_capability(device_index)),
            )
        else:
            profile = CudaDeviceProfile(
                name="CPU / CUDA unavailable",
                capability=(0, 0),
                architecture="cpu",
                minimum_cuda="0.0",
                supports_bf16=False,
                supports_tf32=False,
                supports_fp8=False,
                supports_fp4=False,
                attention_preference=("torch_sdpa",),
            )

    available = set(available_backends) if available_backends is not None else detect_attention_backends(torch_module)
    runtime = getattr(getattr(torch_module, "version", None), "cuda", None)
    if runtime:
        # Device support and extension support have different minimum versions.
        # Older source builds may work, but are not implicitly qualified here.
        try:
            version_pair = tuple(int(part) for part in runtime.split(".")[:2])
            for backend, minimum in (("flash_attn2", (12, 0)), ("flash_attn3", (12, 3)), ("flash_attn4", (12, 8))):
                if version_pair < minimum:
                    available.discard(backend)
        except (TypeError, ValueError):
            available.intersection_update({"torch_sdpa"})
    strict_mode = bool(config.get("strict_cuda_backend", False)) if strict is None else strict
    changes: dict[str, tuple[str, str]] = {}

    def _resolve(container: MutableMapping[str, Any], key: str, display_key: str) -> None:
        requested = str(container.get(key, ""))
        if requested not in DENSE_ATTENTION_BACKENDS:
            return
        validated = ()
        if requested == "flash_attn4" and requested in available and torch_module is not None:
            report = probe_flash_attention4(torch_module, profile)
            config.setdefault("attention_runtime_probes", {})["flash_attn4"] = report
            if report["passed"]:
                validated = ("flash_attn4",)
        resolved = select_attention_backend(
            requested=requested,
            profile=profile,
            available_backends=available,
            strict=strict_mode,
            validated_backends=validated,
        )
        if resolved != requested:
            container[key] = resolved
            changes[display_key] = (requested, resolved)

    for key in DENSE_ATTENTION_CONFIG_KEYS:
        if key in config:
            _resolve(config, key, key)

    parallel = config.get("parallel")
    if isinstance(parallel, MutableMapping) and "seq_p_attn_type" in parallel:
        _resolve(parallel, "seq_p_attn_type", "parallel.seq_p_attn_type")

    config["resolved_cuda_profile"] = profile.architecture
    config["available_dense_attention_backends"] = sorted(available)
    return changes


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
        native_arch = f"sm_{profile.capability[0]}{profile.capability[1]}"
        compiled_arches = set(report["compiled_arches"])
        device["native_arch"] = native_arch
        device["native_arch_compiled"] = native_arch in compiled_arches if compiled_arches else None
        report["devices"].append(device)
        if issue is not None:
            report["issues"].append(issue)
        if compiled_arches and native_arch not in compiled_arches:
            report["issues"].append(
                f"PyTorch does not list a native {native_arch} target for {profile.name}; "
                "a PTX JIT path may exist, but it is not native-architecture validation."
            )

    return report
