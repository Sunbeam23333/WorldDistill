from __future__ import annotations

"""Dependency-light CUDA compatibility policy shared by train and inference.

The policy separates hardware capability from optional-kernel availability.
Callers can therefore select a safe backend before importing FlashAttention,
SageAttention, or project-specific CUDA extensions.
"""

import importlib
import re
from copy import deepcopy
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
    supports_fp16: bool = True
    known_architecture: bool = True
    maximum_cuda_major: int | None = None
    installation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_cuda_device(name: str, capability: Capability) -> CudaDeviceProfile:
    capability = tuple(capability)
    major, minor = capability
    normalized_name = str(name or "Unknown NVIDIA GPU")

    legacy = {
        (5, 0): ("maxwell", "6.5"), (5, 2): ("maxwell", "6.5"),
        (5, 3): ("maxwell-jetson", "7.0"),
        (6, 0): ("pascal", "8.0"), (6, 1): ("pascal", "8.0"),
        (6, 2): ("pascal-jetson", "8.0"),
        (7, 0): ("volta", "9.0"), (7, 2): ("volta-jetson", "9.0"),
        (7, 5): ("turing", "10.0"),
    }
    if capability in legacy:
        architecture, minimum = legacy[capability]
        return CudaDeviceProfile(
            name=normalized_name, capability=capability, architecture=architecture,
            minimum_cuda=minimum, supports_bf16=False, supports_tf32=False,
            supports_fp8=False, supports_fp4=False, attention_preference=("torch_sdpa",),
            supports_fp16=capability >= (5, 3),
            maximum_cuda_major=12 if capability < (7, 5) else None,
            installation_note=(
                "Legacy CUDA <=12.x libraries/build required; PyTorch 2.11 cu128+ "
                "wheels exclude pre-Turing. Check a compatible legacy wheel and its "
                "cuDNN/architecture list; Jetson also requires its JetPack-specific build."
                if capability < (7, 5) else
                "Use Torch SDPA (math fallback allowed); FA2 and native BF16/TF32 are not enabled."
            ),
        )
    if capability in {(8, 0), (8, 6), (8, 7)}:
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="ampere-jetson" if minor == 7 else "ampere",
            minimum_cuda={(8, 0): "11.0", (8, 6): "11.1", (8, 7): "11.4"}[capability],
            supports_bf16=True,
            supports_tf32=True,
            supports_fp8=False,
            supports_fp4=False,
            attention_preference=(("torch_sdpa",) if minor == 7 else
                                  ("flash_attn2", "sage_attn2", "torch_sdpa")),
            installation_note=("Jetson Orin requires a matching JetPack/aarch64 PyTorch build; "
                               "optional kernels are not enabled by the desktop Ampere policy."
                               if minor == 7 else ""),
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
    if capability == (11, 0):
        # CUDA 13 renamed Thor's former sm_101 target to sm_110. Do not map
        # arbitrary SM101 devices or authorize desktop Blackwell extensions.
        return CudaDeviceProfile(
            name=normalized_name, capability=capability, architecture="blackwell-jetson",
            minimum_cuda="13.0", supports_bf16=True, supports_tf32=True,
            supports_fp8=True, supports_fp4=True, attention_preference=("torch_sdpa",),
            installation_note="Thor SM110 requires the CUDA 13 / JetPack 7 platform stack; "
                              "the former SM101 name and optional kernels are not auto-qualified.",
        )
    if (major, minor) in {(12, 0), (12, 1)}:
        return CudaDeviceProfile(
            name=normalized_name,
            capability=capability,
            architecture="blackwell",
            minimum_cuda="12.9" if minor == 1 else "12.8",
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
        supports_fp16=False,
        known_architecture=False,
        installation_note="Unverified architecture: only a conservative FP32/SDPA smoke is attempted; "
                          "no hardware support or optional-kernel qualification is implied.",
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
    if not cuda.is_available() or not getattr(torch_module.version, "cuda", None) or not callable(fn) or profile.capability not in {(9, 0), (10, 0), (10, 3)}:
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
        if cuda.is_available() and getattr(getattr(torch_module, "version", None), "cuda", None):
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
                supports_fp16=False,
                known_architecture=False,
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
    if profile.maximum_cuda_major is not None and current[0] > profile.maximum_cuda_major:
        return (
            f"{profile.name} ({profile.architecture}) requires a legacy CUDA <= "
            f"{profile.maximum_cuda_major}.x build; CUDA 13 removed pre-Turing "
            "offline compilation and library support. Use a compatible legacy "
            "PyTorch/CUDA/cuDNN package, not just a newer NVIDIA driver."
        )
    return None


def inspect_torch_cuda(torch_module: Any) -> dict[str, Any]:
    """Return a JSON-serializable report without importing optional kernels."""
    cuda = torch_module.cuda
    report: dict[str, Any] = {
        "cuda_available": bool(cuda.is_available() and getattr(getattr(torch_module, "version", None), "cuda", None)),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_runtime": getattr(getattr(torch_module, "version", None), "cuda", None),
        "compiled_arches": list(cuda.get_arch_list()) if hasattr(cuda, "get_arch_list") else [],
        "devices": [],
        "issues": [],
        "warnings": [],
        "native_arch_issues": [],
        "qualified": False,
        "scope": "metadata-only; no CUDA kernel execution",
    }
    if not report["cuda_available"]:
        report["issues"].append("NVIDIA CUDA is not available in this PyTorch runtime (CPU/ROCm is not CUDA qualification).")
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
        device["qualified"] = False
        device["compatibility_evidence"] = "metadata-only"
        report["devices"].append(device)
        if issue is not None:
            report["issues"].append(issue)
        if compiled_arches and native_arch not in compiled_arches:
            report["native_arch_issues"].append(
                f"PyTorch does not list a native {native_arch} target for {profile.name}; "
                "compatible cubin or PTX JIT may still run. Use a kernel probe; "
                "missing native SASS alone is not proof of runtime failure."
            )
        if not profile.known_architecture:
            report["warnings"].append(f"{profile.name}: {profile.installation_note}")
        elif profile.installation_note:
            report["warnings"].append(f"{profile.name}: {profile.installation_note}")

    return report


def policy_supported_precisions(profile: CudaDeviceProfile) -> tuple[str, ...]:
    """Candidates only; neither a dtype object nor BF16 emulation proves native support."""
    candidates = ["no"]
    if profile.supports_fp16:
        candidates.append("fp16")
    if profile.supports_bf16:
        candidates.append("bf16")
    return tuple(candidates)


def select_mixed_precision(requested: str, supported_precisions: Iterable[str]) -> str:
    """Choose from observed support. Explicit requests never silently downgrade."""
    if requested not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError(f"Unknown mixed precision {requested!r}; use auto, no, fp16, or bf16")
    supported = set(supported_precisions).intersection({"no", "fp16", "bf16"})
    if requested != "auto":
        if requested not in supported:
            raise RuntimeError(f"Explicit mixed precision {requested!r} did not pass on every device; "
                               f"common supported precisions: {sorted(supported)}. No automatic downgrade was made.")
        return requested
    for candidate in ("bf16", "fp16", "no"):
        if candidate in supported:
            return candidate
    raise RuntimeError("No common runnable precision was observed. Check each rank's CUDA precision probe report.")


def common_supported_precisions(reports: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Intersect live per-rank reports before constructing models/collectives.

    Gathering reports is the distributed caller's responsibility. This helper
    performs no collectives and is not a signed hardware-attestation mechanism.
    """
    reports = list(reports)
    if not reports:
        return ()
    supported = {"no", "fp16", "bf16"}
    for report in reports:
        if report.get("status") in {"unavailable", "failed"}:
            return ()
        supported.intersection_update(report.get("supported_precisions", ()))
    return tuple(p for p in ("no", "fp16", "bf16") if p in supported)


def validate_tf32_request(enabled: bool, reports_or_single) -> bool:
    """Validate an explicit TF32 toggle without mutating Torch global settings."""
    if not enabled:
        return False
    reports = [reports_or_single] if isinstance(reports_or_single, dict) else list(reports_or_single)
    if not reports or any(not report.get("tf32_available", False) or
                          report.get("status") in {"failed", "unavailable"} for report in reports):
        raise RuntimeError("TF32 was explicitly enabled, but is not available on every actual device. "
                           "Volta/Turing, CPU, unverified devices, and failed CUDA probes cannot enable TF32.")
    return True


_PRECISION_PROBE_RESULTS: dict[tuple, dict[str, Any]] = {}


def _run_precision_case(torch_module: Any, device, precision: str) -> dict[str, Any]:
    """Small real-device GEMM + math-SDPA fwd/bwd, independent of global RNG.

    CPU FP32 references avoid accidentally using the same CUDA kernel as the
    oracle. Math SDPA deliberately remains usable on Volta/Turing. This does
    not attest to fused attention, convolution, AMP scaling or a whole model.
    """
    torch = torch_module
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    generator = torch.Generator(device="cpu").manual_seed(7321)
    tolerance = {"no": 0.005, "fp16": 0.015, "bf16": 0.06}[precision]
    errors = {}

    def check(name, actual, reference):
        actual = actual.detach().float().cpu()
        reference = reference.detach().float().cpu()
        if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
            raise RuntimeError(f"{name} contains non-finite values")
        error = float(torch.linalg.vector_norm(actual - reference) /
                      torch.linalg.vector_norm(reference).clamp_min(1e-7))
        errors[name] = error
        if error > tolerance:
            raise RuntimeError(f"{name} relative L2 {error:.6g} exceeds {tolerance}")

    # Quantize the reference inputs identically, without requiring low-precision
    # CPU matmul support. Local generators preserve the training seed/state.
    with torch.inference_mode(False), torch.enable_grad(), \
            torch.autocast(device_type="cuda", enabled=False), torch.autocast(device_type="cpu", enabled=False):
        for operation, shapes in (("gemm", ((31, 32), (32, 23))),
                                  ("sdpa", ((2, 2, 16, 32),) * 3)):
            source = [torch.randn(shape, generator=generator).to(dtype).float() for shape in shapes]
            reference_inputs = [x.detach().requires_grad_() for x in source]
            device_inputs = [x.to(device=device, dtype=dtype).detach().requires_grad_() for x in source]
            if operation == "gemm":
                reference = reference_inputs[0] @ reference_inputs[1]
                actual = device_inputs[0] @ device_inputs[1]
            else:
                # Force the portable math implementation, not an optimized
                # kernel that happens to be selected on the current GPU.
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.MATH):
                    reference = torch.nn.functional.scaled_dot_product_attention(*reference_inputs, is_causal=True)
                    actual = torch.nn.functional.scaled_dot_product_attention(*device_inputs, is_causal=True)
            upstream = torch.randn(reference.shape, generator=generator).to(dtype).float()
            if actual.dtype != dtype:
                raise RuntimeError(f"{operation} returned {actual.dtype}, but {dtype} was requested")
            reference_grads = torch.autograd.grad(reference, reference_inputs, upstream)
            actual_grads = torch.autograd.grad(actual, device_inputs, upstream.to(device=device, dtype=dtype))
            torch.cuda.synchronize(device)
            check(f"{operation}_forward", actual, reference)
            for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, reference_grads)):
                check(f"{operation}_backward_{index}", actual_grad, expected_grad)
    return {"status": "passed", "relative_l2": errors, "relative_l2_limit": tolerance}


def probe_cuda_precision(torch_module: Any, device=None) -> dict[str, Any]:
    """Memoized exact-device precision smoke, not full GPU/model qualification.

    Missing native SASS is only metadata: compatible cubin/PTX execution can
    pass. Missing binary/library support, invalid numerics or launch failures
    cannot. A CPU/ROCm host never runs the probe and never reports a pass.
    """
    torch = torch_module
    cuda = torch.cuda
    runtime = getattr(getattr(torch, "version", None), "cuda", None)
    report = {"status": "unavailable", "qualified": False,
              "scope": "small GEMM + math SDPA forward/backward precision smoke; not full-model/optional-kernel qualification",
              "torch_version": str(getattr(torch, "__version__", "unknown")),
              "cuda_runtime": runtime, "supported_precisions": [], "tf32_available": False, "cases": {}}
    if not runtime or not cuda.is_available():
        report["reason"] = "An available NVIDIA CUDA runtime is required; CPU/ROCm is not GPU validation."
        return report
    resolved = torch.device("cuda", device) if isinstance(device, int) else torch.device(device or "cuda")
    if resolved.type != "cuda":
        report["reason"] = f"Requested device {resolved} is not NVIDIA CUDA."
        return report
    index = resolved.index if resolved.index is not None else cuda.current_device()
    resolved = torch.device("cuda", index)
    try:
        with cuda.device(index):
            profile = profile_cuda_device(cuda.get_device_name(index), tuple(cuda.get_device_capability(index)))
            arches = tuple(cuda.get_arch_list())
            key = (str(torch.__version__), str(runtime), index, profile.name, profile.capability, arches)
            if key in _PRECISION_PROBE_RESULTS:
                return deepcopy(_PRECISION_PROBE_RESULTS[key])
            report.update(device_index=index, profile=profile.to_dict(), compiled_arches=list(arches),
                          policy_precisions=list(policy_supported_precisions(profile)))
            issue = minimum_cuda_issue(profile, runtime)
            if issue:
                report.update(status="failed", reason=issue)
            else:
                for precision in report["policy_precisions"]:
                    try:
                        result = _run_precision_case(torch, resolved, precision)
                        report["cases"][precision] = result
                        if result["status"] == "passed":
                            report["supported_precisions"].append(precision)
                    except (RuntimeError, ValueError, TypeError, AttributeError, ImportError, OSError) as exc:
                        report["cases"][precision] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                report["tf32_available"] = profile.supports_tf32 and "no" in report["supported_precisions"]
                report["tf32_evidence"] = "known actual-device capability + FP32 launch; TF32 kernel use/performance not measured"
                all_passed = len(report["supported_precisions"]) == len(report["policy_precisions"])
                report["qualified"] = all_passed and profile.known_architecture
                report["status"] = ("passed" if report["qualified"] else
                                    "unverified" if all_passed else
                                    "partial" if report["supported_precisions"] else "failed")
            _PRECISION_PROBE_RESULTS[key] = deepcopy(report)
    except (RuntimeError, ValueError, TypeError, AttributeError, ImportError, OSError) as exc:
        report.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return report
