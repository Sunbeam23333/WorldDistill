"""Quantized-operator prerequisites, not a hardware-validation certificate.

Architecture eligibility and importable callables are necessary but not
sufficient: each installed binary still needs the GPU smoke/sanitizer gate.
Q8F's upstream setup.py targets sm_89 only; the DeepGEMM entry used here is
documented for SM90/SM100. Unknown architectures fail closed.
"""

from collections.abc import Mapping

from cuda_compat import LIGHTX2V_SM120_QUANT_SCHEMES, validate_lightx2v_quant_backend

_INT8_CAPABILITIES = frozenset({(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)})
_FP8_CAPABILITIES = _INT8_CAPABILITIES - {(8, 0), (8, 6)}
_INT8_BACKENDS = frozenset({"int8-triton", "int8-torchao", "int8-vllm", "int8-sgl", "int4-g128-marlin"})
_FP8_BACKENDS = frozenset({"fp8-triton", "fp8-torchao", "fp8-vllm", "fp8-sgl", "fp8-pertensor"})


def validate_quant_backend(scheme: str, capability: tuple[int, int], operators: Mapping[str, object]) -> None:
    """Validate exact architecture policy and every required operator callable."""
    capability = tuple(capability)
    if scheme in LIGHTX2V_SM120_QUANT_SCHEMES:
        validate_lightx2v_quant_backend(scheme, capability, callables_available=bool(operators) and all(callable(op) for op in operators.values()))
        return
    if scheme in _INT8_BACKENDS:
        supported = _INT8_CAPABILITIES
    elif scheme in _FP8_BACKENDS:
        supported = _FP8_CAPABILITIES
    elif scheme in {"int8-q8f", "fp8-q8f"}:
        supported = {(8, 9)}
    elif scheme == "fp8-b128-deepgemm":
        supported = {(9, 0), (10, 0)}
    else:
        raise ValueError(f"No quantization compatibility policy exists for {scheme!r}")
    if capability not in supported:
        targets = ", ".join(f"sm_{major}{minor}" for major, minor in sorted(supported))
        raise RuntimeError(
            f"Quantization backend {scheme!r} is not enabled for sm_{capability[0]}{capability[1]}; "
            f"its architecture prerequisites are {targets}. Use unquantized weights or a compatible backend."
        )
    missing = [name for name, operator in operators.items() if not callable(operator)]
    if not operators or missing:
        raise RuntimeError(
            f"Quantization backend {scheme!r} is missing callable operators: {', '.join(missing) or '(none supplied)'}. "
            "Install/build its optional dependency for this PyTorch/CUDA environment or choose another backend."
        )


def require_quant_backend(scheme: str, operators: Mapping[str, object], *, device=None) -> None:
    """Use the actual tensor device when available, never a different current GPU."""
    import torch

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError(f"Quantization backend {scheme!r} requires an NVIDIA CUDA device")
    resolved = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    if resolved.type != "cuda":
        raise RuntimeError(f"Quantization backend {scheme!r} cannot execute on {resolved}")
    capability = tuple(torch.cuda.get_device_capability(resolved))
    validate_quant_backend(scheme, capability, operators)


def optional_attr(module, name):
    """Handle missing/lazily registered extension symbols without AttributeError."""
    try:
        return getattr(module, name, None)
    except (AttributeError, ImportError, OSError, RuntimeError):
        return None


def fp8_per_token_quantize(x):
    """Reference activation quantizer; FP32 scales keep FP16 zero rows finite."""
    import torch

    values = x.float()
    scale = values.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    quantized = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale
