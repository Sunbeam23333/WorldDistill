import ast
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from quant_compat import fp8_per_token_quantize, require_quant_backend, validate_quant_backend

_PATH = Path(__file__).resolve().parents[2] / "inference/lightx2v/common/ops/mm/triton_kernels.py"
_SPEC = importlib.util.spec_from_file_location("quant_kernel_contract_subject", _PATH)
subject = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(subject)


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0), (10, 3)])
def test_int8_prerequisite_profiles(capability):
    validate_quant_backend("int8-triton", capability, {"jit": lambda: None})


@pytest.mark.parametrize("scheme", ["fp8-triton", "fp8-vllm", "fp8-sgl", "fp8-torchao", "fp8-pertensor"])
def test_a100_rejects_native_fp8(scheme):
    with pytest.raises(RuntimeError, match="sm_80"):
        validate_quant_backend(scheme, (8, 0), {"operator": lambda: None})


@pytest.mark.parametrize("scheme", ["int8-vllm", "int8-sgl", "int8-torchao", "fp8-vllm", "fp8-sgl", "fp8-pertensor"])
def test_imported_module_does_not_replace_callable_check(scheme):
    with pytest.raises(RuntimeError, match="missing callable"):
        validate_quant_backend(scheme, (9, 0), {"gemm": object()})


def test_exact_architecture_extensions_fail_closed():
    validate_quant_backend("int8-q8f", (8, 9), {"gemm": lambda: None})
    with pytest.raises(RuntimeError, match="sm_90"):
        validate_quant_backend("int8-q8f", (9, 0), {"gemm": lambda: None})
    with pytest.raises(RuntimeError, match="sm_103"):
        validate_quant_backend("fp8-b128-deepgemm", (10, 3), {"gemm": lambda: None})
    with pytest.raises(RuntimeError, match="sm_130"):
        validate_quant_backend("int8-triton", (13, 0), {"jit": lambda: None})


def test_actual_tensor_device_is_checked():
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.version, "cuda", "12.8"), patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)) as capability:
        require_quant_backend("fp8-triton", {"jit": lambda: None}, device="cuda:3")
        capability.assert_called_once_with(torch.device("cuda:3"))
        with pytest.raises(RuntimeError, match="cannot execute"):
            require_quant_backend("int8-triton", {"jit": lambda: None}, device="cpu")


def test_rocm_cuda_namespace_does_not_qualify_nvidia_operators():
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.version, "cuda", None):
        with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
            require_quant_backend("int8-triton", {"jit": lambda: None}, device="cuda:0")


def test_quant_checker_never_qualifies_skipped_or_empty_cases():
    from tools.check_quant_kernels import qualification_status
    assert qualification_status([]) == "partial"
    assert qualification_status([{"status": "passed"}, {"status": "skipped"}]) == "partial"
    assert qualification_status([{"status": "passed"}, {"status": "failed"}]) == "failed"
    assert qualification_status([{"status": "passed"}]) == "passed"


def test_fp16_zero_row_reference_fp8_quantizer_has_finite_scales():
    quantized, scales = fp8_per_token_quantize(torch.zeros(3, 17, dtype=torch.float16))
    assert scales.dtype == torch.float32
    assert torch.isfinite(scales).all() and (scales > 0).all()
    assert torch.isfinite(quantized.float()).all() and torch.count_nonzero(quantized.float()) == 0


def inputs():
    return torch.ones(3, 129, dtype=torch.int8), torch.ones(7, 129, dtype=torch.int8), torch.ones(3), torch.ones(7)


def test_gemm_shapes_and_per_row_scales():
    a, b, sa, sb = inputs()
    assert subject._validate_gemm_inputs(a, b, sa, sb, None, int8=True, output_dtype=torch.float32) == (3, 7)
    with pytest.raises(ValueError, match="matching K"):
        subject._validate_gemm_inputs(a, b[:, :-1], sa, sb, None, int8=True, output_dtype=torch.float32)
    with pytest.raises(ValueError, match="per-row scales"):
        subject._validate_gemm_inputs(a, b, sa[:1], sb, None, int8=True, output_dtype=torch.float32)
    with pytest.raises(ValueError, match="bias"):
        subject._validate_gemm_inputs(a, b, sa, sb, torch.ones(8), int8=True, output_dtype=torch.float32)


def test_all_device_loads_have_explicit_masks():
    # Structural guard is deliberately separate from the real GPU numerical gate.
    tree = ast.parse(_PATH.read_text())
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
    assert len(loads) == 6
    assert all(any(keyword.arg == "mask" for keyword in node.keywords) for node in loads)
    assert "EVEN_K=True" not in _PATH.read_text()


@pytest.mark.skipif(not torch.cuda.is_available() or subject.triton is None, reason="real CUDA + Triton required")
@pytest.mark.parametrize("int8", [True, False])
@pytest.mark.parametrize("shape", [(1, 1, 1), (37, 71, 129), (3, 5, 63), (129, 131, 127)])
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_real_cuda_tail_gemm(int8, shape, bias_enabled):
    if not int8 and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native FP8 requires SM89+")
    m, n, k = shape
    torch.manual_seed(23)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(n, k, device="cuda")
    a[0].zero_()
    quantize = subject.int8_quantize_triton if int8 else subject.fp8_quantize_triton
    qa, sa = quantize(a)
    qb, sb = quantize(b)
    assert torch.isfinite(sa).all() and torch.isfinite(sb).all()
    assert torch.count_nonzero(qa[0].float()) == 0
    bias = torch.randn(n, device="cuda") if bias_enabled else None
    # Double accumulation avoids TF32 influencing the reference.
    reference = (qa.double() @ qb.double().T) * sa.double()[:, None] * sb.double()[None, :]
    if bias is not None:
        reference += bias.double()
    result = subject._gemm(qa, qb, sa, sb, bias, int8=int8, output_dtype=torch.float32)
    torch.testing.assert_close(result, reference.float(), atol=1e-4, rtol=2e-4)
    torch.cuda.synchronize()
