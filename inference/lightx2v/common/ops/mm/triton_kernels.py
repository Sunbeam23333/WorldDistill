"""Tail-safe per-row INT8/FP8 quantization and inference linear operators.

All GEMM entry points share a masked kernel so bias/scales and M/N/K tails
cannot diverge. These operators are inference-only, not autograd GEMMs.
"""

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError, RuntimeError):
    triton = None
    tl = None

from quant_compat import require_quant_backend


if triton is not None:

    @triton.jit
    def _row_quantize_kernel(X, OUT, SCALES, K: tl.constexpr, BLOCK: tl.constexpr, INT8: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        values = tl.load(X + row * K + col, mask=col < K, other=0.0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-8)
        if INT8:
            scale = absmax / 127.0
            scaled = values / scale
            # Round before casting: casting +/-0.5 to int8 would give zero.
            quantized = tl.where(scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
            quantized = tl.minimum(tl.maximum(quantized, -127.0), 127.0)
        else:
            scale = absmax / 448.0
            quantized = tl.minimum(tl.maximum(values / scale, -448.0), 448.0)
        tl.store(OUT + row * K + col, quantized, mask=col < K)
        tl.store(SCALES + row, scale)

    @triton.autotune(
        configs=[
            triton.Config({"BM": 32, "BN": 64, "BK": 64}, num_stages=3, num_warps=4),
            triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_stages=3, num_warps=4),
            triton.Config({"BM": 64, "BN": 128, "BK": 128}, num_stages=3, num_warps=8),
        ],
        key=["M", "N", "K", "INT8", "HAS_BIAS", "FUSE_GELU"],
    )
    @triton.jit
    def _quantized_gemm_kernel(
        A, B, A_SCALES, B_SCALES, BIAS, OUT, M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk,
        INT8: tl.constexpr, HAS_BIAS: tl.constexpr, FUSE_GELU: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        reduction = tl.arange(0, BK)
        if INT8:
            acc = tl.zeros((BM, BN), tl.int32)
        else:
            acc = tl.zeros((BM, BN), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + reduction
            a = tl.load(
                A + rows[:, None] * stride_am + k[None, :] * stride_ak,
                mask=(rows[:, None] < M) & (k[None, :] < K), other=0,
            )
            b = tl.load(
                B + cols[None, :] * stride_bn + k[:, None] * stride_bk,
                mask=(cols[None, :] < N) & (k[:, None] < K), other=0,
            )
            acc += tl.dot(a, b)
        a_scale = tl.load(A_SCALES + rows, mask=rows < M, other=0.0).to(tl.float32)
        b_scale = tl.load(B_SCALES + cols, mask=cols < N, other=0.0).to(tl.float32)
        result = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(BIAS + cols, mask=cols < N, other=0.0).to(tl.float32)
            result += bias[None, :]
        if FUSE_GELU:
            result = result * tl.sigmoid(result * 1.702)
        tl.store(OUT + rows[:, None] * N + cols[None, :], result,
                 mask=(rows[:, None] < M) & (cols[None, :] < N))


def _quantize(x, *, int8):
    if x.ndim < 1 or x.shape[-1] == 0 or not x.is_floating_point():
        raise ValueError("quantization requires floating input with a nonempty last dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("quantization supports FP16, BF16 or FP32 input")
    scheme = "int8-triton" if int8 else "fp8-triton"
    require_quant_backend(scheme, {"triton": triton.jit if triton is not None else None}, device=x.device)
    x = x.contiguous()
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    output = torch.empty_like(x, dtype=torch.int8 if int8 else torch.float8_e4m3fn)
    scales = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    if rows:
        _row_quantize_kernel[(rows,)](x, output, scales, K=width,
                                     BLOCK=triton.next_power_of_2(width), INT8=int8)
    return output, scales


def int8_quantize_triton(x):
    return _quantize(x, int8=True)


def fp8_quantize_triton(x):
    return _quantize(x, int8=False)


def _validate_gemm_inputs(a, b, a_scales, b_scales, bias, *, int8, output_dtype):
    """Check the pointer/shape contract before launching a device kernel."""
    expected = torch.int8 if int8 else getattr(torch, "float8_e4m3fn", None)
    if a.ndim < 2 or b.ndim != 2 or a.shape[-1] != b.shape[1]:
        raise ValueError("linear GEMM requires A[..., K], B[N, K] with matching K")
    if a.dtype != expected or b.dtype != expected:
        raise TypeError(f"linear GEMM requires both inputs to have dtype {expected}")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("quantized GEMM output must be FP16, BF16 or FP32")
    if a.shape[-1] == 0:
        raise ValueError("quantized GEMM requires nonempty K")
    m, n = a.numel() // a.shape[-1], b.shape[0]
    if a_scales.numel() != m or b_scales.numel() != n:
        raise ValueError(f"per-row scales must have {m} activation and {n} weight entries")
    if bias is not None and (bias.ndim != 1 or bias.numel() != n):
        raise ValueError(f"bias must be a vector of length {n}")
    for name, tensor in (("B", b), ("A scales", a_scales), ("B scales", b_scales), ("bias", bias)):
        if tensor is not None and tensor.device != a.device:
            raise ValueError(f"{name} must be on the same device as A")
    for tensor in (a_scales, b_scales, bias):
        if tensor is not None and not tensor.is_floating_point():
            raise TypeError("scales and bias must be floating-point tensors")
        if tensor is not None and tensor.requires_grad:
            raise ValueError("quantized Triton GEMM is inference-only; use an autograd-aware training operator")
    return m, n


def _gemm(a, b, a_scales, b_scales, bias=None, *, int8, fuse_gelu=False, output_dtype=None):
    output_dtype = torch.float16 if output_dtype is None else output_dtype
    m, n = _validate_gemm_inputs(a, b, a_scales, b_scales, bias, int8=int8, output_dtype=output_dtype)
    scheme = "int8-triton" if int8 else "fp8-triton"
    require_quant_backend(scheme, {"triton": triton.jit if triton is not None else None}, device=a.device)
    shape = a.shape[:-1] + (n,)
    a = a.reshape(-1, a.shape[-1])
    # The kernel expects dense scales, including when callers supply a strided view.
    a_scales = a_scales.contiguous().view(-1)
    b_scales = b_scales.contiguous().view(-1)
    if bias is not None:
        bias = bias.contiguous()
    output = torch.empty(shape, dtype=output_dtype, device=a.device)
    if m and n:
        grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))
        _quantized_gemm_kernel[grid](
            a, b, a_scales, b_scales, bias if bias is not None else a_scales, output,
            m, n, a.shape[-1], a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            INT8=int8, HAS_BIAS=bias is not None, FUSE_GELU=fuse_gelu,
        )
    return output


def int8_gemm_triton(a, b, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    return _gemm(a, b, a_scales, b_scales, int8=True, fuse_gelu=fuse_gelu, output_dtype=output_dtype)


def int8_gemm_bias_triton(a, b, bias, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    return _gemm(a, b, a_scales, b_scales, bias, int8=True, fuse_gelu=fuse_gelu, output_dtype=output_dtype)


def fp8_gemm_triton(a, b, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    return _gemm(a, b, a_scales, b_scales, int8=False, fuse_gelu=fuse_gelu, output_dtype=output_dtype)


def fp8_gemm_bias_triton(a, b, bias, a_scales, b_scales, fuse_gelu=False, output_dtype=None):
    return _gemm(a, b, a_scales, b_scales, bias, int8=False, fuse_gelu=fuse_gelu, output_dtype=output_dtype)
