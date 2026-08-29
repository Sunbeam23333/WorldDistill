from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAGE_ATTN_PATH = PROJECT_ROOT / "inference/lightx2v/common/ops/attn/sage_attn.py"


def _load_sage_attn_module():
    package_names = (
        "lightx2v",
        "lightx2v.common",
        "lightx2v.common.ops",
        "lightx2v.common.ops.attn",
        "lightx2v.utils",
    )
    stubs = {}
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        stubs[name] = module

    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.ATTN_WEIGHT_REGISTER = lambda _name: lambda cls: cls
    stubs[registry.__name__] = registry

    template = types.ModuleType("lightx2v.common.ops.attn.template")
    template.AttnWeightTemplate = object
    stubs[template.__name__] = template

    sageattention = types.ModuleType("sageattention")
    sageattention.sageattn = lambda q, k, v, **_kwargs: q
    sageattention.sageattn_qk_int8_pv_fp16_triton = (
        lambda q, k, v, **_kwargs: q
    )
    stubs[sageattention.__name__] = sageattention

    sageattn3 = types.ModuleType("sageattn3")
    sageattn3.sageattn3_blackwell = lambda q, k, v, **_kwargs: q
    stubs[sageattn3.__name__] = sageattn3

    module_name = "lightx2v.common.ops.attn.sage_attn"
    spec = importlib.util.spec_from_file_location(module_name, SAGE_ATTN_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


SAGE_ATTN = _load_sage_attn_module()


class _FakeSageKernel:
    def __init__(self, layout: str):
        self.layout = layout
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tensor_layout: str | None = None,
        is_causal: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        if self.layout == "NHD":
            if tensor_layout != "NHD":
                raise AssertionError(f"expected NHD layout, got {tensor_layout!r}")
            q_hnd = q.transpose(1, 2)
            k_hnd = k.transpose(1, 2)
            v_hnd = v.transpose(1, 2)
        else:
            if tensor_layout is not None:
                raise AssertionError("SageAttention3 does not take tensor_layout")
            q_hnd, k_hnd, v_hnd = q, k, v

        self.calls.append(
            {
                "is_causal": is_causal,
                "q_len": q_hnd.shape[2],
                "kv_len": k_hnd.shape[2],
                "batch": q_hnd.shape[0],
            }
        )
        output = F.scaled_dot_product_attention(
            q_hnd,
            k_hnd,
            v_hnd,
            is_causal=is_causal,
        )
        return output.transpose(1, 2) if self.layout == "NHD" else output


class SageAttentionVarlenTests(unittest.TestCase):
    def _operator(self, backend: str):
        if backend == "sage_attn2":
            kernel = _FakeSageKernel("NHD")
            SAGE_ATTN.sageattn = kernel
            return SAGE_ATTN.SageAttn2Weight(), kernel
        kernel = _FakeSageKernel("HND")
        SAGE_ATTN.sageattn3_blackwell = kernel
        return SAGE_ATTN.SageAttn3Weight(), kernel

    def test_packed_segments_do_not_attend_across_boundaries(self) -> None:
        q = torch.zeros(4, 1, 2)
        k = torch.zeros(4, 1, 2)
        v = torch.tensor(
            [[[1.0, 1.0]], [[1.0, 1.0]], [[100.0, 100.0]], [[100.0, 100.0]]]
        )
        offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
        expected = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [100.0, 100.0], [100.0, 100.0]]
        )

        for backend in ("sage_attn2", "sage_attn3"):
            with self.subTest(backend=backend):
                operator, kernel = self._operator(backend)
                output = operator.apply(
                    q,
                    k,
                    v,
                    cu_seqlens_q=offsets,
                    cu_seqlens_kv=offsets,
                    max_seqlen_q=2,
                    max_seqlen_kv=2,
                )

                torch.testing.assert_close(output, expected)
                self.assertEqual(len(kernel.calls), 2)
                self.assertEqual(
                    [(call["q_len"], call["kv_len"]) for call in kernel.calls],
                    [(2, 2), (2, 2)],
                )

    def test_dense_and_packed_4d_outputs_flatten_without_batch_leakage(self) -> None:
        dense_q = torch.zeros(2, 3, 1, 2)
        dense_k = torch.zeros(2, 2, 1, 2)
        dense_v = torch.tensor(
            [
                [[[1.0, 1.0]], [[3.0, 3.0]]],
                [[[100.0, 100.0]], [[200.0, 200.0]]],
            ]
        )
        expected_dense = torch.tensor(
            [
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [150.0, 150.0],
                [150.0, 150.0],
                [150.0, 150.0],
            ]
        )
        packed_q = torch.zeros(2, 2, 1, 2)
        packed_k = torch.zeros(2, 2, 1, 2)
        packed_v = torch.tensor(
            [
                [[[1.0, 1.0]], [[1.0, 1.0]]],
                [[[100.0, 100.0]], [[100.0, 100.0]]],
            ]
        )
        offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
        expected_packed = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [100.0, 100.0], [100.0, 100.0]]
        )

        for backend in ("sage_attn2", "sage_attn3"):
            with self.subTest(backend=backend):
                operator, kernel = self._operator(backend)
                dense_output = operator.apply(dense_q, dense_k, dense_v)
                self.assertEqual(dense_output.shape, (6, 2))
                torch.testing.assert_close(dense_output, expected_dense)
                self.assertEqual(kernel.calls[0]["batch"], 2)

                kernel.calls.clear()
                packed_output = operator.apply(
                    packed_q,
                    packed_k,
                    packed_v,
                    cu_seqlens_q=offsets,
                    cu_seqlens_kv=offsets,
                )
                self.assertEqual(packed_output.shape, (4, 2))
                torch.testing.assert_close(packed_output, expected_packed)
                self.assertEqual(len(kernel.calls), 2)
                self.assertTrue(all(call["batch"] == 1 for call in kernel.calls))

    def test_causal_is_forwarded_for_every_packed_segment(self) -> None:
        q = torch.zeros(4, 1, 1)
        k = torch.zeros(4, 1, 1)
        v = torch.tensor([[[1.0]], [[3.0]], [[10.0]], [[14.0]]])
        offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
        expected = torch.tensor([[1.0], [2.0], [10.0], [12.0]])

        for backend in ("sage_attn2", "sage_attn3"):
            with self.subTest(backend=backend):
                operator, kernel = self._operator(backend)
                output = operator.apply(
                    q,
                    k,
                    v,
                    cu_seqlens_q=offsets,
                    cu_seqlens_kv=offsets,
                    causal=True,
                )

                torch.testing.assert_close(output, expected)
                self.assertEqual(
                    [call["is_causal"] for call in kernel.calls],
                    [True, True],
                )

    def test_noncausal_packed_cross_attention_accepts_different_lengths(self) -> None:
        q = torch.zeros(4, 1, 1)
        k = torch.zeros(4, 1, 1)
        v = torch.tensor([[[2.0]], [[4.0]], [[10.0]], [[14.0]]])

        for backend in ("sage_attn2", "sage_attn3"):
            with self.subTest(backend=backend):
                operator, kernel = self._operator(backend)
                output = operator.apply(
                    q,
                    k,
                    v,
                    cu_seqlens_q=torch.tensor([0, 1, 4], dtype=torch.int32),
                    cu_seqlens_kv=torch.tensor([0, 2, 4], dtype=torch.int32),
                )

                torch.testing.assert_close(
                    output,
                    torch.tensor([[3.0], [12.0], [12.0], [12.0]]),
                )
                self.assertEqual(
                    [(call["q_len"], call["kv_len"]) for call in kernel.calls],
                    [(1, 2), (3, 2)],
                )

    def test_invalid_packed_contracts_fail_before_launch(self) -> None:
        q = k = v = torch.zeros(4, 1, 2)
        offsets = torch.tensor([0, 2, 4], dtype=torch.int32)

        for backend in ("sage_attn2", "sage_attn3"):
            with self.subTest(backend=backend):
                operator, kernel = self._operator(backend)
                with self.assertRaisesRegex(ValueError, "provided together"):
                    operator.apply(q, k, v, cu_seqlens_q=offsets)
                with self.assertRaisesRegex(ValueError, "integer dtype"):
                    operator.apply(
                        q,
                        k,
                        v,
                        cu_seqlens_q=offsets.float(),
                        cu_seqlens_kv=offsets.float(),
                    )
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    operator.apply(
                        q,
                        k,
                        v,
                        cu_seqlens_q=torch.tensor([0, 2, 2, 4]),
                        cu_seqlens_kv=torch.tensor([0, 2, 2, 4]),
                    )
                with self.assertRaisesRegex(ValueError, "same number of segments"):
                    operator.apply(
                        q,
                        k,
                        v,
                        cu_seqlens_q=offsets,
                        cu_seqlens_kv=torch.tensor([0, 4]),
                    )
                with self.assertRaisesRegex(ValueError, "preserve every batch boundary"):
                    operator.apply(
                        q.reshape(2, 2, 1, 2),
                        k.reshape(2, 2, 1, 2),
                        v.reshape(2, 2, 1, 2),
                        cu_seqlens_q=torch.tensor([0, 1, 4]),
                        cu_seqlens_kv=torch.tensor([0, 1, 4]),
                    )
                with self.assertRaisesRegex(ValueError, "every packed segment"):
                    operator.apply(
                        torch.zeros(3, 1, 2),
                        k,
                        v,
                        cu_seqlens_q=torch.tensor([0, 1, 3]),
                        cu_seqlens_kv=offsets,
                        causal=True,
                    )
                with self.assertRaisesRegex(ValueError, "smaller than the packed maximum"):
                    operator.apply(
                        q,
                        k,
                        v,
                        cu_seqlens_q=offsets,
                        cu_seqlens_kv=offsets,
                        max_seqlen_q=1,
                    )
                with self.assertRaisesRegex(ValueError, "custom attention mask"):
                    operator.apply(q, k, v, attn_mask=torch.ones(4, 4))
                with self.assertRaisesRegex(ValueError, "attention dropout"):
                    operator.apply(q, k, v, drop_rate=0.1)
                self.assertEqual(kernel.calls, [])


if __name__ == "__main__":
    unittest.main()
