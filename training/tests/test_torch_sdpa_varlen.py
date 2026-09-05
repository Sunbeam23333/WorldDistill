from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SDPA_PATH = PROJECT_ROOT / "inference/lightx2v/common/ops/attn/torch_sdpa.py"


def _load_torch_sdpa_class():
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

    module_name = "lightx2v.common.ops.attn.torch_sdpa"
    spec = importlib.util.spec_from_file_location(module_name, SDPA_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module.TorchSDPAWeight


TorchSDPAWeight = _load_torch_sdpa_class()


class TorchSDPAVarlenTests(unittest.TestCase):
    def test_dense_4d_output_is_flattened_and_batch_isolated(self) -> None:
        operator = TorchSDPAWeight()
        q = torch.zeros(2, 3, 1, 2)
        k = torch.zeros(2, 2, 1, 2)
        v = torch.tensor(
            [
                [[[1.0, 1.0]], [[3.0, 3.0]]],
                [[[100.0, 100.0]], [[200.0, 200.0]]],
            ]
        )

        output = operator.apply(q, k, v)

        self.assertEqual(output.shape, (6, 2))
        torch.testing.assert_close(
            output,
            torch.tensor(
                [
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [150.0, 150.0],
                    [150.0, 150.0],
                    [150.0, 150.0],
                ]
            ),
        )

    def test_4d_single_batch_accepts_packed_offsets(self) -> None:
        operator = TorchSDPAWeight()
        q = torch.zeros(1, 4, 1, 2)
        k = torch.zeros(1, 4, 1, 2)
        v = torch.tensor(
            [[[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[4.0, 4.0]]]]
        )
        offsets = torch.tensor([0, 4], dtype=torch.int32)

        output = operator.apply(
            q,
            k,
            v,
            cu_seqlens_q=offsets,
            cu_seqlens_kv=offsets,
        )

        self.assertEqual(output.shape, (4, 2))
        torch.testing.assert_close(output, torch.full((4, 2), 2.5))

    def test_4d_two_batch_segments_do_not_leak(self) -> None:
        operator = TorchSDPAWeight()
        q = torch.zeros(2, 2, 1, 2)
        k = torch.zeros(2, 2, 1, 2)
        v = torch.tensor(
            [
                [[[1.0, 1.0]], [[1.0, 1.0]]],
                [[[100.0, 100.0]], [[100.0, 100.0]]],
            ]
        )
        offsets = torch.tensor([0, 2, 4], dtype=torch.int32)

        output = operator.apply(
            q,
            k,
            v,
            cu_seqlens_q=offsets,
            cu_seqlens_kv=offsets,
        )

        self.assertEqual(output.shape, (4, 2))
        torch.testing.assert_close(
            output,
            torch.tensor(
                [[1.0, 1.0], [1.0, 1.0], [100.0, 100.0], [100.0, 100.0]]
            ),
        )

    def test_packed_segments_do_not_attend_across_boundaries(self) -> None:
        operator = TorchSDPAWeight()
        q = torch.zeros(4, 1, 2)
        k = torch.zeros(4, 1, 2)
        v = torch.tensor(
            [[[1.0, 1.0]], [[1.0, 1.0]], [[100.0, 100.0]], [[100.0, 100.0]]]
        )
        offsets = torch.tensor([0, 2, 4], dtype=torch.int32)

        output = operator.apply(
            q,
            k,
            v,
            cu_seqlens_q=offsets,
            cu_seqlens_kv=offsets,
        )

        expected = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [100.0, 100.0], [100.0, 100.0]]
        )
        torch.testing.assert_close(output, expected)

    def test_cross_attention_supports_different_segment_lengths(self) -> None:
        operator = TorchSDPAWeight()
        q = torch.zeros(4, 1, 1)
        k = torch.zeros(4, 1, 1)
        v = torch.tensor([[[2.0]], [[4.0]], [[10.0]], [[14.0]]])

        output = operator.apply(
            q,
            k,
            v,
            cu_seqlens_q=torch.tensor([0, 1, 4], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 2, 4], dtype=torch.int32),
        )

        torch.testing.assert_close(output, torch.tensor([[3.0], [12.0], [12.0], [12.0]]))

    def test_invalid_offsets_and_global_mask_fail_early(self) -> None:
        operator = TorchSDPAWeight()
        q = k = v = torch.zeros(4, 1, 2)
        offsets = torch.tensor([0, 2, 4], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            operator.apply(
                q,
                k,
                v,
                cu_seqlens_q=torch.tensor([0, 2, 2, 4], dtype=torch.int32),
                cu_seqlens_kv=torch.tensor([0, 2, 2, 4], dtype=torch.int32),
            )
        with self.assertRaisesRegex(ValueError, "cannot be mapped safely"):
            operator.apply(
                q,
                k,
                v,
                cu_seqlens_q=offsets,
                cu_seqlens_kv=offsets,
                attn_mask=torch.ones(4, 4, dtype=torch.bool),
            )

        with self.assertRaisesRegex(ValueError, "end at 4"):
            operator.apply(
                q.reshape(2, 2, 1, 2),
                k.reshape(2, 2, 1, 2),
                v.reshape(2, 2, 1, 2),
                cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
                cu_seqlens_kv=torch.tensor([0, 2], dtype=torch.int32),
            )

        with self.assertRaisesRegex(ValueError, "preserve every batch boundary"):
            operator.apply(
                q.reshape(2, 2, 1, 2),
                k.reshape(2, 2, 1, 2),
                v.reshape(2, 2, 1, 2),
                cu_seqlens_q=torch.tensor([0, 1, 4], dtype=torch.int32),
                cu_seqlens_kv=torch.tensor([0, 1, 4], dtype=torch.int32),
            )

        with self.assertRaisesRegex(ValueError, "provided together"):
            operator.apply(q, k, v, cu_seqlens_q=offsets)

        with self.assertRaisesRegex(ValueError, "same rank"):
            operator.apply(
                q.reshape(1, 4, 1, 2),
                k,
                v,
                cu_seqlens_q=offsets,
                cu_seqlens_kv=offsets,
            )


if __name__ == "__main__":
    unittest.main()
