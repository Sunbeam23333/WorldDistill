from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
ATTN_NO_PAD_PATH = ROOT / "inference/lightx2v/models/networks/hunyuan_video/infer/attn_no_pad.py"
PRE_INFER_PATH = ROOT / "inference/lightx2v/models/networks/hunyuan_video/infer/pre_infer.py"


def _load_attn_no_pad_module():
    einops = types.ModuleType("einops")
    einops.rearrange = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("rearrange is not used by the native SDPA test")
    )
    spec = importlib.util.spec_from_file_location("worlddistill_hunyuan_attn_no_pad", ATTN_NO_PAD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"einops": einops}):
        spec.loader.exec_module(module)
    return module


def test_hunyuan_native_sdpa_masked_attention_is_batch_isolated():
    module = _load_attn_no_pad_module()
    query = torch.zeros(2, 4, 1, 1)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [
            [[1.0], [3.0], [100.0], [100.0]],
            [[10.0], [20.0], [30.0], [100.0]],
        ]
    ).unsqueeze(2)
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    qkv = torch.stack([query, key, value], dim=2)

    output = module.torch_sdpa_no_pad(qkv, mask)

    expected = torch.tensor(
        [
            [[2.0], [2.0], [0.0], [0.0]],
            [[20.0], [20.0], [20.0], [0.0]],
        ]
    ).unsqueeze(2)
    assert output.shape == (2, 4, 1, 1)
    torch.testing.assert_close(output, expected)


def test_hunyuan_token_refiner_uses_the_resolved_dense_backend():
    tree = ast.parse(PRE_INFER_PATH.read_text(encoding="utf-8"))
    refiner = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_individual_token_refiner"
    )
    attention_calls = [
        node
        for node in ast.walk(refiner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "attention"
    ]

    assert attention_calls
    for call in attention_calls:
        backend = next(keyword.value for keyword in call.keywords if keyword.arg == "attn_type")
        assert ast.unparse(backend) == "self.attn_type"
