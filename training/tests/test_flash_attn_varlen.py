from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
FLASH_ATTN_PATH = ROOT / "inference/lightx2v/common/ops/attn/flash_attn.py"


def _fake_flash_kernel(q, _k, v, *_args, **_kwargs):
    return q + v


def _load_flash_attn_module():
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
        module.__path__ = [str(ROOT / "inference" / name.replace(".", "/"))]
        stubs[name] = module

    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.ATTN_WEIGHT_REGISTER = lambda _name: lambda cls: cls
    stubs[registry.__name__] = registry

    template = types.ModuleType("lightx2v.common.ops.attn.template")
    template.AttnWeightTemplate = object
    stubs[template.__name__] = template

    flash_package = types.ModuleType("flash_attn")
    flash_package.__path__ = []
    stubs[flash_package.__name__] = flash_package
    flash_v2 = types.ModuleType("flash_attn.flash_attn_interface")
    flash_v2.flash_attn_varlen_func = _fake_flash_kernel
    stubs[flash_v2.__name__] = flash_v2
    flash_v3 = types.ModuleType("flash_attn_interface")
    flash_v3.flash_attn_varlen_func = _fake_flash_kernel
    stubs[flash_v3.__name__] = flash_v3

    module_name = "lightx2v.common.ops.attn.flash_attn"
    spec = importlib.util.spec_from_file_location(module_name, FLASH_ATTN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def test_flash_adapters_preserve_total_tokens_for_3d_multisegment_input():
    module = _load_flash_attn_module()
    query = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    key = torch.zeros_like(query)
    value = torch.flip(query, dims=(0,))
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    expected = (query + value).reshape(5, 6)

    for operator_cls in (module.FlashAttn2Weight, module.FlashAttn3Weight):
        output = operator_cls().apply(
            query,
            key,
            value,
            cu_seqlens_q=offsets,
            cu_seqlens_kv=offsets,
            max_seqlen_q=3,
            max_seqlen_kv=3,
        )

        assert output.shape == (5, 6)
        torch.testing.assert_close(output, expected)
