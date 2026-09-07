"""CPU semantic tests; optional-kernel stubs do not qualify CUDA hardware."""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from cuda_compat import import_attention_callable, profile_cuda_device, resolve_attention_config, select_attention_backend, probe_flash_attention4


ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "inference"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, INFERENCE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load("lightx2v/utils/attention.py", "worlddistill_dense_attention_contract")


def load_adapter(name):
    stubs = {}
    for package in ("lightx2v", "lightx2v.common", "lightx2v.common.ops", "lightx2v.common.ops.attn", "lightx2v.utils"):
        module = types.ModuleType(package)
        module.__path__ = [str(INFERENCE / package.replace(".", "/"))]
        stubs[package] = module
    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.ATTN_WEIGHT_REGISTER = lambda _name: lambda cls: cls
    stubs[registry.__name__] = registry
    stubs["lightx2v.utils.attention"] = contract
    with patch.dict(sys.modules, stubs):
        return load(f"lightx2v/common/ops/attn/{name}.py", f"lightx2v.common.ops.attn.{name}")


def extract_method(path, class_name, method_name, **scope):
    tree = ast.parse((INFERENCE / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    namespace = {"torch": torch, "dense_attention": contract.attention, "AI_DEVICE": "cpu", **scope}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("q_len,k_len", [(2, 5), (5, 2)])
def test_native_cached_causal_alignment_and_gqa(q_len, k_len):
    q = torch.zeros(2, q_len, 4, 2)
    k = torch.zeros(2, k_len, 2, 2)
    v = torch.arange(1, k_len + 1, dtype=torch.float32).reshape(1, k_len, 1, 1).expand(2, -1, 2, 2)
    out = contract.native_attention(q, k, v, causal=True)
    expected = [0.0 if i + k_len - q_len < 0 else (i + k_len - q_len + 2) / 2 for i in range(q_len)]
    torch.testing.assert_close(out[0, :, 0, 0], torch.tensor(expected))
    assert out.shape == q.shape


def test_packed_segments_preserve_isolation_and_gradients():
    q = torch.zeros(3, 2, 2, requires_grad=True)
    k = torch.zeros(5, 1, 2, requires_grad=True)
    v = torch.tensor([1., 3., 10., 20., 30.]).reshape(5, 1, 1).expand(-1, -1, 2).clone().requires_grad_()
    out = contract.attention(q, k, v, backend="torch_sdpa", causal=True,
        cu_seqlens_q=torch.tensor([0, 1, 3]), cu_seqlens_k=torch.tensor([0, 2, 5]))
    torch.testing.assert_close(out[:, 0, 0], torch.tensor([2., 15., 20.]))
    out[0].sum().backward()
    assert torch.count_nonzero(v.grad[2:]) == 0


def test_flash_dense_animate_offsets_and_semantic_arguments():
    module = load_adapter("flash_attn")
    calls = []
    def kernel(q, k, v, cq, ck, mq, mk, **kwargs):
        calls.append((cq.tolist(), ck.tolist(), mq, mk, kwargs))
        return contract.attention(q, k, v, backend="torch_sdpa", cu_seqlens_q=cq,
            cu_seqlens_k=ck, causal=kwargs["causal"], softmax_scale=kwargs["softmax_scale"])
    module.flash_attn_varlen_func = kernel
    module.flash_attn_varlen_func_v3 = kernel
    q = torch.zeros(2, 3, 2, 2)
    k = torch.zeros(2, 2, 1, 2)
    v = torch.tensor([1., 3., 10., 20.]).reshape(2, 2, 1, 1).expand(-1, -1, -1, 2)
    for cls in (module.FlashAttn2Weight, module.FlashAttn3Weight):
        out = cls().apply(q, k, v, max_seqlen_q=3, is_causal=True, scale=0.7)
        expected = contract.native_attention(q, k, v, causal=True, softmax_scale=0.7)
        torch.testing.assert_close(out, expected.reshape(6, 4))
        assert calls[-1][:4] == ([0, 3, 6], [0, 2, 4], 3, 2)
        assert calls[-1][4]["causal"] is True
        assert calls[-1][4]["softmax_scale"] == 0.7
    with pytest.raises(ValueError, match="batch boundary"):
        module.FlashAttn2Weight().apply(q, k, v, cu_seqlens_q=torch.tensor([0, 6]), cu_seqlens_kv=torch.tensor([0, 4]))
    with pytest.raises(ValueError, match="dropout"):
        module.FlashAttn3Weight().apply(q, k, v, dropout_p=0.5)


def test_fa4_uses_keyword_metadata_not_qv_position():
    module = load_adapter("flash_attn")
    def kernel(q, k, v, qv=None, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        assert qv is None
        assert cu_seqlens_q.tolist() == [0, 2, 4]
        return q + v
    module.flash_attn_varlen_func_v4 = kernel
    q = torch.zeros(2, 2, 1, 2)
    torch.testing.assert_close(module.FlashAttn4Weight().apply(q, q, q), torch.zeros(4, 2))


@pytest.mark.parametrize("name", ["sage_attn_no_pad_v2", "sage_attn_no_pad_v3", "flash_attn_no_pad", "flash_attn_no_pad_v3", "flash_attn_no_pad_v4"])
def test_hunyuan_no_pad_backends_do_not_mix_samples_or_require_fa_padding(name):
    module = load("lightx2v/models/networks/hunyuan_video/infer/attn_no_pad.py", "worlddistill_no_pad_contract")
    qkv = torch.zeros(3, 3, 3, 1, 1)
    qkv[0, :, 2] = torch.tensor([1., 3., 100.]).reshape(3, 1, 1)
    qkv[1, :, 2] = torch.tensor([10., 20., 30.]).reshape(3, 1, 1)
    mask = torch.tensor([[True, True, False], [True, False, True], [False, False, False]])
    with patch.dict(sys.modules, {"lightx2v.utils.attention": contract}):
        out = getattr(module, name)(qkv, mask, causal=True)
    torch.testing.assert_close(out[..., 0, 0], torch.tensor([[1., 2., 0.], [10., 0., 20.], [0., 0., 0.]]))


@pytest.mark.parametrize("causal", [False, True])
def test_ring_portable_lse_matches_dense_and_merge(causal):
    generator = torch.Generator().manual_seed(41)
    q = torch.randn(2, 7, 4, 3, generator=generator)
    k = torch.randn(2, 5, 2, 3, generator=generator)
    v = torch.randn(2, 5, 2, 3, generator=generator)
    out, lse = contract.attention_with_lse(q, k, v, backend="torch_sdpa", causal=causal, query_chunk=3, key_chunk=2)
    torch.testing.assert_close(out, contract.native_attention(q, k, v, causal=causal), atol=1e-6, rtol=1e-5)
    scores = q.transpose(1, 2) @ k.repeat_interleave(2, dim=2).transpose(1, 2).transpose(-1, -2) / 3 ** .5
    if causal:
        scores = scores.masked_fill(~(torch.arange(5)[None, :] <= torch.arange(7)[:, None] - 2), float("-inf"))
    torch.testing.assert_close(lse, scores.logsumexp(-1))
    if not causal:
        first, lse1 = contract.attention_with_lse(q, k[:, :2], v[:, :2], backend="torch_sdpa")
        last, lse2 = contract.attention_with_lse(q, k[:, 2:], v[:, 2:], backend="torch_sdpa")
        weight = torch.sigmoid(lse1 - lse2).transpose(1, 2).unsqueeze(-1)
        torch.testing.assert_close(first * weight + last * (1 - weight), out, atol=1e-6, rtol=1e-5)


def test_matrix_action_calls_work_with_no_flash_packages():
    path = "lightx2v/models/networks/wan/infer/matrix_game2/transformer_infer.py"
    method = extract_method(path, "WanMtxg2TransformerInfer", "_action_attention")
    q = torch.zeros(2, 2, 1, 2)
    v = torch.tensor([1., 3., 10., 30.]).reshape(2, 2, 1, 1).expand_as(q)
    out = method(types.SimpleNamespace(config={"attn_type": "torch_sdpa"}), q, q, v)
    torch.testing.assert_close(out[:, 0, 0, 0], torch.tensor([2., 20.]))
    tree = ast.parse((INFERENCE / path).read_text())
    assert not any(isinstance(node, ast.Name) and node.id in {"flash_attn_func", "flash_attn_interface", "FLASH_ATTN_3_AVAILABLE"} for node in ast.walk(tree))


def test_audio_adapter_forward_uses_the_dense_contract():
    method = extract_method("lightx2v/models/networks/wan/infer/audio/transformer_infer.py", "WanAudioTransformerInfer", "perceiver_attention_ca")
    identity = types.SimpleNamespace(apply=lambda x: x)
    phase = types.SimpleNamespace(norm_kv=identity, norm_q=identity, to_q=identity,
        to_kv=types.SimpleNamespace(apply=lambda x: torch.cat((torch.zeros_like(x), x), dim=-1)),
        to_out=identity, shift_scale_gate=types.SimpleNamespace(tensor=torch.zeros(1, 3, 2)))
    owner = types.SimpleNamespace(config={"attn_type": "torch_sdpa"}, num_heads=1, head_dim=2,
        perceiver_attn_cu_seqlens_q=torch.tensor([0, 2]), perceiver_attn_cu_seqlens_k=torch.tensor([0, 3]),
        max_seqlen_q=2, max_seqlen_k=3)
    temb = torch.zeros(1, 3, 2); temb[:, 2] = 1
    result = method(owner, phase, torch.tensor([[1., 1.], [3., 3.], [5., 5.]]), torch.zeros(2, 2), temb)
    torch.testing.assert_close(result, torch.full((2, 2), 3.))


def test_bagel_cached_causal_prefill_no_longer_requires_flash():
    method = extract_method("lightx2v/models/networks/bagel/infer/transformer_infer.py", "BagelTransformerInfer", "self_attn",
        apply_rotary_pos_emb=lambda q, k, *_args, **_kwargs: (q, k))
    identity = types.SimpleNamespace(apply=lambda x: x)
    zero = types.SimpleNamespace(apply=lambda x: torch.zeros_like(x))
    weights = types.SimpleNamespace(q_proj=zero, k_proj=zero, v_proj=identity, q_norm=identity, k_norm=identity, o_proj=identity)
    owner = types.SimpleNamespace(config={"attn_type": "torch_sdpa"}, num_heads=1, num_key_value_heads=1, head_dim=2, hidden_size=2)
    past = types.SimpleNamespace(key_cache=[torch.zeros(2, 1, 2, dtype=torch.bfloat16)],
        value_cache=[torch.tensor([1., 3.], dtype=torch.bfloat16).reshape(2, 1, 1).expand(2, 1, 2)])
    out, _ = method(owner, weights, torch.tensor([[5., 5.]]), torch.tensor([1]), (None, None),
        torch.tensor([2]), past, torch.tensor([2]), torch.tensor([0, 1]), False, True, "und", None, None, 0)
    torch.testing.assert_close(out, torch.full((1, 2), 3., dtype=torch.bfloat16))


def test_modern_fa3_namespace_and_broken_legacy_are_handled():
    sentinel = lambda: None
    def importer(name):
        if name == "flash_attn_3.flash_attn_interface":
            return types.SimpleNamespace(flash_attn_varlen_func=sentinel)
        raise OSError("incompatible binary")
    with patch("cuda_compat.importlib.import_module", side_effect=importer):
        assert import_attention_callable("flash_attn3") is sentinel
        assert import_attention_callable("flash_attn2") is None


def test_fa4_importability_is_not_hardware_qualification():
    profile = profile_cuda_device("B200", (10, 0))
    assert select_attention_backend("flash_attn4", profile, {"flash_attn4", "torch_sdpa"}) == "torch_sdpa"
    with pytest.raises(RuntimeError):
        select_attention_backend("flash_attn4", profile, {"flash_attn4", "torch_sdpa"}, strict=True)
    assert not probe_flash_attention4(torch, profile)["passed"]  # CPU run, not mocked CUDA success.
    config = {"attn_type": "flash_attn4"}
    with patch("cuda_compat.probe_flash_attention4", return_value={"passed": False, "reason": "simulated failure"}):
        resolve_attention_config(config, torch_module=torch, profile=profile, available_backends={"flash_attn4", "torch_sdpa"})
    assert config["attn_type"] == "torch_sdpa"


def test_fa3_extension_version_minimum_is_not_hopper_device_minimum():
    config = {"attn_type": "flash_attn3"}
    fake_torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda="12.2"))
    resolve_attention_config(config, torch_module=fake_torch, profile=profile_cuda_device("H100", (9, 0)),
        available_backends={"flash_attn3", "torch_sdpa"})
    assert config["attn_type"] == "torch_sdpa"


@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_ring_single_rank_padding_and_fusion_use_portable_subkernel(fusion, padding):
    path = INFERENCE / "lightx2v/common/ops/attn/ring_attn.py"
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]
    namespace = {"torch": torch, "AttnWeightTemplate": object,
        "ATTN_WEIGHT_REGISTER": lambda _name: lambda cls: cls,
        "dist": types.SimpleNamespace(get_rank=lambda *_: 0, get_world_size=lambda *_: 1),
        "RingComm": lambda *_: None, "GET_DTYPE": lambda: torch.float32,
        "dense_attention": contract.attention, "attention_with_lse": contract.attention_with_lse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    operator = namespace["RingAttnWeight"]()
    size = 5 if padding else 3
    q = torch.zeros(size, 1, 2)
    v = torch.tensor([1., 3., 5., 100., 200.][:size]).reshape(size, 1, 1).expand(size, 1, 2)
    offsets = torch.tensor([0, 3, 5] if padding else [0, 3], dtype=torch.int32)
    actual = operator.apply(q, q, v, slice_qkv_len=2, cu_seqlens_qkv=offsets,
        attention_type="torch_sdpa", use_tensor_fusion=fusion)
    expected = torch.tensor([3., 3., 3., 150., 150.][:size]).reshape(size, 1).expand(size, 2)
    torch.testing.assert_close(actual, expected)


def test_cuda_acceptance_cli_cannot_qualify_cpu(monkeypatch, tmp_path, capsys):
    tool = load("../tools/check_attention_kernels.py", "worlddistill_attention_acceptance")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "unavailable.json"
    assert tool.main(["--backends", "all", "--output", str(output)]) == 2
    import json
    report = json.loads(output.read_text())
    assert report["status"] == "unavailable"
    assert report["qualified"] is False
    assert not report["results"]
    capsys.readouterr()


def test_acceptance_error_metric_rejects_nonfinite_values():
    tool = load("../tools/check_attention_kernels.py", "worlddistill_attention_acceptance_metrics")
    assert tool._error(torch.tensor([float("nan")]), torch.ones(1))["finite"] is False
    assert tool._error(torch.zeros(2), torch.ones(2))["relative_l2"] == 1.0
