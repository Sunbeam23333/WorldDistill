"""CPU policy/control-flow tests; mocked CUDA results are not hardware evidence."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

import cuda_compat as subject
from quant_compat import quant_backend_capabilities, validate_quant_backend
from tools import check_cuda_compat as cli


@pytest.mark.parametrize("cc,architecture,minimum", [
    ((5, 0), "maxwell", "6.5"), ((5, 2), "maxwell", "6.5"),
    ((5, 3), "maxwell-jetson", "7.0"), ((6, 0), "pascal", "8.0"),
    ((6, 1), "pascal", "8.0"), ((6, 2), "pascal-jetson", "8.0"),
    ((7, 0), "volta", "9.0"), ((7, 2), "volta-jetson", "9.0"),
    ((7, 5), "turing", "10.0"), ((8, 6), "ampere", "11.1"),
    ((8, 7), "ampere-jetson", "11.4"), ((11, 0), "blackwell-jetson", "13.0"),
    ((12, 1), "blackwell", "12.9"),
])
def test_official_architecture_rows(cc, architecture, minimum):
    profile = subject.profile_cuda_device("test", cc)
    assert profile.architecture == architecture
    assert profile.minimum_cuda == minimum
    assert profile.known_architecture
    if cc < (8, 0):
        assert not profile.supports_bf16 and not profile.supports_tf32
        assert "bf16" not in subject.policy_supported_precisions(profile)
        assert profile.attention_preference == ("torch_sdpa",)


@pytest.mark.parametrize("cc", [(5, 0), (6, 1), (7, 0), (7, 2)])
def test_legacy_cuda_lifecycle_is_not_fixed_by_newer_driver(cc):
    profile = subject.profile_cuda_device("legacy", cc)
    assert subject.minimum_cuda_issue(profile, "12.6") is None
    assert "legacy CUDA" in subject.minimum_cuda_issue(profile, "13.0")


def test_sm86_is_not_unknown_but_jetson_and_future_kernels_stay_closed():
    ampere = subject.profile_cuda_device("RTX 3090", (8, 6))
    assert ampere.supports_bf16 and ampere.supports_tf32
    assert subject.select_attention_backend("auto", ampere, {"flash_attn2", "torch_sdpa"}) == "flash_attn2"
    for cc in ((8, 7), (11, 0), (13, 0)):
        profile = subject.profile_cuda_device("test", cc)
        assert subject.select_attention_backend("auto", profile, {"flash_attn2", "flash_attn4", "torch_sdpa"}) == "torch_sdpa"
        with pytest.raises(RuntimeError):
            subject.select_attention_backend("flash_attn4", profile, {"flash_attn4", "torch_sdpa"}, strict=True, validated_backends={"flash_attn4"})
    future = subject.profile_cuda_device("future", (13, 0))
    assert not future.known_architecture
    assert subject.policy_supported_precisions(future) == ("no",)


def test_common_precision_and_no_silent_explicit_downgrade():
    reports = [{"status": "passed", "supported_precisions": ["no", "fp16"]},
               {"status": "passed", "supported_precisions": ["no", "fp16", "bf16"]}]
    common = subject.common_supported_precisions(reports)
    assert common == ("no", "fp16")
    assert subject.select_mixed_precision("auto", common) == "fp16"
    assert subject.select_mixed_precision("no", common) == "no"
    with pytest.raises(RuntimeError, match="No automatic downgrade"):
        subject.select_mixed_precision("bf16", common)
    assert subject.common_supported_precisions([]) == ()
    assert subject.common_supported_precisions([*reports, {"status": "unavailable", "supported_precisions": ["no"]}]) == ()
    with pytest.raises(RuntimeError, match="No common"):
        subject.select_mixed_precision("auto", [])
    with pytest.raises(ValueError):
        subject.select_mixed_precision("fp8", ["no"])


def test_tf32_requires_every_actual_device_and_never_mutates_flags():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    reports = [{"tf32_available": True}, {"tf32_available": False}]
    with pytest.raises(RuntimeError, match="every actual device"):
        subject.validate_tf32_request(True, reports)
    assert subject.validate_tf32_request(False, reports) is False
    assert subject.validate_tf32_request(True, reports[0]) is True
    with pytest.raises(RuntimeError):
        subject.validate_tf32_request(True, {"tf32_available": True, "status": "failed"})
    assert torch.backends.cuda.matmul.allow_tf32 == old_matmul


@pytest.mark.parametrize("available,runtime", [(False, None), (False, "12.8"), (True, None)])
def test_cpu_or_rocm_never_launches_or_qualifies(monkeypatch, available, runtime):
    fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=runtime),
                           cuda=SimpleNamespace(is_available=lambda: available, get_arch_list=lambda: []))
    monkeypatch.setattr(subject, "_run_precision_case", lambda *a: pytest.fail("CPU/ROCm launched a probe"))
    report = subject.probe_cuda_precision(fake)
    assert report["status"] == "unavailable" and not report["qualified"]
    assert report["supported_precisions"] == []
    assert not subject.inspect_torch_cuda(fake)["cuda_available"]


def simulated_torch(cc=(8, 6), runtime="12.8", index=2):
    """Metadata-only test double; the numerical launch is explicitly mocked."""
    return SimpleNamespace(__version__="simulated", version=SimpleNamespace(cuda=runtime), device=torch.device,
                           cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: index,
                             device=lambda i: nullcontext(), get_device_name=lambda i: f"test device {i}",
                             get_device_capability=lambda i: cc, get_arch_list=lambda: ["sm_80"]))


def test_exact_device_smoke_cache_isolated_and_native_arch_not_required(monkeypatch):
    subject._PRECISION_PROBE_RESULTS.clear()
    calls = []
    monkeypatch.setattr(subject, "_run_precision_case", lambda t, d, p: calls.append((str(d), p)) or {"status": "passed"})
    fake = simulated_torch()
    report = subject.probe_cuda_precision(fake, 3)
    assert calls == [("cuda:3", "no"), ("cuda:3", "fp16"), ("cuda:3", "bf16")]
    assert report["compiled_arches"] == ["sm_80"] and report["qualified"]
    report["supported_precisions"].clear()
    assert subject.probe_cuda_precision(fake, "cuda:3")["supported_precisions"] == ["no", "fp16", "bf16"]
    assert len(calls) == 3
    subject.probe_cuda_precision(fake, 2)
    assert len(calls) == 6
    json.dumps(report)


def test_failed_dtype_excluded_from_auto_and_explicit_is_an_error(monkeypatch):
    subject._PRECISION_PROBE_RESULTS.clear()
    def run(t, d, p):
        if p == "bf16":
            raise RuntimeError("simulated missing binary")
        return {"status": "passed"}
    monkeypatch.setattr(subject, "_run_precision_case", run)
    report = subject.probe_cuda_precision(simulated_torch())
    assert report["status"] == "partial" and not report["qualified"]
    assert report["cases"]["bf16"]["status"] == "failed"
    common = subject.common_supported_precisions([report])
    assert subject.select_mixed_precision("auto", common) == "fp16"
    with pytest.raises(RuntimeError):
        subject.select_mixed_precision("bf16", common)


def test_unknown_device_remains_unverified_after_fp32_smoke(monkeypatch):
    subject._PRECISION_PROBE_RESULTS.clear()
    monkeypatch.setattr(subject, "_run_precision_case", lambda *a: {"status": "passed"})
    report = subject.probe_cuda_precision(simulated_torch((13, 0), "13.0"))
    assert report["status"] == "unverified" and not report["qualified"]
    assert report["supported_precisions"] == ["no"]
    assert not report["tf32_available"]


def test_legacy_new_runtime_refuses_before_any_kernel(monkeypatch):
    subject._PRECISION_PROBE_RESULTS.clear()
    monkeypatch.setattr(subject, "_run_precision_case", lambda *a: pytest.fail("Unsupported runtime launched"))
    report = subject.probe_cuda_precision(simulated_torch((7, 0), "13.0"))
    assert report["status"] == "failed" and report["supported_precisions"] == []


@pytest.mark.parametrize("precision", ["no", "fp16", "bf16"])
def test_numerical_reference_logic_on_cpu_preserves_rng(monkeypatch, precision):
    # Testing the helper's math on CPU is not calling/qualifying the CUDA gate.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    state = torch.random.get_rng_state().clone()
    result = subject._run_precision_case(torch, torch.device("cpu"), precision)
    assert result["status"] == "passed"
    assert len(result["relative_l2"]) == 7
    assert torch.equal(state, torch.random.get_rng_state())


def test_probe_helper_ignores_and_restores_outer_autocast_and_inference_mode(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert subject._run_precision_case(torch, torch.device("cpu"), "no")["status"] == "passed"
        assert torch.is_inference_mode_enabled()
        assert torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("cc", [(7, 0), (7, 5), (8, 7), (11, 0)])
@pytest.mark.parametrize("scheme", ["int8-triton", "fp8-triton", "nvfp4"])
def test_fallback_gpu_policy_does_not_widen_quantization(cc, scheme):
    assert cc not in quant_backend_capabilities(scheme)
    with pytest.raises(RuntimeError):
        validate_quant_backend(scheme, cc, {"callable": lambda: None})


def test_cli_cpu_is_exit_two_and_never_qualified(capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.main(["--json", "--probe-precision", "--precision", "bf16"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "unavailable" and not report["qualified"]
    assert "precision_probes" not in report


def test_cli_separates_metadata_from_native_arch_requirement(capsys, monkeypatch):
    def metadata(t):
        return {"cuda_available": True, "torch_version": "test", "cuda_runtime": "12.8",
                "compiled_arches": ["sm_80"], "devices": [{"index": 0, "known_architecture": True}],
                "issues": [], "warnings": [], "native_arch_issues": ["missing exact sm_86"], "qualified": False}
    monkeypatch.setattr(cli, "inspect_torch_cuda", metadata)
    assert cli.main(["--json", "--strict"]) == 0
    assert not json.loads(capsys.readouterr().out)["qualified"]
    assert cli.main(["--json", "--strict-native-arch"]) == 1
    assert not json.loads(capsys.readouterr().out)["qualified"]


@pytest.mark.parametrize("strict_flag,arches,native_issues,known", [
    ("--strict-native-arch", ["sm_80"], ["missing exact sm_86"], True),
    ("--strict-native-arch", [], [], True),
    ("--strict", ["sm_130"], [], False),
])
def test_cli_passing_probe_cannot_override_failed_strict_requirement(
        capsys, monkeypatch, strict_flag, arches, native_issues, known):
    def metadata(t):
        return {"cuda_available": True, "compiled_arches": arches,
                "devices": [{"index": 0, "known_architecture": known}],
                "issues": [], "native_arch_issues": native_issues, "qualified": False}
    monkeypatch.setattr(cli, "inspect_torch_cuda", metadata)
    monkeypatch.setattr(cli, "probe_cuda_precision", lambda *args: {
        "qualified": True, "status": "passed", "supported_precisions": ["no"], "tf32_available": False})
    assert cli.main(["--json", "--probe-precision", strict_flag]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["qualified"] is False and report["status"] == "partial_or_failed"
    # Without a strict installation request, passing PTX/cubin launches are
    # independently admissible; exact sm_* metadata is not a runtime test.
    assert cli.main(["--json", "--probe-precision"]) == 0
    assert json.loads(capsys.readouterr().out)["qualified"] is True


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.cuda,
                    reason="real NVIDIA CUDA required; CPU/mock tests are not qualification")
def test_real_cuda_precision_smoke():
    subject._PRECISION_PROBE_RESULTS.clear()
    report = subject.probe_cuda_precision(torch)
    assert report["qualified"], json.dumps(report, indent=2)
    assert report["status"] == "passed"
    assert all(case["status"] == "passed" for case in report["cases"].values())
