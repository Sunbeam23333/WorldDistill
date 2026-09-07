"""CPU guard tests + opt-in actual-device stream checks; skips never qualify."""
import json

import pytest
import torch

from cuda_compat import probe_cuda_precision
from tools import check_runtime_streams as subject


@pytest.mark.parametrize("cuda_available,runtime", [(False, None), (True, None), (False, "12.8")])
def test_cpu_rocm_and_unavailable_never_qualify(monkeypatch, tmp_path, capsys, cuda_available, runtime):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.version, "cuda", runtime)
    monkeypatch.setattr(subject, "_run_case", lambda *a: pytest.fail("Unavailable GPU ran a case"))
    output = tmp_path / "streams.json"
    trace = tmp_path / "trace.json"
    assert subject.main(["--output", str(output), "--profile", str(trace)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert report["status"] == "unavailable" and not report["qualified"]
    assert not report["overlap_or_speedup_claimed"] and report["cases"] == []
    assert not trace.exists()


def test_case_cannot_fall_back_to_a_cpu_runtime():
    with pytest.raises(RuntimeError, match="serial CPU fallback"):
        subject._run_case(torch, torch.device("cpu"), "no", "same", 1)


@pytest.mark.parametrize("actual,expected", [([float("nan")], [1.]), ([1.], [float("inf")]), ([1., 2.], [1.])])
def test_nonfinite_and_shape_mismatch_never_pass(actual, expected):
    report = subject._error(torch.tensor(actual), torch.tensor(expected), 1e-3)
    assert not report["passed"] and not report["finite"]


def test_numerical_error_threshold_is_enforced():
    assert subject._error(torch.ones(3), torch.ones(3), 1e-5)["passed"]
    assert not subject._error(torch.ones(3) * 2, torch.ones(3), 1e-5)["passed"]


def test_synthetic_model_is_differentiable_and_preserves_rng_on_cpu():
    before = torch.random.get_rng_state().clone()
    model = subject._tiny_model(torch, torch.device("cpu"), torch.float32, .9)
    x = torch.ones(2, 3, 64)
    model(x, {"offset": (torch.zeros(64),)}).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.execution_streams == [None]  # explicitly no CUDA evidence.
    assert torch.equal(before, torch.random.get_rng_state())


def test_record_stream_observer_delegates_instead_of_replacing_behavior():
    calls = []
    class RuntimeDouble:
        def _record_payload_stream(self, payload, stream):
            calls.append((payload, stream))
    class StreamDouble:
        cuda_stream = 72
    runtime = RuntimeDouble()
    observations = subject._observe_record_stream(runtime)
    payload = {"cpu": [torch.ones(1)]}
    stream = StreamDouble()
    runtime._record_payload_stream(payload, stream)
    assert calls == [(payload, stream)]
    assert observations == [{"stream": 72, "cuda_tensor_count": 0}]


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.cuda,
                    reason="real NVIDIA CUDA required; CPU fallback/skip is not stream qualification")
@pytest.mark.parametrize("precision", ["no", "fp16", "bf16"])
@pytest.mark.parametrize("consumer_mode", ["same", "separate"])
def test_real_runtime_serial_vs_streams(precision, consumer_mode):
    device = torch.device("cuda", torch.cuda.current_device())
    preflight = probe_cuda_precision(torch, device)
    if precision not in preflight["supported_precisions"]:
        pytest.skip(f"{precision} is not executable on this device; {preflight['status']}")
    report = subject._run_case(torch, device, precision, consumer_mode, 3)
    assert report["status"] == "passed", json.dumps(report, indent=2)
    assert report["runtime_counters"]["async_teacher_launches"] == 3
    assert report["runtime_counters"]["async_teacher_waits"] == 3
    assert report["record_stream_cuda_inputs"] >= 6
    assert report["record_stream_cuda_outputs"] >= 3
    assert report["released_temporary_inputs"] == 6
    assert report["released_temporary_outputs"] == 3
    assert len(report["ready_events"]) == 3 and all(report["ready_events"])
    assert report["teacher_stream"] != report["producer_stream"]
    assert report["teacher_stream"] != report["consumer_stream"]
    if consumer_mode == "separate":
        assert report["consumer_stream"] != report["producer_stream"]
