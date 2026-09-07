"""Synthetic CUDA-stream correctness gate, NOT a speed/overlap benchmark.

Compare the real TeacherStudentRuntime serial and DPP paths, including teacher
outputs, student loss and gradients. Exercise temporary input release and a
different consumer stream under allocator pressure. A trace records scheduling
but does not prove useful overlap or a full-model throughput improvement.

Exit 0: every selected case passed; 1: incomplete/failure; 2: NVIDIA CUDA absent.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cuda_compat import common_supported_precisions, probe_cuda_precision, select_mixed_precision


def _error(actual, expected, tolerance):
    import torch
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"passed": False, "finite": False, "max_abs": None, "relative_l2": None}
    delta = a - b
    relative = float(delta.norm() / b.norm().clamp_min(1e-7))
    return {"passed": relative <= tolerance, "finite": True,
            "max_abs": float(delta.abs().max()), "relative_l2": relative}


def _tiny_model(torch, device, dtype, scale):
    class SyntheticDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # No nn.Linear default initialization/global RNG changes.
            weight = torch.eye(64) * scale
            weight += torch.sin(torch.arange(64 * 64).reshape(64, 64).float()) * .003
            self.weight = torch.nn.Parameter(weight.to(device=device, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.linspace(-.01, .01, 64).to(device=device, dtype=dtype))
            self.execution_streams = []

        def forward(self, hidden_states, context):
            self.execution_streams.append(int(torch.cuda.current_stream(hidden_states.device).cuda_stream)
                                          if hidden_states.is_cuda else None)
            x = hidden_states + context["offset"][0]
            for _ in range(8):
                x = torch.tanh(x @ self.weight + self.bias)
            return x
    return SyntheticDenoiser()


def _args(dpp):
    return SimpleNamespace(runtime_enable_dpp=dpp, runtime_teacher_stream_priority=0,
                           runtime_cache_backend="none", runtime_teacher_cache_mode="disabled",
                           runtime_cache_identity="synthetic-runtime-stream-check-v1")


def _observe_record_stream(runtime):
    """Observe, then delegate to the real implementation; never fake a CUDA call."""
    import torch
    original = runtime._record_payload_stream
    calls = []
    def tensors(payload):
        if isinstance(payload, torch.Tensor):
            return [payload] if payload.is_cuda else []
        if isinstance(payload, dict):
            return [tensor for value in payload.values() for tensor in tensors(value)]
        if isinstance(payload, (list, tuple)):
            return [tensor for value in payload for tensor in tensors(value)]
        return []
    def observed(payload, stream):
        values = tensors(payload)
        calls.append({"stream": int(stream.cuda_stream), "cuda_tensor_count": len(values)})
        return original(payload, stream)
    runtime._record_payload_stream = observed
    return calls


def _pressure(torch, shape, device, dtype):
    # Repeatedly free/reallocate same-sized tensors on the storage's original
    # stream. The other stream must be protected by runtime record_stream calls.
    for _ in range(12):
        scratch = torch.empty(shape, device=device, dtype=dtype)
        scratch.fill_(19.0)
        del scratch


def _run_case(torch, device, precision, consumer_mode, iterations):
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError("A real NVIDIA CUDA device is required; serial CPU fallback cannot pass")
    from training.runtime.teacher_student_runtime import TeacherStudentRuntime

    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    tolerance = {"no": 1e-5, "fp16": .002, "bf16": .01}[precision]
    producer = torch.cuda.current_stream(device)
    consumer = producer if consumer_mode == "same" else torch.cuda.Stream(device=device)
    teacher = _tiny_model(torch, device, dtype, .92).eval().requires_grad_(False)
    student_serial = _tiny_model(torch, device, dtype, .81)
    student_async = _tiny_model(torch, device, dtype, .81)
    serial = TeacherStudentRuntime(_args(False), teacher, student_serial, device)
    dpp = TeacherStudentRuntime(_args(True), teacher, student_async, device)
    if serial.can_pipeline_teacher_student() or not dpp.can_pipeline_teacher_student():
        raise RuntimeError("DPP stream creation fell back to serial; this case cannot qualify")
    teacher_stream = dpp._teacher_stream
    if int(teacher_stream.cuda_stream) in {int(producer.cuda_stream), int(consumer.cuda_stream)}:
        raise RuntimeError("Teacher did not receive an independent CUDA stream")
    # Parameters were created on producer; user-owned consumer must wait for
    # that initialization. Runtime itself waits for its teacher inputs.
    consumer.wait_stream(producer)
    records = _observe_record_stream(dpp)
    generator = torch.Generator(device="cpu").manual_seed(4817)
    comparisons = []
    observed_events = []
    released_inputs = 0
    released_outputs = 0
    try:
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast("cuda", enabled=False):
            for step in range(iterations):
                cpu_input = torch.randn(4, 64, 64, generator=generator) * .2
                cpu_offset = torch.randn(64, generator=generator) * .01
                batch = {"sample_id": [f"synthetic-{step}-{index}" for index in range(4)]}
                with torch.cuda.stream(producer), torch.profiler.record_function("runtime_serial_reference"):
                    student_serial.zero_grad(set_to_none=True)
                    serial_input = {"hidden_states": cpu_input.to(device=device, dtype=dtype),
                                    "context": {"offset": (cpu_offset.to(device=device, dtype=dtype),)}}
                    expected_teacher = serial.run_teacher(serial_input, batch, step, allow_cache=False)
                    expected_student = serial.run_student(student_serial, serial_input, batch, step)
                    expected_loss = (expected_student.float() - expected_teacher.float()).square().mean()
                    expected_loss.backward()
                    torch.cuda.synchronize(device)
                    reference = {"teacher": expected_teacher.detach().cpu(), "student": expected_student.detach().cpu(),
                                 "loss": expected_loss.detach().cpu(),
                                 "gradients": {name: p.grad.detach().cpu() for name, p in student_serial.named_parameters()}}
                    del serial_input, expected_teacher, expected_student, expected_loss
                teacher.execution_streams.clear()
                student_async.execution_streams.clear()
                with torch.cuda.stream(producer), torch.profiler.record_function("runtime_dpp_teacher_launch"):
                    temporary_input = cpu_input.to(device=device, dtype=dtype)
                    temporary_offset = cpu_offset.to(device=device, dtype=dtype)
                    released_refs = [weakref.ref(temporary_input), weakref.ref(temporary_offset)]
                    kwargs = {"hidden_states": temporary_input, "context": {"offset": (temporary_offset,)}}
                    handle = dpp.launch_teacher(kwargs, batch, step, allow_cache=False)
                    if handle.ready_event is None or handle.from_cache:
                        raise RuntimeError("No real teacher ready event was recorded")
                    ready_event = handle.ready_event
                    observed_events.append(int(ready_event.cuda_event))
                    del kwargs, temporary_input, temporary_offset
                    if any(ref() is not None for ref in released_refs):
                        raise RuntimeError("Temporary teacher inputs remained strongly referenced; lifetime stress was not exercised")
                    released_inputs += len(released_refs)
                    _pressure(torch, cpu_input.shape, device, dtype)
                    _pressure(torch, cpu_offset.shape, device, dtype)
                with torch.cuda.stream(consumer), torch.profiler.record_function("runtime_dpp_student_loss_backward"):
                    student_async.zero_grad(set_to_none=True)
                    student_input = {"hidden_states": cpu_input.to(device=device, dtype=dtype),
                                     "context": {"offset": (cpu_offset.to(device=device, dtype=dtype),)}}
                    actual_student = dpp.run_student(student_async, student_input, batch, step)
                    actual_teacher = dpp.wait_teacher(handle)
                    output_ref = weakref.ref(actual_teacher)
                    teacher_snapshot = actual_teacher.detach().clone()
                    actual_loss = (actual_student.float() - actual_teacher.float()).square().mean()
                    # Drop the producer-owned output before the consumer's
                    # queued clone/loss has completed; record_stream must keep
                    # its storage alive across the streams.
                    del actual_teacher, handle
                    if output_ref() is not None:
                        raise RuntimeError("Teacher output remained strongly referenced; output lifetime stress was not exercised")
                    released_outputs += 1
                    actual_loss.backward()
                    del student_input
                with torch.cuda.stream(teacher_stream), torch.profiler.record_function("runtime_teacher_allocator_pressure"):
                    _pressure(torch, cpu_input.shape, device, dtype)
                torch.cuda.synchronize(device)
                if not ready_event.query() or not observed_events[-1]:
                    raise RuntimeError("Recorded CUDA ready event did not complete")
                if teacher.execution_streams != [int(teacher_stream.cuda_stream)]:
                    raise RuntimeError("Teacher forward did not execute in the runtime's teacher stream")
                if student_async.execution_streams != [int(consumer.cuda_stream)]:
                    raise RuntimeError("Student forward did not execute in the selected consumer stream")
                result = {"iteration": step, "teacher": _error(teacher_snapshot, reference["teacher"], tolerance),
                          "student": _error(actual_student, reference["student"], tolerance),
                          "loss": _error(actual_loss, reference["loss"], tolerance),
                          "gradients": {name: _error(p.grad, reference["gradients"][name], tolerance)
                                        for name, p in student_async.named_parameters()}}
                result["passed"] = all(result[key]["passed"] for key in ("teacher", "student", "loss")) and all(
                    value["passed"] for value in result["gradients"].values())
                comparisons.append(result)
                dpp.finish_step()
                serial.finish_step()
            counters = dpp.stats()
            if counters["async_teacher_launches"] != iterations or counters["async_teacher_waits"] != iterations:
                raise RuntimeError("DPP launch/wait counters did not match actual iterations")
            if serial.stats()["async_teacher_launches"] or serial.stats()["async_teacher_waits"]:
                raise RuntimeError("The reference runtime unexpectedly used an asynchronous stream")
            input_records = sum(c["cuda_tensor_count"] for c in records if c["stream"] == int(teacher_stream.cuda_stream))
            output_records = sum(c["cuda_tensor_count"] for c in records if c["stream"] == int(consumer.cuda_stream))
            if input_records < 2 * iterations or output_records < iterations:
                raise RuntimeError("Real runtime input/output record_stream calls were not observed")
            return {"status": "passed" if all(c["passed"] for c in comparisons) else "failed",
                    "precision": precision, "consumer_mode": consumer_mode, "iterations": iterations,
                    "relative_l2_limit": tolerance, "comparisons": comparisons, "runtime_counters": counters,
                    "producer_stream": int(producer.cuda_stream), "teacher_stream": int(teacher_stream.cuda_stream),
                    "consumer_stream": int(consumer.cuda_stream), "ready_events": observed_events,
                    "record_stream_cuda_inputs": input_records, "record_stream_cuda_outputs": output_records,
                    "released_temporary_inputs": released_inputs, "released_temporary_outputs": released_outputs,
                    "allocator_pressure_allocations_per_iteration": 36,
                    "overlap_or_speedup_claimed": False}
    finally:
        torch.cuda.synchronize(device)
        serial.close()
        dpp.close()


def _emit(report, output, exit_code):
    payload = json.dumps(report, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=("all", "auto", "no", "fp16", "bf16"), default="all",
                        help="all means every mode that passed the actual precision probe; explicit modes never downgrade")
    parser.add_argument("--consumer-stream", choices=("same", "separate", "both"), default="both")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path, nargs="?", const=Path("results/runtime_streams_trace.json"),
                        help="Optionally export a Chrome trace (no overlap/speedup assertion)")
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 100:
        parser.error("--iterations must be between 1 and 100")
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "unavailable", "qualified": False, "requested_precision": args.precision,
              "scope": "synthetic TeacherStudentRuntime stream/event/lifetime correctness; not model, overlap, or throughput qualification",
              "cases": [], "overlap_or_speedup_claimed": False}
    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        report["working_tree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        report["commit"] = None
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        report["reason"] = f"PyTorch unavailable: {exc}"
        return _emit(report, args.output, 2)
    report.update(torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda)
    if not torch.cuda.is_available() or not torch.version.cuda:
        report["reason"] = "NVIDIA CUDA unavailable; CPU/ROCm fallback cannot qualify streams"
        return _emit(report, args.output, 2)
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("Requested CUDA device index is out of range")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    report["device"] = {"index": args.device, "name": torch.cuda.get_device_name(device),
                        "capability": list(torch.cuda.get_device_capability(device)), "compiled_arches": torch.cuda.get_arch_list()}
    try:
        report["driver_inventory"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        report["driver_inventory"] = None
    probe = probe_cuda_precision(torch, device)
    report["precision_probe"] = probe
    available = common_supported_precisions([probe])
    try:
        precisions = list(available) if args.precision == "all" else [select_mixed_precision(args.precision, available)]
        if not precisions:
            raise RuntimeError("No precision passed actual-device preflight")
    except RuntimeError as exc:
        report.update(status="failed", reason=str(exc))
        return _emit(report, args.output, 1)
    report["selected_precisions"] = precisions
    report["precision_selection_scope"] = "only actual probe-passed modes; no claim for excluded modes"
    modes = ["same", "separate"] if args.consumer_stream == "both" else [args.consumer_stream]
    try:
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                           record_shapes=True, profile_memory=True) if args.profile else nullcontext())
        with profiler:
            for precision in precisions:
                for mode in modes:
                    try:
                        row = _run_case(torch, device, precision, mode, args.iterations)
                    except Exception as exc:
                        row = {"precision": precision, "consumer_mode": mode, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                    report["cases"].append(row)
        if args.profile:
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.profile))
            report["trace"] = str(args.profile.resolve())
        report["qualified"] = bool(report["cases"]) and all(row["status"] == "passed" for row in report["cases"])
        report["status"] = "passed" if report["qualified"] else "failed"
    except Exception as exc:
        report.update(status="failed", qualified=False, reason=f"Profiler/runtime failure: {exc}")
    return _emit(report, args.output, 0 if report["qualified"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
