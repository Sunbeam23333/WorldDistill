"""Parameter residency, state semantics and stream ownership for teacher offload."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.runtime.distill_cache import MemoryDistillCache
from training.runtime.teacher_offload import TeacherParameterLease, TeacherParameterOffloader
from training.runtime.teacher_student_runtime import TeacherStudentRuntime
from training.runtime import build_runtime


class TiedTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.tied = self.weight
        self.register_buffer("calls", torch.zeros(()), persistent=False)
        self.register_buffer("assigned", torch.zeros(()))

    def forward(self, value):
        assert self.weight is self.tied
        self.calls.add_(1)
        self.assigned = self.assigned + 1
        return value * (self.weight + self.tied) + self.calls + self.assigned


def runtime_args(tmp_path, **extra):
    return SimpleNamespace(
        output_dir=str(tmp_path), runtime_teacher_offload="cpu", **extra,
    )


@pytest.mark.parametrize("target", ["cpu", "nvme"])
def test_offload_preserves_ties_nonpersistent_buffers_and_mutations(tmp_path, target):
    model = TiedTeacher()
    offloader = TeacherParameterOffloader(model, torch.device("cpu"), target=target, directory=tmp_path)
    original_parameter = model.weight
    first, first_lease = offloader.forward({"value": torch.tensor(3.0)})
    second, _ = offloader.forward({"value": torch.tensor(3.0)})
    torch.testing.assert_close(first, torch.tensor(14.0))
    torch.testing.assert_close(second, torch.tensor(16.0))
    assert model.weight is model.tied is original_parameter
    assert not model.weight.requires_grad
    assert first_lease.released
    assert model.calls.item() == model.assigned.item() == 2
    assert offloader.stats()["teacher_offload_forwards"] == 2
    path = offloader.storage_path
    if target == "nvme":
        assert path.exists()
    offloader.close()
    offloader.close()
    assert model.weight.device.type == "cpu"
    if target == "nvme":
        assert not path.exists()
    torch.testing.assert_close(model(torch.tensor(3.0)), torch.tensor(18.0))
    with pytest.raises(RuntimeError, match="closed"):
        offloader.forward({"value": torch.tensor(1.0)})


def test_nvme_cleanup_owns_only_its_temporary_directory(tmp_path):
    unrelated = tmp_path / "user-checkpoint.pt"
    torch.save({"keep": torch.ones(1)}, unrelated)
    offloader = TeacherParameterOffloader(nn.Linear(2, 2), torch.device("cpu"), target="nvme", directory=tmp_path)
    offloader.close()
    assert unrelated.exists()


def test_nvme_teacher_revision_republishes_current_weights(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path), runtime_teacher_offload="nvme")
    teacher = nn.Linear(2, 2, bias=False)
    runtime = TeacherStudentRuntime(args, teacher, nn.Identity(), torch.device("cpu"))
    before_path = runtime._teacher_offloader.storage_path
    with torch.no_grad():
        teacher.weight.fill_(3)
    runtime.advance_teacher_revision("handoff")
    after_path = runtime._teacher_offloader.storage_path
    assert after_path != before_path and not before_path.exists()
    stored = torch.load(after_path, weights_only=True)
    torch.testing.assert_close(stored["weight"], torch.full((2, 2), 3.0))
    result = runtime.run_teacher({"input": torch.ones(1, 2)}, {}, 0)
    torch.testing.assert_close(result, torch.full((1, 2), 6.0))
    runtime.close()


def test_offload_does_not_swallow_forward_error_or_leave_functional_parameters(tmp_path):
    class RaisingTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, value):
            raise RuntimeError("deliberate forward failure")

    model = RaisingTeacher()
    parameter = model.weight
    executor = TeacherParameterOffloader(model, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="deliberate forward failure"):
        executor.forward({"value": torch.ones(1)})
    assert model.weight is parameter
    assert not executor._active
    executor.close()


def test_offload_rejects_foreign_thread_and_unmaterialized_weights():
    executor = TeacherParameterOffloader(nn.Linear(1, 1), torch.device("cpu"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.forward, {"input": torch.ones(1, 1)})
        with pytest.raises(RuntimeError, match="owning training thread"):
            future.result()
    executor.close()
    with pytest.raises(ValueError, match="materialized"):
        TeacherParameterOffloader(nn.Linear(1, 1, device="meta"), torch.device("cpu"))


def test_lazy_and_resized_registered_cache_buffers_stay_on_host():
    class LazyCache(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", None, persistent=False)

        def forward(self, value):
            if self.cache is None or self.cache.shape != value.shape:
                self.cache = torch.ones_like(value)
            return value + self.cache

    model = LazyCache()
    executor = TeacherParameterOffloader(model, torch.device("cpu"))
    for size in (2, 5):
        result, _ = executor.forward({"value": torch.ones(size)})
        torch.testing.assert_close(result, torch.full((size,), 2.0))
        assert model.cache.shape == (size,) and model.cache.device.type == "cpu"
    executor.close()


def test_parameter_lease_retains_storage_until_event_completes():
    class Event:
        complete = False
        waits = 0

        def query(self):
            return self.complete

        def synchronize(self):
            self.waits += 1
            self.complete = True

    event = Event()
    lease = TeacherParameterLease({"weight": torch.ones(2)}, event)
    assert not lease.release(wait=False)
    assert lease.tensors
    assert lease.release(wait=True)
    assert not lease.tensors
    assert lease.released
    assert event.waits == 1
    assert lease.release(wait=True)
    assert event.waits == 1


def test_parameter_lease_propagates_cuda_event_failure():
    class Event:
        def synchronize(self):
            raise RuntimeError("CUDA illegal memory access")

    lease = TeacherParameterLease({"weight": torch.ones(2)}, Event())
    with pytest.raises(RuntimeError, match="illegal memory access"):
        lease.release(wait=True)
    assert not lease.released
    assert lease.tensors


def test_runtime_offloads_without_metadata_only_heterogeneous_flag_and_cache_hits(tmp_path):
    teacher = nn.Linear(2, 2, bias=False)
    cache = MemoryDistillCache(max_entries=4, pin_memory=False)
    runtime = TeacherStudentRuntime(
        runtime_args(tmp_path, runtime_teacher_cache_mode="teacher_output"),
        teacher, nn.Identity(), torch.device("cpu"), cache,
    )
    x = torch.ones(1, 2)
    first = runtime.run_teacher({"input": x}, {"sample_id": ["a"]}, 0)
    second = runtime.run_teacher({"input": x}, {"sample_id": ["a"]}, 0)
    torch.testing.assert_close(first, second)
    assert runtime.stats()["teacher_offload_forwards"] == 1
    assert runtime.stats()["async_teacher_cache_hits"] == 1
    assert runtime._build_teacher_placement_plan("teacher_output").offload_target == "cpu"
    runtime.synchronize_teacher()
    with torch.no_grad():
        teacher.weight.add_(1)
    runtime.advance_teacher_revision("new-weights")
    third = runtime.run_teacher({"input": x}, {"sample_id": ["a"]}, 1)
    torch.testing.assert_close(third, first + 2)
    assert runtime.stats()["teacher_offload_forwards"] == 2
    runtime.close()


def test_runtime_preserves_joint_audio_video_outputs_during_offload(tmp_path):
    class JointTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, video, audio):
            return {"sample": video * self.weight, "audio_sample": audio * self.weight}

    runtime = TeacherStudentRuntime(runtime_args(tmp_path), JointTeacher(), nn.Identity(), torch.device("cpu"))
    actual = runtime.run_teacher({"video": torch.ones(1, 2), "audio": torch.ones(1, 3)}, {}, 0)
    torch.testing.assert_close(actual["sample"], torch.full((1, 2), 2.0))
    torch.testing.assert_close(actual["audio_sample"], torch.full((1, 3), 2.0))
    runtime.close()


def test_compiled_teacher_uses_current_offloaded_parameters_after_revision(tmp_path):
    teacher = torch.compile(nn.Linear(2, 2, bias=False), backend="eager", fullgraph=True)
    runtime = TeacherStudentRuntime(runtime_args(tmp_path), teacher, nn.Identity(), torch.device("cpu"))
    value = torch.ones(1, 2)
    first = runtime.run_teacher({"input": value}, {}, 0)
    runtime.synchronize_teacher()
    with torch.no_grad():
        teacher.weight.add_(1)
    runtime.advance_teacher_revision("compiled-teacher-update")
    second = runtime.run_teacher({"input": value}, {}, 1)
    torch.testing.assert_close(second, first + 2)
    runtime.close()


@pytest.mark.parametrize("name", ["none", "noop"])
def test_offload_requires_active_runtime(tmp_path, name):
    with pytest.raises(ValueError, match="requires an active runtime"):
        build_runtime(runtime_args(tmp_path, runtime_name=name), nn.Linear(2, 2), nn.Identity(), torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("target", ["cpu", "nvme"])
@pytest.mark.parametrize("dpp", [False, True])
def test_cuda_offload_stream_handoff_state_residency_and_student_gradients(tmp_path, target, dpp):
    args = SimpleNamespace(output_dir=str(tmp_path), runtime_teacher_offload=target, runtime_enable_dpp=dpp)
    teacher = TiedTeacher()
    student = nn.Linear(4, 4, bias=False).cuda()
    runtime = TeacherStudentRuntime(args, teacher, student, torch.device("cuda"))
    device = torch.device("cuda")
    consumer = torch.cuda.Stream(device=device)
    value = torch.randn(4, 4, device=device)
    expected = value * 4 + 2
    handle = runtime.launch_teacher({"value": value}, {}, 0)
    assert all(p.device.type == "cpu" for p in teacher.parameters())
    assert handle.parameter_lease is not None
    assert not handle.parameter_lease.tensors  # stream-ordered release, not full weights held until the next step
    # The output may be consumed on any stream; neither deleting inputs nor
    # allocator pressure may invalidate teacher weights/input storage.
    producer = torch.cuda.current_stream(device)
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        target_value = runtime.wait_teacher(handle)
        prediction = student(torch.ones_like(target_value))
        loss = (prediction - target_value).square().mean()
        loss.backward()
        target_value.record_stream(consumer)
    del value
    junk = [torch.empty(1024, device=device) for _ in range(16)]
    consumer.synchronize()
    torch.testing.assert_close(target_value, expected)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    runtime.synchronize_teacher()
    assert teacher.calls.item() == teacher.assigned.item() == 1
    assert runtime.stats()["teacher_offload_h2d_bytes"] > 0
    second = runtime.run_teacher({"value": expected}, {}, 1)
    torch.cuda.synchronize()
    torch.testing.assert_close(second, expected * 4 + 4)
    runtime.close()
    assert all(p.device.type == "cpu" for p in teacher.parameters())
    del junk
