from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import cuda_compat
from training.trainer_args import TrainerArgs
from training.utils.precision import resolve_training_precision


def report(precisions, tf32=False):
    return {"status": "passed", "supported_precisions": precisions, "tf32_available": tf32}


def test_auto_cpu_is_not_cuda_qualification():
    args = TrainerArgs()
    result = resolve_training_precision(args, "cpu")
    assert args.mixed_precision == "no" and result["gpu_qualified"] is False


def test_single_gpu_resolution_rechecks_settings_for_the_trainer(monkeypatch):
    args = TrainerArgs()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with patch.object(cuda_compat, "probe_cuda_precision", return_value=report(["no", "fp16", "bf16"])) as probe:
        result = resolve_training_precision(args, "cuda:0")
        assert args.mixed_precision == "bf16"
        assert resolve_training_precision(args, "cuda:0")["selected"] == result["selected"]
        assert probe.call_count == 2
        args.enable_tf32 = True
        with pytest.raises(RuntimeError, match="TF32"):
            resolve_training_precision(args, "cuda:0")


def test_auto_does_not_choose_missing_fsdp_or_auxiliary_scaler(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(cuda_compat, "probe_cuda_precision", lambda *a, **k: report(["no", "fp16"]))
    for fields in ({"parallel_mode": "fsdp"}, {"distill_method": "dmd_distill"}, {"distill_method": "adversarial_distill"}):
        args = TrainerArgs(**fields)
        assert resolve_training_precision(args, "cuda:0")["selected"] == "no"


def test_explicit_unavailable_precision_and_tf32_are_not_downgraded(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(cuda_compat, "probe_cuda_precision", lambda *a, **k: report(["no", "fp16"]))
    with pytest.raises(RuntimeError, match="Explicit mixed precision"):
        resolve_training_precision(TrainerArgs(mixed_precision="bf16"), "cuda:0")
    with pytest.raises(RuntimeError, match="TF32"):
        resolve_training_precision(TrainerArgs(enable_tf32=True), "cuda:0")


def test_mixed_cards_use_the_observed_all_rank_intersection(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(cuda_compat, "probe_cuda_precision", lambda *a, **k: report(["no", "fp16", "bf16"]))
    def gather(peers, local):
        peers[:] = [local, {**local, "probe": report(["no", "fp16"])}]
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    assert resolve_training_precision(TrainerArgs(), "cuda:0")["selected"] == "fp16"
    def mismatch(peers, local):
        peers[:] = [local, {**local, "controls": {**local["controls"], "batch_size": 7}}]
    monkeypatch.setattr(torch.distributed, "all_gather_object", mismatch)
    with pytest.raises(ValueError, match="Ranks disagree"):
        resolve_training_precision(TrainerArgs(), "cuda:0")


def test_default_matmul_precision_cannot_reenable_tf32(monkeypatch):
    from training.train_distill import _configure_cuda_backend
    matmul, cudnn = SimpleNamespace(allow_tf32=True), SimpleNamespace(allow_tf32=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul)
    monkeypatch.setattr(torch.backends, "cudnn", cudnn)
    seen = []
    def set_precision(value):
        seen.append(value)
        matmul.allow_tf32 = value != "highest"
    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_precision)
    _configure_cuda_backend(TrainerArgs(enable_tf32=False, float32_matmul_precision="high"))
    assert seen == ["highest"] and not matmul.allow_tf32 and not cudnn.allow_tf32
