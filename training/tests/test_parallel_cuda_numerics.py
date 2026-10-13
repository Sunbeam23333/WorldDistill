"""NCCL/GPU execution of the exact TP/PP numerical-reference contracts.

Run on a host with 2/4 visible GPUs:
    pytest -q training/tests/test_parallel_cuda_numerics.py
"""
import pytest
import torch
import torch.multiprocessing as mp
from training.tests.test_tensor_parallel_plan import _worker as tp_worker
from training.tests.test_pipeline_parallel_plan import _worker as pp_worker


@pytest.mark.parametrize("world,kind,worker", [(2, "wan", tp_worker), (2, "wan_bf16", tp_worker),
    (2, "wan_lora", tp_worker), (2, "wan_checkpoint", pp_worker), (4, "tp_pp_lora_checkpoint", pp_worker)])
def test_cuda_real_shards_match_full_reference(tmp_path, world, kind, worker):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} visible CUDA GPUs")
    if "bf16" in kind and not torch.cuda.is_bf16_supported():
        pytest.skip("device lacks native bfloat16")
    pytest.importorskip("diffusers")
    mp.spawn(worker, args=(world, str(tmp_path / "init"), kind, "cuda"), nprocs=world, join=True)
