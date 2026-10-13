"""Actual changed-world-size checkpoint continuation and cursor contracts."""
import copy
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader

from training.distributed_config import rank_environment
from training.trainers.base_distill_trainer import BaseDistillTrainer, EMAModel
from training.utils.elastic import ElasticBatchSampler, elastic_seed, latest_elastic_checkpoint, publish_elastic_checkpoint
from training.utils.elastic_deepspeed import configure_elastic_deepspeed, resize_amp_scaler_state
from training.utils.sharded_state import ShardedEMA


class _ElasticTrainer(BaseDistillTrainer):
    def compute_distill_loss(self, *args):
        raise NotImplementedError

    def prepare_teacher_input(self, *args):
        raise NotImplementedError


def test_cursor_is_committed_not_prefetched_and_reshards_only_unconsumed_suffix():
    sampler = ElasticBatchSampler(range(13), batch_size=2, rank=0, world_size=2, shuffle=False)
    prefetched = list(sampler)
    assert prefetched == [[0, 1], [4, 5], [8, 9], [12]]
    assert sampler.global_cursor == 0
    sampler.commit_batch()
    state = sampler.state_dict()
    resumed = ElasticBatchSampler(range(13), batch_size=2, rank=0, world_size=1, shuffle=False)
    resumed.load_state_dict(state)
    assert list(resumed) == [[4, 5], [6, 7], [8, 9], [10, 11], [12]]
    resumed.commit_batch()
    enlarged = [ElasticBatchSampler(range(13), batch_size=2, rank=rank, world_size=3, shuffle=False) for rank in range(3)]
    for replica in enlarged:
        replica.load_state_dict(resumed.state_dict())
    assert [list(replica) for replica in enlarged] == [[[6, 7], [12]], [[8, 9], [12]], [[10, 11], [12]]]


@pytest.mark.parametrize("mutation", [{"global_cursor": 99}, {"global_cursor": -1}, {"fingerprint": "changed"}, {"global_batches": 3}])
def test_cursor_rejects_malformed_or_changed_data(mutation):
    sampler = ElasticBatchSampler(range(8), batch_size=2)
    with pytest.raises(ValueError):
        sampler.load_state_dict({**sampler.state_dict(), **mutation})


def test_manifest_reordering_changes_cursor_identity():
    class Dataset:
        data = [{"path": "a"}, {"path": "b"}]
        def __len__(self):
            return len(self.data)
    original = ElasticBatchSampler(Dataset(), batch_size=1)
    changed = Dataset()
    changed.data = list(reversed(changed.data))
    with pytest.raises(ValueError, match="differs"):
        ElasticBatchSampler(changed, batch_size=1).load_state_dict(original.state_dict())


def test_elastic_seed_reproducible_but_new_generation_changes_stream():
    assert elastic_seed(42, 10, 0, 1) == elastic_seed(42, 10, 0, 1)
    assert len({elastic_seed(42, 10, rank, generation) for rank in range(2) for generation in range(2)}) == 4


def test_only_complete_checkpoints_are_discoverable(tmp_path):
    complete = tmp_path / "checkpoint-4"
    complete.mkdir()
    torch.save({"model": torch.ones(2)}, complete / "trainer_state.pt")
    partial = tmp_path / "checkpoint-5"
    partial.mkdir()
    torch.save({}, partial / "trainer_state.pt")
    assert latest_elastic_checkpoint(tmp_path) is None
    publish_elastic_checkpoint(complete, 4)
    assert latest_elastic_checkpoint(tmp_path) == str(complete)
    (complete / "trainer_state.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        latest_elastic_checkpoint(tmp_path)


def test_elastic_rank_environment_requires_opt_in_and_range():
    env = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1",
           "MASTER_ADDR": "localhost", "MASTER_PORT": "29500", "GROUP_RANK": "1", "GROUP_WORLD_SIZE": "2",
           "TORCHELASTIC_RESTART_COUNT": "1", "WORLD_DISTILL_FIXED_WORLD_SIZE": "0",
           "WORLD_DISTILL_ELASTIC_RESUME": "1", "WORLD_DISTILL_MIN_NODES": "1", "WORLD_DISTILL_MAX_NODES": "3"}
    assert rank_environment(env).world_size == 2
    for mutation in ({"WORLD_DISTILL_MAX_NODES": "1"}, {"WORLD_DISTILL_ELASTIC_RESUME": "0"},
                     {"WORLD_DISTILL_EXPECTED_WORLD_SIZE": "2"}):
        with pytest.raises(ValueError):
            rank_environment({**env, **mutation})


def test_deepspeed_elastic_requires_universal_and_rejects_nvme_swap():
    assert configure_elastic_deepspeed({"zero_optimization": {"stage": 3}}, enabled=True)["checkpoint"]["load_universal"]
    for zero in ({"stage": 0}, {"stage": 3, "offload_optimizer": {"device": "nvme"}}):
        with pytest.raises(ValueError):
            configure_elastic_deepspeed({"zero_optimization": zero}, enabled=True)
    saved = {"schema_version": 1, "world_size": 2,
             "rank_states": [{"rank": rank, "state": {"scale": 1024, "_growth_tracker": 7}} for rank in range(2)]}
    assert resize_amp_scaler_state(saved, 1)["rank_states"] == saved["rank_states"][:1]
    saved["rank_states"][1]["state"]["scale"] = 512
    with pytest.raises(ValueError, match="disagree"):
        resize_amp_scaler_state(saved, 3)


def test_launcher_forwards_elastic_range_and_model_parallel_flags():
    from training.tests.test_multinode_launch import dry_launch
    result = dry_launch("--nnodes", "1:3", "--gpus", "4", "--rdzv_backend", "c10d",
                        "--rdzv_endpoint", "node0:29500", "--rdzv_id", "elastic-test",
                        "--elastic_resume", "--elastic_auto_resume", "--tp_size", "2", "--pp_size", "2")
    assert result.returncode == 0, result.stderr
    for text in ("--nnodes=1:3", "WORLD_DISTILL_FIXED_WORLD_SIZE=0", "WORLD_DISTILL_MIN_NODES=1",
                 "WORLD_DISTILL_MAX_NODES=3", "--elastic_resume", "--elastic_auto_resume", "--tp_size 2", "--pp_size 2"):
        assert text in result.stdout
    assert "WORLD_DISTILL_EXPECTED_WORLD_SIZE=" not in result.stdout
    for arguments in (("--nnodes", "3:1", "--elastic_resume"), ("--elastic_auto_resume",), ("--tp_size", "0")):
        assert dry_launch(*arguments).returncode != 0


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_real_c10d_elastic_node_range_accepts_current_worker_world():
    from training.tests.test_multinode_launch import ROOT, parse_launch_probe_records
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {key: value for key, value in os.environ.items()
           if key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE", "WORLD_DISTILL_EXPECTED_WORLD_SIZE"}
           and not key.startswith("TORCHELASTIC_")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", WORLD_DISTILL_DIST_BACKEND="gloo",
               WORLD_DISTILL_DIST_TIMEOUT="30", WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE="2",
               WORLD_DISTILL_FIXED_WORLD_SIZE="0", WORLD_DISTILL_ELASTIC_RESUME="1",
               WORLD_DISTILL_MIN_NODES="1", WORLD_DISTILL_MAX_NODES="2", PROBE_HYBRID_GROUPS="0",
               GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1:2", "--nproc_per_node=2",
                             "--local_addr=127.0.0.1", "--rdzv_backend=c10d", f"--rdzv_endpoint=127.0.0.1:{port}",
                             f"--rdzv_id=elastic-{port}", "--max_restarts=1",
                             "--rdzv_conf=join_timeout=20,last_call_timeout=1,read_timeout=20",
                             "-m", "training.tests.distributed_launch_probe"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert result.returncode == 0, result.stdout
    parse_launch_probe_records([result.stdout], 2, 2)


def _assert_nested_equal(actual, expected, path="state"):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, msg=lambda detail: f"{path}: {detail}")
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for index, (value, wanted) in enumerate(zip(actual, expected)):
            _assert_nested_equal(value, wanted, f"{path}[{index}]")
    else:
        assert actual == expected


def _elastic_worker(rank, world, rendezvous, directory, resume, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(17)
        raw = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        consistency = mode == "consistency_fsdp"
        auxiliary = mode == "dmd"
        if consistency:
            from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer
            trainer = object.__new__(ConsistencyDistillTrainer)
            target = copy.deepcopy(raw)
        elif auxiliary:
            from training.trainers.dmd_distill_trainer import DMDDistillTrainer
            trainer = object.__new__(DMDDistillTrainer)
        else:
            trainer = object.__new__(_ElasticTrainer)
        trainer.device = torch.device("cpu")
        trainer.args = SimpleNamespace(elastic_resume=True, seed=91, sp_size=1, tp_size=1, pp_size=1)
        trainer.parallel_mode = "fsdp" if consistency else "ddp" if auxiliary else mode
        trainer._deepspeed_engine = None
        trainer.runtime = None
        trainer.global_step = trainer.epoch = trainer._batches_in_epoch = 0
        trainer._pending_rng_state = None
        trainer.best_loss = float("inf")
        trainer._refresh_distributed_state()
        if trainer.parallel_mode == "fsdp":
            # This explicit CPU device exercises real FSDP state/reshard APIs;
            # it does not change production's CUDA-only FSDP placement policy.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            trainer.student_model = FSDP(raw, use_orig_params=True, device_id=torch.device("cpu"))
            trainer.ema = ShardedEMA(trainer.student_model, decay=.5)
            if consistency:
                trainer._ema_offloader = None
                trainer.ema_model = FSDP(target, use_orig_params=True, device_id=torch.device("cpu"))
                trainer.consistency_ema = ShardedEMA(trainer.student_model, decay=.7)
        else:
            trainer.student_model = torch.nn.parallel.DistributedDataParallel(raw)
            trainer.ema = EMAModel(raw, decay=.5)
        trainer.optimizer = torch.optim.AdamW(trainer.student_model.parameters(), lr=.01)
        trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 2, gamma=.8)
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        if auxiliary:
            trainer.enable_fake_score = trainer.use_gan = True
            trainer.fake_score_model = torch.nn.parallel.DistributedDataParallel(nn.Linear(2, 1))
            trainer.discriminator = torch.nn.parallel.DistributedDataParallel(nn.Linear(2, 1))
            trainer.fake_score_optimizer = torch.optim.AdamW(trainer.fake_score_model.parameters(), lr=.02)
            trainer.disc_optimizer = torch.optim.AdamW(trainer.discriminator.parameters(), lr=.03)
            trainer.fake_score_scaler = torch.amp.GradScaler("cpu", enabled=False)
            trainer.disc_scaler = torch.amp.GradScaler("cpu", enabled=False)
            trainer.fake_score_ema = EMAModel(trainer.fake_score_model.module, decay=.8)
        sampler = ElasticBatchSampler(range(20), batch_size=1, rank=rank, world_size=world, shuffle=False, seed=91)
        trainer.train_dataloader = DataLoader(list(range(20)), batch_sampler=sampler)
        if resume is not None:
            trainer.load_checkpoint(resume)
            expected = torch.load(Path(resume) / "trainer_state.pt", map_location="cpu", weights_only=False)
            assert trainer.global_step == expected["step"]
            assert trainer.lr_scheduler.state_dict() == expected["lr_scheduler"]
            assert trainer.ema.step_count == expected["step"]
            if trainer.parallel_mode == "ddp":
                _assert_nested_equal(raw.state_dict(), expected["student_model"])
                _assert_nested_equal(trainer.optimizer.state_dict(), expected["optimizer"])
                _assert_nested_equal(trainer.ema.state_dict(), expected["ema"])
                if auxiliary:
                    saved_aux = torch.load(Path(resume) / "dmd_state.pt", weights_only=False)
                    _assert_nested_equal(trainer.fake_score_model.module.state_dict(), saved_aux["fake_score_model"])
                    _assert_nested_equal(trainer.fake_score_optimizer.state_dict(), saved_aux["fake_score_optimizer"])
                    _assert_nested_equal(trainer.fake_score_ema.state_dict(), saved_aux["fake_score_ema"])
                    _assert_nested_equal(trainer.discriminator.module.state_dict(), saved_aux["discriminator"])
                    _assert_nested_equal(trainer.disc_optimizer.state_dict(), saved_aux["disc_optimizer"])
            else:
                from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
                model_state, optimizer_state = get_state_dict(trainer.student_model, trainer.optimizer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=False))
                _assert_nested_equal(model_state, expected["student_model"])
                _assert_nested_equal(optimizer_state, expected["optimizer"])
                portable = trainer.ema.portable_state_dict(trainer.student_model)
                _assert_nested_equal(portable, torch.load(Path(resume) / "ema_portable.pt", weights_only=True))
                if consistency:
                    from torch.distributed.checkpoint.state_dict import get_model_state_dict
                    saved_target = torch.load(Path(resume) / "consistency_portable.pt", weights_only=True)
                    _assert_nested_equal(trainer.consistency_ema.portable_state_dict(trainer.student_model), saved_target["consistency_ema"])
                    _assert_nested_equal(get_model_state_dict(trainer.ema_model, options=StateDictOptions(full_state_dict=True)), saved_target["target"])
        trainer._initialize_data_iterator(resuming=resume is not None)
        visited = []
        for _ in range(2):
            identifiers = trainer._next_batch()
            visited.extend(identifiers.tolist())
            x = torch.stack([identifiers.float() / 20, torch.ones_like(identifiers).float()], dim=-1)
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.student_model(x).square().mean().backward()
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.global_step += 1
            trainer.ema.update(trainer._unwrap_model(trainer.student_model))
            assert all(torch.isfinite(parameter).all() for parameter in trainer.student_model.parameters())
            if consistency:
                trainer.consistency_ema.update(trainer._unwrap_model(trainer.student_model))
                trainer.consistency_ema.copy_to(trainer.ema_model)
            if auxiliary:
                for model, optimizer in ((trainer.fake_score_model, trainer.fake_score_optimizer),
                                         (trainer.discriminator, trainer.disc_optimizer)):
                    optimizer.zero_grad(set_to_none=True)
                    model(x).square().mean().backward()
                    optimizer.step()
                trainer.fake_score_ema.update(trainer.fake_score_model.module)
        trainer.save_checkpoint(trainer.global_step, directory)
        all_visited = [None] * world
        dist.all_gather_object(all_visited, visited)
        if rank == 0:
            torch.save({"visited": sorted(value for values in all_visited for value in values),
                        "cursor": sampler.global_cursor, "generation": getattr(trainer, "_elastic_generation", 0)},
                       Path(directory) / f"report-{trainer.global_step}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["ddp", "fsdp", "consistency_fsdp", "dmd"])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_real_checkpoint_optimizer_ema_and_data_resume_two_to_one_to_two(tmp_path, mode):
    resume = None
    for phase, world in enumerate((2, 1, 2)):
        mp.spawn(_elastic_worker, args=(world, str(tmp_path / f"rdzv-{phase}"), str(tmp_path), resume, mode),
                 nprocs=world, join=True)
        resume = latest_elastic_checkpoint(tmp_path)
        assert resume == str(tmp_path / f"checkpoint-{2 * (phase + 1)}")
    reports = [torch.load(tmp_path / f"report-{step}.pt", weights_only=True) for step in (2, 4, 6)]
    assert [row["visited"] for row in reports] == [list(range(4)), [4, 5], list(range(6, 10))]
    assert [row["cursor"] for row in reports] == [4, 6, 10]
    assert [row["generation"] for row in reports] == [0, 1, 2]


def _zero_canonical_state(engine):
    from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_optimizer_state
    from training.utils.elastic_deepspeed import capture_deepspeed_supplement
    if engine.zero_optimization_stage() < 3:
        # Upstream's lazy debug-state linkage runs after the first parameter
        # group step, before subsequent groups own Adam state. Refresh these
        # inspection views after ALL groups have stepped/restored; no tensors
        # or optimizer values are changed by this public linkage utility.
        from deepspeed.utils import lazy_init_hp_params_optimizer_state
        for parameters, master in zip(engine.optimizer.bit16_groups, engine.optimizer.single_partition_of_fp32_groups):
            lazy_init_hp_params_optimizer_state(parameters, master, engine.optimizer.optimizer.state)
    result = {"supplement": capture_deepspeed_supplement(engine), "parameters": {}}
    for name, parameter in engine.module.named_parameters():
        if parameter.requires_grad:
            result["parameters"][name] = {
                "fp32": safe_get_full_fp32_param(parameter).detach().cpu().clone(),
                **{field: safe_get_full_optimizer_state(parameter, field).detach().cpu().clone()
                   for field in ("exp_avg", "exp_avg_sq")},
            }
    return result


def _zero_elastic_worker(rank, world, rendezvous, directory, resume, stage):
    import deepspeed
    from training.utils.elastic_deepspeed import configure_elastic_deepspeed
    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(17)
        raw = nn.Linear(2, 1)
        raw.register_parameter("frozen", nn.Parameter(torch.tensor([7.]), requires_grad=False))
        raw.register_buffer("counter", torch.tensor([3.]))
        optimizer = torch.optim.AdamW([{"params": [raw.weight], "weight_decay": .1},
                                       {"params": [raw.bias], "weight_decay": 0.}], lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=.8)
        config = configure_elastic_deepspeed({"train_batch_size": world, "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 1, "zero_optimization": {"stage": stage},
                    "zero_allow_untested_optimizer": True, "steps_per_print": 10000}, enabled=True)
        engine, optimizer, _, scheduler = deepspeed.initialize(model=raw, optimizer=optimizer,
            lr_scheduler=scheduler, config=config, dist_init_required=False)
        trainer = object.__new__(_ElasticTrainer)
        trainer.device = torch.device("cpu")
        trainer.args = SimpleNamespace(elastic_resume=True, seed=91, sp_size=1, tp_size=1, pp_size=1,
                                       deepspeed_stage=stage, mixed_precision="no")
        trainer.parallel_mode = "deepspeed"
        trainer._deepspeed_engine = trainer.student_model = engine
        trainer.optimizer, trainer.lr_scheduler = optimizer, scheduler
        trainer.runtime = None
        trainer.global_step = trainer.epoch = trainer._batches_in_epoch = 0
        trainer._pending_rng_state = None
        trainer.best_loss = float("inf")
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer._refresh_distributed_state()
        trainer.ema = ShardedEMA(raw, parallel_mode="deepspeed", decay=.5) if stage == 3 else EMAModel(raw, decay=.5)
        sampler = ElasticBatchSampler(range(20), batch_size=1, rank=rank, world_size=world, shuffle=False, seed=91)
        trainer.train_dataloader = DataLoader(list(range(20)), batch_sampler=sampler)
        if resume is not None:
            trainer.load_checkpoint(resume)
            expected = torch.load(Path(directory) / f"zero-reference-{trainer.global_step}.pt", weights_only=False)
            actual = _zero_canonical_state(engine)
            actual["ema"] = trainer.ema.portable_state_dict(raw) if stage == 3 else trainer.ema.state_dict()
            _assert_nested_equal(actual, expected)
            assert trainer.ema.step_count == trainer.global_step
        trainer._initialize_data_iterator(resuming=resume is not None)
        for _ in range(2):
            identifiers = trainer._next_batch()
            x = torch.stack([identifiers.float() / 20, torch.ones_like(identifiers).float()], dim=-1)
            engine.backward(engine(x).square().mean())
            engine.step()
            trainer.global_step += 1
            trainer.ema.update(raw)
        with deepspeed.zero.GatheredParameters([raw.frozen], modifier_rank=0, enabled=hasattr(raw.frozen, "ds_id")):
            raw.frozen.data.add_(1)
        raw.counter.add_(2)
        # Preserve different optimizer-group clocks as well as weight decay.
        master_groups = (engine.optimizer.fp32_partitioned_groups_flat if stage == 3
                         else engine.optimizer.single_partition_of_fp32_groups)
        engine.optimizer.optimizer.state[master_groups[-1]]["step"] += 1
        state = _zero_canonical_state(engine)
        state["ema"] = trainer.ema.portable_state_dict(raw) if stage == 3 else trainer.ema.state_dict()
        trainer.save_checkpoint(trainer.global_step, directory)
        if rank == 0:
            torch.save(state, Path(directory) / f"zero-reference-{trainer.global_step}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.skipif(importlib.util.find_spec("deepspeed") is None, reason="DeepSpeed optional dependency")
def test_real_cpu_zero_universal_resize_preserves_moments_groups_frozen_and_ema(tmp_path, monkeypatch, stage):
    # Real ZeRO engine / official Universal converter, with the official CPU
    # accelerator. CUDA placement and kernel execution have separate tests.
    monkeypatch.setenv("DS_ACCELERATOR", "cpu")
    resume = None
    for phase, world in enumerate((2, 1, 2)):
        mp.spawn(_zero_elastic_worker, args=(world, str(tmp_path / f"rdzv-{phase}"), str(tmp_path), resume, stage),
                 nprocs=world, join=True)
        resume = latest_elastic_checkpoint(tmp_path)
        assert resume == str(tmp_path / f"checkpoint-{2 * (phase + 1)}")
