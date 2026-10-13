from types import SimpleNamespace

import pytest
import torch

from training.train_distill import _build_validation_dataloader
from training.trainer_args import TrainerArgs
from training.trainers.step_distill_trainer import StepDistillTrainer
from training.tests.test_bucket_shape_contracts import _cached_rows


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_validation_rejects_invalid_batch_limit(value):
    with pytest.raises(ValueError, match="eval_batches"):
        TrainerArgs(eval_batches=value)


def test_validation_loader_uses_exact_buckets_and_retains_partial_batches(tmp_path):
    dataset = _cached_rows(tmp_path, [{}, {}, {"frames": 5, "audio": 7, "fps": 24}])
    args = TrainerArgs(use_bucket_sampler=True, batch_size=2, num_workers=0)
    batches = list(_build_validation_dataloader(dataset, args, 0, 1))
    assert [batch["latents"].shape[0] for batch in batches] == [2, 1]
    assert [batch["latents"].shape[2] for batch in batches] == [3, 5]
    assert batches[-1]["audio_latents"].shape == (1, 7, 4)


def test_evaluation_weights_partial_batches_by_sample_count():
    trainer = object.__new__(StepDistillTrainer)
    trainer.student_model = torch.nn.Linear(1, 1)
    trainer.val_dataloader = [{"latents": torch.ones(2, 1)}, {"latents": torch.full((1, 1), 3.)}]
    trainer.args = SimpleNamespace(eval_batches=2)
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False
    trainer._move_batch_to_device = lambda batch: batch
    trainer._prepare_batch_for_model = lambda batch: batch
    trainer._prepare_sequence_parallel_batch = lambda batch: batch
    trainer.validation_step = lambda batch: batch["latents"].mean()
    assert trainer.evaluate()["eval_loss"] == pytest.approx(5 / 3)
    assert trainer.student_model.training


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("world_size", [2, 4])
def test_validation_loader_counts_each_sample_once(tmp_path, bucketed, world_size):
    dataset = _cached_rows(tmp_path, [{} for _ in range(5)])
    args = TrainerArgs(use_bucket_sampler=bucketed, batch_size=2, num_workers=0)
    counts, forwards = [], []
    for rank in range(world_size):
        loader = _build_validation_dataloader(dataset, args, rank, world_size)
        count = 0
        batches = list(loader)
        for index, batch in enumerate(batches):
            if not loader.batch_sampler.is_padding_batch(index):
                count += batch["latents"].shape[0]
        counts.append(count)
        forwards.append(len(batches))
    assert sum(counts) == 5
    assert len(set(forwards)) == 1


def test_evaluation_executes_padding_but_excludes_its_loss():
    class ValidationBatches(list):
        batch_sampler = SimpleNamespace(is_padding_batch=lambda index: index == 1)

    trainer = object.__new__(StepDistillTrainer)
    trainer.student_model = torch.nn.Linear(1, 1)
    trainer.val_dataloader = ValidationBatches([
        {"latents": torch.ones(2, 1)}, {"latents": torch.full((2, 1), float("nan"))}])
    trainer.args = SimpleNamespace(eval_batches=2)
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False
    trainer._move_batch_to_device = lambda batch: batch
    trainer._prepare_batch_for_model = lambda batch: batch
    trainer._prepare_sequence_parallel_batch = lambda batch: batch
    calls = []
    def validation_step(batch):
        calls.append(batch)
        return batch["latents"].mean()
    trainer.validation_step = validation_step
    assert trainer.evaluate()["eval_loss"] == pytest.approx(1.)
    assert len(calls) == 2
