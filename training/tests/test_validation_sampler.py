import itertools

import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, TensorDataset

from training.data.validation_sampler import DistributedValidationBatchSampler


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sample_count", [0, 1, 5])
def test_validation_shards_have_equal_forwards_and_exact_unique_metric_weight(
        sample_count, batch_size, world_size):
    data = TensorDataset(torch.arange(1, sample_count + 1, dtype=torch.float32))
    base = BatchSampler(SequentialSampler(data), batch_size=batch_size, drop_last=False)
    rank_batches, real_indices, summed_loss, summed_count = [], [], 0., 0
    for rank in range(world_size):
        sampler = DistributedValidationBatchSampler(base, rank=rank, world_size=world_size)
        expected_padding = [sampler.is_padding_batch(index) for index in range(len(sampler))]
        batches = list(sampler)
        rank_batches.append(batches)
        assert len(batches) == len(sampler)
        assert [sampler.is_padding_batch(index) for index in range(len(batches))] == expected_padding
        for index, (values,) in enumerate(DataLoader(data, batch_sampler=sampler)):
            padding = sampler.is_padding_batch(index)
            if not padding:
                real_indices.extend(batches[index])
                # This is the sample-weighted evaluation accumulation, including
                # real partial batches but excluding whole-batch padding copies.
                summed_loss += values.mean().item() * values.shape[0]
                summed_count += values.shape[0]
    assert len({len(batches) for batches in rank_batches}) == 1
    assert sorted(real_indices) == list(range(sample_count))
    assert summed_count == sample_count
    assert summed_loss == pytest.approx(sample_count * (sample_count + 1) / 2)
    if sample_count:
        assert summed_loss / summed_count == pytest.approx((sample_count + 1) / 2)
        original_batches = list(base)
        # Forward-only padding must copy a whole existing batch, never append
        # duplicate examples to a real partial batch.
        assert all(batch in original_batches for batch in itertools.chain.from_iterable(rank_batches))


def test_validation_sharding_preserves_heterogeneous_bucket_boundaries():
    base = [[0, 1], [2], [3, 4], [5], [6, 7]]
    first = DistributedValidationBatchSampler(base, rank=0, world_size=2)
    second = DistributedValidationBatchSampler(base, rank=1, world_size=2)
    assert list(first) == [[0, 1], [3, 4], [6, 7]]
    assert list(second) == [[2], [5], [0, 1]]
    assert [first.is_padding_batch(index) for index in range(3)] == [False, False, False]
    assert [second.is_padding_batch(index) for index in range(3)] == [False, False, True]


@pytest.mark.parametrize("rank,world_size", [(-1, 2), (2, 2), (0, 0), (0, True), (False, 2)])
def test_validation_sampler_rejects_invalid_distributed_coordinates(rank, world_size):
    with pytest.raises(ValueError, match="world_size"):
        DistributedValidationBatchSampler([[0]], rank=rank, world_size=world_size)


def test_validation_sampler_rejects_empty_base_batch_and_invalid_local_index():
    sampler = DistributedValidationBatchSampler([[0], []], rank=0, world_size=2)
    with pytest.raises(ValueError, match="empty batches"):
        list(sampler)
    empty = DistributedValidationBatchSampler([], rank=0, world_size=2)
    assert list(empty) == []
    with pytest.raises(IndexError, match="out of range"):
        empty.is_padding_batch(0)
    sampler = DistributedValidationBatchSampler([[0]], rank=0, world_size=2)
    for index in (-1, 1, True):
        with pytest.raises(IndexError, match="out of range"):
            sampler.is_padding_batch(index)
