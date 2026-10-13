import pytest
from torch.utils.data.distributed import DistributedSampler

from training.train_distill import _data_parallel_coordinates
from training.data.bucket_sampler import BucketSampler


class Dataset:
    data = [{"resolution": [4, 4], "num_frames": 16} for _ in range(16)]

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize("bucket", [False, True])
def test_context_ranks_share_batches_and_data_groups_partition_dataset(bucket):
    samples = []
    for rank in range(4):
        data_rank, data_size = _data_parallel_coordinates(rank, 4, 2)
        if bucket:
            sampler = BucketSampler(Dataset(), batch_size=2, shuffle=False, seed=7,
                                    rank=data_rank, world_size=data_size)
            samples.append([index for batch in sampler for index in batch])
        else:
            sampler = DistributedSampler(Dataset(), num_replicas=data_size, rank=data_rank,
                                         shuffle=False, seed=7)
            samples.append(list(sampler))
    assert samples[0] == samples[1] and samples[2] == samples[3]
    assert set(samples[0]).isdisjoint(samples[2])
    assert set(samples[0] + samples[2]) == set(range(16))


@pytest.mark.parametrize("rank,world,context", [(0, 3, 2), (0, 1, 0), (4, 4, 2), (-1, 4, 2)])
def test_invalid_context_mesh_fails_before_data_loading(rank, world, context):
    with pytest.raises(ValueError, match="sp_size"):
        _data_parallel_coordinates(rank, world, context)
