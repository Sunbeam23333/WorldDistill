"""Equal-forward-count validation sharding without counting padding twice."""

import math

from torch.utils.data import Sampler


class DistributedValidationBatchSampler(Sampler):
    """Shard complete, unpartitioned validation batches across data ranks.

    Every rank must receive an identically ordered, deterministic base sampler
    (for example a non-shuffled BucketSampler with rank=0, world_size=1).
    Whole-batch copies equalize forward counts for FSDP/ZeRO collectives; callers
    must exclude those copies from metric sums using ``is_padding_batch``.
    Real and padding examples are never mixed in the same local batch.
    """

    def __init__(self, base_batch_sampler, *, rank=0, world_size=1):
        if (not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1
                or not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size):
            raise ValueError("Validation sampler requires a positive world_size and a valid integer rank")
        self.base_batch_sampler = base_batch_sampler
        self.rank = rank
        self.world_size = world_size
        self._global_batch_count = None

    def __iter__(self):
        # Keep only indices, never loaded samples. Materializing the list gives
        # every rank the same original batch count, including partial buckets.
        batches = [list(batch) for batch in self.base_batch_sampler]
        if any(not batch for batch in batches):
            raise ValueError("Validation base batch sampler cannot emit empty batches")
        self._global_batch_count = len(batches)
        if not batches:
            return
        padded_count = math.ceil(len(batches) / self.world_size) * self.world_size
        for global_index in range(self.rank, padded_count, self.world_size):
            yield list(batches[global_index % len(batches)])

    def __len__(self):
        return math.ceil(len(self.base_batch_sampler) / self.world_size)

    def is_padding_batch(self, local_batch_index):
        """Whether this zero-based local forward exists only for collectives."""
        total = (len(self.base_batch_sampler) if self._global_batch_count is None
                 else self._global_batch_count)
        local_count = math.ceil(total / self.world_size)
        if (not isinstance(local_batch_index, int) or isinstance(local_batch_index, bool)
                or not 0 <= local_batch_index < local_count):
            raise IndexError("Validation local batch index is out of range")
        return self.rank + local_batch_index * self.world_size >= total
