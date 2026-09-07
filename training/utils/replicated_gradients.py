"""Replicated-gradient synchronization for dynamically routed multi-forward graphs.

DDP's per-forward unused-parameter traversal cannot describe an iteration that
uses different experts in several forwards before one backward. These helpers
reduce the completed iteration's gradients instead, keeping globally unused
parameters as None. This is a correctness-oriented fallback, not FSDP/sharding.
"""

import torch
import torch.distributed as dist


@torch.no_grad()
def broadcast_replicated_model(model, src=0):
    if not dist.is_initialized():
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor, src=src)


@torch.no_grad()
def synchronize_replicated_gradients(model, bucket_bytes=25 * 1024 * 1024):
    """Average local gradients, including locally absent but globally used ones.

    Call before GradScaler.unscale_: all ranks must see the same scaled Inf/NaN
    values before deciding whether to skip the optimizer step.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    has_sparse = torch.tensor(int(any(p.grad is not None and p.grad.is_sparse for p in parameters)),
                              dtype=torch.int32, device=parameters[0].device)
    dist.all_reduce(has_sparse, op=dist.ReduceOp.MAX)
    if has_sparse.item():
        raise ValueError("Replicated routed-gradient synchronization requires dense gradients on every rank")
    usage = torch.tensor([int(p.grad is not None) for p in parameters],
                         dtype=torch.int32, device=parameters[0].device)
    dist.all_reduce(usage, op=dist.ReduceOp.MAX)
    world_size = dist.get_world_size()
    bucket, size, signature = [], 0, None

    def flush():
        if not bucket:
            return
        packed = torch.cat([gradient.reshape(-1) for gradient in bucket])
        dist.all_reduce(packed)
        packed.div_(world_size)
        offset = 0
        for gradient in bucket:
            count = gradient.numel()
            gradient.copy_(packed[offset:offset + count].reshape(gradient.shape))
            offset += count
        bucket.clear()

    for parameter, used in zip(parameters, usage.cpu().tolist()):
        if not used:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient = parameter.grad
        current_signature = (gradient.dtype, gradient.device)
        current_bytes = gradient.numel() * gradient.element_size()
        if bucket and (signature != current_signature or size + current_bytes > bucket_bytes):
            flush()
            size = 0
        signature = current_signature
        if current_bytes > bucket_bytes:
            # A single large weight must not force an unbounded concatenation.
            flattened = gradient.contiguous().view(-1)
            elements = max(1, bucket_bytes // gradient.element_size())
            for offset in range(0, flattened.numel(), elements):
                shard = flattened[offset:offset + elements]
                dist.all_reduce(shard)
                shard.div_(world_size)
            gradient.copy_(flattened.reshape(gradient.shape))
        else:
            bucket.append(gradient)
            size += current_bytes
    flush()
