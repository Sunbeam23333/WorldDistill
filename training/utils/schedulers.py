"""Learning rate schedulers for distillation training."""

import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _cosine_schedule(step, warmup, total, min_ratio=0.0):
    if step < warmup:
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _linear_schedule(step, warmup, total):
    if step < warmup:
        return float(step) / float(max(1, warmup))
    return max(0.0, float(total - step) / float(max(1, total - warmup)))


def _constant_schedule(step, warmup):
    return min(1.0, float(step) / float(max(1, warmup)))


def build_lr_scheduler(optimizer: Optimizer, scheduler_type: str = "cosine",
                       warmup_steps: int = 1000, total_steps: int = 100000,
                       min_lr_ratio: float = 0.1) -> LambdaLR:
    if scheduler_type == "cosine":
        fn = partial(_cosine_schedule, warmup=warmup_steps, total=total_steps, min_ratio=min_lr_ratio)
    elif scheduler_type == "linear":
        fn = partial(_linear_schedule, warmup=warmup_steps, total=total_steps)
    elif scheduler_type in ("constant", "constant_with_warmup"):
        fn = partial(_constant_schedule, warmup=warmup_steps)
    elif scheduler_type == "cosine_with_min_lr":
        fn = partial(_cosine_schedule, warmup=warmup_steps, total=total_steps, min_ratio=min_lr_ratio)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    return LambdaLR(optimizer, fn)
