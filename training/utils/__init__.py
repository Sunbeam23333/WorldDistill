"""Training utilities.

- optimizers: AdamW + Muon optimizer
- schedulers: LR schedulers (cosine, linear, constant, etc.)
- distributed: Distributed training helpers (DDP, FSDP, DeepSpeed, Sequence Parallel)
"""

from training.utils.optimizers import build_optimizer
from training.utils.schedulers import build_lr_scheduler
from training.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    all_reduce_mean,
    is_main_process,
    wrap_model,
    wrap_model_ddp,
    wrap_model_fsdp,
    get_deepspeed_config,
    init_deepspeed,
    init_sequence_parallel,
    scatter_sequence,
    gather_sequence,
)

__all__ = [
    "build_optimizer",
    "build_lr_scheduler",
    "setup_distributed",
    "cleanup_distributed",
    "all_reduce_mean",
    "is_main_process",
    "wrap_model",
    "wrap_model_ddp",
    "wrap_model_fsdp",
    "get_deepspeed_config",
    "init_deepspeed",
    "init_sequence_parallel",
    "scatter_sequence",
    "gather_sequence",
]
