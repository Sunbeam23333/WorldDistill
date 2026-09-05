"""Distillation Trainers.

Each trainer implements a specific distillation method:
- StepDistillTrainer: Fixed N-step distillation
- StreamDistillTrainer: Diffusion Forcing streaming distillation
- ProgressiveDistillTrainer: Progressive halving of denoising steps
- ConsistencyDistillTrainer: Trajectory Consistency Distillation (TCD)
- ContextForcingTrainer: Memory-aware Context Forcing distillation
- AdversarialDistillTrainer: Adversarial Diffusion Distillation (ADD/LADD)
- DMDDistillTrainer: Distribution Matching Distillation (DMD/DMD2)
"""

from distill_capabilities import REGISTRY_BACKED_DISTILL_METHODS

from training.trainers.base_distill_trainer import BaseDistillTrainer
from training.trainers.step_distill_trainer import StepDistillTrainer
from training.trainers.stream_distill_trainer import StreamDistillTrainer
from training.trainers.progressive_distill_trainer import ProgressiveDistillTrainer
from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer
from training.trainers.context_forcing_trainer import ContextForcingTrainer
from training.trainers.adversarial_distill_trainer import AdversarialDistillTrainer
from training.trainers.dmd_distill_trainer import DMDDistillTrainer

TRAINER_REGISTRY = {
    "step_distill": StepDistillTrainer,
    "stream_distill": StreamDistillTrainer,
    "progressive_distill": ProgressiveDistillTrainer,
    "consistency_distill": ConsistencyDistillTrainer,
    "context_forcing": ContextForcingTrainer,
    "adversarial_distill": AdversarialDistillTrainer,
    "dmd_distill": DMDDistillTrainer,
}

_missing_registry_methods = [method for method in REGISTRY_BACKED_DISTILL_METHODS if method not in TRAINER_REGISTRY]
_extra_registry_methods = [method for method in TRAINER_REGISTRY if method not in REGISTRY_BACKED_DISTILL_METHODS]
if _missing_registry_methods or _extra_registry_methods:  # pragma: no cover - import-time contract check
    raise RuntimeError(
        "TRAINER_REGISTRY drifted from REGISTRY_BACKED_DISTILL_METHODS: "
        f"missing={_missing_registry_methods}, extra={_extra_registry_methods}"
    )


def build_trainer(method: str, **kwargs) -> BaseDistillTrainer:
    """Build a trainer instance by method name."""
    if method not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown distill method: {method}. "
            f"Available: {list(TRAINER_REGISTRY.keys())}"
        )
    return TRAINER_REGISTRY[method](**kwargs)

__all__ = [
    "BaseDistillTrainer",
    "StepDistillTrainer",
    "StreamDistillTrainer",
    "ProgressiveDistillTrainer",
    "ConsistencyDistillTrainer",
    "ContextForcingTrainer",
    "AdversarialDistillTrainer",
    "DMDDistillTrainer",
    "build_trainer",
    "TRAINER_REGISTRY",
]
