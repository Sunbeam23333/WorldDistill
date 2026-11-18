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
from training.trainers.cfg_distill_trainer import CFGDistillTrainer
from training.trainers.stream_distill_trainer import StreamDistillTrainer
from training.trainers.progressive_distill_trainer import ProgressiveDistillTrainer
from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer
from training.trainers.consistency_variants import PCMTrainer, TCMTrainer
from training.trainers.advanced_distill_trainers import (
    SelfForcingTrainer, CausalForcingTrainer, RCMTrainer, SCMTrainer,
    DiagonalDistillTrainer, ReflowTrainer, TMDTrainer, SiDDiTTrainer,
    VSDRLTrainer, DataFreeDMDTrainer,
)
from training.trainers.context_forcing_trainer import ContextForcingTrainer
from training.trainers.adversarial_distill_trainer import AdversarialDistillTrainer
from training.trainers.dmd_distill_trainer import DMDDistillTrainer
from training.trainers.mean_flow_distill_trainer import MeanFlowDistillTrainer
from training.trainers.joint_av_distill_trainer import JointAVDistillTrainer

TRAINER_REGISTRY = {
    "step_distill": StepDistillTrainer,
    "cfg_distill": CFGDistillTrainer,
    "self_forcing": SelfForcingTrainer,
    "causal_forcing": CausalForcingTrainer,
    "stream_distill": StreamDistillTrainer,
    "progressive_distill": ProgressiveDistillTrainer,
    "consistency_distill": ConsistencyDistillTrainer,
    "pcm_distill": PCMTrainer,
    "tcm_distill": TCMTrainer,
    "rcm_distill": RCMTrainer,
    "scm_distill": SCMTrainer,
    "diagonal_distill": DiagonalDistillTrainer,
    "reflow": ReflowTrainer,
    "tmd_distill": TMDTrainer,
    "sid_dit_distill": SiDDiTTrainer,
    "vsd_rl_distill": VSDRLTrainer,
    "data_free_dmd_distill": DataFreeDMDTrainer,
    "context_forcing": ContextForcingTrainer,
    "adversarial_distill": AdversarialDistillTrainer,
    "dmd_distill": DMDDistillTrainer,
    "mean_flow_distill": MeanFlowDistillTrainer,
    "joint_av_distill": JointAVDistillTrainer,
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
    "CFGDistillTrainer",
    "SelfForcingTrainer",
    "CausalForcingTrainer",
    "StreamDistillTrainer",
    "ProgressiveDistillTrainer",
    "ConsistencyDistillTrainer",
    "PCMTrainer",
    "TCMTrainer",
    "RCMTrainer",
    "SCMTrainer",
    "DiagonalDistillTrainer",
    "ReflowTrainer",
    "TMDTrainer",
    "SiDDiTTrainer",
    "VSDRLTrainer",
    "DataFreeDMDTrainer",
    "ContextForcingTrainer",
    "AdversarialDistillTrainer",
    "DMDDistillTrainer",
    "MeanFlowDistillTrainer",
    "JointAVDistillTrainer",
    "build_trainer",
    "TRAINER_REGISTRY",
]
