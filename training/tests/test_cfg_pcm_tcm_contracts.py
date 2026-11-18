from types import SimpleNamespace

import pytest
import torch

from distill_capabilities import REGISTRY_BACKED_DISTILL_METHODS
from training.trainer_args import TrainerArgs, build_training_arg_parser
from training.trainers import TRAINER_REGISTRY
from training.trainers.cfg_distill_trainer import CFGDistillTrainer
from training.trainers.consistency_variants import PCMTrainer, TCMTrainer


def test_new_methods_are_registered_and_parseable():
    assert "cfg_distill" in TRAINER_REGISTRY
    assert "pcm_distill" in TRAINER_REGISTRY
    assert "tcm_distill" in TRAINER_REGISTRY
    assert set(REGISTRY_BACKED_DISTILL_METHODS) <= set(TRAINER_REGISTRY)
    parser = build_training_arg_parser()
    for method in ("cfg_distill", "pcm_distill", "tcm_distill"):
        args = parser.parse_args(["--teacher_model_path", "teacher", "--data_json", "data", "--distill_method", method])
        assert args.distill_method == method


def test_cfg_guided_target_matches_standard_formula():
    trainer = CFGDistillTrainer.__new__(CFGDistillTrainer)
    trainer.cfg_scale = 3.0
    identity = lambda value, noisy, timesteps, source, target: value
    trainer.noise_objective = SimpleNamespace(convert_prediction=identity, prediction_type="velocity")
    trainer.teacher_noise_objective = SimpleNamespace(prediction_type="velocity")
    conditional = torch.tensor([[[[[3.0]]]]])
    unconditional = torch.tensor([[[[[1.0]]]]])
    target = trainer._guided_teacher_target(
        conditional, unconditional, torch.zeros_like(conditional), torch.ones(1)
    )
    assert torch.equal(target, torch.tensor([[[[[7.0]]]]]))


def test_cfg_requires_explicit_unconditional_branch():
    trainer = CFGDistillTrainer.__new__(CFGDistillTrainer)
    trainer.unconditional_prefix = "uncond_"
    with pytest.raises(ValueError, match="unconditional text embeddings"):
        trainer._unconditional_batch({"encoder_hidden_states": torch.ones(1, 2, 3)})


def test_pcm_samples_inside_declared_phase_boundaries():
    trainer = PCMTrainer.__new__(PCMTrainer)
    trainer.args = SimpleNamespace(pcm_num_phases=4, pcm_phase_sampling="uniform")
    trainer.device = torch.device("cpu")
    trainer.num_train_timesteps = 1000
    high, low = trainer._sample_consistency_pair(256)
    assert torch.all(high >= low)
    assert torch.all(high >= 0) and torch.all(high <= 1000)
    assert torch.all(low >= 0) and torch.all(low <= 1000)
    assert torch.all((high - low) <= 250 + 1e-5)


def test_tcm_stage_and_truncation_contract():
    trainer = TCMTrainer.__new__(TCMTrainer)
    trainer.args = SimpleNamespace(tcm_stage=2, tcm_truncation_start=0.25, tcm_truncation_end=0.9)
    trainer.device = torch.device("cpu")
    trainer.num_train_timesteps = 1000
    trainer._get_delta = lambda: 50.0
    high, low = trainer._sample_consistency_pair(256)
    assert torch.all(high >= 250) and torch.all(high <= 900)
    assert torch.all(low >= 250) and torch.all(low <= high)


def test_trainer_args_validate_new_method_fields():
    assert TrainerArgs(distill_method="cfg_distill", report_to="none").cfg_scale >= 1
    with pytest.raises(ValueError, match="cfg_scale"):
        TrainerArgs(distill_method="cfg_distill", cfg_scale=0.5, report_to="none")
