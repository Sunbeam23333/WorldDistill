from types import SimpleNamespace

import torch

from training.trainers.advanced_distill_trainers import (
    DataFreeDMDTrainer,
    DiagonalDistillTrainer,
    RCMTrainer,
    SCMTrainer,
    SelfForcingTrainer,
    CausalForcingTrainer,
    SiDDiTTrainer,
    TMDTrainer,
    VSDRLTrainer,
    ReflowTrainer,
)
from training.trainer_args import build_training_arg_parser
from training.trainers import TRAINER_REGISTRY


def test_advanced_trainers_are_importable():
    assert all(cls is not None for cls in (
        SelfForcingTrainer, CausalForcingTrainer, RCMTrainer, SCMTrainer,
        DiagonalDistillTrainer, ReflowTrainer, TMDTrainer, SiDDiTTrainer,
        VSDRLTrainer, DataFreeDMDTrainer,
    ))


def test_advanced_method_cli_and_registry_contract():
    parser = build_training_arg_parser()
    methods = (
        "self_forcing", "causal_forcing", "rcm_distill", "scm_distill",
        "diagonal_distill", "reflow", "tmd_distill", "sid_dit_distill",
        "vsd_rl_distill", "data_free_dmd_distill",
    )
    for method in methods:
        parsed = parser.parse_args([
            "--teacher_model_path", "teacher", "--data_json", "data",
            "--distill_method", method,
        ])
        assert parsed.distill_method == method
        assert method in TRAINER_REGISTRY


def test_diagonal_schedule_is_monotone_and_bounded():
    trainer = DiagonalDistillTrainer.__new__(DiagonalDistillTrainer)
    trainer.device = torch.device("cpu")
    trainer.num_train_timesteps = 1000
    trainer.diagonal_slope = 1.0
    trainer.diagonal_jitter = 0.0
    schedule = trainer._sample_per_frame_timesteps(32, 9)
    assert schedule.shape == (32, 9)
    assert torch.all(schedule >= 0)
    assert torch.all(schedule <= 1000)
    assert torch.all(schedule[:, 1:] >= schedule[:, :-1])


def test_data_free_dmd_synthesizes_video_latents():
    trainer = DataFreeDMDTrainer.__new__(DataFreeDMDTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(batch_size=2)
    trainer.data_free_latent_shape = None
    trainer.data_free_latent_channels = 4
    trainer.data_free_num_frames = 3
    trainer.data_free_height = 8
    trainer.data_free_width = 8
    latents = trainer._synthetic_latents({"text": ["a", "b"]})
    assert latents.shape == (2, 4, 3, 8, 8)
    assert torch.isfinite(latents).all()
