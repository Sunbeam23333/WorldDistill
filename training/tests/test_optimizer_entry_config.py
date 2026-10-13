import torch

from training.trainer_args import TrainerArgs, build_training_arg_parser
from training.train_distill import _build_training_optimizer


def test_entry_preserves_adam_epsilon_and_betas():
    args = TrainerArgs(adam_epsilon=0.003, adam_beta1=0.7, adam_beta2=0.8)
    optimizer = _build_training_optimizer(torch.nn.Linear(3, 2), args)
    assert all(group["eps"] == 0.003 and group["betas"] == (0.7, 0.8)
               for group in optimizer.param_groups)


def test_entry_preserves_muon_configuration():
    parsed = build_training_arg_parser().parse_args([
        "--teacher_model_path", "teacher", "--data_json", "data.json",
        "--optimizer", "muon", "--muon_momentum", "0.8", "--no-muon_nesterov", "--muon_ns_steps", "3"])
    args = TrainerArgs(optimizer=parsed.optimizer, muon_momentum=parsed.muon_momentum,
                       muon_nesterov=parsed.muon_nesterov, muon_ns_steps=parsed.muon_ns_steps,
                       adam_epsilon=0.004)
    optimizer = _build_training_optimizer(torch.nn.Linear(3, 2), args)
    for group in optimizer.param_groups:
        assert group["momentum"] == 0.8 and group["nesterov"] is False and group["ns_steps"] == 3
    assert optimizer.adamw_optimizer.param_groups[0]["eps"] == 0.004
