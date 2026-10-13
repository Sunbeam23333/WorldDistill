"""Two-rank regression: different MeanFlow experts, identical reduced updates."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from torch import nn
from training.model_adapter import NoiseRoutedDenoiser
from training.trainer_args import TrainerArgs
from training.trainers.mean_flow_distill_trainer import MeanFlowDistillTrainer
from training.utils.distributed import setup_distributed
from training.utils.meanflow import MeanFlowTimeAdapter


class Expert(nn.Module):
    config = {"in_channels": 2, "out_channels": 2}
    def __init__(self, value):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))
    def forward(self, hidden_states, timestep, **kwargs):
        return hidden_states * self.scale


def main():
    torch.set_num_threads(1)
    rank, world = setup_distributed(backend="gloo", timeout_seconds=60)
    try:
        torch.manual_seed(43)
        teacher = NoiseRoutedDenoiser(Expert(0.8), Expert(0.9), 0.5)
        student = MeanFlowTimeAdapter(NoiseRoutedDenoiser(Expert(0.4), Expert(0.3), 0.5))
        args = TrainerArgs(distill_method="mean_flow_distill", mixed_precision="no", report_to="none",
            max_train_steps=2, output_dir=sys.argv[1], save_every=100, log_every=100, batch_size=1,
            meanflow_norm_power=0, gradient_checkpointing=False)
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
        trainer = MeanFlowDistillTrainer(args=args, teacher_model=teacher, student_model=student,
            optimizer=optimizer, lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            train_dataloader=[{"latents": torch.ones(1, 2, 2, 2, 2)} for _ in range(2)], device=torch.device("cpu"))
        trainer._sample_time_pair = lambda size: (torch.full((size,), 0.8 if rank == 0 else 0.2), torch.full((size,), 0.1))
        assert trainer._manual_replicated_sync
        trainer.train()
        flat = torch.cat([parameter.detach().flatten() for parameter in student.parameters()])
        minimum, maximum = flat.clone(), flat.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        torch.testing.assert_close(minimum, maximum, rtol=0, atol=0)
        assert not torch.equal(student.base_model.high.scale.detach(), torch.tensor(0.4))
        assert not torch.equal(student.base_model.low.scale.detach(), torch.tensor(0.3))
        if rank == 0:
            print("routed-meanflow-two-rank-passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
