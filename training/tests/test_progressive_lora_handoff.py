"""Real PEFT safe merge, stage handoff, and exact interrupted training."""
import copy
import threading
import pytest
import torch
from torch import nn

from training.trainer_args import TrainerArgs
from training.trainers.progressive_distill_trainer import ProgressiveDistillTrainer
from training.trainers.step_distill_trainer import StepDistillTrainer
from training.utils.peft_handoff import merge_lora_student_state
from training.tests.test_auxiliary_accumulation import assert_nested_equal


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)

    def forward(self, hidden_states, timestep, **kwargs):
        value = hidden_states.movedim(1, -1)
        return self.head(self.proj(value).tanh()).movedim(-1, 1)


def make_peft(base, **kwargs):
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(r=2, lora_alpha=3, target_modules=["proj"],
                             modules_to_save=["head"], **kwargs)
    model = peft.get_peft_model(base, config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.03)
    return model


@pytest.mark.parametrize("variant", [{}, {"use_dora": True}, {"use_rslora": True}])
def test_safe_cpu_merge_preserves_teacher_and_student_and_matches_prediction(variant):
    torch.manual_seed(34)
    teacher = TinyFlow()
    student = make_peft(copy.deepcopy(teacher), **variant)
    source = {key: value.detach().clone() for key, value in student.state_dict().items()}
    teacher_before = copy.deepcopy(teacher.state_dict())
    identities = [id(parameter) for parameter in teacher.parameters()]
    state = merge_lora_student_state(teacher, student, source)
    assert [id(parameter) for parameter in teacher.parameters()] == identities
    assert_nested_equal(teacher.state_dict(), teacher_before)
    assert_nested_equal(student.state_dict(), source)
    teacher.load_state_dict(state, strict=True)
    x, t = torch.randn(2, 2, 2, 3, 3), torch.ones(2)
    teacher.eval(), student.eval()
    torch.testing.assert_close(teacher(x, t), student(x, t), rtol=1e-5, atol=1e-6)
    assert [id(parameter) for parameter in teacher.parameters()] == identities


def build_trainer(path, steps=4, resume="", method="progressive_distill"):
    torch.manual_seed(77)
    teacher = TinyFlow()
    student = make_peft(copy.deepcopy(teacher))
    args = TrainerArgs(distill_method=method, output_dir=str(path), mixed_precision="no",
        use_lora=True, use_dual_model=method == "step_distill", progressive_stages=[8, 4, 2],
        progressive_stage_steps=2, max_train_steps=steps, resume_from=resume, report_to="none",
        warmup_steps=0, num_workers=0, batch_size=1, gradient_checkpointing=False,
        log_every=100, save_every=100, learning_rate=0.01)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=0.01)
    cls = ProgressiveDistillTrainer if method == "progressive_distill" else StepDistillTrainer
    trainer = cls(args=args, teacher_model=teacher, student_model=student, optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        train_dataloader=[{"latents": torch.randn(1, 2, 2, 3, 3)} for _ in range(3)],
        device=torch.device("cpu"))
    torch.manual_seed(94)
    return trainer


def test_progressive_lora_stage_handoff_resume_exact_weights_optimizer_and_rng(tmp_path):
    continuous = build_trainer(tmp_path / "continuous")
    teacher_ids = [id(p) for p in continuous.teacher_model.parameters()]
    continuous.train()
    expected_rng = torch.rand(4)
    interrupted = build_trainer(tmp_path / "interrupted", steps=2)
    interrupted.train()
    resumed = build_trainer(tmp_path / "resumed", resume=str(tmp_path / "interrupted" / "checkpoint-2"))
    resumed.train()
    assert continuous.current_stage == resumed.current_stage == 1
    assert [id(p) for p in continuous.teacher_model.parameters()] == teacher_ids
    for attribute in ("student_model", "teacher_model", "optimizer", "lr_scheduler"):
        assert_nested_equal(getattr(continuous, attribute).state_dict(), getattr(resumed, attribute).state_dict())
    torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)


def test_dual_student_lora_keeps_identical_frozen_parameter_contract(tmp_path):
    trainer = build_trainer(tmp_path, method="step_distill")
    high = dict(trainer.student_high.named_parameters())
    low = dict(trainer.student_low.named_parameters())
    assert high.keys() == low.keys()
    assert any(not parameter.requires_grad for parameter in high.values())
    assert all(parameter.requires_grad == low[name].requires_grad for name, parameter in high.items())
    expected = {id(p) for model in (trainer.student_high, trainer.student_low)
                for p in model.parameters() if p.requires_grad}
    actual = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert expected == actual


def test_merge_skeleton_does_not_copy_runtime_hook_ownership():
    teacher = TinyFlow()
    student = make_peft(copy.deepcopy(teacher))
    class RuntimeOwner:
        def __init__(self):
            self.lock = threading.Lock()
        def forward(self, *args):
            pass
    owner = RuntimeOwner()
    handle = teacher.proj.register_forward_hook(owner.forward)
    try:
        state = merge_lora_student_state(teacher, student,
            {key: value.detach().cpu().clone() for key, value in student.state_dict().items()})
        assert set(state) == set(teacher.state_dict())
        assert next(iter(teacher.proj._forward_hooks.values())).__self__ is owner
    finally:
        handle.remove()


def test_merge_skeleton_drops_process_groups_and_mesh_registries_without_mutating_teacher():
    teacher = TinyFlow()
    student = make_peft(copy.deepcopy(teacher))
    class UncopyableRuntime:
        def __deepcopy__(self, memo):
            raise AssertionError("ProcessGroup/DeviceMesh runtime must not be deep-copied")
    group, mesh, registry = UncopyableRuntime(), UncopyableRuntime(), UncopyableRuntime()
    teacher._context_parallel_group = group
    teacher.proj._parallel_config = mesh
    teacher.head._diffusers_hook = registry
    merged = merge_lora_student_state(teacher, student,
        {key: value.detach().clone() for key, value in student.state_dict().items()})
    assert set(merged) == set(teacher.state_dict())
    assert teacher._context_parallel_group is group
    assert teacher.proj._parallel_config is mesh
    assert teacher.head._diffusers_hook is registry
