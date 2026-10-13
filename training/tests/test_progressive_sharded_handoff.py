from types import SimpleNamespace
from contextlib import contextmanager
import sys
import torch
from torch import nn

from training.trainer_args import TrainerArgs
from training.trainers.progressive_distill_trainer import ProgressiveDistillTrainer


def trainer(mode="ddp"):
    student, teacher = nn.Linear(2, 2), nn.Linear(2, 2)
    args = TrainerArgs(distill_method="progressive_distill", parallel_mode=mode,
        mixed_precision="no", report_to="none", progressive_stages=[8, 4, 2],
        progressive_stage_steps=2, warmup_steps=0, learning_rate=0.03)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.03)
    return ProgressiveDistillTrainer(args=args, student_model=student, teacher_model=teacher,
        optimizer=optimizer, lr_scheduler=None, train_dataloader=[], device=torch.device("cpu"))


def test_sharded_stage_reset_preserves_parameter_ownership_and_master_weights():
    instance = trainer("deepspeed")
    base_optimizer = instance.optimizer
    instance.student_model(torch.ones(1, 2)).sum().backward()
    base_optimizer.step()
    params_before = list(base_optimizer.param_groups[0]["params"])
    weights_before = [p.detach().clone() for p in params_before]
    assert base_optimizer.state
    wrapper = SimpleNamespace(optimizer=base_optimizer)
    engine = SimpleNamespace(lr_scheduler=None)
    instance.optimizer, instance._deepspeed_engine = wrapper, engine
    instance._reset_stage_optimizer()
    assert instance.optimizer is wrapper and not base_optimizer.state
    assert instance.lr_scheduler is engine.lr_scheduler
    assert instance.lr_scheduler.optimizer is base_optimizer
    assert all(a is b for a, b in zip(params_before, base_optimizer.param_groups[0]["params"]))
    for weight, parameter in zip(weights_before, params_before):
        torch.testing.assert_close(weight, parameter, atol=0, rtol=0)
    instance.student_model(torch.ones(1, 2)).sum().backward()
    base_optimizer.step()
    assert base_optimizer.state  # fresh moments initialize correctly after reset


def test_zero3_teacher_state_is_cloned_inside_gather_and_stays_full(monkeypatch):
    instance = trainer("deepspeed")
    instance.args.deepspeed_stage = 3
    raw = instance.student_model
    instance.student_model = SimpleNamespace(module=raw)
    instance._deepspeed_engine = instance.student_model
    entered = []
    @contextmanager
    def gather(parameters, modifier_rank):
        entered.append((list(parameters), modifier_rank))
        yield
        with torch.no_grad():
            for parameter in raw.parameters():
                parameter.zero_()
    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(zero=SimpleNamespace(GatheredParameters=gather)))
    state = instance._full_student_state_for_teacher()
    assert len(entered) == 1 and entered[0][1] is None
    assert any(value.abs().sum() > 0 for value in state.values())
    assert all(parameter.abs().sum() == 0 for parameter in raw.parameters())
    instance.teacher_model.load_state_dict(state)
    for key, value in instance.teacher_model.state_dict().items():
        torch.testing.assert_close(value, state[key])


def test_fsdp_handoff_requests_full_state_on_every_rank(monkeypatch):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
    instance = trainer("fsdp")
    instance._is_fsdp_wrapped = lambda model: True
    entered = []
    @contextmanager
    def state_dict_type(model, kind, config):
        entered.append((model, kind, config))
        yield
    monkeypatch.setattr(FSDP, "state_dict_type", state_dict_type)
    state = instance._full_student_state_for_teacher()
    assert entered[0][1] == StateDictType.FULL_STATE_DICT
    assert entered[0][2].offload_to_cpu and not entered[0][2].rank0_only
    instance.teacher_model.load_state_dict(state)


def test_progressive_handoff_matches_keys_for_student_only_and_teacher_only_compile():
    for compiled_role in ("student", "teacher"):
        instance = trainer()
        name = f"{compiled_role}_model"
        setattr(instance, name, torch.compile(getattr(instance, name), backend="eager"))
        source = instance._full_student_state_for_teacher()
        assert set(source) == {"weight", "bias"}
        instance._teacher_state_model().load_state_dict(source, strict=True)
        for key, value in instance._teacher_state_model().state_dict().items():
            torch.testing.assert_close(value, source[key])
