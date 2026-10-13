"""Native declarative TP×PP and real action/camera PP numerical contracts."""
import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from training.native_model_adapter import NativeTrainingAdapter
from training.parallel.plans import apply_model_parallel_plan
from training.parallel.state import collect_full_model_state, collect_full_named_tensors, load_full_model_state


class VideoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in, self.proj_out = nn.Linear(2, 8), nn.Linear(8, 2)
        self.gradient_checkpointing = True
    def forward(self, hidden_states, timestep):
        value = hidden_states.permute(0, 2, 3, 4, 1)
        return self.proj_out(torch.nn.functional.gelu(self.proj_in(value))).permute(0, 4, 1, 2, 3)


class BufferedProjection(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.register_buffer("scale", torch.tensor(0.7))
    def forward(self, value):
        return self.linear(value) * self.scale


def _model_and_inputs(kind):
    inputs = dict(hidden_states=torch.randn(1, 2, 3, 4, 4), timestep=torch.tensor([30.]))
    if kind == "declarative":
        descriptor = {"parallel_plan": {"schema_version": 1,
            "tensor": {"modules": {"proj_in": "column", "proj_out": "row"}},
            "pipeline": {"modules": {"proj_in": 0, "proj_out": 1},
                "gradient_checkpointing_attribute": "gradient_checkpointing"}}}
        return NativeTrainingAdapter(VideoMLP(), descriptor), inputs
    from training.native_adapters.world_models import load_builtin_world_model
    config = dict(model_type="i2v" if kind.startswith("matrix") else "t2v", in_dim=6 if kind.startswith("matrix") else 2,
                  out_dim=2, dim=24, ffn_dim=32, freq_dim=8, text_dim=8, text_len=3, num_heads=2, num_layers=2)
    if kind.startswith("matrix"):
        config["action_config"] = dict(blocks=[0, 1], enable_keyboard=True, enable_mouse=True, heads_num=2,
            hidden_size=8, img_hidden_size=24, keyboard_dim_in=4, keyboard_hidden_dim=16, mouse_dim_in=2,
            mouse_hidden_dim=16, mouse_qk_dim_list=[4, 2, 2], patch_size=[1, 2, 2], qk_norm=True,
            qkv_bias=False, rope_dim_list=[4, 2, 2], rope_theta=256, vae_time_compression_ratio=4, windows_size=3)
        inputs.update(image_cond=torch.randn(1, 4, 3, 4, 4), image_embeds=torch.randn(1, 3, 1280),
                      keyboard_actions=torch.randn(1, 9, 4), mouse_actions=torch.randn(1, 9, 2))
        descriptor = {"architecture": "matrix_game2", "model_config": config,
                      "variant": "foundation" if "foundation" in kind else "causal"}
    else:
        inputs.update(encoder_hidden_states=torch.randn(1, 3, 8), camera_features=torch.randn(1, 384, 3, 4, 4))
        descriptor = {"architecture": "lingbot_cam", "model_config": config}
    model = load_builtin_world_model(".", descriptor, weights=False)
    model.enable_gradient_checkpointing()
    with torch.no_grad():
        model.model.head.head.weight.normal_(std=0.02)
        for name, parameter in model.named_parameters():
            if "action_model.proj_" in name and name.endswith("weight"):
                parameter.normal_(std=0.02)
    return model, inputs


def _worker(rank, world, path, kind):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=100))
    try:
        tp_group, pp_group = None, dist.group.WORLD
        if world == 4:
            for ranks in ([0, 1], [2, 3]):
                group = dist.new_group(ranks)
                if rank in ranks:
                    tp_group = group
            for ranks in ([0, 2], [1, 3]):
                group = dist.new_group(ranks)
                if rank in ranks:
                    pp_group = group
        torch.manual_seed(927)
        reference, inputs = _model_and_inputs(kind)
        model = apply_model_parallel_plan(copy.deepcopy(reference), tp_group=tp_group, pp_group=pp_group)
        expected, actual = reference(**inputs), model(**inputs)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
        expected.square().mean().backward()
        actual.square().mean().backward()
        gradients = collect_full_named_tensors(model, {name: value.grad for name, value in model.named_parameters()}, dist.group.WORLD)
        for name, value in reference.named_parameters():
            assert (name in gradients) == (value.grad is not None), name
            if value.grad is not None:
                torch.testing.assert_close(gradients[name], value.grad, atol=4e-6, rtol=2e-4,
                                           msg=lambda message: f"{name}: {message}")
        assert sum(value.numel() for value in model.parameters()) < sum(value.numel() for value in reference.parameters())
        torch.optim.SGD(model.parameters(), lr=0.02).step()
        torch.optim.SGD(reference.parameters(), lr=0.02).step()
        state = collect_full_model_state(model, dist.group.WORLD)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(state[name], value, atol=1e-6, rtol=1e-4)
        with torch.no_grad():
            for value in model.parameters():
                value.zero_()
        load_full_model_state(model, state)
        recovered = collect_full_model_state(model, dist.group.WORLD)
        for name, value in state.items():
            torch.testing.assert_close(recovered[name], value, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,kind", [(4, "declarative"), (2, "matrix_causal"),
    (2, "matrix_foundation"), (2, "lingbot")])
def test_native_stage_owned_forward_backward_update_restore(tmp_path, world, kind):
    pytest.importorskip("diffusers")
    mp.spawn(_worker, args=(world, str(tmp_path / "init"), kind), nprocs=world, join=True)


def _offload_worker(rank, path, directory):
    from training.runtime.teacher_offload import TeacherParameterOffloader
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(621)
        reference, inputs = _model_and_inputs("declarative")
        reference.model.proj_in = BufferedProjection(2, 8)
        reference.model.proj_out = BufferedProjection(8, 2)
        reference.eval()
        model = apply_model_parallel_plan(copy.deepcopy(reference), pp_group=dist.group.WORLD)
        original = collect_full_model_state(model, dist.group.WORLD)
        # Module._apply replaces buffers on device/dtype moves; custom Tensor
        # attributes do not survive that replacement. Layout must still work.
        model.to(dtype=torch.float64).to(dtype=torch.float32)
        moved = collect_full_model_state(model, dist.group.WORLD)
        assert moved.keys() == original.keys()
        offloader = TeacherParameterOffloader(model, torch.device("cpu"), target="nvme", directory=directory)
        try:
            with torch.no_grad():
                expected = reference(**inputs)
                actual, _ = offloader.forward(inputs)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            recovered = collect_full_model_state(model, dist.group.WORLD)
            assert recovered.keys() == original.keys()
            for name in original:
                torch.testing.assert_close(recovered[name], original[name], atol=0, rtol=0)
        finally:
            offloader.close()
        after_close = collect_full_model_state(model, dist.group.WORLD)
        assert after_close.keys() == original.keys()
    finally:
        dist.destroy_process_group()


def test_native_pp_nvme_functional_call_preserves_buffer_ownership(tmp_path):
    mp.spawn(_offload_worker, args=(str(tmp_path / "offload_init"), str(tmp_path / "weights")),
             nprocs=2, join=True)
