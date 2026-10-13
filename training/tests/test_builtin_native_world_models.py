"""Real author architecture tests: no mock denoisers, no downloaded weights."""
import copy
import json

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("einops")
pytest.importorskip("safetensors.torch")

from training.native_adapters.world_models import load_builtin_world_model, lingbot_camera_features
from training.native_model_adapter import load_native_training_model


def tiny_descriptor(family, variant="causal"):
    config = dict(model_type="i2v" if family == "matrix_game2" else "t2v", in_dim=6 if family == "matrix_game2" else 2,
                  out_dim=2, dim=24, ffn_dim=32, freq_dim=8, text_dim=8, text_len=3, num_heads=2, num_layers=1)
    if family == "matrix_game2":
        config["action_config"] = dict(blocks=[0],enable_keyboard=True,enable_mouse=True,heads_num=2,
            hidden_size=8,img_hidden_size=24,keyboard_dim_in=4,keyboard_hidden_dim=16,
            mouse_dim_in=2,mouse_hidden_dim=16,mouse_qk_dim_list=[4,2,2],patch_size=[1,2,2],
            qk_norm=True,qkv_bias=False,rope_dim_list=[4,2,2],rope_theta=256,
            vae_time_compression_ratio=4,windows_size=3)
    return dict(schema_version=1,architecture=family,model_config=config,variant=variant)


def tiny_inputs(family, batch=2, frames=3):
    x = torch.randn(batch,2,frames,4,4)
    conditions = dict(image_cond=torch.randn(batch,4,frames,4,4),image_embeds=torch.randn(batch,3,1280),
        keyboard_actions=torch.randn(batch,(frames-1)*4+1,4),mouse_actions=torch.randn(batch,(frames-1)*4+1,2)) if family == "matrix_game2" else dict(
        encoder_hidden_states=torch.randn(batch,3,8),camera_features=torch.randn(batch,384,frames,4,4))
    return dict(hidden_states=x,timestep=torch.linspace(700,200,frames).expand(batch,-1),**conditions)


@pytest.mark.parametrize("family", ["matrix_game2","lingbot_cam"])
@pytest.mark.parametrize("checkpointing", [False,True])
def test_author_architecture_backward_and_strict_reload(tmp_path,family,checkpointing):
    torch.manual_seed(77)
    adapter = load_builtin_world_model(tmp_path,tiny_descriptor(family),weights=False)
    if checkpointing:
        adapter.enable_gradient_checkpointing()
    # Author zero-initialized output projection yields a zero initial latent
    # gradient. A trained-shaped projection exercises the entire graph.
    with torch.no_grad():
        adapter.model.head.head.weight.normal_(std=.03)
    inputs = tiny_inputs(family)
    inputs["hidden_states"].requires_grad_()
    output = adapter(**inputs)
    assert output.shape == inputs["hidden_states"].shape
    output.square().mean().backward()
    assert inputs["hidden_states"].grad.norm() > 0
    keys = adapter.model.state_dict()
    if family == "matrix_game2":
        assert "blocks.0.action_model.proj_mouse.weight" in keys
        assert adapter.model.blocks[0].action_model.proj_mouse.weight.grad.norm() > 0
        assert adapter.model.blocks[0].action_model.proj_keyboard.weight.grad.norm() > 0
    else:
        assert "patch_embedding_wancamctrl.weight" in keys
        assert adapter.model.blocks[0].cam_injector_layer2.weight.grad.norm() > 0
    destination=tmp_path/"saved"
    adapter.save_pretrained(destination)
    loaded=load_native_training_model(destination,dtype=torch.float32,allow_native_code=False)
    torch.testing.assert_close(loaded(**inputs),output)
    assert loaded.model.state_dict().keys()==keys.keys()


def test_matrix_foundation_and_batch_independence(tmp_path):
    model=load_builtin_world_model(tmp_path,tiny_descriptor("matrix_game2","foundation"),weights=False)
    with torch.no_grad():
        model.model.head.head.weight.normal_(std=.03)
    inputs=tiny_inputs("matrix_game2")
    inputs["timestep"]=torch.ones(2)*500
    out=model(**inputs)
    single=torch.cat([model(**{k:v[i:i+1] for k,v in inputs.items()}) for i in range(2)])
    torch.testing.assert_close(out,single,atol=2e-6,rtol=1e-5)
    out.square().mean().backward()


def test_matrix_changes_shape_and_rejects_latent_rate_actions(tmp_path):
    model=load_builtin_world_model(tmp_path,tiny_descriptor("matrix_game2"),weights=False)
    for frames in (3,2,32):
        data=tiny_inputs("matrix_game2",batch=1,frames=frames)
        assert model(**data).shape==data["hidden_states"].shape
    data["keyboard_actions"]=torch.randn(1,32,4)
    with pytest.raises(ValueError,match="keyboard_actions"):
        model(**data)


def test_camera_ray_origin_direction_protocol():
    pose=torch.eye(4).expand(1,3,4,4).clone()
    pose[0,:,0,3]=torch.tensor([0.,2.,3.])
    intrinsics=torch.tensor([8.,8.,4.,4.]).expand(1,3,4)
    feat=lingbot_camera_features(pose,intrinsics,1,1)
    rays=feat.reshape(1,6,8,8,3,1,1).permute(0,4,5,2,6,3,1).reshape(1,3,8,8,6)
    torch.testing.assert_close(rays[0,:,0,0,0],torch.tensor([0.,1.,.5]))
    torch.testing.assert_close(rays[...,3:].norm(dim=-1),torch.ones(1,3,8,8))
    assert torch.equal(rays[0,0,...,:3],torch.zeros(8,8,3))


def test_strict_author_checkpoint_keys_and_explicit_prefix(tmp_path):
    from safetensors.torch import save_file
    descriptor=tiny_descriptor("matrix_game2")
    model=load_builtin_world_model(tmp_path,descriptor,weights=False)
    state={"model."+key:value.clone().contiguous() for key,value in model.model.state_dict().items()}
    save_file(state,tmp_path/"weights.safetensors")
    descriptor.update(weights="weights.safetensors",weights_prefix="model.")
    loaded=load_builtin_world_model(tmp_path,descriptor)
    assert loaded.model.state_dict().keys()==model.model.state_dict().keys()
    state.pop(next(iter(state)))
    save_file(state,tmp_path/"weights.safetensors")
    with pytest.raises(RuntimeError,match="Missing key"):
        load_builtin_world_model(tmp_path,descriptor)


def test_native_unknown_architecture_or_extra_config_rejected(tmp_path):
    descriptor=tiny_descriptor("matrix_game2")
    descriptor["model_config"]["not_a_real_parameter"]=3
    with pytest.raises(ValueError,match="Unknown native"):
        load_builtin_world_model(tmp_path,descriptor,weights=False)


def test_official_matrix_descriptor_uses_only_real_constructor_fields():
    import inspect
    from pathlib import Path
    from training.native_adapters.matrix_game2.causal_model import CausalWanModel
    path=Path(__file__).resolve().parents[2]/'configs/native_models/matrix_game2_universal.example.json'
    config=json.loads(path.read_text())['model_config']
    assert not set(config)-set(inspect.signature(CausalWanModel.__init__).parameters)


def test_matrix_causal_tail_mask_excludes_padded_keys():
    from training.native_adapters.matrix_game2.causal_model import CausalWanModel
    mask=CausalWanModel._prepare_blockwise_causal_attn_mask("cpu",3,4,num_frame_per_block=2)
    fn=mask.mask_mod
    assert fn(torch.tensor(0),torch.tensor(0),torch.tensor(10),torch.tensor(11))
    assert not fn(torch.tensor(0),torch.tensor(0),torch.tensor(10),torch.tensor(12))


@pytest.mark.parametrize("family",["matrix_game2","lingbot_cam"])
@pytest.mark.parametrize("meanflow",[False,True])
def test_native_student_export_and_sampling_roundtrip(tmp_path,family,meanflow):
    from training.model_adapter import load_diffusers_training_model
    from training.student_export import export_student, sample_student_latents
    from training.utils.meanflow import MeanFlowTimeAdapter
    base=tmp_path/'base'
    original=load_builtin_world_model(tmp_path,tiny_descriptor(family),weights=False)
    with torch.no_grad():
        original.model.head.head.weight.normal_(std=.03)
    original.save_pretrained(base)
    model=MeanFlowTimeAdapter(original,latent_channels=2,hidden_dim=8) if meanflow else original
    data=tiny_inputs(family,batch=1)
    time=torch.tensor([700.])
    data['timestep']=time
    if meanflow:
        data['timestep_r']=torch.tensor([200.])
    expected=model(**data)
    args=dict(distill_method='mean_flow_distill' if meanflow else 'step_distill',
              denoising_step_list=[1000,500],meanflow_latent_channels=2,meanflow_hidden_dim=8)
    checkpoint=tmp_path/'trainer_state.pt'
    torch.save({'student_model':model.state_dict(),'step':1,'args':args},checkpoint)
    dest=tmp_path/'export'
    export_student(str(base),str(checkpoint),str(dest),num_steps=2)
    restored=load_diffusers_training_model(dest)
    torch.testing.assert_close(restored(**data),expected)
    conditions={key:value for key,value in data.items() if key not in {'hidden_states','timestep','timestep_r'}}
    noise=torch.randn_like(data['hidden_states'])
    a=sample_student_latents(model,noise.clone(),num_steps=2,conditions=conditions)
    b=sample_student_latents(restored,noise.clone(),num_steps=2,conditions=conditions)
    torch.testing.assert_close(a,b)


@pytest.mark.parametrize('family',['matrix_game2','lingbot_cam'])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_native_low_precision_storage_and_sampling_dtype(tmp_path,family,dtype):
    from training.student_export import sample_student_latents
    model=load_builtin_world_model(tmp_path,tiny_descriptor(family),weights=False,dtype=dtype)
    with torch.no_grad():
        model.model.head.head.weight.normal_(std=.03)
    inputs={key:value.to(dtype=dtype) for key,value in tiny_inputs(family,batch=1).items()}
    inputs['hidden_states'].requires_grad_()
    output=model(**inputs)
    assert output.dtype==dtype and torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert inputs['hidden_states'].grad is not None and torch.isfinite(inputs['hidden_states'].grad).all()
    conditions={key:value for key,value in inputs.items() if key not in {'hidden_states','timestep'}}
    sampled=sample_student_latents(model,inputs['hidden_states'].detach(),num_steps=2,conditions=conditions)
    assert sampled.dtype==dtype and torch.isfinite(sampled).all()
