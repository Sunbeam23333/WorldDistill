import pytest
import torch

diffusers=pytest.importorskip("diffusers")
from training.native_adapters.diffusers_controls import (
    WanVACETrainingAdapter, WanAnimateTrainingAdapter, SkyReelsV2TrainingAdapter)


def architecture(name):
    cls=getattr(diffusers,name,None)
    if cls is None:
        pytest.skip(f"Installed Diffusers does not include {name}")
    return cls


def basic():
    return dict(num_attention_heads=2,attention_head_dim=12,in_channels=2,out_channels=2,
                text_dim=8,freq_dim=8,ffn_dim=32,num_layers=1)


def test_vace_actual_architecture_batched_scale_backward_and_reload(tmp_path):
    cls=architecture("WanVACETransformer3DModel")
    model=cls(**basic(),vace_layers=[0],vace_in_channels=6)
    wrapper=WanVACETrainingAdapter(model)
    x=torch.randn(2,2,2,4,4,requires_grad=True)
    context=torch.randn(2,6,2,4,4,requires_grad=True)
    args=dict(hidden_states=x,timestep=torch.ones(2)*400,encoder_hidden_states=torch.randn(2,3,8),
              vace_context=context,vace_scale=torch.tensor([[.4],[.8]]))
    output=wrapper(**args)
    assert output.shape==x.shape
    output.square().mean().backward()
    assert x.grad.norm()>0
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.vace_blocks.parameters())
    model.save_pretrained(tmp_path)
    loaded=WanVACETrainingAdapter(cls.from_pretrained(tmp_path))
    torch.testing.assert_close(loaded(**args),output)
    with pytest.raises(ValueError,match="vace_context"):
        wrapper(x,args['timestep'],encoder_hidden_states=args['encoder_hidden_states'])


def test_animate_real_motion_pose_face_backward_and_reload(tmp_path):
    cls=architecture("WanAnimateTransformer3DModel")
    config=basic()
    config.update(in_channels=8,latent_channels=2,image_dim=None,motion_encoder_size=8,
        motion_encoder_channel_sizes={"8":8,"4":8},motion_style_dim=8,motion_dim=4,
        motion_encoder_dim=8,face_encoder_hidden_dim=8,face_encoder_num_heads=2,
        inject_face_latents_blocks=1)
    model=cls(**config)
    wrapper=WanAnimateTrainingAdapter(model)
    x=torch.randn(1,2,3,4,4,requires_grad=True)
    pose=torch.randn(1,2,2,4,4,requires_grad=True)
    face=torch.rand(1,3,5,8,8,requires_grad=True)
    args=dict(hidden_states=x,timestep=torch.ones(1)*400,encoder_hidden_states=torch.randn(1,3,8),
              image_cond=torch.randn(1,6,3,4,4),pose_latents=pose,face_pixels=face)
    output=wrapper(**args)
    output.square().mean().backward()
    assert x.grad.norm()>0 and pose.grad.norm()>0
    assert face.grad is not None and torch.isfinite(face.grad).all()
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.motion_encoder.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.face_adapter.parameters())
    model.save_pretrained(tmp_path)
    loaded=WanAnimateTrainingAdapter(cls.from_pretrained(tmp_path))
    torch.testing.assert_close(loaded(**args),output)
    with pytest.raises(ValueError,match="face_pixels"):
        wrapper(**{**args,"face_pixels":torch.rand(1,3,9,8,8)})


@pytest.mark.parametrize("inject_fps",[False,True])
def test_skyreels_actual_per_frame_time_and_batch_independence(tmp_path,inject_fps):
    cls=architecture("SkyReelsV2Transformer3DModel")
    model=cls(**basic(),inject_sample_info=inject_fps,num_frame_per_block=2)
    wrapper=SkyReelsV2TrainingAdapter(model)
    x=torch.randn(2,2,4,4,4,requires_grad=True)
    args=dict(hidden_states=x,timestep=torch.tensor([[800.,700.,300.,200.],[900.,800.,400.,200.]]),
              encoder_hidden_states=torch.randn(2,3,8))
    if inject_fps:
        args['fps_condition']=torch.tensor([16.,24.])
    output=wrapper(**args)
    torch.testing.assert_close(output,torch.cat([wrapper(**{k:v[i:i+1] for k,v in args.items()}) for i in range(2)]))
    output.square().mean().backward()
    assert x.grad.norm()>0
    if inject_fps:
        assert model.fps_embedding.weight.grad.norm()>0
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(SkyReelsV2TrainingAdapter(cls.from_pretrained(tmp_path))(**args),output)
    with pytest.raises(ValueError,match="divide latent frames"):
        wrapper(**{**args,"hidden_states":x[:,:,:3],"timestep":args['timestep'][:,:3]})
