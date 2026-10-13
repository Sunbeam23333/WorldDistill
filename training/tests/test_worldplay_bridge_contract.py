"""Bridge contracts only; these are NOT author-runtime numerical tests.

The optional licensed training runtime is deliberately not replaced by a fake
architecture claiming native HY-WorldPlay forward/backward coverage.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.native_adapters import worldplay


class ContractFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter=nn.Parameter(torch.ones(()))
        self.config={}
        self.out_channels=2
        self.patch_size=(1,2,2)
        self.img_in=SimpleNamespace(proj=SimpleNamespace(in_channels=5))
        self.attn_mode='torch_causal'
    def forward(self,**kwargs):
        raise AssertionError('A contract test must not substitute fake author numerical execution')


def inputs():
    return dict(hidden_states=torch.zeros(1,2,4,4,4),image_cond=torch.zeros(1,3,4,4,4),
        timestep=torch.ones(1),encoder_hidden_states=torch.zeros(1,3,8),encoder_attention_mask=torch.ones(1,3),
        worldplay_actions=torch.zeros(1,4),camera_view_matrices=torch.eye(4).expand(1,4,4,4),
        camera_intrinsics_normalized=torch.eye(3).expand(1,4,3,3))


def test_worldplay_opt_in_before_external_import(tmp_path,monkeypatch):
    monkeypatch.setattr(worldplay,'_official_training_class',lambda descriptor:pytest.fail('must not import'))
    with pytest.raises(ValueError,match='allow_native_model_code'):
        worldplay.load_hy_worldplay(tmp_path,{'model_config':{}},weights=False)


def test_worldplay_only_the_pinned_official_training_class(tmp_path,monkeypatch):
    seen=[]
    sentinel=object()
    monkeypatch.setattr(worldplay.subprocess,'check_output',lambda *a,**kw:worldplay.WORLDPLAY_REVISION+'\n')
    module=SimpleNamespace(__file__=str(tmp_path/'trainer/model.py'))
    setattr(module,worldplay.TRAINING_CLASS,sentinel)
    monkeypatch.setattr(worldplay.importlib,'import_module',lambda name:(seen.append(name),module)[1])
    assert worldplay._official_training_class({'source_root':str(tmp_path)}) is sentinel
    assert seen==['trainer.models.hyvideo.models.transformers.ar_action_hunyuanvideo_1_5_transformer']
    with pytest.raises(ValueError,match='reviewed official revision'):
        worldplay._official_training_class({'source_root':str(tmp_path),'source_revision':'wrong'})


@pytest.mark.parametrize('field,value,message',[
    ('worldplay_actions',torch.zeros(1,4,2),'worldplay_actions'),
    ('worldplay_actions',torch.full((1,4),.5),'discrete'),
    ('camera_view_matrices',torch.zeros(1,4,3,3),'camera_view_matrices'),
    ('encoder_attention_mask',torch.zeros(1,3),'valid token'),
    ('camera_intrinsics_normalized',torch.zeros(1,4,3,3),'focal'),
])
def test_worldplay_rejects_incorrect_conditions_before_runtime(field,value,message):
    model=worldplay.HYWorldPlayTrainingAdapter(ContractFixture(),{'model_config':{}})
    data=inputs();data[field]=value
    with pytest.raises(ValueError,match=message):
        model(**data)


def test_worldplay_causal_resolution_and_inert_checkpoint_flag_are_explicit():
    model=worldplay.HYWorldPlayTrainingAdapter(ContractFixture(),{'model_config':{}})
    with pytest.raises(ValueError,match='1560 spatial tokens'):
        model(**inputs())
    with pytest.raises(ValueError,match='external activation'):
        model.enable_gradient_checkpointing()
