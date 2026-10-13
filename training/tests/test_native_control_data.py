import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import default_collate

from training.data.video_dataset import VideoDataset, CachedLatentDataset
from training.native_adapters.temporal_conditions import slice_native_temporal_conditions
from training.utils.batch_encoder import DiffusersRawBatchEncoder


def test_raw_pixel_controls_not_downsampled_and_intrinsics_resize(tmp_path,monkeypatch):
    torch.save(torch.arange(18).reshape(9,2).float(),tmp_path/'mouse.pt')
    torch.save(torch.arange(36).reshape(9,4).float(),tmp_path/'keys.pt')
    torch.save(torch.tensor([100.,80.,40.,30.]).expand(11,4),tmp_path/'k.pt')
    row={'path':'sample.mp4','num_frames':9,'text':'a','mouse_action_path':'mouse.pt',
         'keyboard_action_path':'keys.pt','matrix_control_alignment':'sampled_video',
         'camera_intrinsics_path':'k.pt','camera_intrinsics_resolution':[60,80]}
    manifest=tmp_path/'raw.json'
    manifest.write_text(json.dumps([row]))
    monkeypatch.setattr(VideoDataset,'_load_video',lambda *a,**kw:(torch.ones(3,9,6,8),torch.arange(9),11))
    ds=VideoDataset(str(manifest),video_dir=str(tmp_path))
    data=default_collate([ds[0]])
    assert data['mouse_actions'].shape==(1,9,2)
    torch.testing.assert_close(data['camera_intrinsics'][0,0],torch.tensor([10.,8.,4.,3.]))
    encoder=DiffusersRawBatchEncoder.__new__(DiffusersRawBatchEncoder)
    encoder.device=torch.device('cpu')
    encoder.temporal_compression_ratio=4
    encoder._encode_pixels=lambda values: values[:,:2,::4]
    encoder._encode_text=lambda text:torch.ones(1,3,4)
    encoded=encoder.encode_batch(data)
    assert encoded['latents'].shape[2]==3
    assert encoded['mouse_actions'].shape==(1,9,2)
    assert encoded['keyboard_actions'].shape==(1,9,4)
    assert encoded['camera_intrinsics'].shape==(1,3,4)
    ds.data[0].pop('matrix_control_alignment')
    with pytest.raises(ValueError,match='matrix_control_alignment'):
        ds[0]


def test_cached_all_native_conditions_roundtrip(tmp_path):
    latent=torch.randn(2,3,4,4)
    conditions={'keyboard_actions':torch.randn(9,4),'mouse_actions':torch.randn(9,2),
                'camera_features':torch.randn(384,3,4,4),'vace_context':torch.randn(6,3,4,4),
                'vace_scale':torch.ones(2),'pose_latents':torch.randn(2,2,4,4),
                'face_pixels':torch.rand(3,5,8,8),'fps_condition':torch.tensor(16.)}
    torch.save(latent,tmp_path/'latent.pt');torch.save(conditions,tmp_path/'conditions.pt')
    manifest=tmp_path/'cache.json'
    manifest.write_text(json.dumps([{'latent_path':'latent.pt','conditioning_path':'conditions.pt'}]))
    ds=CachedLatentDataset(str(manifest),cache_dir=str(tmp_path))
    result=default_collate([ds[0]])
    for key,value in conditions.items():
        torch.testing.assert_close(result[key][0],value)
    signature=ds.get_metadata(0)['collation_signature']
    assert signature['keyboard_actions']['shape']==[9,4]


def test_explicit_native_temporal_selection():
    batch={'vace_context':torch.randn(2,6,5,2,2),'camera_features':torch.randn(2,384,5,2,2),
           'camera_intrinsics':torch.randn(2,5,4),'fps_condition':torch.tensor([16.,24.])}
    idx=torch.tensor([0,3,4])
    out=slice_native_temporal_conditions(batch,idx,5)
    torch.testing.assert_close(out['camera_features'],batch['camera_features'][:,:,idx])
    torch.testing.assert_close(out['camera_intrinsics'],batch['camera_intrinsics'][:,idx])
    assert out['fps_condition'] is batch['fps_condition']
    for key in ('keyboard_actions','mouse_actions','pose_latents','face_pixels'):
        with pytest.raises(ValueError,match=key):
            slice_native_temporal_conditions({key:torch.ones(1)},idx,5)
        assert slice_native_temporal_conditions({key:torch.ones(1)},torch.arange(5),5)[key].numel()==1
    with pytest.raises(ValueError,match='precomputed camera_features'):
        slice_native_temporal_conditions({**batch,'camera_poses':torch.randn(2,5,4,4)},idx,5)
    four={'camera_poses':torch.randn(2,4,4,4),'camera_intrinsics':torch.randn(2,4,4)}
    for key,value in four.items():
        torch.testing.assert_close(slice_native_temporal_conditions(four,torch.arange(4),4)[key],value)
