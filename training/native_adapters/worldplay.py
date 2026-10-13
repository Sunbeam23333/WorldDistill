"""Concrete HY-WorldPlay author-training bridge (no redistributed Tencent code).

Install the official, licensed HY-WorldPlay training repository and dependencies
separately. This binds its AR action/camera *training* transformer, not the
hyvideo inference runner, and strictly preserves that class's checkpoint keys.
"""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import torch

from .world_models import _ConcreteWorldAdapter, _condition, _read_native_weights, _times, _video

WORLDPLAY_REVISION = "1588e1336e842b03b0a7860c654ebd7c46bb065e"
TRAINING_MODULE = "trainer.models.hyvideo.models.transformers.ar_action_hunyuanvideo_1_5_transformer"
TRAINING_CLASS = "ARHunyuanVideo_1_5_DiffusionTransformer"


def _official_training_class(descriptor):
    source = descriptor.get("source_root") or os.environ.get("WORLDDISTILL_HY_WORLDPLAY_SOURCE")
    if not source:
        raise ValueError("HY-WorldPlay requires its separately licensed official source checkout; set "
                         "WORLDDISTILL_HY_WORLDPLAY_SOURCE (or descriptor source_root)")
    root = Path(source).expanduser().resolve()
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("HY-WorldPlay source_root must be the official pinned Git checkout") from error
    if revision != WORLDPLAY_REVISION or descriptor.get("source_revision",revision) != revision:
        raise ValueError(f"HY-WorldPlay requires reviewed official revision {WORLDPLAY_REVISION}")
    previous = sys.modules.get("trainer")
    if previous is not None and not Path(previous.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Another package named trainer is imported; start with HY-WorldPlay on PYTHONPATH")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(TRAINING_MODULE)
    except ImportError as error:
        raise RuntimeError("Install the official HY-WorldPlay training requirements in this environment; "
                           f"its training module could not import: {error}") from error
    finally:
        sys.path.remove(str(root))
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("HY-WorldPlay training module did not resolve inside the pinned source checkout")
    return getattr(module, TRAINING_CLASS)


class HYWorldPlayTrainingAdapter(_ConcreteWorldAdapter):
    """Flow velocity, discrete latent-rate actions and normalized PRoPE cameras.

The author module packs text and frame modulation with batch-one semantics.
We execute independent examples separately, retaining their autograd graphs.
Its training runtime must be initialized by the launcher, with official SP=1;
WorldDistill DP/FSDP may still wrap this adapter externally.
"""
    def __init__(self, model, descriptor):
        super().__init__(model, descriptor)
        self.supports_meanflow = bool(descriptor["model_config"].get("use_meanflow", False))

    @property
    def config(self):
        return {**dict(self.model.config), "in_channels":self.model.out_channels,
                "out_channels":self.model.out_channels}

    def enable_gradient_checkpointing(self):
        # The pinned author forward does not use its gradient_checkpointing
        # flag; do not report that an inert flag enabled activation savings.
        raise ValueError("Pinned HY-WorldPlay training forward requires the author's external activation-"
                         "checkpoint wrapper; disable WorldDistill gradient_checkpointing for this bridge")

    def forward(self, hidden_states, timestep, image_cond=None, encoder_hidden_states=None,
                encoder_attention_mask=None, image_embeds=None, worldplay_actions=None,
                camera_view_matrices=None, camera_intrinsics_normalized=None,
                byt5_text_states=None, byt5_text_mask=None, guidance=None, timestep_r=None, **conditions):
        unknown = [key for key,value in conditions.items() if value is not None]
        if unknown:
            raise ValueError(f"HY-WorldPlay does not accept conditions {sorted(unknown)}")
        _video(hidden_states, self.model.patch_size)
        b,c,t,h,w = hidden_states.shape
        if self.model.patch_size[0] != 1:
            raise ValueError("HY-WorldPlay frame-wise action conditioning requires temporal patch_size=1")
        expected = self.model.img_in.proj.in_channels - c
        if expected:
            _condition(image_cond,(b,expected,t,h,w),"image_cond")
            inputs=torch.cat((hidden_states,image_cond.to(hidden_states)),1)
        elif image_cond is not None:
            raise ValueError("This HY-WorldPlay configuration has no concatenated condition channels")
        else:
            inputs=hidden_states
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != b:
            raise ValueError("HY-WorldPlay requires B,L,D encoder_hidden_states")
        _condition(encoder_attention_mask,encoder_hidden_states.shape[:2],"encoder_attention_mask")
        if not ((encoder_attention_mask==0)|(encoder_attention_mask==1)).all() or not encoder_attention_mask.bool().any(dim=1).all():
            raise ValueError("HY-WorldPlay text attention mask must be binary with a valid token per sample")
        _condition(worldplay_actions,(b,t),"worldplay_actions")
        if not torch.equal(worldplay_actions,worldplay_actions.round()) or (worldplay_actions<0).any():
            raise ValueError("worldplay_actions must be nonnegative discrete author action codes, not vectors")
        _condition(camera_view_matrices,(b,t,4,4),"camera_view_matrices")
        _condition(camera_intrinsics_normalized,(b,t,3,3),"camera_intrinsics_normalized")
        if (camera_intrinsics_normalized[...,(0,1),(0,1)]<=0).any():
            raise ValueError("HY-WorldPlay normalized camera focal lengths must be positive")
        times=_times(timestep,hidden_states,per_frame=True)
        if timestep_r is not None:
            if not self.supports_meanflow:
                raise ValueError("This HY-WorldPlay checkpoint has no MeanFlow second-time embedding")
            time_r=_times(timestep_r,hidden_states,per_frame=True)
        else:
            time_r=None
        if self.model.attn_mode in {"torch_causal","flex_causal"} and ((h//self.model.patch_size[1])*(w//self.model.patch_size[2]) !=1560 or t%4):
            raise ValueError("Pinned HY-WorldPlay causal attention uses 1560 spatial tokens and four-latent chunks; "
                             "use the checkpoint's 480x832 decoded resolution and aligned clip length")
        if self.model.attn_mode not in {"torch","torch_causal","flex_causal","flash2","flash3"}:
            raise ValueError("HY-WorldPlay training requires a differentiable author attention backend")
        from trainer.distributed.parallel_state import model_parallel_is_initialized, get_sp_world_size
        if not model_parallel_is_initialized() or get_sp_world_size()!=1:
            raise RuntimeError("Initialize the official HY-WorldPlay training distributed runtime with SP=1 "
                               "before using this bridge; its internal SP is not WorldDistill's CP protocol")
        if self.model.glyph_byT5_v2 and (byt5_text_states is None or byt5_text_mask is None):
            raise ValueError("This HY-WorldPlay checkpoint requires byt5_text_states and byt5_text_mask")
        if self.model.vision_in is not None and image_embeds is None:
            raise ValueError("This HY-WorldPlay checkpoint requires vision image_embeds")
        results=[]
        for i in range(b):
            take=lambda value: None if value is None else value[i:i+1]
            result=self.model(hidden_states=inputs[i:i+1],timestep=times[i],
                timestep_txt=times.new_zeros(1),text_states=encoder_hidden_states[i:i+1],text_states_2=None,
                encoder_attention_mask=encoder_attention_mask[i:i+1],vision_states=take(image_embeds),
                action=worldplay_actions[i],viewmats=camera_view_matrices[i:i+1],
                Ks=camera_intrinsics_normalized[i:i+1],timestep_r=None if time_r is None else time_r[i],
                guidance=take(guidance),mask_type="i2v",return_dict=False,
                extra_kwargs={"byt5_text_states":take(byt5_text_states),"byt5_text_mask":take(byt5_text_mask)})
            results.append(result[0])
        output=torch.cat(results)
        if output.shape != hidden_states.shape:
            raise ValueError("HY-WorldPlay prediction does not match noisy video latents")
        return output


def load_hy_worldplay(path, descriptor, *, dtype=torch.float32, weights=True, allow_native_code=False):
    if not allow_native_code:
        raise ValueError("HY-WorldPlay imports separately licensed installed Python; set allow_native_model_code explicitly")
    if descriptor.get("noise_process", "flow") != "flow" or descriptor.get("prediction_type", "velocity") != "velocity":
        raise ValueError("HY-WorldPlay requires flow velocity metadata")
    cls=_official_training_class(descriptor)
    config=dict(descriptor.get("model_config",{}))
    unknown=set(config)-set(inspect.signature(cls.__init__).parameters)
    if unknown:
        raise ValueError(f"Unknown official HY-WorldPlay constructor fields {sorted(unknown)}")
    model=cls(**config)
    model.add_discrete_action_parameters()
    if weights:
        model.load_state_dict(_read_native_weights(Path(path).resolve(),descriptor),strict=True)
    metadata={**descriptor,"source_revision":WORLDPLAY_REVISION,"model_config":config,
              "noise_process":"flow","prediction_type":"velocity"}
    return HYWorldPlayTrainingAdapter(model.to(dtype=dtype),metadata)
