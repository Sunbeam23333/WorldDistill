"""Joint audiovisual flow distillation with separate, balanced modality losses."""
from __future__ import annotations

import math
import torch

from training.trainers.base_distill_trainer import BaseDistillTrainer


def audiovisual_prediction(output, *, video_shape=None, audio_shape=None):
    if not isinstance(output, dict) or set(output) != {"sample", "audio_sample"}:
        raise TypeError("Joint AV training requires explicit sample and audio_sample predictions")
    for key, shape in (("sample", video_shape), ("audio_sample", audio_shape)):
        value = output[key]
        if not torch.is_tensor(value) or (shape is not None and value.shape != shape):
            raise ValueError(f"Joint AV {key} has an invalid prediction layout")
    return output


class JointAVDistillTrainer(BaseDistillTrainer):
    """Match both teacher velocities at synchronized audiovisual noise levels.

    Each modality is mean-reduced independently before weighting, so the much
    larger video tensor does not implicitly suppress audio. Native LTX-2 joint
    attention and both output heads are trained; audio is never a side feature.
    Inputs are normalized video latents and packed normalized audio latents.
    """

    def __init__(self, **kwargs):
        for role in ("teacher_model", "student_model"):
            if not getattr(kwargs[role], "supports_joint_av", False):
                raise ValueError(f"{role} requires an explicit joint audiovisual adapter")
            if getattr(kwargs[role], "training_noise_process", "flow") != "flow":
                raise ValueError("Joint AV distillation requires flow-matching denoisers")
        if getattr(kwargs["args"], "prediction_type", "velocity") != "velocity":
            raise ValueError("Joint AV distillation requires velocity prediction")
        super().__init__(**kwargs)
        self._require_velocity_prediction("Joint audiovisual distillation")
        self.video_weight = float(getattr(self.args, "joint_av_video_weight", 1.0))
        self.audio_weight = float(getattr(self.args, "joint_av_audio_weight", 1.0))
        if not all(math.isfinite(w) and w > 0 for w in (self.video_weight, self.audio_weight)):
            raise ValueError("Both joint audiovisual loss weights must be finite and positive")

    def prepare_teacher_input(self, batch, noisy_latents, timesteps):
        kwargs = self._augment_model_input({"hidden_states": noisy_latents, "timestep": timesteps}, batch)
        for key in ("audio_encoder_hidden_states", "audio_encoder_attention_mask", "fps",
                    "video_coords", "audio_coords", "use_cross_timestep"):
            if key in batch:
                kwargs[key] = batch[key]
        # Metadata collates to a B-vector; LTX's RoPE takes one rate for the batch.
        if torch.is_tensor(kwargs.get("fps")):
            fps = kwargs["fps"]
            if not torch.all(fps == fps.flatten()[0]):
                raise ValueError("Joint AV batch members must share their frame rate")
            kwargs["fps"] = float(fps.flatten()[0])
        return kwargs

    def _forward_and_loss(self, batch):
        video, audio = batch["latents"], batch.get("audio_latents")
        if video.ndim != 5 or not torch.is_tensor(audio) or audio.ndim != 3 or audio.shape[0] != video.shape[0]:
            raise ValueError("Joint AV batches require video B,C,T,H,W and packed audio B,L,D latents")
        t = self._sample_timesteps(video.shape[0])
        sigma = t / self.num_train_timesteps
        video_sigma = sigma.reshape(-1, 1, 1, 1, 1)
        audio_sigma = sigma.reshape(-1, 1, 1)
        noisy_video = (1 - video_sigma) * video + video_sigma * torch.randn_like(video)
        noisy_audio = (1 - audio_sigma) * audio + audio_sigma * torch.randn_like(audio)
        inputs = self.prepare_teacher_input(batch, noisy_video, t)
        inputs.update(audio_hidden_states=noisy_audio, audio_timestep=t)
        # Runtime recursively records/caches both output tensors and preserves
        # its teacher stream, teacher offload, and payload lifetime guarantees.
        if self.runtime is not None:
            handle = self.runtime.launch_teacher(inputs, batch, self.global_step,
                cache_namespace="joint_av_teacher", cache_extra={"timesteps": t})
            student = self.student_model(**inputs)
            teacher = self.runtime.wait_teacher(handle)
        else:
            with torch.no_grad():
                teacher = self.teacher_model(**inputs)
            student = self.student_model(**inputs)
        teacher = audiovisual_prediction(teacher, video_shape=video.shape, audio_shape=audio.shape)
        student = audiovisual_prediction(student, video_shape=video.shape, audio_shape=audio.shape)
        return self.compute_distill_loss(teacher, student, batch, t)

    def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
        teacher = audiovisual_prediction(teacher_output)
        student = audiovisual_prediction(student_output)
        video = self.compute_supervision_loss(student["sample"], teacher["sample"].detach(),
                                              mask=batch.get("loss_mask"))
        audio = self.compute_supervision_loss(student["audio_sample"], teacher["audio_sample"].detach(),
                                              mask=batch.get("audio_loss_mask"))
        return self.video_weight * video + self.audio_weight * audio

    def _forward_and_loss_eval(self, batch):
        return self._forward_and_loss(batch)
