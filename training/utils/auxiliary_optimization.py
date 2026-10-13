"""AMP and microbatch accumulation for alternating discriminator/score updates."""
from contextlib import nullcontext
import torch
from training.utils.replicated_gradients import synchronize_replicated_gradients


class AuxiliaryTrainingMixin:
    """Finish auxiliary optimizer turns before accumulating student gradients.

    Only encoded microbatch tensors are buffered, not activation graphs. Each
    auxiliary optimizer sees the mean over the same accumulation window, and
    DMD's K score updates remain K optimizer steps (not K * GAS updates).

    FSDP/ZeRO apply only to the student. Auxiliary models, gradients, and Adam
    states remain independent full replicas on each worker. Their memory cost
    is explicitly reported; student sharding never implies auxiliary sharding.
    """
    auxiliary_parallel_mode = "replicated"

    def _validate_auxiliary_ownership(self, names):
        """Check device/dtype/optimizer isolation before sharding the student."""
        occupied = {id(parameter) for model in (self.student_model, self.teacher_model)
                    for parameter in model.parameters()}
        target_device = self.device
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        for model_name, optimizer_name in names:
            model, optimizer = getattr(self, model_name), getattr(self, optimizer_name)
            if model is None:
                continue
            model_parameters = list(model.parameters())
            parameters = {id(parameter) for parameter in model_parameters}
            owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            if parameters & occupied or owned != parameters:
                raise ValueError(f"{model_name} needs a disjoint optimizer covering its full replica")
            if any(parameter.dtype != torch.float32 or parameter.device != target_device
                   for parameter in model_parameters):
                raise ValueError(f"{model_name} parameters must be full FP32 replicas on {self.device}")
            occupied.update(parameters)
        self._auxiliary_names = tuple(names)

    def auxiliary_placement_report(self):
        """Live per-worker tensor storage, excluding activation and allocator peaks."""
        report = {"parallel_mode": "replicated", "student_parallel_mode": self.parallel_mode,
                  "replicas": {}}
        for model_name, optimizer_name in self._auxiliary_names:
            model, optimizer = getattr(self, model_name), getattr(self, optimizer_name)
            if model is None:
                continue
            parameters = list(model.parameters())
            report["replicas"][model_name] = {
                "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
                "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
                "gradient_bytes": sum(p.grad.numel() * p.grad.element_size() for p in parameters if p.grad is not None),
                "optimizer_tensor_bytes": sum(value.numel() * value.element_size()
                    for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)),
            }
        ema = getattr(self, "fake_score_ema", None)
        ema_model = getattr(self, "fake_score_ema_model", None)
        report["ema_shadow_bytes"] = sum(value.numel() * value.element_size() for value in ema.shadow.values()) if ema is not None else 0
        report["ema_target_bytes"] = sum(value.numel() * value.element_size()
            for value in list(ema_model.parameters()) + list(ema_model.buffers())) if ema_model is not None else 0
        return report

    def _training_microbatches(self, batch):
        batches = list(super()._training_microbatches(batch))
        self._run_auxiliary_updates(batches)
        self._auxiliary_prepared = True
        try:
            yield from batches
        finally:
            self._auxiliary_prepared = False

    def _auxiliary_autocast(self):
        return torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp)

    def _new_auxiliary_scaler(self):
        return torch.amp.GradScaler("cuda", enabled=self.args.mixed_precision == "fp16" and self.device.type == "cuda")


def auxiliary_sync_context(model, is_last):
    return model.no_sync() if not is_last and hasattr(model, "no_sync") else nullcontext()


def finish_auxiliary_step(model, optimizer, scaler, max_grad_norm, *, manual_sync=False):
    """Unscale exactly once, then clip/step/update; report whether weights moved."""
    if manual_sync:
        synchronize_replicated_gradients(model)
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(torch.stack([torch.isfinite(parameter.grad).all() for parameter in parameters]).all().item()) if parameters else True
    if finite and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    previous_scale = scaler.get_scale()
    if finite or scaler.is_enabled():
        # Enabled GradScaler independently checks inf and skips on overflow.
        scaler.step(optimizer)
    scaler.update()
    if not finite:
        optimizer.zero_grad(set_to_none=True)
    return finite and scaler.get_scale() >= previous_scale


def restore_auxiliary_scaler(scaler, state, key):
    if key not in state:
        if scaler.is_enabled():
            raise ValueError(f"FP16 resume requires the saved auxiliary {key}")
        return  # older FP32/BF16 checkpoints did not need scaling state
    scaler.load_state_dict(state[key])
