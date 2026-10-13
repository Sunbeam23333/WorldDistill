"""Export trained denoisers and reload them in the original Diffusers pipeline.

Bundles intentionally reference (rather than redistribute) the base model's
licensed VAE/text/tokenizer assets. Provenance ties inference to an exact
training checkpoint; no inference-family weight-name guessing is performed.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import os
from pathlib import Path

import torch

from training.model_adapter import (DiffusersTrainingAdapter, NoiseRoutedDenoiser,
                                    load_diffusers_training_model, save_training_component)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if (path / "trainer_state.pt").is_file():
        return path / "trainer_state.pt"
    candidates = [p for p in path.glob("checkpoint-*/trainer_state.pt") if p.parent.name[11:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No consolidated trainer_state.pt under {path}; consolidate ZeRO shards before export")
    return max(candidates, key=lambda p: int(p.parent.name[11:]))


def _resolve_inference_steps(args: dict, state: dict, source: Path, num_steps: int | None) -> tuple[int, dict]:
    """Infer only the step count actually specified by this training method.

    This records a sampling default, not validation that the base pipeline's
    scheduler reproduces the trainer's sampling algorithm.
    """
    if num_steps is not None:
        return num_steps, {"source": "explicit_num_steps"}

    def positive_integer(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def ambiguous(reason):
        raise ValueError(f"Cannot infer student inference steps: {reason}; supply explicit --num_steps")

    method = args.get("distill_method", "unknown")
    if method == "step_distill":
        schedule = args.get("denoising_step_list")
        if not isinstance(schedule, (list, tuple)) or not schedule or not all(positive_integer(step) for step in schedule):
            ambiguous("step_distill requires a recorded non-empty denoising_step_list")
        return len(schedule), {"source": "training_args.denoising_step_list"}
    if method in {"dmd_distill", "stream_distill"}:
        key = "dmd_student_steps" if method == "dmd_distill" else "denoising_steps_per_frame"
        if not positive_integer(args.get(key)):
            ambiguous(f"{method} requires a recorded positive {key}")
        return args[key], {"source": f"training_args.{key}"}
    if method != "progressive_distill":
        ambiguous(f"{method} does not define a unique trained sampling step count")

    # The final configured stage may never have been reached. Only the saved
    # stage state identifies the objective used by this particular checkpoint.
    stages = args.get("progressive_stages")
    stage_steps = args.get("progressive_stage_steps")
    if (not isinstance(stages, (list, tuple)) or len(stages) < 2
            or not all(positive_integer(step) for step in stages)
            or any(teacher != 2 * student for teacher, student in zip(stages, stages[1:]))
            or not positive_integer(stage_steps)):
        ambiguous("progressive_distill requires its recorded halving schedule and stage duration")
    progressive_path = source.parent / "progressive_state.pt"
    if not progressive_path.is_file():
        ambiguous("progressive_state.pt is missing from the checkpoint directory")
    progressive = torch.load(progressive_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(progressive, dict):
        ambiguous("progressive_state.pt does not contain a stage record")
    stage = progressive.get("current_stage")
    if not isinstance(stage, int) or isinstance(stage, bool) or not 0 <= stage < len(stages) - 1:
        ambiguous("progressive checkpoint stage is outside the recorded schedule")
    if (progressive.get("current_teacher_steps") != stages[stage]
            or progressive.get("current_student_steps") != stages[stage + 1]):
        ambiguous("progressive checkpoint step counts disagree with its recorded stage")
    global_step = state.get("step")
    if not positive_integer(global_step) or stage != min(global_step // stage_steps, len(stages) - 2):
        ambiguous("progressive checkpoint stage and completed optimizer steps are inconsistent")
    stage_updates = global_step - stage * stage_steps
    if stage_updates <= 0:
        # on_train_step_end advances the stage *before* saving the boundary
        # checkpoint. Its newly advertised target has not received an update.
        ambiguous("this progressive boundary checkpoint has no updates in its newly selected stage")
    return stages[stage + 1], {
        "source": "progressive_state.current_student_steps",
        "current_stage": stage,
        "stage_optimizer_steps": stage_updates,
        "progressive_state_sha256": file_sha256(progressive_path),
    }


def read_student_manifest(bundle: str | Path) -> dict:
    """Validate the complete exported components before any weights are loaded."""
    root = Path(bundle).resolve()
    manifest = json.loads((root / "worlddistill_export.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Student bundle requires schema version 2; re-export to include configuration integrity records")
    steps = manifest.get("num_inference_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("Student bundle num_inference_steps must be a positive integer")
    components = manifest.get("components")
    allowed = ({"transformer"}, {"unet"}, {"transformer", "transformer_2"})
    if not isinstance(components, list) or not all(isinstance(c, str) for c in components) or len(set(components)) != len(components) or set(components) not in allowed:
        raise ValueError("Invalid student component topology")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Student bundle has no complete artifact integrity record")
    actual = set()
    for component in components:
        directory = root / component
        markers = [name for name in ("config.json", "worlddistill_model.json", "worlddistill_adapter.json")
                   if (directory / name).is_file()]
        if len(markers) != 1 or not any(directory.rglob("*.safetensors")):
            raise ValueError(f"Student component {component} lacks its config or safetensors weights")
        actual.update(str(path.relative_to(root)) for path in directory.rglob("*") if path.is_file())
    if actual != set(artifacts):
        raise ValueError("Student artifact inventory differs from its manifest")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError(f"Student artifact integrity check failed: {relative}")
    expected_weights = {name: digest for name, digest in artifacts.items() if name.endswith(".safetensors")}
    if manifest.get("weights") != expected_weights:
        raise ValueError("Student weight inventory does not match the complete artifact record")
    return manifest


def _step_distill_timesteps(args, model):
    """Reconstruct the trainer's actual times, including native-scale defaults."""
    scale = int(getattr(model, "num_train_timesteps", 1000))
    schedule, shift = args.get("denoising_step_list"), args.get("sample_shift", 1.0)
    if (not isinstance(schedule, (list, tuple)) or not schedule
            or not all(isinstance(t, int) and not isinstance(t, bool) for t in schedule)
            or not isinstance(shift, (int, float)) or not math.isfinite(shift) or shift <= 0):
        raise ValueError("Step-student export requires its exact denoising schedule and finite positive shift")
    schedule = list(schedule)
    if schedule == [1000, 750, 500, 250] and scale != 1000:
        schedule = [max(1, round(scale * fraction)) for fraction in (1, .75, .5, .25)]
    if any(t < 0 or t > scale for t in schedule):
        raise ValueError("Recorded step schedule is outside the native training timestep range")
    if getattr(model, "training_noise_process", "flow") == "vp":
        if shift != 1.0:
            raise ValueError("VP step schedules do not use a flow sampling shift")
        return torch.tensor(schedule, dtype=torch.float32).clamp(max=scale - 1).tolist()
    grid = torch.linspace(1.0, 0.0, scale + 1)[:-1]
    grid = shift * grid / (1 + (shift - 1) * grid)
    indices = torch.tensor([scale - t for t in schedule]).clamp(0, scale - 1)
    return (grid[indices] * scale).tolist()


def _dual_student_boundary(args, model):
    times = _step_distill_timesteps(args, model)
    index, scale = args.get("boundary_step_index"), getattr(model, "num_train_timesteps", 1000)
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(times):
        raise ValueError("Dual-student checkpoint needs its exact boundary step index")
    return times[index] / scale, scale


def export_student(base_model: str, checkpoint: str, output_dir: str, *, num_steps: int | None = None,
                   student_architecture: str | None = None, allow_native_model_code: bool = False) -> dict:
    if num_steps is not None and (not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0):
        raise ValueError("num_steps must be a positive integer")
    source = resolve_checkpoint(checkpoint).resolve()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty student bundle: {destination}")
    # Trusted local training artifact: includes Python/numpy RNG and optimizer
    # state, so it is not a weights-only interchange file.
    state = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(state, dict) or "student_model" not in state:
        raise ValueError("Checkpoint lacks the trained student_model state")
    args = state.get("args", {})
    if not isinstance(args, dict):
        args = {}
    layout = args.get("dual_student_layout", "")
    if layout not in {"", "unified_v1"}:
        raise ValueError(f"Unknown dual_student_layout: {layout}")
    unified = layout == "unified_v1"
    if unified and not args.get("use_dual_model"):
        raise ValueError("Unified dual-student metadata requires use_dual_model=True")
    low_checkpoint = source.parent / "student_low.pt"
    if args.get("use_dual_model") and not unified and not low_checkpoint.is_file():
        raise ValueError("Dual-student export requires its matching student_low.pt; an incomplete legacy checkpoint "
                         "requires an explicit merge")
    if args.get("use_lora"):
        rank, alpha, targets = args.get("lora_rank"), args.get("lora_alpha"), args.get("lora_target_modules")
        if (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1
                or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0
                or not isinstance(targets, list) or not targets or not all(isinstance(v, str) and v for v in targets)):
            raise ValueError("PEFT export requires recorded LoRA rank, alpha, and target modules; "
                             "legacy incomplete metadata requires an explicit merge")
    inferred_steps, step_provenance = _resolve_inference_steps(args, state, source, num_steps)
    # A user-supplied student directory can legitimately have fewer layers than
    # the teacher. Reconstruct that exact architecture, never force its trained
    # tensors into the base teacher's config just because both use the adapter.
    declared_student = args.get("student_model_path")
    if student_architecture is not None:
        architecture_source = student_architecture
    elif declared_student and Path(declared_student).is_dir():
        architecture_source = declared_student
    elif declared_student and not Path(declared_student).is_file():
        raise FileNotFoundError("The recorded student architecture is unavailable; supply --student_architecture")
    else:
        architecture_source = base_model
    with torch.device("meta"):
        model = load_diffusers_training_model(architecture_source, weights=False,
                                             allow_native_code=allow_native_model_code)
        if args.get("use_dual_model") and (args.get("distill_method") != "step_distill"
                                           or isinstance(model, NoiseRoutedDenoiser)):
            raise ValueError("Explicit dual-student export requires the single-architecture step_distill contract")
        if args.get("distill_method") == "mean_flow_distill" and not getattr(model, "supports_meanflow", False):
            from training.utils.meanflow import MeanFlowTimeAdapter
            model = MeanFlowTimeAdapter(model, latent_channels=args.get("meanflow_latent_channels") or None,
                                        hidden_dim=args.get("meanflow_hidden_dim", 128))
        if args.get("use_lora"):
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets,
                lora_dropout=0.0, bias="none", modules_to_save=["time_conditioner"]
                if hasattr(model, "time_conditioner") else None))
        low_model = copy.deepcopy(model) if args.get("use_dual_model") else None
    if unified:
        boundary, scale = _dual_student_boundary(args, model)
        routing_path = source.parent / "dual_student_state.pt"
        if not routing_path.is_file():
            raise FileNotFoundError("Unified dual-student export requires its matching dual_student_state.pt")
        routing = torch.load(routing_path, map_location="cpu", weights_only=True, mmap=True)
        expected_routing = {"dual_student_layout": "unified_v1", "num_train_timesteps": scale,
                            "distill_timesteps": _step_distill_timesteps(args, model),
                            "boundary_step_index": args["boundary_step_index"]}
        if not isinstance(routing, dict) or any(routing.get(key) != value for key, value in expected_routing.items()):
            raise ValueError("Unified dual-student routing state disagrees with its recorded training configuration")
        model = NoiseRoutedDenoiser(model, low_model, boundary, scale)
    student_state = state["student_model"]
    # Checkpoint resume keeps the compiled wrapper's keys unchanged. Export is
    # deliberately eager, so remove exactly one *declared* outer compile layer;
    # never rewrite arbitrary module names or accept mixed/partial prefixes.
    if args.get("enable_torch_compile") and args.get("torch_compile_scope", "student") in {"student", "both"}:
        prefix = "_orig_mod."
        if student_state and all(isinstance(name, str) and name.startswith(prefix) for name in student_state):
            student_state = {name[len(prefix):]: value for name, value in student_state.items()}
        if unified:
            # Unified trainers may contain separately compiled high/low experts.
            # Strip a branch prefix only if every key in that branch agrees.
            for branch in ("high", "low"):
                branch_keys = [name for name in student_state if name.startswith(branch + ".")]
                compiled_prefix = branch + "._orig_mod."
                if branch_keys and all(name.startswith(compiled_prefix) for name in branch_keys):
                    student_state = {(branch + "." + name[len(compiled_prefix):])
                        if name in branch_keys else name: value for name, value in student_state.items()}
    model.load_state_dict(student_state, strict=True, assign=True)
    if args.get("use_lora"):
        # The checkpoint contains the exact training-time base parameters too;
        # never merge the adapter into a fresh teacher's possibly changed weights.
        if unified:
            model.high = model.high.merge_and_unload(safe_merge=True)
            model.low = model.low.merge_and_unload(safe_merge=True)
        else:
            model = model.merge_and_unload(safe_merge=True)
    if low_model is not None and not unified:
        low_state = torch.load(low_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if args.get("enable_torch_compile") and args.get("torch_compile_scope", "student") in {"student", "both"}:
            if low_state and all(isinstance(name, str) and name.startswith("_orig_mod.") for name in low_state):
                low_state = {name[len("_orig_mod."):]: value for name, value in low_state.items()}
        low_model.load_state_dict(low_state, strict=True, assign=True)
        if args.get("use_lora"):
            low_model = low_model.merge_and_unload(safe_merge=True)
        boundary, scale = _dual_student_boundary(args, model)
        model = NoiseRoutedDenoiser(model, low_model, boundary, scale)
    sampling_timesteps = None
    if args.get("distill_method") == "step_distill":
        try:
            sampling_timesteps = _step_distill_timesteps(args, model)
        except ValueError:
            # A legacy single/native-routed checkpoint may specify an explicit
            # export step count without recording its training schedule. Keep
            # that existing contract, but never label an invented grid replay.
            # Separate legacy high/low checkpoints were validated above because
            # their actual routing boundary cannot be recovered without it.
            if num_steps is None:
                raise
    if isinstance(model, NoiseRoutedDenoiser):
        components = {"transformer": model.high, "transformer_2": model.low}
    else:
        index = Path(base_model) / "model_index.json"
        component = "transformer"
        if index.is_file():
            pipeline_config = json.loads(index.read_text())
            component = "transformer" if pipeline_config.get("transformer", [None])[0] else "unet"
        components = {component: model}
    destination.mkdir(parents=True, exist_ok=True)
    for name, adapter in components.items():
        save_training_component(adapter, destination / name)
    artifacts = {str(p.relative_to(destination)): file_sha256(p) for name in components
                 for p in (destination / name).rglob("*") if p.is_file()}
    pipeline_index = Path(base_model) / "model_index.json"
    base_pipeline_class = json.loads(pipeline_index.read_text()).get("_class_name") if pipeline_index.is_file() else None
    base_pipeline_config = json.loads(pipeline_index.read_text()) if pipeline_index.is_file() else {}
    manifest = {
        "schema_version": 2,
        "base_model": str(Path(base_model).resolve()),
        "student_architecture_source": str(Path(architecture_source).resolve()),
        "checkpoint": str(source),
        "checkpoint_sha256": file_sha256(source),
        "global_step": int(state.get("step", 0)),
        "distill_method": args.get("distill_method", "unknown"),
        "lora_merged": bool(args.get("use_lora", False)),
        "legacy_dual_consolidated": low_model is not None and not unified,
        "dual_student_layout": layout or None,
        "dual_routing_state_sha256": file_sha256(routing_path) if unified else None,
        "low_checkpoint_sha256": file_sha256(low_checkpoint) if low_model is not None and not unified else None,
        "requires_adapter_sampling": "transformer_2" in components and not base_pipeline_config.get("transformer_2", [None])[0],
        "trained_sampling_timesteps": sampling_timesteps,
        "training_noise_process": getattr(model, "training_noise_process", "flow"),
        "training_sampling_config": {key: args[key] for key in (
            "denoising_step_list", "sample_shift", "timestep_sampling", "guidance_scale", "cfg_scale",
            "teacher_guidance_scale", "boundary_step_index", "denoising_steps_per_frame",
            "dmd_student_steps", "progressive_stages", "progressive_stage_steps",
        ) if key in args},
        "sampling_parity": {
            "enforced": False,
            "note": "Pipeline sampling uses the base scheduler and guidance. Adapter sampling replays recorded step times when the step count matches and no explicit shift overrides them; otherwise it uses the requested Euler/MeanFlow or VP DDIM grid.",
        },
        "num_inference_steps": inferred_steps,
        "inference_steps_provenance": step_provenance,
        "base_pipeline_class": base_pipeline_class,
        "components": list(components),
        "boundary_ratio": getattr(model, "boundary_ratio", None),
        "num_train_timesteps": getattr(model, "num_train_timesteps", 1000),
        "artifacts": artifacts,
        "weights": {name: digest for name, digest in artifacts.items() if name.endswith(".safetensors")},
    }
    temp = destination / ".worlddistill_export.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp, destination / "worlddistill_export.json")
    return manifest


def load_student_pipeline(bundle: str, *, base_model: str | None = None, dtype=torch.float32):
    from diffusers import DiffusionPipeline
    root = Path(bundle)
    manifest = read_student_manifest(root)
    from training.model_adapter import load_diffusers_denoiser
    adapters = {name: load_diffusers_denoiser(root / name, dtype=dtype) for name in manifest["components"]}
    if any(not isinstance(adapter, DiffusersTrainingAdapter) for adapter in adapters.values()):
        raise ValueError("Native/MeanFlow wrapper bundles use load_diffusers_training_model and sample_student_latents; "
                         "they cannot be substituted into an unmodified Diffusers pipeline")
    overrides = {name: adapter.model for name, adapter in adapters.items()}
    source = base_model or manifest["base_model"]
    if not (Path(source) / "model_index.json").is_file():
        raise ValueError("Video sampling requires the original full pipeline; supply --base_model if the bundle moved")
    pipeline_config = json.loads((Path(source) / "model_index.json").read_text())
    if manifest.get("base_pipeline_class") and pipeline_config.get("_class_name") != manifest["base_pipeline_class"]:
        raise ValueError("Replacement base pipeline class differs from the exported student's pipeline")
    for name in overrides:
        entry = pipeline_config.get(name)
        if not isinstance(entry, (list, tuple)) or not entry or not entry[0]:
            raise ValueError(f"Base pipeline does not declare the trained student component {name}")
    pipe = DiffusionPipeline.from_pretrained(source, torch_dtype=dtype, **overrides)
    for name, trained_component in overrides.items():
        if getattr(pipe, name, None) is not trained_component:
            raise RuntimeError(f"Pipeline ignored the trained student override {name}; refusing to sample base-model weights")
    if manifest["boundary_ratio"] is not None:
        pipe.register_to_config(boundary_ratio=manifest["boundary_ratio"])
    return pipe, manifest


@torch.no_grad()
def sample_student_latents(model, latents, *, num_steps: int, conditions: dict | None = None,
                          audio_latents=None, shift: float = 1.0, timesteps=None):
    """Euler/MeanFlow or deterministic VP DDIM on explicit model-native times.

    This returns normalized latents. Decode them with the original licensed
    VAE(s); conditioning uses the exact training adapter's public inputs.
    Both AV modalities advance on the same explicit shifted sigma schedule.
    VP DDIM uses the checkpoint's cumulative alpha table and a final clean x0
    endpoint, not a guessed flow schedule or an implicit pipeline scheduler.
    """
    import math
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be finite and positive")
    process = getattr(model, "training_noise_process", "flow")
    if process not in {"flow", "vp"} or (process == "vp" and shift != 1.0):
        raise ValueError("Adapter sampling supports flow or VP; VP schedules require shift=1")
    joint = bool(getattr(model, "supports_joint_av", False))
    if joint != (audio_latents is not None):
        raise ValueError("Joint AV sampling requires both modalities, and video-only sampling cannot discard audio")
    if latents.ndim != 5 or (joint and (audio_latents.ndim != 3 or audio_latents.shape[0] != latents.shape[0])):
        raise ValueError("Sampling expects video B,C,T,H,W and optional packed audio B,L,D")
    condition = dict(conditions or {})
    reserved = {"hidden_states", "timestep", "timestep_r", "audio_hidden_states", "audio_timestep"}
    if reserved & condition.keys():
        raise ValueError("Sampling conditions cannot override the latent trajectory or timestep inputs")
    scale = getattr(model, "num_train_timesteps", 1000)
    if timesteps is not None:
        times = torch.as_tensor(timesteps, device=latents.device, dtype=torch.float32)
        if (times.ndim != 1 or len(times) != num_steps or not torch.isfinite(times).all()
                or (times < 0).any() or (times > scale - int(process == "vp")).any()
                or (times[1:] > times[:-1]).any()):
            raise ValueError("Explicit sampling timesteps must be finite descending native times matching num_steps")
    elif process == "vp":
        if num_steps > scale:
            raise ValueError("VP sampling steps cannot exceed the explicit alpha schedule length")
        times = torch.linspace(scale - 1, 0, num_steps, device=latents.device).round()
    else:
        sigma = torch.linspace(1, 0, num_steps + 1, device=latents.device, dtype=torch.float32)[:-1]
        times = (shift * sigma / (1 + (shift - 1) * sigma)) * scale
    if process == "vp":
        from training.utils.noise_objective import NoiseObjective
        if joint or getattr(model, "supports_meanflow", False):
            raise ValueError("Joint AV and MeanFlow adapters require a flow process")
        objective = NoiseObjective(process, scale, getattr(model, "training_prediction_type", "v_prediction"),
                                   getattr(model, "training_alphas_cumprod", None))
        if not torch.equal(times, times.round()):
            raise ValueError("VP sampling timesteps must be integer alpha-table indices")
    elif getattr(model, "training_prediction_type", "velocity") != "velocity":
        from training.utils.noise_objective import NoiseObjective
        if joint or getattr(model, "supports_meanflow", False):
            raise ValueError("Joint AV and MeanFlow sampling require velocity predictions")
        objective = NoiseObjective("flow", scale, model.training_prediction_type)
    model_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for index, current in enumerate(times):
            following = times[index + 1] if index + 1 < len(times) else current.new_zeros(())
            t = current.expand(latents.shape[0])
            inputs = dict(condition, hidden_states=latents, timestep=t)
            if getattr(model, "supports_meanflow", False):
                inputs["timestep_r"] = following.expand_as(t)
            if joint:
                inputs.update(audio_hidden_states=audio_latents, audio_timestep=t)
                result = model(**inputs)
                from training.trainers.joint_av_distill_trainer import audiovisual_prediction
                result = audiovisual_prediction(result, video_shape=latents.shape, audio_shape=audio_latents.shape)
                audio_latents = audio_latents + ((following - current) / scale) * result["audio_sample"]
                prediction = result["sample"]
            else:
                from training.utils.model_output import extract_prediction_tensor
                prediction = extract_prediction_tensor(model(**inputs), tag="exported student")
                if prediction.shape != latents.shape:
                    raise ValueError("Student sampling prediction does not match video latents")
            if process == "vp":
                if index + 1 == len(times):
                    latents = objective.to_x0_epsilon(prediction, latents, t)[0].to(latents.dtype)
                else:
                    latents = objective.ddim_step(prediction, latents, t, following.expand_as(t))
            else:
                if getattr(model, "training_prediction_type", "velocity") != "velocity":
                    prediction = objective.convert_prediction(prediction, latents, t,
                        model.training_prediction_type, "velocity")
                latents = latents + ((following - current) / scale) * prediction
    finally:
        for module, training in model_states:
            module.training = training
    return {"latents": latents, "audio_latents": audio_latents} if joint else latents
