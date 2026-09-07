"""Export trained denoisers and reload them in the original Diffusers pipeline.

Bundles intentionally reference (rather than redistribute) the base model's
licensed VAE/text/tokenizer assets. Provenance ties inference to an exact
training checkpoint; no inference-family weight-name guessing is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from training.model_adapter import DiffusersTrainingAdapter, NoiseRoutedDenoiser, load_diffusers_training_model


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
        if not (directory / "config.json").is_file() or not any(directory.rglob("*.safetensors")):
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


def export_student(base_model: str, checkpoint: str, output_dir: str, *, num_steps: int | None = None,
                   student_architecture: str | None = None) -> dict:
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
    if args.get("use_dual_model") or args.get("use_lora"):
        raise ValueError("Legacy dual-student/PEFT checkpoints require explicit merge; export native noise-routed full weights")
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
        model = load_diffusers_training_model(architecture_source, weights=False)
    student_state = state["student_model"]
    # Checkpoint resume keeps the compiled wrapper's keys unchanged. Export is
    # deliberately eager, so remove exactly one *declared* outer compile layer;
    # never rewrite arbitrary module names or accept mixed/partial prefixes.
    if args.get("enable_torch_compile") and args.get("torch_compile_scope", "student") in {"student", "both"}:
        prefix = "_orig_mod."
        if student_state and all(isinstance(name, str) and name.startswith(prefix) for name in student_state):
            student_state = {name[len(prefix):]: value for name, value in student_state.items()}
    model.load_state_dict(student_state, strict=True, assign=True)
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
        if not isinstance(adapter, DiffusersTrainingAdapter):
            raise TypeError("Export requires a Diffusers-backed differentiable model adapter")
        adapter.model.save_pretrained(destination / name, safe_serialization=True)
    artifacts = {str(p.relative_to(destination)): file_sha256(p) for name in components
                 for p in (destination / name).rglob("*") if p.is_file()}
    pipeline_index = Path(base_model) / "model_index.json"
    base_pipeline_class = json.loads(pipeline_index.read_text()).get("_class_name") if pipeline_index.is_file() else None
    manifest = {
        "schema_version": 2,
        "base_model": str(Path(base_model).resolve()),
        "student_architecture_source": str(Path(architecture_source).resolve()),
        "checkpoint": str(source),
        "checkpoint_sha256": file_sha256(source),
        "global_step": int(state.get("step", 0)),
        "distill_method": args.get("distill_method", "unknown"),
        "training_sampling_config": {key: args[key] for key in (
            "denoising_step_list", "sample_shift", "timestep_sampling", "guidance_scale", "cfg_scale",
            "teacher_guidance_scale", "boundary_step_index", "denoising_steps_per_frame",
            "dmd_student_steps", "progressive_stages", "progressive_stage_steps",
        ) if key in args},
        "sampling_parity": {
            "enforced": False,
            "note": "The training settings are recorded, not replayed. Sampling uses the base pipeline scheduler/guidance unless explicitly overridden; objective and scheduler parity require model-specific validation.",
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
    overrides = {name: load_diffusers_denoiser(root / name, dtype=dtype).model for name in manifest["components"]}
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
