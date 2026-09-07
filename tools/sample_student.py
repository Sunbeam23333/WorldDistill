"""Generate from an exported student, never an unrelated catalog checkpoint."""
from pathlib import Path
import argparse
import inspect
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.student_export import file_sha256, load_student_pipeline


def resolve_sampling_options(args, manifest):
    """Explicit CLI values override JSON; absent flags never erase controls."""
    kwargs = json.loads(Path(args.conditions).read_text()) if args.conditions else {}
    if not isinstance(kwargs, dict):
        raise ValueError("--conditions must contain an object")
    if "generator" in kwargs:
        raise ValueError("Supply the random generator through --seed, not --conditions")
    kwargs["prompt"] = args.prompt
    defaults = {"num_frames": 81, "height": 480, "width": 832,
                "num_inference_steps": manifest["num_inference_steps"]}
    for key, default in defaults.items():
        explicit = getattr(args, "num_steps" if key == "num_inference_steps" else key)
        if explicit is not None:
            kwargs[key] = explicit
        else:
            kwargs.setdefault(key, default)
        if not isinstance(kwargs[key], int) or isinstance(kwargs[key], bool) or kwargs[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--base_model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image_path")
    parser.add_argument("--conditions", help="JSON of pipeline-specific keyword values; unsupported controls fail")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--num_frames", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    for name in ("num_frames", "height", "width", "fps", "num_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    output = Path(args.save_path)
    provenance = output.with_suffix(".provenance.json")
    if output.exists() or provenance.exists():
        raise FileExistsError(f"Refusing to overwrite sample or provenance at {output}")
    pipe, manifest = load_student_pipeline(args.bundle, base_model=args.base_model, dtype=getattr(torch, args.dtype))
    pipe.to(args.device)
    kwargs = resolve_sampling_options(args, manifest)
    effective_options = dict(kwargs)
    kwargs["generator"] = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.image_path:
        from diffusers.utils import load_image
        kwargs["image"] = load_image(args.image_path)
    unknown = set(kwargs) - set(inspect.signature(pipe.__call__).parameters)
    if unknown:
        raise ValueError(f"{type(pipe).__name__} has no adapters for inputs {sorted(unknown)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    start = time.perf_counter()
    with torch.inference_mode():
        generated = pipe(**kwargs)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    elapsed = time.perf_counter() - start
    from diffusers.utils import export_to_video
    export_to_video(generated.frames[0], str(output), fps=args.fps)
    record = {"student_bundle": str(Path(args.bundle).resolve()), "checkpoint_sha256": manifest["checkpoint_sha256"],
              "global_step": manifest["global_step"], "pipeline": type(pipe).__name__, "latency_seconds": elapsed,
              "seed": args.seed, "num_inference_steps": kwargs["num_inference_steps"], "output": str(output),
              "device": args.device, "dtype": args.dtype, "torch_version": torch.__version__,
              "effective_options": effective_options, "fps": args.fps,
              "sampling_parity": manifest.get("sampling_parity", {"enforced": False}),
              "output_sha256": file_sha256(output)}
    if args.conditions:
        record["conditions_sha256"] = file_sha256(Path(args.conditions))
    if args.image_path:
        record["image_sha256"] = file_sha256(Path(args.image_path))
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
