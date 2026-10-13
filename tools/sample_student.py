"""Generate from an exported student, never an unrelated catalog checkpoint."""
from pathlib import Path
import argparse
import importlib
import inspect
import json
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.student_export import (file_sha256, load_student_pipeline, read_student_manifest,
                                     sample_student_latents)


def parse_latent_shape(value, dimensions):
    try:
        shape = tuple(int(part) for part in value.split(","))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Latent shape must be comma-separated positive dimensions") from exc
    if len(shape) != dimensions or any(size <= 0 for size in shape) or shape[0] != 1:
        raise ValueError(f"Sampling CLI expects {dimensions} positive dimensions with batch size 1")
    return shape


def denormalize_video_latents(latents, vae):
    config = vae.config
    mean = getattr(vae, "latents_mean", getattr(config, "latents_mean", None))
    std = getattr(vae, "latents_std", getattr(config, "latents_std", None))
    scale = getattr(config, "scaling_factor", 1.0)
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("VAE scaling factor must be finite and positive")
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("VAE must declare both normalization statistics")
        shape = (1, -1) + (1,) * (latents.ndim - 2)
        mean = torch.as_tensor(mean, device=latents.device, dtype=latents.dtype).reshape(shape)
        std = torch.as_tensor(std, device=latents.device, dtype=latents.dtype).reshape(shape)
        if (mean.shape[1] != latents.shape[1] or std.shape[1] != latents.shape[1] or (std <= 0).any()
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()):
            raise ValueError("Invalid VAE channel normalization")
        return latents * std / (scale if "LTX" in type(vae).__name__ else 1.0) + mean
    return latents / scale + (getattr(config, "shift_factor", 0.0) or 0.0)


def write_generated_media(frames, output, *, fps, audio=None, sample_rate=None):
    if audio is not None:
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("Audiovisual output requires the vocoder's positive audio sample rate")
        from diffusers.utils import encode_video
        encode_video(frames, fps=fps, audio=audio.float().cpu(), audio_sample_rate=sample_rate,
                     output_path=str(output))
    else:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(output), fps=fps)


def decode_adapter_output(result, args, manifest):
    """Decode actual sampled latents, using original VAE assets or author code."""
    source = args.base_model or manifest["base_model"]
    if args.decoder_factory:
        if not args.allow_native_model_code:
            raise ValueError("--decoder_factory executes installed code and requires --allow_native_model_code")
        module, separator, name = args.decoder_factory.partition(":")
        if not separator or not name.isidentifier() or not all(p.isidentifier() for p in module.split(".")):
            raise ValueError("Decoder factory must be installed.module:factory")
        decoder = getattr(importlib.import_module(module), name)(base_model=source, device=args.device,
                                                               dtype=getattr(torch, args.dtype))
        decoded = decoder(result)
        if not isinstance(decoded, dict) or "frames" not in decoded:
            raise ValueError("Native decoder must return {'frames': ..., optional 'audio', 'audio_sample_rate'}")
        if isinstance(result, dict) and "audio_latents" in result and "audio" not in decoded:
            raise ValueError("Joint AV decoder cannot discard the audio modality")
        return decoded
    if isinstance(result, dict):
        raise ValueError("Joint AV adapter decoding requires an author --decoder_factory; alternatively use the "
                         "LTX2 pipeline backend (audio included) or save .safetensors latents")
    import diffusers
    directory = Path(source) / "vae"
    config = json.loads((directory / "config.json").read_text())
    vae_class = getattr(diffusers, config.get("_class_name", ""), None)
    if not isinstance(vae_class, type) or not issubclass(vae_class, diffusers.ModelMixin):
        raise ValueError("Base VAE architecture is not a Diffusers model; provide an author --decoder_factory")
    vae = vae_class.from_pretrained(directory, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    kwargs = {"return_dict": False}
    if getattr(vae.config, "timestep_conditioning", False):
        if args.decode_timestep is None:
            raise ValueError("This VAE requires an explicit --decode_timestep in [0,1]")
        noise_scale = args.decode_timestep if args.decode_noise_scale is None else args.decode_noise_scale
        generator = torch.Generator(device=result.device).manual_seed(args.seed)
        noise = torch.randn(result.shape, generator=generator, device=result.device, dtype=result.dtype)
        result = (1 - noise_scale) * result + noise_scale * noise
        kwargs["timestep"] = torch.full((result.shape[0],), args.decode_timestep, device=result.device, dtype=result.dtype)
    latent = denormalize_video_latents(result, vae)
    with torch.no_grad():
        frames = vae.decode(latent.to(dtype=vae.dtype), **kwargs)[0]
    from diffusers.video_processor import VideoProcessor
    return {"frames": VideoProcessor().postprocess_video(frames, output_type="pil")[0]}


def sample_adapter(args, manifest, output):
    from safetensors.torch import load_file, save_file
    from training.model_adapter import load_diffusers_training_model
    shape = parse_latent_shape(args.latent_shape, 5)
    if any(getattr(args, key, None) is not None for key in ("num_frames", "height", "width")):
        raise ValueError("Adapter dimensions are specified by --latent_shape, not pipeline frame/pixel dimensions")
    model = load_diffusers_training_model(args.bundle, dtype=getattr(torch, args.dtype),
        allow_native_code=args.allow_native_model_code).to(args.device).eval()
    conditions = json.loads(Path(args.conditions).read_text()) if args.conditions else {}
    if not isinstance(conditions, dict):
        raise ValueError("--conditions must contain an object")
    if args.conditioning_tensors:
        tensors = load_file(args.conditioning_tensors, device=args.device)
        if conditions.keys() & tensors.keys():
            raise ValueError("Scalar and tensor conditioning names cannot overlap")
        conditions.update({key: value.to(dtype=getattr(torch, args.dtype)) if value.is_floating_point() else value
                           for key, value in tensors.items()})
    if args.prompt and "encoder_hidden_states" not in conditions:
        raise ValueError("Adapter sampling takes pre-encoded --conditioning_tensors; encode the supplied prompt "
                         "with the checkpoint's text encoders or use the pipeline backend")
    if args.image_path:
        raise ValueError("Adapter sampling uses encoded image_cond/image_embeds in --conditioning_tensors")
    if getattr(model, "supports_joint_av", False):
        conditions.setdefault("fps", args.fps)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    latent = torch.randn(shape, device=args.device, dtype=getattr(torch, args.dtype), generator=generator)
    audio = None
    if args.audio_latent_shape:
        audio = torch.randn(parse_latent_shape(args.audio_latent_shape, 3), device=args.device,
                            dtype=latent.dtype, generator=generator)
    count = args.num_steps or manifest["num_inference_steps"]
    shift = args.shift if args.shift is not None else manifest.get("training_sampling_config", {}).get("sample_shift", 1.0)
    trained_times = manifest.get("trained_sampling_timesteps")
    times = trained_times if trained_times is not None and len(trained_times) == count and args.shift is None else None
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    start = time.perf_counter()
    result = sample_student_latents(model, latent, audio_latents=audio, conditions=conditions,
                                   num_steps=count, shift=shift, timesteps=times)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    elapsed = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".safetensors":
        tensors = result if isinstance(result, dict) else {"latents": result}
        save_file({key: tensor.detach().cpu().contiguous() for key, tensor in tensors.items()}, str(output))
        artifact_type = "normalized_generated_latents"
    else:
        decoded = decode_adapter_output(result, args, manifest)
        write_generated_media(decoded["frames"], output, fps=args.fps, audio=decoded.get("audio"),
                              sample_rate=decoded.get("audio_sample_rate"))
        artifact_type = "generated_video"
    return {"student_bundle": str(Path(args.bundle).resolve()), "checkpoint_sha256": manifest["checkpoint_sha256"],
        "global_step": manifest["global_step"], "backend": "adapter", "artifact_type": artifact_type,
        "latency_scope": "denoising_only", "latency_seconds": elapsed, "num_inference_steps": count,
        "sigma_schedule": {"type": "ddim_clean_endpoint" if getattr(model, "training_noise_process", "flow") == "vp"
                           else "meanflow" if getattr(model, "supports_meanflow", False) else "euler",
                           "shift": shift, "timesteps": times, "trained_times_used": times is not None}, "seed": args.seed,
        "prompt": args.prompt,
        "latent_shape": shape, "audio_latent_shape": args.audio_latent_shape, "device": args.device,
        "dtype": args.dtype, "torch_version": torch.__version__, "output": str(output),
        "output_sha256": file_sha256(output), "conditioning_sha256": file_sha256(Path(args.conditioning_tensors))
            if args.conditioning_tensors else None, "conditions_sha256": file_sha256(Path(args.conditions))
            if args.conditions else None, "decoder_factory": args.decoder_factory,
        "decode_timestep": args.decode_timestep, "decode_noise_scale": args.decode_noise_scale,
        "fps": conditions.get("fps", args.fps)}


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
    parser.add_argument("--prompt")
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
    parser.add_argument("--backend", choices=["auto", "pipeline", "adapter"], default="auto")
    parser.add_argument("--conditioning_tensors", help="Safetensors of pre-encoded batched native model conditions")
    parser.add_argument("--latent_shape", help="Adapter initial Gaussian shape B,C,T,H,W (B=1)")
    parser.add_argument("--audio_latent_shape", help="Joint AV initial Gaussian shape B,L,D (B=1)")
    parser.add_argument("--shift", type=float, help="Adapter flow shift; defaults to the recorded training schedule")
    parser.add_argument("--decoder_factory", help="Trusted installed.module:factory returning a latent-to-media decoder")
    parser.add_argument("--decode_timestep", type=float, help="Explicit timestep for timestep-conditioned video VAEs")
    parser.add_argument("--decode_noise_scale", type=float, help="VAE decode noise amount; defaults to decode_timestep")
    parser.add_argument("--allow_native_model_code", action="store_true")
    args = parser.parse_args()
    for name in ("num_frames", "height", "width", "fps", "num_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    output = Path(args.save_path)
    provenance = output.with_suffix(".provenance.json")
    if output.exists() or provenance.exists():
        raise FileExistsError(f"Refusing to overwrite sample or provenance at {output}")
    if args.decode_timestep is not None and not 0 <= args.decode_timestep <= 1:
        parser.error("--decode_timestep must be in [0,1]")
    if args.decode_noise_scale is not None and not 0 <= args.decode_noise_scale <= 1:
        parser.error("--decode_noise_scale must be in [0,1]")
    manifest = read_student_manifest(args.bundle)
    wrapper = any((Path(args.bundle) / component / marker).is_file()
        for component in manifest["components"] for marker in ("worlddistill_model.json", "worlddistill_adapter.json"))
    adapter_backend = args.backend == "adapter" or (args.backend == "auto" and (
        wrapper or args.latent_shape or args.conditioning_tensors or manifest.get("requires_adapter_sampling", False)))
    if adapter_backend:
        record = sample_adapter(args, manifest, output)
        provenance.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return
    if not args.prompt:
        parser.error("Pipeline sampling requires --prompt")
    if any((args.conditioning_tensors, args.latent_shape, args.audio_latent_shape, args.decoder_factory,
            args.decode_timestep is not None, args.decode_noise_scale is not None, args.shift is not None)):
        parser.error("Adapter-specific sampling options require --backend adapter")
    pipe, manifest = load_student_pipeline(args.bundle, base_model=args.base_model, dtype=getattr(torch, args.dtype))
    pipe.to(args.device)
    kwargs = resolve_sampling_options(args, manifest)
    if "frame_rate" in inspect.signature(pipe.__call__).parameters:
        kwargs.setdefault("frame_rate", args.fps)
    effective_fps = kwargs.get("frame_rate", kwargs.get("fps", args.fps))
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
    audio = getattr(generated, "audio", None)
    sample_rate = getattr(getattr(getattr(pipe, "vocoder", None), "config", None), "output_sampling_rate", None)
    write_generated_media(generated.frames[0], output, fps=effective_fps,
                          audio=audio[0] if audio is not None else None, sample_rate=sample_rate)
    record = {"student_bundle": str(Path(args.bundle).resolve()), "checkpoint_sha256": manifest["checkpoint_sha256"],
              "global_step": manifest["global_step"], "pipeline": type(pipe).__name__, "latency_seconds": elapsed,
              "seed": args.seed, "num_inference_steps": kwargs["num_inference_steps"], "output": str(output),
              "device": args.device, "dtype": args.dtype, "torch_version": torch.__version__,
              "effective_options": effective_options, "fps": effective_fps,
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
