"""Evaluate generated videos, supplied I3D features, and recovered control tracks.

No randomly initialized feature extractor or replacement benchmark is used.
FVD consumes safetensors feature matrices with matching extractor metadata;
VBench invokes the installed official package on actual video files.
References: https://github.com/Vchitect/VBench
https://github.com/google-research/google-research/tree/master/frechet_video_distance
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from training.student_export import file_sha256


def frechet_feature_distance(reference, generated):
    """Unbiased sample covariances and PSD-symmetric Frechet distance."""
    for features in (reference, generated):
        if not torch.is_tensor(features) or features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
            raise ValueError("Frechet distance needs N,D feature matrices with at least two samples")
        if not torch.isfinite(features).all():
            raise ValueError("Feature matrices must be finite")
    if reference.shape[1] != generated.shape[1]:
        raise ValueError("Reference and generated feature dimensions must agree")
    a, b = reference.detach().cpu().double(), generated.detach().cpu().double()
    mean_a, mean_b = a.mean(0), b.mean(0)
    a, b = a - mean_a, b - mean_b
    cov_a, cov_b = a.T @ a / (a.shape[0] - 1), b.T @ b / (b.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(cov_a)
    root_a = (vectors * eigenvalues.clamp_min(0).sqrt()) @ vectors.T
    product = root_a @ cov_b @ root_a
    cross_trace = torch.linalg.eigvalsh((product + product.T) * 0.5).clamp_min(0).sqrt().sum()
    distance = (mean_a - mean_b).square().sum() + cov_a.trace() + cov_b.trace() - 2 * cross_trace
    return float(distance.clamp_min(0))


def trajectory_metrics(prediction, target, valid=None):
    """ADE/FDE in the caller's coordinate units; no hidden alignment or scaling."""
    if prediction.shape != target.shape or prediction.ndim != 3 or min(prediction.shape) < 1:
        raise ValueError("Trajectories must share nonempty B,T,D shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Trajectories must be finite")
    if valid is None:
        valid = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    if valid.dtype != torch.bool or valid.shape != prediction.shape[:2] or not valid.any(1).all():
        raise ValueError("Trajectory mask must be B,T bool with at least one valid point per sample")
    difference = prediction.double() - target.double()
    distances = difference.norm(dim=-1)
    last = torch.arange(valid.shape[1], device=valid.device).expand_as(valid).masked_fill(~valid, -1).max(1).values
    final = distances.gather(1, last[:, None]).mean()
    return {"ade": float(distances[valid].mean()), "fde": float(final),
            "coordinate_rmse": float(difference[valid].square().mean().sqrt()),
            "valid_points": int(valid.sum()), "trajectories": prediction.shape[0],
            "aggregation": {"ade": "mean_over_valid_points", "fde": "mean_over_trajectories"}}


def action_accuracy(logits, labels):
    if logits.ndim < 2 or logits.shape[:-1] != labels.shape or labels.numel() == 0:
        raise ValueError("Action logits must have an extra final class dimension relative to labels")
    if labels.dtype not in {torch.int32, torch.int64} or not torch.isfinite(logits).all():
        raise ValueError("Action labels must be integer and logits finite")
    if ((labels < 0) | (labels >= logits.shape[-1])).any():
        raise ValueError("Action labels are outside the classifier vocabulary")
    return {"accuracy": float((logits.argmax(-1) == labels).double().mean()), "examples": labels.numel()}


def _feature_bundle(path):
    from safetensors.torch import load_file
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    extractor = record.get("extractor", {})
    if (extractor.get("name") != "i3d" or not re.fullmatch(r"[a-fA-F0-9]{64}", extractor.get("weights_sha256", ""))
            or not isinstance(extractor.get("preprocessing"), dict) or not extractor["preprocessing"]):
        raise ValueError("FVD feature metadata must identify i3d, its weight SHA256, and explicit preprocessing")
    feature_path = path.parent / record["features"]
    features = load_file(str(feature_path))["features"]
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("FVD features must be a nonempty N,D matrix with N >= 2")
    videos = record.get("videos")
    if not isinstance(videos, list) or len(videos) != features.shape[0] or not videos:
        raise ValueError("Each feature row must identify its corresponding source video")
    resolved = [(path.parent / video).resolve() for video in videos]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Feature video inventory must not contain duplicate samples")
    hashes = {str(video): file_sha256(video) for video in resolved}
    return features, {"extractor": extractor, "feature_sha256": file_sha256(feature_path),
                      "metadata_sha256": file_sha256(path), "videos": hashes}


def evaluate_fvd(reference_metadata, generated_metadata):
    reference, reference_record = _feature_bundle(reference_metadata)
    generated, generated_record = _feature_bundle(generated_metadata)
    if reference_record["extractor"] != generated_record["extractor"]:
        raise ValueError("FVD feature extractor weights and preprocessing must match exactly")
    return {"fvd": frechet_feature_distance(reference, generated), "reference_samples": reference.shape[0],
            "generated_samples": generated.shape[0], "feature_dimension": reference.shape[1],
            "reference": reference_record, "generated": generated_record,
            "feature_source": "caller-supplied I3D features"}


def evaluate_vbench(videos, full_info, output_dir, dimensions, *, device="cuda", prompts=None, local=True):
    from vbench import VBench
    videos, destination = Path(videos).resolve(), Path(output_dir)
    files = [videos] if videos.is_file() else sorted(p for p in videos.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".gif"})
    if not files or any(p.suffix.lower() not in {".mp4", ".gif"} for p in files):
        raise ValueError("VBench requires actual .mp4/.gif videos")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("VBench output directory must be empty")
    if not Path(full_info).is_file():
        raise FileNotFoundError("Provide the official VBench full-info JSON")
    if prompts:
        if videos.is_file():
            if not isinstance(prompts, list) or len(prompts) != 1 or not isinstance(prompts[0], str):
                raise ValueError("Single-video VBench prompts must be a one-item string list")
        else:
            if not isinstance(prompts, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()):
                raise ValueError("Directory VBench prompts must map video filenames to strings")
            named = {(videos / name).resolve() for name in prompts}
            if named != set(files):
                raise ValueError("VBench prompt mapping must match the video inventory exactly")
    evaluator = VBench(torch.device(device), str(full_info), str(destination))
    supported = set(evaluator.build_full_dimension_list())
    if not dimensions or len(set(dimensions)) != len(dimensions) or not set(dimensions) <= supported:
        raise ValueError("Unknown or empty VBench dimensions")
    evaluator.check_dimension_requires_extra_info(dimensions)
    evaluator.evaluate(videos_path=str(videos), name="worlddistill", dimension_list=dimensions,
                       mode="custom_input", prompt_list=prompts or [], local=local)
    results = destination / "worlddistill_eval_results.json"
    if not results.is_file():
        raise RuntimeError("Official VBench evaluator did not write its result file")
    scores = json.loads(results.read_text())
    if set(scores) != set(dimensions):
        raise ValueError("Official VBench result does not cover the requested dimensions")
    for name, value in scores.items():
        if not isinstance(value, list) or not value or not isinstance(value[0], (int, float)) or not math.isfinite(value[0]):
            raise ValueError(f"Official VBench returned invalid score for {name}")
    return {"vbench": scores, "vbench_version": importlib.metadata.version("vbench"),
            "dimensions": dimensions, "mode": "custom_input", "device": device,
            "full_info_sha256": file_sha256(Path(full_info)), "prompts": prompts,
            "videos": {str(p): file_sha256(p) for p in files}, "official_result_sha256": file_sha256(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    subparsers = parser.add_subparsers(dest="metric", required=True)
    fvd = subparsers.add_parser("fvd")
    fvd.add_argument("--reference_metadata", required=True)
    fvd.add_argument("--generated_metadata", required=True)
    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("--tensors", required=True, help="Safetensors with prediction/target/optional bool valid")
    trajectory.add_argument("--coordinate_units", required=True)
    action = subparsers.add_parser("action")
    action.add_argument("--tensors", required=True, help="Safetensors with logits/labels from recovered video actions")
    vbench = subparsers.add_parser("vbench")
    vbench.add_argument("--videos", required=True)
    vbench.add_argument("--full_info", required=True)
    vbench.add_argument("--result_dir", required=True)
    vbench.add_argument("--dimensions", nargs="+", required=True)
    vbench.add_argument("--device", default="cuda")
    vbench.add_argument("--prompts", help="Official VBench custom prompt JSON (mapping filenames, or one-item list)")
    vbench.add_argument("--allow_download", action="store_true", help="Use VBench's remote rather than local checkpoint selection")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result {destination}")
    if args.metric == "fvd":
        report = evaluate_fvd(args.reference_metadata, args.generated_metadata)
    elif args.metric == "vbench":
        prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else None
        report = evaluate_vbench(args.videos, args.full_info, args.result_dir, args.dimensions,
                                 device=args.device, prompts=prompts, local=not args.allow_download)
    else:
        from safetensors.torch import load_file
        tensors = load_file(args.tensors)
        if args.metric == "trajectory":
            report = trajectory_metrics(tensors["prediction"], tensors["target"], tensors.get("valid"))
            report["coordinate_units"] = args.coordinate_units
        else:
            report = action_accuracy(tensors["logits"], tensors["labels"])
        report["input_sha256"] = file_sha256(Path(args.tensors))
        report["input_scope"] = "caller-supplied recovered tracks/actions, not a built-in visual estimator"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(json.dumps({"metric": args.metric, **report}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
