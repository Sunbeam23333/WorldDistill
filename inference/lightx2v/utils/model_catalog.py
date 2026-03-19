from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_ZOO_PATH = _REPO_ROOT / "configs" / "model_zoo.yaml"
_DEFAULT_CONFIG_BASE = _REPO_ROOT / "inference" / "configs"

_TASK_IO_SIGNATURES: dict[str, dict[str, list[str]]] = {
    "t2v": {"inputs": ["text"], "outputs": ["video"]},
    "i2v": {"inputs": ["image", "text"], "outputs": ["video"]},
    "flf2v": {"inputs": ["image", "text"], "outputs": ["video"]},
    "vace": {"inputs": ["image", "video", "mask", "text"], "outputs": ["video"]},
    "s2v": {"inputs": ["audio", "image", "text"], "outputs": ["video"]},
    "rs2v": {"inputs": ["audio", "image", "text"], "outputs": ["video"]},
    "animate": {"inputs": ["image", "pose", "face", "mask", "text"], "outputs": ["video"]},
    "t2i": {"inputs": ["text"], "outputs": ["image"]},
    "i2i": {"inputs": ["image", "text"], "outputs": ["image"]},
    "t2av": {"inputs": ["text"], "outputs": ["audio", "video"]},
    "i2av": {"inputs": ["image", "text"], "outputs": ["audio", "video"]},
    "game": {"inputs": ["image", "text", "action", "pose", "memory"], "outputs": ["video"]},
    "open_world": {"inputs": ["image", "text", "action", "memory"], "outputs": ["video"]},
    "long_video": {"inputs": ["image", "text", "memory"], "outputs": ["video"]},
    "ti2v": {"inputs": ["image", "text"], "outputs": ["video"]},
    "audio": {"inputs": ["text"], "outputs": ["audio"]},
}

_DEFAULT_MODEL_ALIASES: dict[str, str] = {
    "wan21": "wan2.1",
    "wan-2.1": "wan2.1",
    "wan2_1": "wan2.1",
    "wan2.1-t2v": "wan2.1",
    "wan2.1-i2v": "wan2.1",
    "wan21_distill": "wan2.1_distill",
    "wan2.1-distill": "wan2.1_distill",
    "wan2.1-mean-flow-distill": "wan2.1_mean_flow_distill",
    "wan21_meanflow_distill": "wan2.1_mean_flow_distill",
    "wan2.1-vace": "wan2.1_vace",
    "wan21_vace": "wan2.1_vace",
    "wan2.1-sf": "wan2.1_sf",
    "wan21_sf": "wan2.1_sf",
    "wan2.2-a14b": "wan2.2_moe",
    "wan22_a14b": "wan2.2_moe",
    "wan22": "wan2.2",
    "wan-2.2": "wan2.2",
    "wan2_2": "wan2.2",
    "wan22_moe": "wan2.2_moe",
    "wan-2.2-moe": "wan2.2_moe",
    "wan2_2_moe": "wan2.2_moe",
    "wan2.2_moe_diffusers": "wan2.2_moe",
    "wan2.2-diffusers": "wan2.2_moe",
    "wan22_diffusers": "wan2.2_moe",
    "wan2.2_distill": "wan2.2_moe_distill",
    "wan2.2-moe-distill": "wan2.2_moe_distill",
    "wan22_distill": "wan2.2_moe_distill",
    "wan2.2-moe-vace": "wan2.2_moe_vace",
    "wan22_moe_vace": "wan2.2_moe_vace",
    "wan2.2-audio": "wan2.2_audio",
    "wan22_audio": "wan2.2_audio",
    "hyvideo": "hunyuan_video_1.5",
    "hy-video": "hunyuan_video_1.5",
    "hunyuan-video": "hunyuan_video_1.5",
    "hunyuan-video-1.5": "hunyuan_video_1.5",
    "hunyuan_video_15": "hunyuan_video_1.5",
    "hunyuan_video_1_5": "hunyuan_video_1.5",
    "hyvideo_distill": "hunyuan_video_1.5_distill",
    "hunyuan-video-distill": "hunyuan_video_1.5_distill",
    "worldplay": "worldplay_distill",
    "hy-worldplay": "worldplay_distill",
    "hy_worldplay": "worldplay_distill",
    "qwen-image": "qwen_image",
    "qwen_image_edit": "qwen_image",
    "qwen-image-edit": "qwen_image",
    "qwen-image-edit-2509": "qwen_image",
    "qwen-image-edit-2511": "qwen_image",
    "qwen-image-2512": "qwen_image",
    "longcat-image": "longcat_image",
    "z-image": "z_image",
    "zimage": "z_image",
    "lingbot": "lingbot_cam_moe",
    "lingbot-cam": "lingbot_cam_moe",
    "skyreels-v2": "skyreels_v2",
    "matrix-game-2": "matrix_game_2",
    "matrix_game2": "matrix_game_2",
    "gamecraft-2": "gamecraft",
    "seko-talk": "seko_talk",
}


def _spec(
    architecture: str,
    tasks: list[str],
    runner_cls: str,
    *,
    status: str = "supported",
    model_family: str = "",
    checkpoint_formats: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "architecture": architecture,
        "tasks": tasks,
        "runner_cls": runner_cls,
        "status": status,
    }
    if model_family:
        spec["model_family"] = model_family
    if checkpoint_formats is not None:
        spec["checkpoint_formats"] = checkpoint_formats
    spec.update(extra)
    return spec


_FALLBACK_MODEL_ZOO: dict[str, dict[str, Any]] = {
    "wan2.1": _spec(
        "wan",
        ["t2v", "i2v", "flf2v"],
        "wan2.1",
        model_family="video",
        checkpoint_formats=["directory", "original", "diffusers", "state_dict"],
        default_configs={
            "t2v": "wan/wan_t2v.json",
            "i2v": "wan/wan_i2v.json",
            "flf2v": "wan/wan_flf2v.json",
        },
    ),
    "wan2.1_distill": _spec(
        "wan",
        ["t2v", "i2v"],
        "wan2.1_distill",
        model_family="video",
        checkpoint_formats=["directory", "original", "state_dict"],
        teacher_model_cls="wan2.1",
        distill_stage="student",
        default_configs={
            "t2v": "distill/wan21/wan_t2v_distill_model_4step_cfg.json",
            "i2v": "distill/wan21/wan_i2v_distill_model_4step_cfg.json",
        },
    ),
    "wan2.1_mean_flow_distill": _spec(
        "wan",
        ["t2v"],
        "wan2.1_mean_flow_distill",
        model_family="video",
        checkpoint_formats=["directory", "original", "state_dict"],
        teacher_model_cls="wan2.1",
        distill_stage="student",
        distill_methods=["mean_flow_distill"],
        default_configs={"t2v": "meanflow/wan_t2v_meanflow_distill_2step.json"},
    ),
    "wan2.1_vace": _spec(
        "wan",
        ["vace"],
        "wan2.1_vace",
        model_family="video",
        checkpoint_formats=["directory", "original", "state_dict"],
        features=["video_editing"],
        default_configs={"vace": "wan/wan_vace.json"},
    ),
    "wan2.1_sf": _spec(
        "wan",
        ["t2v"],
        "wan2.1_sf",
        model_family="world_model",
        checkpoint_formats=["directory", "original", "state_dict"],
        features=["self_forcing", "streaming_generation"],
        default_configs={"t2v": "self_forcing/wan_t2v_sf.json"},
    ),
    "wan2.2": _spec(
        "wan_dense",
        ["t2v", "i2v", "ti2v"],
        "wan2.2",
        model_family="video",
        checkpoint_formats=["directory", "original", "diffusers", "state_dict"],
        default_configs={
            "t2v": "wan22/wan_ti2v_t2v.json",
            "i2v": "wan22/wan_ti2v_i2v.json",
        },
    ),
    "wan2.2_moe": _spec(
        "wan_moe",
        ["t2v", "i2v", "flf2v", "t2av", "i2av"],
        "wan2.2_moe",
        model_family="video",
        checkpoint_formats=["directory", "diffusers", "dual_model", "quantized", "state_dict"],
        default_configs={
            "t2v": "wan22/wan_moe_t2v.json",
            "i2v": "wan22/wan_moe_i2v.json",
            "flf2v": "wan22/wan_moe_flf2v.json",
            "t2av": "wan22/wan_moe_i2v_audio.json",
            "i2av": "wan22/wan_moe_i2v_audio.json",
        },
    ),
    "wan2.2_moe_vace": _spec(
        "wan_moe",
        ["vace"],
        "wan2.2_moe_vace",
        model_family="video",
        checkpoint_formats=["directory", "state_dict"],
        features=["video_editing"],
        default_configs={"vace": "wan22_vace/a800/bf16/wan22_moe_vace.json"},
    ),
    "wan2.2_moe_distill": _spec(
        "wan_moe",
        ["t2v", "i2v", "flf2v"],
        "wan2.2_moe_distill",
        model_family="video",
        checkpoint_formats=["directory", "dual_model", "quantized", "state_dict"],
        teacher_model_cls="wan2.2_moe",
        distill_stage="student",
        default_configs={
            "t2v": "wan22/wan_moe_t2v_distill.json",
            "i2v": "distill/wan_i2v_distill_4step_cfg.json",
            "flf2v": "wan22/wan_distill_moe_flf2v.json",
        },
    ),
    "wan2.2_audio": _spec(
        "wan_dense",
        ["s2v", "rs2v"],
        "wan2.2_audio",
        model_family="audio_video",
        checkpoint_formats=["directory", "state_dict"],
        features=["audio_conditioning", "streaming_generation"],
        default_configs={
            "s2v": "wan22/wan_moe_i2v_audio.json",
            "rs2v": "wan22/wan_moe_i2v_audio.json",
        },
    ),
    "wan2.2_animate": _spec(
        "wan_moe",
        ["animate"],
        "wan2.2_animate",
        model_family="video",
        checkpoint_formats=["directory", "state_dict"],
        features=["motion_guidance", "pose_conditioning"],
        default_configs={"animate": "wan22/wan_animate.json"},
    ),
    "seko_talk": _spec(
        "wan",
        ["s2v", "rs2v"],
        "seko_talk",
        model_family="audio_video",
        checkpoint_formats=["directory", "state_dict"],
        features=["audio_conditioning", "reference_speech"],
        default_configs={
            "s2v": "seko_talk/shot/stream/s2v.json",
            "rs2v": "seko_talk/shot/rs2v/rs2v.json",
        },
    ),
    "hunyuan_video_1.5": _spec(
        "hunyuan_video",
        ["t2v", "i2v"],
        "hunyuan_video_1.5",
        model_family="video",
        checkpoint_formats=["directory", "diffusers", "state_dict"],
        default_configs={
            "t2v": "hunyuan_video_15/hunyuan_video_t2v_480p.json",
            "i2v": "hunyuan_video_15/hunyuan_video_i2v_480p.json",
        },
    ),
    "hunyuan_video_1.5_distill": _spec(
        "hunyuan_video",
        ["t2v", "i2v"],
        "hunyuan_video_1.5_distill",
        model_family="video",
        checkpoint_formats=["directory", "diffusers", "state_dict"],
        teacher_model_cls="hunyuan_video_1.5",
        distill_stage="student",
        default_configs={"t2v": "hunyuan_video_15/hunyuan_video_t2v_480p_distill.json"},
    ),
    "worldplay_distill": _spec(
        "hunyuan_video",
        ["t2v", "i2v", "game"],
        "worldplay_distill",
        model_family="world_model",
        checkpoint_formats=["directory", "diffusers", "state_dict"],
        distill_stage="student",
        features=["action_conditioning", "rope_camera", "context_memory"],
        default_configs={
            "i2v": "worldplay/worldplay_distill_i2v_480p.json",
            "game": "worldplay/worldplay_distill_i2v_480p.json",
        },
    ),
    "worldplay_ar": _spec(
        "hunyuan_video",
        ["i2v", "game"],
        "worldplay_ar",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
        default_configs={
            "i2v": "worldplay/worldplay_ar_i2v_480p.json",
            "game": "worldplay/worldplay_ar_i2v_480p.json",
        },
    ),
    "worldplay_bi": _spec(
        "hunyuan_video",
        ["i2v", "game"],
        "worldplay_bi",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
        default_configs={
            "i2v": "worldplay/worldplay_bi_i2v_480p.json",
            "game": "worldplay/worldplay_bi_i2v_480p.json",
        },
    ),
    "skyreels_v2": _spec(
        "wan",
        ["t2v", "i2v"],
        "skyreels_v2",
        status="stub",
        model_family="video",
        checkpoint_formats=["directory", "diffusers", "state_dict"],
    ),
    "lingbot_cam_moe": _spec(
        "wan_moe",
        ["i2v"],
        "lingbot_cam_moe",
        model_family="world_model",
        checkpoint_formats=["directory", "dual_model", "quantized", "state_dict"],
        features=["plucker_camera_control"],
        default_configs={"i2v": "lingbot/lingbot_cam_moe_i2v.json"},
    ),
    "matrix_game_2": _spec(
        "wan",
        ["game"],
        "wan2.1_sf_mtxg2",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
        features=["self_forcing", "autoregressive"],
        default_configs={"game": "matrix_game2/matrix_game2_universal.json"},
    ),
    "gamecraft": _spec(
        "hunyuan_video",
        ["game"],
        "gamecraft",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "gamefactory": _spec(
        "custom",
        ["game"],
        "gamefactory",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "infinite_world": _spec(
        "custom",
        ["game", "open_world"],
        "infinite_world",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "genie": _spec(
        "custom_autoregressive",
        ["game"],
        "genie",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "gamegen_x": _spec(
        "custom",
        ["game"],
        "gamegen_x",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "vmem": _spec(
        "memory_augmented",
        ["long_video", "game"],
        "vmem",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "spmem": _spec(
        "memory_augmented",
        ["long_video", "game"],
        "spmem",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "cam": _spec(
        "memory_augmented",
        ["long_video"],
        "cam",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "mirage": _spec(
        "custom",
        ["game"],
        "mirage",
        status="stub",
        model_family="world_model",
        checkpoint_formats=["directory", "state_dict"],
    ),
    "qwen_image": _spec(
        "dit",
        ["t2i", "i2i"],
        "qwen_image",
        model_family="image",
        checkpoint_formats=["directory", "diffusers", "quantized", "state_dict"],
        default_configs={
            "t2i": "qwen_image/qwen_image_t2i_2512.json",
            "i2i": "qwen_image/qwen_image_i2i_2511.json",
        },
    ),
    "longcat_image": _spec(
        "dit",
        ["t2i", "i2i"],
        "longcat_image",
        model_family="image",
        checkpoint_formats=["directory", "diffusers", "state_dict"],
        default_configs={
            "t2i": "longcat_image/longcat_image_t2i.json",
            "i2i": "longcat_image/longcat_image_i2i.json",
        },
    ),
    "z_image": _spec(
        "dit",
        ["t2i", "i2i"],
        "z_image",
        model_family="image",
        checkpoint_formats=["directory", "diffusers", "quantized", "state_dict"],
        default_configs={"t2i": "z_image/z_image_turbo_t2i.json"},
    ),
    "ltx2": _spec(
        "dit",
        ["t2v", "i2v", "t2av", "i2av"],
        "ltx2",
        model_family="video",
        checkpoint_formats=["directory", "diffusers", "state_dict"],
        features=["audio_video_generation"],
        default_configs={
            "t2v": "ltx2/ltx2.json",
            "i2v": "ltx2/ltx2.json",
            "t2av": "ltx2/ltx2.json",
            "i2av": "ltx2/ltx2.json",
        },
    ),
    "bagel": _spec(
        "dit",
        ["t2i", "i2i"],
        "bagel",
        model_family="image",
        checkpoint_formats=["directory", "diffusers", "state_dict"],
        default_configs={
            "t2i": "bagel/bagel_t2i.json",
            "i2i": "bagel/bagel_t2i.json",
        },
    ),
}


def _canonical_key(value: str) -> str:
    return str(value or "").strip().lower().replace("/", "_")


@lru_cache(maxsize=1)
def load_model_zoo() -> dict[str, dict[str, Any]]:
    merged = copy.deepcopy(_FALLBACK_MODEL_ZOO)
    if yaml is not None and _MODEL_ZOO_PATH.exists():
        with _MODEL_ZOO_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        models = raw.get("models", {}) if isinstance(raw, dict) else {}
        if isinstance(models, dict):
            for name, spec in models.items():
                if isinstance(spec, dict):
                    merged[str(name)] = {**merged.get(str(name), {}), **copy.deepcopy(spec)}
    return merged


@lru_cache(maxsize=1)
def get_model_alias_map() -> dict[str, str]:
    alias_map = {_canonical_key(alias): canonical for alias, canonical in _DEFAULT_MODEL_ALIASES.items()}
    for canonical, spec in load_model_zoo().items():
        alias_map[_canonical_key(canonical)] = canonical
        for alias in spec.get("aliases", []) or []:
            alias_map[_canonical_key(alias)] = canonical
    return alias_map


def normalize_model_cls(model_cls: str) -> str:
    key = _canonical_key(model_cls)
    if not key:
        return model_cls
    return get_model_alias_map().get(key, model_cls)


@lru_cache(maxsize=1)
def get_supported_model_inputs() -> tuple[str, ...]:
    return tuple(sorted(get_model_alias_map().keys()))


def infer_checkpoint_format(model_path: str | os.PathLike[str] | None) -> str:
    if not model_path:
        return ""
    path = Path(model_path)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            return "gguf"
        if suffix in {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}:
            return "state_dict"
        return "file"

    if not path.exists():
        return ""
    if (path / "model_index.json").exists():
        return "diffusers"
    if (path / "high_noise_model").exists() or (path / "low_noise_model").exists():
        return "dual_model"
    if (path / "transformer").exists() and (path / "vae").exists():
        return "modular_directory"
    if (path / "transformer" / "config.json").exists() or (path / "config.json").exists():
        return "directory"
    if any(path.glob("*.safetensors")) or any(path.glob("*.pt")) or any(path.glob("*.bin")):
        return "state_dict_directory"
    return "directory"


def _infer_default_formats(canonical: str, architecture: str) -> list[str]:
    if canonical.startswith("wan2.2") and "moe" in canonical:
        return ["directory", "dual_model", "diffusers", "quantized", "state_dict"]
    if architecture in {"wan", "wan_dense", "wan_moe", "hunyuan_video", "dit"}:
        return ["directory", "diffusers", "state_dict"]
    return ["directory", "state_dict"]


def _infer_model_family(canonical: str, architecture: str, tasks: list[str]) -> str:
    if any(task in {"game", "open_world", "long_video"} for task in tasks):
        return "world_model"
    if any(task in {"t2i", "i2i"} for task in tasks) and not any(task in {"t2v", "i2v", "t2av", "i2av"} for task in tasks):
        return "image"
    if canonical.endswith("_audio") or "audio" in canonical or any(task in {"s2v", "rs2v", "t2av", "i2av"} for task in tasks):
        return "audio_video"
    if architecture in {"wan", "wan_dense", "wan_moe", "hunyuan_video"}:
        return "video"
    if architecture in {"dit", "unet"}:
        return "image"
    return "generic"


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _task_modalities(tasks: list[str], field: str) -> list[str]:
    values: list[str] = []
    for task in tasks:
        signature = _TASK_IO_SIGNATURES.get(task, {})
        for value in signature.get(field, []):
            values.append(value)
    return _unique_preserve_order(values)


def _normalize_default_configs(default_configs: Any) -> dict[str, list[str]]:
    if not isinstance(default_configs, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for task, value in default_configs.items():
        key = str(task)
        if isinstance(value, str):
            normalized[key] = [value]
        elif isinstance(value, (list, tuple)):
            normalized[key] = [str(item) for item in value if item]
    return normalized


def resolve_default_config_candidates(model_cls: str, task: str | None = None) -> list[str]:
    canonical = normalize_model_cls(model_cls)
    spec = copy.deepcopy(load_model_zoo().get(canonical, {}))
    default_configs = _normalize_default_configs(spec.get("default_configs"))
    ordered_keys: list[str] = []
    if task:
        ordered_keys.append(str(task))
    ordered_keys.extend(["default", "*"])

    candidates: list[str] = []
    seen: set[str] = set()
    for key in ordered_keys:
        for relative_path in default_configs.get(key, []):
            candidate = Path(relative_path)
            if not candidate.is_absolute():
                candidate = _DEFAULT_CONFIG_BASE / candidate
            normalized = str(candidate)
            if normalized not in seen:
                candidates.append(normalized)
                seen.add(normalized)
    return candidates


def resolve_default_config_path(model_cls: str, task: str | None = None) -> str:
    for candidate in resolve_default_config_candidates(model_cls, task=task):
        if os.path.exists(candidate):
            return candidate
    return ""


def _derive_teacher_model_cls(canonical: str, spec: dict[str, Any], zoo: dict[str, dict[str, Any]]) -> str:
    explicit_teacher = str(spec.get("teacher_model_cls") or "").strip()
    if explicit_teacher:
        normalized_teacher = normalize_model_cls(explicit_teacher)
        if normalized_teacher in zoo and normalized_teacher != canonical:
            return normalized_teacher
        if explicit_teacher in zoo and explicit_teacher != canonical:
            return explicit_teacher

    if canonical.endswith("_mean_flow_distill"):
        candidate = canonical.removesuffix("_mean_flow_distill")
        if candidate in zoo and candidate != canonical:
            return candidate
    if canonical.endswith("_distill"):
        candidate = canonical.removesuffix("_distill")
        if candidate in zoo and candidate != canonical:
            return candidate
    return ""


def _collect_related_distill_models(canonical: str, zoo: dict[str, dict[str, Any]]) -> list[str]:
    related: list[str] = []
    for candidate, candidate_spec in zoo.items():
        if candidate == canonical:
            continue
        if _derive_teacher_model_cls(candidate, candidate_spec, zoo) == canonical:
            related.append(candidate)
    return sorted(set(related))


def _infer_distill_stage(canonical: str, spec: dict[str, Any], zoo: dict[str, dict[str, Any]]) -> str:
    explicit_stage = str(spec.get("distill_stage") or "").strip()
    if explicit_stage:
        return explicit_stage
    if canonical.endswith("_distill") or canonical.endswith("_mean_flow_distill"):
        return "student"
    if _collect_related_distill_models(canonical, zoo):
        return "teacher"
    return "base"


def _infer_primary_modality(model_family: str, output_modalities: list[str]) -> str:
    output_set = set(output_modalities)
    if "video" in output_set and "audio" in output_set:
        return "audio_video"
    if "video" in output_set:
        return "video"
    if "image" in output_set:
        return "image"
    if "audio" in output_set:
        return "audio"
    if model_family == "world_model":
        return "video"
    return model_family or "generic"


def resolve_model_metadata(
    model_cls: str,
    *,
    model_path: str | os.PathLike[str] | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    canonical = normalize_model_cls(model_cls)
    zoo = load_model_zoo()
    spec = copy.deepcopy(zoo.get(canonical, {}))
    tasks = list(spec.get("tasks") or [])
    architecture = str(spec.get("architecture") or "unknown")
    checkpoint_formats = list(spec.get("checkpoint_formats") or _infer_default_formats(canonical, architecture))
    model_family = str(spec.get("model_family") or _infer_model_family(canonical, architecture, tasks))
    resolved_checkpoint_format = infer_checkpoint_format(model_path) or (checkpoint_formats[0] if checkpoint_formats else "unknown")
    alias_inputs = sorted(alias for alias, target in get_model_alias_map().items() if target == canonical)
    features = list(spec.get("features") or [])
    input_modalities = list(spec.get("input_modalities") or _task_modalities(tasks, "inputs"))
    output_modalities = list(spec.get("output_modalities") or _task_modalities(tasks, "outputs"))
    modalities = list(spec.get("modalities") or _unique_preserve_order(input_modalities + output_modalities))
    distill_stage = _infer_distill_stage(canonical, spec, zoo)
    teacher_model_cls = _derive_teacher_model_cls(canonical, spec, zoo)
    related_distill_models = _collect_related_distill_models(canonical, zoo)
    distill_methods = list(spec.get("distill_methods") or [])
    default_config_candidates = resolve_default_config_candidates(canonical, task=task)
    default_config_path = resolve_default_config_path(canonical, task=task)
    distill_runtime_hints: list[str] = []
    if distill_stage == "student":
        distill_runtime_hints.append("student_inference")
    if teacher_model_cls:
        distill_runtime_hints.append("teacher_student_pair")
    if "dual_model" in checkpoint_formats:
        distill_runtime_hints.append("dual_student_checkpoint")
    if any(method in {"progressive_distill", "stream_distill", "context_forcing"} for method in distill_methods):
        distill_runtime_hints.append("online_distill_candidate")
    if any(feature in features for feature in ("action_conditioning", "context_memory", "plucker_camera_control")):
        distill_runtime_hints.append("world_model_runtime")

    metadata = {
        **spec,
        "input_model_cls": model_cls,
        "canonical_model_cls": canonical,
        "runner_cls": spec.get("runner_cls", canonical),
        "architecture": architecture,
        "tasks": tasks,
        "status": spec.get("status", "unknown"),
        "model_family": model_family,
        "checkpoint_formats": checkpoint_formats,
        "checkpoint_format": resolved_checkpoint_format,
        "features": features,
        "aliases": alias_inputs,
        "supports_task": (task in tasks) if task and tasks else True,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "modalities": modalities,
        "primary_modality": str(spec.get("primary_modality") or _infer_primary_modality(model_family, output_modalities)),
        "distill_stage": distill_stage,
        "teacher_model_cls": teacher_model_cls,
        "related_distill_models": related_distill_models,
        "distill_methods": distill_methods,
        "distill_runtime_hints": distill_runtime_hints,
        "default_config_candidates": default_config_candidates,
        "default_config_path": default_config_path,
        "is_world_model": model_family == "world_model",
        "is_distilled_model": distill_stage == "student",
        "supports_action_conditioning": "action_conditioning" in features,
        "supports_camera_conditioning": any(
            feature in features for feature in ("plucker_camera_control", "rope_camera", "prope_camera")
        ),
        "supports_memory_context": any(
            feature in features for feature in ("context_memory", "visual_memory", "spatial_memory", "fov_retrieval")
        ),
        "supports_audio_conditioning": "audio" in input_modalities,
        "supports_audio_generation": "audio" in output_modalities,
        "supports_video_generation": "video" in output_modalities,
        "supports_image_generation": "image" in output_modalities,
        "supports_opd_like_runtime": any(
            method in {"progressive_distill", "stream_distill", "context_forcing"} for method in distill_methods
        ),
    }
    return metadata


def apply_model_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = resolve_model_metadata(
        str(config.get("model_cls", "")),
        model_path=config.get("model_path", ""),
        task=config.get("task"),
    )
    config["model_cls"] = metadata["canonical_model_cls"]
    config["canonical_model_cls"] = metadata["canonical_model_cls"]
    config["resolved_runner_cls"] = metadata["runner_cls"]
    config["model_architecture"] = metadata["architecture"]
    config["model_family"] = metadata["model_family"]
    config["model_status"] = metadata["status"]
    config["checkpoint_format"] = metadata["checkpoint_format"]
    config["supported_tasks"] = metadata["tasks"]
    config["model_features"] = metadata["features"]
    config["model_modalities"] = metadata["modalities"]
    config["input_modalities"] = metadata["input_modalities"]
    config["output_modalities"] = metadata["output_modalities"]
    config["primary_modality"] = metadata["primary_modality"]
    config["distill_stage"] = metadata["distill_stage"]
    config["teacher_model_cls"] = metadata["teacher_model_cls"]
    config["related_distill_models"] = metadata["related_distill_models"]
    config["distill_methods"] = metadata["distill_methods"]
    config["distill_runtime_hints"] = metadata["distill_runtime_hints"]
    config["default_config_candidates"] = metadata["default_config_candidates"]
    config["default_config_path"] = metadata["default_config_path"]
    config["is_world_model"] = metadata["is_world_model"]
    return metadata
