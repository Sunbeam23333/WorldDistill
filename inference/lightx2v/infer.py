import argparse
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.runners.bagel.bagel_runner import BagelRunner  # noqa: F401
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_distill_runner import HunyuanVideo15DistillRunner  # noqa: F401
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner  # noqa: F401
from lightx2v.models.runners.longcat_image.longcat_image_runner import LongCatImageRunner  # noqa: F401
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner  # noqa: F401
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_matrix_game2_runner import WanSFMtxg2Runner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.lingbot_runner import LingBotCamMoeRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import Wan22MoeVaceRunner, WanVaceRunner  # noqa: F401
from lightx2v.models.runners.worldplay.worldplay_ar_runner import WorldPlayARRunner  # noqa: F401
from lightx2v.models.runners.worldplay.worldplay_bi_runner import WorldPlayBIRunner  # noqa: F401
from lightx2v.models.runners.worldplay.worldplay_distill_runner import WorldPlayDistillRunner  # noqa: F401
from lightx2v.models.runners.z_image.z_image_runner import ZImageRunner  # noqa: F401
# World model runners (stubs for upcoming models)
from lightx2v.models.runners.world_models import (  # noqa: F401
    SkyReelsV2Runner,
    GameCraftRunner,
    GameFactoryRunner,
    InfiniteWorldRunner,
    GenieRunner,
    GameGenXRunner,
    VMemRunner,
    SPMemRunner,
    CAMRunner,
    MirageRunner,
)
from lightx2v.utils.env_compat import validate_runtime_dependency_versions
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.model_catalog import get_supported_model_inputs, resolve_model_metadata
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_config_paths, validate_task_arguments
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def init_runner(config):
    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The seed for random generator")
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=list(get_supported_model_inputs()),
        default="wan2.1",
        help="Model key or alias. Canonical names and common aliases are both accepted.",
    )

    parser.add_argument("--task", type=str, choices=["t2v", "i2v", "t2i", "i2i", "flf2v", "vace", "animate", "s2v", "rs2v", "t2av", "i2av", "game"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sf_model_path", type=str, required=False)
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--transformer_model_name", type=str, default="")
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="The path to input image file(s) for image-to-video (i2v) or image-to-audio-video (i2av) task. Multiple paths should be comma-separated. Example: 'path1.jpg,path2.jpg'",
    )
    parser.add_argument("--last_frame_path", type=str, default="", help="The path to last frame file for first-last-frame-to-video (flf2v) task")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file or directory for audio-to-video (s2v) task")
    parser.add_argument("--image_strength", type=float, default=1.0, help="The strength of the image-to-audio-video (i2av) task")
    # [Warning] For vace task, need refactor.
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.",
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.",
    )
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument(
        "--src_pose_path",
        type=str,
        default=None,
        help="The file of the source pose. Default None.",
    )
    parser.add_argument(
        "--src_face_path",
        type=str,
        default=None,
        help="The file of the source face. Default None.",
    )
    parser.add_argument(
        "--src_bg_path",
        type=str,
        default=None,
        help="The file of the source background. Default None.",
    )
    parser.add_argument(
        "--src_mask_path",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="Pose string (e.g., 'w-3, right-0.5') or JSON file path for WorldPlay models.",
    )
    parser.add_argument(
        "--action_ckpt",
        type=str,
        default=None,
        help="Path to action model checkpoint for WorldPlay models.",
    )
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="Path to camera pose file (JSON/NPY/NPZ) for LingBot camera control.",
    )
    parser.add_argument("--save_result_path", type=str, default=None, help="The path to save generated result file (video/image/audio-video)")
    parser.add_argument("--return_result_tensor", action="store_true", help="Whether to return result tensor. (Useful for comfyui)")
    parser.add_argument("--target_shape", nargs="+", default=[], help="Set return video or image shape")
    parser.add_argument("--aspect_ratio", type=str, default="")

    args = parser.parse_args()
    validate_runtime_dependency_versions()
    validate_task_arguments(args)

    metadata = resolve_model_metadata(args.model_cls, model_path=args.model_path, task=args.task)
    logger.info(
        "Resolved model metadata | canonical={} | architecture={} | family={} | checkpoint={} | runner={}",
        metadata["canonical_model_cls"],
        metadata["architecture"],
        metadata["model_family"],
        metadata["checkpoint_format"],
        metadata["runner_cls"],
    )
    logger.info(
        "Execution profile | primary_modality={} | inputs={} | outputs={} | distill_stage={} | teacher_ref={} | default_config={}",
        metadata["primary_modality"],
        metadata["input_modalities"],
        metadata["output_modalities"],
        metadata["distill_stage"],
        metadata["teacher_model_cls"] or "-",
        metadata["default_config_path"] or "-",
    )

    seed_all(args.seed)

    # set config
    config = set_config(args)
    # init input_info
    input_info = init_empty_input_info(args.task)

    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    print_config(config)

    validate_config_paths(config)

    with ProfilingContext4DebugL1("Total Cost"):
        # init runner
        runner = init_runner(config)
        # start to infer
        data = args.__dict__
        update_input_info_from_dict(input_info, data)
        runner.run_pipeline(input_info)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
