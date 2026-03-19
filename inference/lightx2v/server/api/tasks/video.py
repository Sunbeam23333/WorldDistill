import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from ...schema import TaskResponse, VideoTaskRequest
from ...task_manager import task_manager
from ..deps import get_services, validate_url_async

router = APIRouter()


def _write_file_sync(file_path: Path, content: bytes) -> None:
    with open(file_path, "wb") as buffer:
        buffer.write(content)


@router.post("/", response_model=TaskResponse)
async def create_video_task(message: VideoTaskRequest):
    try:
        if hasattr(message, "image_path") and message.image_path and message.image_path.startswith("http"):
            if not await validate_url_async(message.image_path):
                raise HTTPException(status_code=400, detail=f"Image URL is not accessible: {message.image_path}")

        task_id = task_manager.create_task(message)
        message.task_id = task_id

        return TaskResponse(
            task_id=task_id,
            task_status="pending",
            save_result_path=message.save_result_path,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create video task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/form", response_model=TaskResponse)
async def create_video_task_form(
    image_file: Optional[UploadFile] = File(default=None),
    prompt: str = Form(default=""),
    save_result_path: str = Form(default=""),
    use_prompt_enhancer: bool = Form(default=False),
    negative_prompt: str = Form(default=""),
    num_fragments: int = Form(default=1),
    infer_steps: int = Form(default=5),
    target_video_length: int = Form(default=81),
    seed: int = Form(default=42),
    audio_file: Optional[UploadFile] = File(default=None),
    video_duration: int = Form(default=5),
    target_fps: int = Form(default=16),
    resize_mode: str = Form(default="adaptive"),
    image_strength: Optional[float] = Form(default=None),
    pose: str = Form(default=""),
    last_frame_file: Optional[UploadFile] = File(default=None),
    src_ref_image_file: Optional[UploadFile] = File(default=None),
    src_video_file: Optional[UploadFile] = File(default=None),
    src_mask_file: Optional[UploadFile] = File(default=None),
    src_pose_file: Optional[UploadFile] = File(default=None),
    src_face_file: Optional[UploadFile] = File(default=None),
    src_bg_file: Optional[UploadFile] = File(default=None),
    src_mask_video_file: Optional[UploadFile] = File(default=None),
    action_file: Optional[UploadFile] = File(default=None),
):
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    async def save_file_async(file: Optional[UploadFile], target_dir: Path) -> str:
        if not file or not file.filename:
            return ""

        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = target_dir / unique_filename

        content = await file.read()
        await asyncio.to_thread(_write_file_sync, file_path, content)

        return str(file_path)

    image_path = await save_file_async(image_file, services.file_service.input_image_dir)
    audio_path = await save_file_async(audio_file, services.file_service.input_audio_dir)
    last_frame_path = await save_file_async(last_frame_file, services.file_service.input_image_dir)
    src_ref_images = await save_file_async(src_ref_image_file, services.file_service.input_image_dir)
    src_video = await save_file_async(src_video_file, services.file_service.input_video_dir)
    src_mask = await save_file_async(src_mask_file, services.file_service.input_video_dir)
    src_pose_path = await save_file_async(src_pose_file, services.file_service.input_video_dir)
    src_face_path = await save_file_async(src_face_file, services.file_service.input_video_dir)
    src_bg_path = await save_file_async(src_bg_file, services.file_service.input_video_dir)
    src_mask_path = await save_file_async(src_mask_video_file, services.file_service.input_video_dir)
    action_path = await save_file_async(action_file, services.file_service.input_video_dir)

    message = VideoTaskRequest(
        prompt=prompt,
        use_prompt_enhancer=use_prompt_enhancer,
        negative_prompt=negative_prompt,
        image_path=image_path,
        num_fragments=num_fragments,
        save_result_path=save_result_path,
        infer_steps=infer_steps,
        target_video_length=target_video_length,
        seed=seed,
        audio_path=audio_path,
        video_duration=video_duration,
        target_fps=target_fps,
        resize_mode=resize_mode,
        image_strength=image_strength,
        last_frame_path=last_frame_path,
        src_ref_images=src_ref_images,
        src_video=src_video,
        src_mask=src_mask,
        src_pose_path=src_pose_path,
        src_face_path=src_face_path,
        src_bg_path=src_bg_path,
        src_mask_path=src_mask_path,
        pose=pose,
        action_path=action_path,
    )

    try:
        task_id = task_manager.create_task(message)
        message.task_id = task_id

        return TaskResponse(
            task_id=task_id,
            task_status="pending",
            save_result_path=message.save_result_path,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create video form task: {e}")
        raise HTTPException(status_code=500, detail=str(e))
