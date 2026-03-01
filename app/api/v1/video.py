"""
Video Generation API 路由

兼容 OpenAI Sora 风格的视频 API:
  POST /v1/videos          — 创建视频生成任务（异步）
  POST /v1/video/create    — 备用端点（同上）
  GET  /v1/videos/{id}     — 查询任务状态 / 获取结果
"""

import asyncio
import re
import time
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_config
from app.core.exceptions import AppException, ErrorType, ValidationException
from app.core.logger import logger
from app.services.grok.services.model import ModelService
from app.services.grok.services.video import VideoService
from app.services.grok.utils.upload import UploadService
from app.services.token import get_token_manager
from app.services.token.manager import BASIC_POOL_NAME

router = APIRouter(tags=["Video"])

# ---------------------------------------------------------------------------
# In-memory task store  (sufficient for single-process deployment)
# ---------------------------------------------------------------------------

_tasks: Dict[str, Dict[str, Any]] = {}
_MAX_TASKS = 500          # Prevent unbounded growth
_TASK_TTL_SECONDS = 3600  # Auto-expire completed tasks after 1 hour

# Maximum reference images for video generation
_MAX_VIDEO_REFERENCE_IMAGES = 3


def _cleanup_expired() -> None:
    """Remove tasks older than TTL."""
    now = time.time()
    expired = [
        tid for tid, t in _tasks.items()
        if t.get("status") in ("completed", "failed")
        and now - t.get("created_at", 0) > _TASK_TTL_SECONDS
    ]
    for tid in expired:
        _tasks.pop(tid, None)


# ---------------------------------------------------------------------------
# Background worker — runs the streaming video generation and collects result
# ---------------------------------------------------------------------------

async def _run_video_task(task_id: str, params: Dict[str, Any]) -> None:
    """Execute video generation in the background and update task store."""
    try:
        _tasks[task_id]["status"] = "in_progress"

        prompt = params["prompt"]
        model = params.get("model", "grok-imagine-1.0-video")
        image_url = params.get("image_url")
        aspect_ratio = params.get("aspect_ratio", "3:2")
        video_length = params.get("video_length", 6)
        resolution = params.get("resolution", "480p")
        preset = params.get("preset", "normal")

        # ---- Token selection ------------------------------------------------
        token_mgr = await get_token_manager()
        await token_mgr.reload_if_stale()

        pool_candidates = ModelService.pool_candidates_for_model(model)
        token_info = token_mgr.get_token_for_video(
            resolution=resolution,
            video_length=video_length,
            pool_candidates=pool_candidates,
        )
        if not token_info:
            _tasks[task_id].update(status="failed", error="No available tokens")
            return

        token = token_info.token
        if token.startswith("sso="):
            token = token[4:]

        # ---- Upload reference image(s) if provided --------------------------
        uploaded_image_urls: List[str] = []
        if image_url:
            # image_url can be a single URL string
            raw_images = [image_url] if isinstance(image_url, str) else list(image_url)
            images_to_upload = raw_images[:_MAX_VIDEO_REFERENCE_IMAGES]

            upload_service = UploadService()
            try:
                for idx, img_data in enumerate(images_to_upload):
                    try:
                        _, file_uri = await upload_service.upload_file(img_data, token)
                        url = f"https://assets.grok.com/{file_uri}"
                        uploaded_image_urls.append(url)
                        logger.info(f"[VideoTask {task_id}] Image {idx+1} uploaded: {url}")
                    except Exception as e:
                        logger.warning(f"[VideoTask {task_id}] Image {idx+1} upload failed: {e}")
            finally:
                await upload_service.close()

        # ---- Generate video --------------------------------------------------
        service = VideoService()
        if uploaded_image_urls:
            response: AsyncIterable[bytes] = await service.generate_from_images(
                token, prompt, uploaded_image_urls,
                aspect_ratio, video_length, resolution, preset,
            )
        else:
            response = await service.generate(
                token, prompt, aspect_ratio, video_length, resolution, preset,
            )

        # ---- Consume the stream and extract video URL ------------------------
        import orjson
        video_url = ""
        thumbnail_url = ""

        async for raw_line in response:
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace").strip()
            else:
                line = str(raw_line).strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
            except Exception:
                continue

            resp = data.get("result", {}).get("response", {})
            video_resp = resp.get("streamingVideoGenerationResponse")
            if video_resp:
                progress = video_resp.get("progress", 0)
                _tasks[task_id]["progress"] = progress
                if progress == 100:
                    video_url = video_resp.get("videoUrl", "")
                    thumbnail_url = video_resp.get("thumbnailImageUrl", "")

        # ---- Upscale if needed -----------------------------------------------
        pool_name = token_mgr.get_pool_name_for_token(token)
        should_upscale = resolution == "720p" and pool_name == BASIC_POOL_NAME
        if video_url and should_upscale:
            try:
                from app.services.reverse.video_upscale import VideoUpscaleReverse
                from app.services.reverse.utils.session import ResettableSession

                vid_match = re.search(r"/generated/([0-9a-fA-F-]{32,36})/", video_url)
                if not vid_match:
                    vid_match = re.search(r"/([0-9a-fA-F-]{32,36})/generated_video", video_url)
                if vid_match:
                    browser = get_config("proxy.browser")
                    session = ResettableSession(impersonate=browser) if browser else ResettableSession()
                    try:
                        resp_upscale = await VideoUpscaleReverse.request(session, token, vid_match.group(1))
                        payload = resp_upscale.json() if resp_upscale else {}
                        hd_url = payload.get("hdMediaUrl") if isinstance(payload, dict) else None
                        if hd_url:
                            video_url = hd_url
                            logger.info(f"[VideoTask {task_id}] Upscaled: {hd_url}")
                    finally:
                        await session.close()
            except Exception as e:
                logger.warning(f"[VideoTask {task_id}] Upscale failed: {e}")

        if video_url:
            _tasks[task_id].update(
                status="completed",
                video_url=video_url,
                thumbnail_url=thumbnail_url,
            )
            logger.info(f"[VideoTask {task_id}] Completed: {video_url}")
        else:
            _tasks[task_id].update(status="failed", error="No video URL in response")
            logger.warning(f"[VideoTask {task_id}] No video URL found")

    except Exception as e:
        logger.error(f"[VideoTask {task_id}] Error: {e}")
        _tasks[task_id].update(
            status="failed",
            error=str(e) or "Video generation failed",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_aspect_ratio(value: Optional[str]) -> str:
    """Accept both pixel sizes and ratio strings."""
    if not value:
        return "3:2"
    ratio_map = {
        "1280x720": "16:9", "720x1280": "9:16",
        "1792x1024": "3:2", "1024x1792": "2:3",
        "1024x1024": "1:1",
        "16:9": "16:9", "9:16": "9:16",
        "3:2": "3:2", "2:3": "2:3", "1:1": "1:1",
        "4:3": "3:2", "3:4": "2:3",
        "21:9": "16:9", "9:21": "9:16",
    }
    return ratio_map.get(value, "3:2")


def _normalize_seconds(value: Any) -> int:
    """Normalise duration to one of the supported values."""
    if value is None:
        return 6
    try:
        s = int(value)
    except (ValueError, TypeError):
        return 6
    if s <= 6:
        return 6
    if s <= 10:
        return 10
    return 15


def _build_task_response(task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Build a response dict matching OpenAI video task format."""
    result: Dict[str, Any] = {
        "id": task_id,
        "object": "video",
        "status": task["status"],
        "created_at": int(task.get("created_at", 0)),
    }
    if task["status"] == "completed":
        result["video_url"] = task.get("video_url", "")
    if task["status"] == "failed":
        result["error"] = {"message": task.get("error", "Unknown error")}
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/videos")
@router.post("/video/create")
async def create_video(request: Request):
    """
    Create a video generation task (async).

    Accepts JSON body with:
      - prompt (str, required)
      - model (str, optional, default grok-imagine-1.0-video)
      - image_url (str, optional — reference image URL or data URI)
      - input_reference (ignored, use image_url for non-SDK callers)
      - seconds / duration (int, optional — 6/10/15)
      - size (str, optional — pixel size like 1280x720)
      - aspect_ratio (str, optional — like 16:9)
    """
    # Clean up old tasks
    _cleanup_expired()
    if len(_tasks) >= _MAX_TASKS:
        _cleanup_expired()
        if len(_tasks) >= _MAX_TASKS:
            raise AppException(
                message="Too many pending tasks. Please try again later.",
                error_type=ErrorType.RATE_LIMIT.value,
                code="rate_limit_exceeded",
                status_code=429,
            )

    # Parse body — handle both JSON and multipart/form-data
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        body: Dict[str, Any] = {}
        for key in form:
            values = form.getlist(key)
            if len(values) == 1:
                v = values[0]
                # Check if it's a file upload (input_reference from OpenAI SDK)
                if hasattr(v, "read") and hasattr(v, "filename"):
                    import base64
                    file_bytes = await v.read()
                    ct = getattr(v, "content_type", "image/png") or "image/png"
                    b64 = base64.b64encode(file_bytes).decode()
                    body[key] = f"data:{ct};base64,{b64}"
                else:
                    body[key] = v
            else:
                body[key] = [v for v in values]
        # Map 'input_reference' (OpenAI SDK field name) to 'image_url'
        if "input_reference" in body and "image_url" not in body:
            body["image_url"] = body.pop("input_reference")
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}

    prompt = body.get("prompt", "")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValidationException(
            message="prompt is required",
            param="prompt",
            code="missing_prompt",
        )
    prompt = prompt.strip()

    model = body.get("model", "grok-imagine-1.0-video")
    if not ModelService.valid(model):
        model = "grok-imagine-1.0-video"

    model_info = ModelService.get(model)
    if not model_info or not model_info.is_video:
        model = "grok-imagine-1.0-video"

    image_url = body.get("image_url")
    size = body.get("size")
    raw_ratio = body.get("aspect_ratio") or body.get("aspectRatio") or size
    aspect_ratio = _normalize_aspect_ratio(raw_ratio)
    seconds = _normalize_seconds(body.get("seconds") or body.get("duration"))

    resolution = "480p"
    if size:
        if "1792" in str(size) or "1080" in str(size):
            resolution = "720p"

    # Create task
    task_id = f"video-{uuid.uuid4().hex[:24]}"
    _tasks[task_id] = {
        "status": "queued",
        "created_at": time.time(),
        "video_url": "",
        "thumbnail_url": "",
        "error": "",
        "progress": 0,
    }

    # Start background task
    params = {
        "prompt": prompt,
        "model": model,
        "image_url": image_url,
        "aspect_ratio": aspect_ratio,
        "video_length": seconds,
        "resolution": resolution,
        "preset": "custom",
    }
    asyncio.create_task(_run_video_task(task_id, params))

    logger.info(
        f"Video task created: id={task_id}, model={model}, "
        f"ratio={aspect_ratio}, length={seconds}s, has_image={bool(image_url)}"
    )

    return JSONResponse(
        content=_build_task_response(task_id, _tasks[task_id]),
        status_code=200,
    )


@router.get("/videos/{video_id}")
async def get_video(video_id: str):
    """
    Query video generation task status.

    Returns OpenAI-compatible response with status, video_url, etc.
    """
    task = _tasks.get(video_id)
    if not task:
        raise AppException(
            message=f"Video task '{video_id}' not found",
            error_type=ErrorType.NOT_FOUND.value,
            code="not_found",
            status_code=404,
        )

    return JSONResponse(content=_build_task_response(video_id, task))


__all__ = ["router"]
