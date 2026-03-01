"""
Image Generation API 路由
"""

import base64
import time
from pathlib import Path
from typing import List, Optional, Union

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from app.services.grok.services.image import ImageGenerationService
from app.services.grok.services.image_edit import ImageEditService
from app.services.grok.services.model import ModelService
from app.services.token import get_token_manager
from app.core.exceptions import ValidationException, AppException, ErrorType
from app.core.config import get_config
from app.core.logger import logger

router = APIRouter(tags=["Images"])

# Canonical sizes supported by this API layer (OpenAI-compatible).
ALLOWED_IMAGE_SIZES = {
    "1280x720",
    "720x1280",
    "1792x1024",
    "1024x1792",
    "1024x1024",
}

SIZE_TO_ASPECT = {
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1792x1024": "3:2",
    "1024x1792": "2:3",
    "1024x1024": "1:1",
}
ALLOWED_ASPECT_RATIOS = {"1:1", "2:3", "3:2", "9:16", "16:9"}

# Reverse mapping (choose a canonical size for a given aspect ratio).
ASPECT_TO_SIZE = {
    "16:9": "1280x720",
    "9:16": "720x1280",
    "3:2": "1792x1024",
    "2:3": "1024x1792",
    "1:1": "1024x1024",
}


class ImageGenerationRequest(BaseModel):
    """图片生成请求 - OpenAI 兼容"""

    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(..., description="图片描述")
    model: Optional[str] = Field("grok-imagine-1.0", description="模型名称")
    n: Optional[int] = Field(1, ge=1, le=10, description="生成数量 (1-10)")
    size: Optional[str] = Field(
        "1024x1024",
        description="图片尺寸: 1280x720, 720x1280, 1792x1024, 1024x1792, 1024x1024 或直接传比例 9:16/16:9/1:1",
    )
    # Non-OpenAI field, accepted for compatibility with upstream callers.
    resolution: Optional[str] = Field(
        None,
        description="尺寸别名，等价于 size（例如 720x1280 或 9:16）",
    )
    # Non-OpenAI field, accepted for compatibility with upstream callers (e.g. waoowaoo)
    aspect_ratio: Optional[str] = Field(
        None,
        alias="aspectRatio",
        description="图片比例: 1:1, 2:3, 3:2, 9:16, 16:9 (优先级高于 size)",
    )

    quality: Optional[str] = Field("standard", description="图片质量 (暂不支持)")
    response_format: Optional[str] = Field(None, description="响应格式")
    style: Optional[str] = Field(None, description="风格 (暂不支持)")
    stream: Optional[bool] = Field(False, description="是否流式输出")


class ImageEditRequest(BaseModel):
    """图片编辑请求 - OpenAI 兼容"""

    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(..., description="编辑描述")
    model: Optional[str] = Field("grok-imagine-1.0-edit", description="模型名称")
    image: Optional[Union[str, List[str]]] = Field(None, description="待编辑图片文件")
    n: Optional[int] = Field(1, ge=1, le=10, description="生成数量 (1-10)")
    size: Optional[str] = Field(
        "1024x1024",
        description="图片尺寸: 1280x720, 720x1280, 1792x1024, 1024x1792, 1024x1024 或直接传比例 9:16/16:9/1:1",
    )
    # Non-OpenAI field, accepted for compatibility with upstream callers.
    resolution: Optional[str] = Field(
        None,
        description="尺寸别名，等价于 size（例如 720x1280 或 9:16）",
    )
    # Non-OpenAI field, accepted for compatibility with upstream callers.
    aspect_ratio: Optional[str] = Field(
        None,
        alias="aspectRatio",
        description="图片比例: 1:1, 2:3, 3:2, 9:16, 16:9 (优先级高于 size)",
    )

    quality: Optional[str] = Field("standard", description="图片质量 (暂不支持)")
    response_format: Optional[str] = Field(None, description="响应格式")
    style: Optional[str] = Field(None, description="风格 (暂不支持)")
    stream: Optional[bool] = Field(False, description="是否流式输出")


def _is_aspect_ratio(value: str) -> bool:
    if not value:
        return False
    v = str(value).strip()
    if v in ALLOWED_ASPECT_RATIOS:
        return True
    if ":" not in v:
        return False
    try:
        left, right = v.split(":", 1)
        left_i = int(left.strip())
        right_i = int(right.strip())
        if left_i <= 0 or right_i <= 0:
            return False
        normalized = f"{left_i}:{right_i}"
        return normalized in ALLOWED_ASPECT_RATIOS
    except (TypeError, ValueError):
        return False


def _normalize_size_and_aspect(*, size: Optional[str], aspect_ratio: Optional[str]):
    """Return (normalized_size, normalized_aspect_ratio).

    - Accepts OpenAI-like size (e.g. 720x1280)
    - Also accepts ratio-like size (e.g. 9:16)
    - Also accepts explicit aspect_ratio/aspectRatio which takes precedence

    This is the key compatibility bridge for callers that only know about
    `aspectRatio` (like storyboard/video pipelines) while still supporting
    OpenAI clients that use `size`.
    """
    size_val = (size or "").strip()
    aspect_val = (aspect_ratio or "").strip()

    if aspect_val and _is_aspect_ratio(aspect_val):
        ar = resolve_aspect_ratio(aspect_val)
        # If caller didn't specify a real pixel size (or sent a ratio in size),
        # choose a canonical pixel size for metadata/validation.
        if (not size_val) or _is_aspect_ratio(size_val) or (size_val not in ALLOWED_IMAGE_SIZES):
            return ASPECT_TO_SIZE.get(ar, "1024x1024"), ar
        return size_val, ar

    # No explicit aspect_ratio provided; interpret size.
    if _is_aspect_ratio(size_val):
        ar = resolve_aspect_ratio(size_val)
        return ASPECT_TO_SIZE.get(ar, "1024x1024"), ar

    # Standard OpenAI size.
    ar = resolve_aspect_ratio(size_val)
    if not size_val:
        size_val = "1024x1024"
    return size_val, ar


def _validate_common_request(
    request: Union[ImageGenerationRequest, ImageEditRequest],
    *,
    allow_ws_stream: bool = False,
):
    """通用参数校验"""
    # 验证 prompt
    if not request.prompt or not request.prompt.strip():
        raise ValidationException(
            message="Prompt cannot be empty", param="prompt", code="empty_prompt"
        )

    # 验证 n 参数范围
    if request.n < 1 or request.n > 10:
        raise ValidationException(
            message="n must be between 1 and 10", param="n", code="invalid_n"
        )

    # 流式只支持 n=1 或 n=2
    if request.stream and request.n not in [1, 2]:
        raise ValidationException(
            message="Streaming is only supported when n=1 or n=2",
            param="stream",
            code="invalid_stream_n",
        )

    if allow_ws_stream:
        if request.stream and request.response_format:
            allowed_stream_formats = {"b64_json", "base64", "url"}
            if request.response_format not in allowed_stream_formats:
                raise ValidationException(
                    message="Streaming only supports response_format=b64_json/base64/url",
                    param="response_format",
                    code="invalid_response_format",
                )

    if request.response_format:
        allowed_formats = {"b64_json", "base64", "url"}
        if request.response_format not in allowed_formats:
            raise ValidationException(
                message=f"response_format must be one of {sorted(allowed_formats)}",
                param="response_format",
                code="invalid_response_format",
            )

    # NOTE: for compatibility, `size` can also be an aspect ratio string like "9:16".
    if request.size:
        size_val = str(request.size).strip()
        if size_val not in ALLOWED_IMAGE_SIZES and not _is_aspect_ratio(size_val):
            raise ValidationException(
                message=(
                    f"size must be one of {sorted(ALLOWED_IMAGE_SIZES)} "
                    f"or an aspect ratio in {sorted(ALLOWED_ASPECT_RATIOS)}"
                ),
                param="size",
                code="invalid_size",
            )

    # resolution 作为 size 的别名同样校验
    if getattr(request, "resolution", None):
        resolution_val = str(getattr(request, "resolution")).strip()
        if resolution_val not in ALLOWED_IMAGE_SIZES and not _is_aspect_ratio(resolution_val):
            raise ValidationException(
                message=(
                    f"resolution must be one of {sorted(ALLOWED_IMAGE_SIZES)} "
                    f"or an aspect ratio in {sorted(ALLOWED_ASPECT_RATIOS)}"
                ),
                param="resolution",
                code="invalid_resolution",
            )

    # Validate optional explicit aspect_ratio as well.
    if getattr(request, "aspect_ratio", None):
        ar_val = str(getattr(request, "aspect_ratio")).strip()
        if ar_val and not _is_aspect_ratio(ar_val):
            raise ValidationException(
                message=f"aspect_ratio must be one of {sorted(ALLOWED_ASPECT_RATIOS)}",
                param="aspect_ratio",
                code="invalid_aspect_ratio",
            )


def validate_generation_request(request: ImageGenerationRequest):
    """验证图片生成请求参数"""
    if request.model != "grok-imagine-1.0":
        raise ValidationException(
            message="The model `grok-imagine-1.0` is required for image generation.",
            param="model",
            code="model_not_supported",
        )
    # 验证模型 - 通过 is_image 检查
    model_info = ModelService.get(request.model)
    if not model_info or not model_info.is_image:
        # 获取支持的图片模型列表
        image_models = [m.model_id for m in ModelService.MODELS if m.is_image]
        raise ValidationException(
            message=(
                f"The model `{request.model}` is not supported for image generation. "
                f"Supported: {image_models}"
            ),
            param="model",
            code="model_not_supported",
        )
    _validate_common_request(request, allow_ws_stream=True)


def resolve_response_format(response_format: Optional[str]) -> str:
    """解析响应格式"""
    fmt = response_format or get_config("app.image_format")
    if isinstance(fmt, str):
        fmt = fmt.lower()
    if fmt in ("b64_json", "base64", "url"):
        return fmt
    raise ValidationException(
        message="response_format must be one of b64_json, base64, url",
        param="response_format",
        code="invalid_response_format",
    )


def response_field_name(response_format: str) -> str:
    """获取响应字段名"""
    return {"url": "url", "base64": "base64"}.get(response_format, "b64_json")


def resolve_aspect_ratio(size: str) -> str:
    """Map OpenAI size to Grok Imagine aspect ratio."""
    value = (size or "").strip()
    if not value:
        return "2:3"
    if value in SIZE_TO_ASPECT:
        return SIZE_TO_ASPECT[value]
    if ":" in value:
        try:
            left, right = value.split(":", 1)
            left_i = int(left.strip())
            right_i = int(right.strip())
            if left_i > 0 and right_i > 0:
                ratio = f"{left_i}:{right_i}"
                if ratio in ALLOWED_ASPECT_RATIOS:
                    return ratio
        except (TypeError, ValueError):
            pass
    return "2:3"


def _detect_image_dimensions(content: bytes, mime: str) -> tuple[Optional[int], Optional[int]]:
    """Best-effort parse image dimensions without extra deps (png/jpeg/webp)."""
    try:
        # PNG
        if content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) >= 24:
            w = int.from_bytes(content[16:20], "big")
            h = int.from_bytes(content[20:24], "big")
            if w > 0 and h > 0:
                return w, h

        # JPEG
        if content.startswith(b"\xff\xd8"):
            i = 2
            end = len(content)
            while i + 9 < end:
                if content[i] != 0xFF:
                    i += 1
                    continue
                marker = content[i + 1]
                i += 2
                # standalone markers
                if marker in (0xD8, 0xD9):
                    continue
                if i + 1 >= end:
                    break
                seg_len = int.from_bytes(content[i:i + 2], "big")
                if seg_len < 2 or i + seg_len > end:
                    break
                # SOF markers
                if marker in {
                    0xC0, 0xC1, 0xC2, 0xC3,
                    0xC5, 0xC6, 0xC7,
                    0xC9, 0xCA, 0xCB,
                    0xCD, 0xCE, 0xCF,
                }:
                    if i + 7 < end:
                        h = int.from_bytes(content[i + 3:i + 5], "big")
                        w = int.from_bytes(content[i + 5:i + 7], "big")
                        if w > 0 and h > 0:
                            return w, h
                    break
                i += seg_len

        # WEBP (basic VP8X / VP8L parse)
        if len(content) >= 30 and content[0:4] == b"RIFF" and content[8:12] == b"WEBP":
            chunk = content[12:16]
            if chunk == b"VP8X" and len(content) >= 30:
                w = 1 + int.from_bytes(content[24:27], "little")
                h = 1 + int.from_bytes(content[27:30], "little")
                if w > 0 and h > 0:
                    return w, h
            if chunk == b"VP8L" and len(content) >= 25:
                b0 = content[21]
                b1 = content[22]
                b2 = content[23]
                b3 = content[24]
                w = 1 + (((b1 & 0x3F) << 8) | b0)
                h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
                if w > 0 and h > 0:
                    return w, h
    except Exception:
        pass

    return None, None


def _closest_allowed_aspect_ratio(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if not width or not height or width <= 0 or height <= 0:
        return None

    target = width / height
    ratio_values = {
        "1:1": 1.0,
        "2:3": 2 / 3,
        "3:2": 3 / 2,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
    }
    best = min(ratio_values.items(), key=lambda kv: abs(kv[1] - target))[0]
    return best


def validate_edit_request(request: ImageEditRequest, images: list):
    """验证图片编辑请求参数"""
    if request.model != "grok-imagine-1.0-edit":
        raise ValidationException(
            message=("The model `grok-imagine-1.0-edit` is required for image edits."),
            param="model",
            code="model_not_supported",
        )
    model_info = ModelService.get(request.model)
    if not model_info or not model_info.is_image_edit:
        edit_models = [m.model_id for m in ModelService.MODELS if m.is_image_edit]
        raise ValidationException(
            message=(
                f"The model `{request.model}` is not supported for image edits. "
                f"Supported: {edit_models}"
            ),
            param="model",
            code="model_not_supported",
        )
    _validate_common_request(request, allow_ws_stream=False)
    if not images:
        raise ValidationException(
            message="Image is required",
            param="image",
            code="missing_image",
        )
    if len(images) > 16:
        raise ValidationException(
            message="Too many images. Maximum is 16.",
            param="image",
            code="invalid_image_count",
        )


async def _get_token(model: str):
    """获取可用 token"""
    token_mgr = await get_token_manager()
    await token_mgr.reload_if_stale()

    token = None
    for pool_name in ModelService.pool_candidates_for_model(model):
        token = token_mgr.get_token(pool_name)
        if token:
            break

    if not token:
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429,
        )

    return token_mgr, token


# ---------------------------------------------------------------------------
# Accepted multipart field names for image uploads.
# Many clients (OpenAI SDK, curl, etc.) send multiple files as "image[]"
# instead of repeating "image".  We accept both.
# ---------------------------------------------------------------------------
_IMAGE_FIELD_NAMES = {"image", "image[]"}


def _is_upload_file(obj) -> bool:
    """Check whether *obj* is an UploadFile using duck typing.

    We cannot rely on ``isinstance(obj, UploadFile)`` because FastAPI
    re-exports ``UploadFile`` from ``fastapi`` while Starlette internally
    creates instances from ``starlette.datastructures.UploadFile``.
    When these two import paths diverge the ``isinstance`` check fails
    even though the object is perfectly usable as an UploadFile.
    """
    return hasattr(obj, "read") and hasattr(obj, "filename")


@router.post("/images/generations")
async def create_image(request: ImageGenerationRequest):
    """
    Image Generation API

    流式响应格式:
    - event: image_generation.partial_image
    - event: image_generation.completed

    非流式响应格式:
    - {"created": ..., "data": [{"b64_json": "..."}], "usage": {...}}
    """
    # stream 默认为 false
    if request.stream is None:
        request.stream = False

    if request.response_format is None:
        request.response_format = resolve_response_format(None)

    # 参数验证
    validate_generation_request(request)

    # 兼容 base64/b64_json
    if request.response_format == "base64":
        request.response_format = "b64_json"

    response_format = resolve_response_format(request.response_format)
    response_field = response_field_name(response_format)

    logger.info(
        f"[request-trace] image_gen.raw_body body={request.model_dump(by_alias=True)}"
    )

    logger.info(
        f"[aspect-ratio-trace] image_gen.request model={request.model} size={request.size} resolution={request.resolution} aspect_ratio={request.aspect_ratio} n={request.n} stream={request.stream}"
    )

    # compatibility: when caller sends only resolution, treat it as size.
    normalize_size_input = request.size
    if request.resolution and str(request.resolution).strip():
        normalize_size_input = str(request.resolution).strip()

    # Normalize size/aspect ratio (compat: accept aspectRatio and ratio-like size)
    normalized_size, aspect_ratio = _normalize_size_and_aspect(
        size=normalize_size_input,
        aspect_ratio=request.aspect_ratio,
    )

    logger.info(
        f"[aspect-ratio-trace] image_gen.normalized model={request.model} normalized_size={normalized_size} normalized_aspect_ratio={aspect_ratio}"
    )

    # 获取 token 和模型信息
    token_mgr, token = await _get_token(request.model)
    model_info = ModelService.get(request.model)

    result = await ImageGenerationService().generate(
        token_mgr=token_mgr,
        token=token,
        model_info=model_info,
        prompt=request.prompt,
        n=request.n,
        response_format=response_format,
        size=normalized_size,
        aspect_ratio=aspect_ratio,
        stream=bool(request.stream),
    )

    if result.stream:
        return StreamingResponse(
            result.data,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    data = [{response_field: img} for img in result.data]
    usage = result.usage_override or {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
    }

    return JSONResponse(
        content={
            "created": int(time.time()),
            "data": data,
            "usage": usage,
        }
    )


@router.post("/images/edits")
async def edit_image(request: Request):
    """
    Image Edits API

    同官方 API 格式，仅支持 multipart/form-data 文件上传。
    兼容 'image' 和 'image[]' 两种字段名，支持单图和多图上传。

    兼容额外字段:
    - aspect_ratio / aspectRatio: 9:16, 16:9 ... (优先级高于 size)
    - size 也支持直接传比例字符串，例如 size=9:16
    """
    # ------------------------------------------------------------------
    # 1. 手动解析 multipart form data
    # ------------------------------------------------------------------
    form = await request.form()

    # 完整请求体日志（文本字段 + 文件元信息）
    raw_text_fields = {}
    raw_file_fields = []
    for key in form:
        values = form.getlist(key)
        text_values = []
        for value in values:
            if _is_upload_file(value):
                raw_file_fields.append(
                    {
                        "field": key,
                        "filename": getattr(value, "filename", None),
                        "content_type": getattr(value, "content_type", None),
                    }
                )
            else:
                text_values.append(str(value))
        if text_values:
            raw_text_fields[key] = text_values if len(text_values) > 1 else text_values[0]

    logger.info(
        f"[request-trace] image_edit.raw_form text_fields={raw_text_fields} file_fields={raw_file_fields}"
    )

    # 提取文本字段（带默认值）
    prompt = form.get("prompt")
    if not prompt or not str(prompt).strip():
        raise ValidationException(
            message="Prompt is required",
            param="prompt",
            code="missing_prompt",
        )
    prompt = str(prompt).strip()

    model = str(form.get("model", "grok-imagine-1.0-edit"))
    n = int(form.get("n", 1))

    size_raw = form.get("size")
    resolution_raw = form.get("resolution")
    effective_size_raw = size_raw if size_raw is not None else resolution_raw
    has_size_field = effective_size_raw is not None
    size = str(effective_size_raw).strip() if effective_size_raw is not None else "1024x1024"

    # Compatibility: accept aspect_ratio/aspectRatio in multipart
    aspect_ratio_raw = form.get("aspect_ratio")
    if aspect_ratio_raw is None:
        aspect_ratio_raw = form.get("aspectRatio")
    aspect_ratio_val = str(aspect_ratio_raw).strip() if aspect_ratio_raw else None

    logger.info(
        f"[aspect-ratio-trace] image_edit.form_keys keys={list(form.keys())}"
    )
    logger.info(
        f"[aspect-ratio-trace] image_edit.form model={model} has_size_field={has_size_field} raw_size={size_raw} raw_resolution={resolution_raw} effective_size={effective_size_raw} raw_aspect_ratio={form.get('aspect_ratio')} raw_aspectRatio={form.get('aspectRatio')} n={n}"
    )

    quality = str(form.get("quality", "standard"))
    response_format_raw = form.get("response_format")
    response_format_val = str(response_format_raw) if response_format_raw else None
    style_raw = form.get("style")
    style = str(style_raw) if style_raw else None
    stream_raw = form.get("stream")
    if stream_raw is None:
        stream = False
    else:
        stream_str = str(stream_raw).lower()
        stream = stream_str in ("true", "1", "yes")

    # ------------------------------------------------------------------
    # 提取上传的图片文件 —— 兼容多种字段名
    # ------------------------------------------------------------------
    uploaded_files = []
    for key in form:
        if key in _IMAGE_FIELD_NAMES:
            values = form.getlist(key)
            logger.debug(f"Found image field {key!r} with {len(values)} item(s)")
            for v in values:
                if _is_upload_file(v):
                    uploaded_files.append(v)
                else:
                    logger.warning(
                        f"Skipping non-file value in field {key!r}: "
                        f"type={type(v).__name__}"
                    )

    logger.info(f"Collected {len(uploaded_files)} uploaded image(s)")

    # ------------------------------------------------------------------
    # 2. 构建 Pydantic 请求对象进行校验
    # ------------------------------------------------------------------
    if response_format_val is None:
        response_format_val = resolve_response_format(None)

    try:
        edit_request = ImageEditRequest(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            resolution=str(resolution_raw).strip() if resolution_raw is not None else None,
            aspect_ratio=aspect_ratio_val,
            quality=quality,
            response_format=response_format_val,
            style=style,
            stream=stream,
        )
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = first.get("loc", [])
            msg = first.get("msg", "Invalid request")
            code = first.get("type", "invalid_value")
            param_parts = [
                str(x) for x in loc if not (isinstance(x, int) or str(x).isdigit())
            ]
            param = ".".join(param_parts) if param_parts else None
            raise ValidationException(message=msg, param=param, code=code)
        raise ValidationException(message="Invalid request", code="invalid_value")

    if edit_request.stream is None:
        edit_request.stream = False

    response_format = resolve_response_format(edit_request.response_format)
    if response_format == "base64":
        response_format = "b64_json"
    edit_request.response_format = response_format
    response_field = response_field_name(response_format)

    # 参数验证
    validate_edit_request(edit_request, uploaded_files)

    # ------------------------------------------------------------------
    # 3. 读取并验证图片内容
    # ------------------------------------------------------------------
    max_image_bytes = 50 * 1024 * 1024
    allowed_types = {"image/png", "image/jpeg", "image/webp", "image/jpg"}

    images: List[str] = []
    inferred_width: Optional[int] = None
    inferred_height: Optional[int] = None

    for item in uploaded_files:
        content = await item.read()
        await item.close()
        if not content:
            raise ValidationException(
                message="File content is empty",
                param="image",
                code="empty_file",
            )
        if len(content) > max_image_bytes:
            raise ValidationException(
                message="Image file too large. Maximum is 50MB.",
                param="image",
                code="file_too_large",
            )
        mime = (item.content_type or "").lower()
        if mime == "image/jpg":
            mime = "image/jpeg"
        ext = Path(item.filename or "").suffix.lower()
        if mime not in allowed_types:
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"
            else:
                raise ValidationException(
                    message="Unsupported image type. Supported: png, jpg, webp.",
                    param="image",
                    code="invalid_image_type",
                )

        if inferred_width is None or inferred_height is None:
            w, h = _detect_image_dimensions(content, mime)
            if w and h:
                inferred_width, inferred_height = w, h

        b64 = base64.b64encode(content).decode()
        images.append(f"data:{mime};base64,{b64}")

    # ------------------------------------------------------------------
    # 4. 归一化比例（新增：在未传 size/aspectRatio 时，自动推断上传图比例）
    # ------------------------------------------------------------------
    normalize_size_input = edit_request.size if has_size_field else None
    normalize_aspect_input = edit_request.aspect_ratio

    inferred_aspect = None
    if not normalize_aspect_input and not has_size_field:
        inferred_aspect = _closest_allowed_aspect_ratio(inferred_width, inferred_height)
        if inferred_aspect:
            normalize_aspect_input = inferred_aspect

    if normalize_size_input is None and not normalize_aspect_input:
        # 保持历史兼容：无任何提示且无法推断时，默认 1:1
        normalized_size, aspect_ratio = "1024x1024", "1:1"
    else:
        normalized_size, aspect_ratio = _normalize_size_and_aspect(
            size=normalize_size_input,
            aspect_ratio=normalize_aspect_input,
        )

    logger.info(
        f"[aspect-ratio-trace] image_edit.infer inferred_width={inferred_width} inferred_height={inferred_height} inferred_aspect={inferred_aspect}"
    )
    logger.info(
        f"[aspect-ratio-trace] image_edit.normalized model={edit_request.model} normalized_size={normalized_size} normalized_aspect_ratio={aspect_ratio} request_size={edit_request.size} request_resolution={edit_request.resolution} request_aspect_ratio={edit_request.aspect_ratio} has_size_field={has_size_field}"
    )

    # ------------------------------------------------------------------
    # 5. 调用编辑服务
    # ------------------------------------------------------------------
    token_mgr, token = await _get_token(edit_request.model)
    model_info = ModelService.get(edit_request.model)

    result = await ImageEditService().edit(
        token_mgr=token_mgr,
        token=token,
        model_info=model_info,
        prompt=edit_request.prompt,
        images=images,
        n=edit_request.n,
        response_format=response_format,
        stream=bool(edit_request.stream),
        aspect_ratio=aspect_ratio,
    )

    if result.stream:
        return StreamingResponse(
            result.data,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    data = [{response_field: img} for img in result.data]

    return JSONResponse(
        content={
            "created": int(time.time()),
            "data": data,
            "usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
            },
        }
    )


__all__ = ["router"]
