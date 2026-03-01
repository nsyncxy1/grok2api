"""
Grok image edit service.
"""

import asyncio
import os
import random
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterable, List, Union, Any, Optional

import orjson
from curl_cffi.requests.errors import RequestsError

from app.core.config import get_config
from app.core.exceptions import (
    AppException,
    ErrorType,
    UpstreamException,
    StreamIdleTimeoutError,
)
from app.core.logger import logger
from app.services.grok.utils.process import (
    BaseProcessor,
    _with_idle_timeout,
    _normalize_line,
    _collect_images,
    _is_http2_error,
)
from app.services.grok.utils.upload import UploadService
from app.services.grok.utils.retry import pick_token, rate_limited
from app.services.grok.utils.response import make_response_id, make_chat_chunk, wrap_image_content
from app.services.grok.services.chat import GrokChatService
from app.services.grok.services.video import VideoService
from app.services.grok.utils.stream import wrap_stream_with_usage
from app.services.token import EffortType


@dataclass
class ImageEditResult:
    stream: bool
    data: Union[AsyncGenerator[str, None], List[str]]


class ImageEditService:
    """Image edit orchestration service."""

    async def edit(
        self,
        *,
        token_mgr: Any,
        token: str,
        model_info: Any,
        prompt: str,
        images: List[str],
        n: int,
        response_format: str,
        stream: bool,
        chat_format: bool = False,
        aspect_ratio: Optional[str] = None,
    ) -> ImageEditResult:

        if len(images) > 3:
            logger.info(
                "Image edit received %d references; using the most recent 3",
                len(images),
            )
            images = images[-3:]

        max_token_retries = int(get_config("retry.max_retry") or 3)
        tried_tokens: set[str] = set()
        last_error: Exception | None = None

        for attempt in range(max_token_retries):
            preferred = token if attempt == 0 else None
            current_token = await pick_token(
                token_mgr, model_info.model_id, tried_tokens, preferred=preferred
            )
            if not current_token:
                if last_error:
                    raise last_error
                raise AppException(
                    message="No available tokens. Please try again later.",
                    error_type=ErrorType.RATE_LIMIT.value,
                    code="rate_limit_exceeded",
                    status_code=429,
                )

            tried_tokens.add(current_token)
            try:
                image_urls = await self._upload_images(images, current_token)
                parent_post_id = await self._get_parent_post_id(
                    current_token, image_urls
                )

                model_config_override = {
                    "modelMap": {
                        "imageEditModel": "imagine",
                        "imageEditModelConfig": {
                            "imageReferences": image_urls,
                        },
                    }
                }

                # Forward aspect_ratio to Grok backend so image edits
                # respect the requested output format (e.g. 9:16 for vertical video)
                if aspect_ratio:
                    model_config_override["modelMap"]["imageEditModelConfig"][
                        "aspectRatio"
                    ] = aspect_ratio

                if parent_post_id:
                    model_config_override["modelMap"]["imageEditModelConfig"][
                        "parentPostId"
                    ] = parent_post_id

                tool_overrides = {"imageGen": True}

                logger.info(
                    "[image_edit] sending to grok: aspect_ratio={} image_urls={} parent_post_id={} model={}",
                    aspect_ratio,
                    [u[:80] for u in image_urls],
                    parent_post_id,
                    model_info.grok_model,
                )

                if stream:
                    response = await GrokChatService().chat(
                        token=current_token,
                        message=prompt,
                        model=model_info.grok_model,
                        mode=None,
                        stream=True,
                        tool_overrides=tool_overrides,
                        model_config_override=model_config_override,
                    )
                    processor = ImageStreamProcessor(
                        model_info.model_id,
                        current_token,
                        n=n,
                        response_format=response_format,
                        chat_format=chat_format,
                    )
                    return ImageEditResult(
                        stream=True,
                        data=wrap_stream_with_usage(
                            processor.process(response),
                            token_mgr,
                            current_token,
                            model_info.model_id,
                        ),
                    )

                images_out = await self._collect_images(
                    token=current_token,
                    prompt=prompt,
                    model_info=model_info,
                    n=n,
                    response_format=response_format,
                    tool_overrides=tool_overrides,
                    model_config_override=model_config_override,
                )
                try:
                    effort = (
                        EffortType.HIGH
                        if (model_info and model_info.cost.value == "high")
                        else EffortType.LOW
                    )
                    await token_mgr.consume(current_token, effort)
                    logger.debug(
                        f"Image edit completed, recorded usage (effort={effort.value})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to record image edit usage: {e}")
                return ImageEditResult(stream=False, data=images_out)

            except UpstreamException as e:
                last_error = e
                if rate_limited(e):
                    await token_mgr.mark_rate_limited(current_token)
                    logger.warning(
                        f"Token {current_token[:10]}... rate limited (429), "
                        f"trying next token (attempt {attempt + 1}/{max_token_retries})"
                    )
                    continue
                raise

        if last_error:
            raise last_error
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429,
        )

    async def _upload_images(self, images: List[str], token: str) -> List[str]:
        image_urls: List[str] = []
        upload_service = UploadService()
        try:
            for idx, image in enumerate(images):
                try:
                    _, file_uri = await upload_service.upload_file(image, token)
                    if file_uri:
                        if file_uri.startswith("http"):
                            image_urls.append(file_uri)
                        else:
                            image_urls.append(
                                f"https://assets.grok.com/{file_uri.lstrip('/')}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to upload image {idx + 1}/{len(images)}: {e}"
                    )
        finally:
            await upload_service.close()

        if not image_urls:
            raise AppException(
                message="Image upload failed",
                error_type=ErrorType.SERVER.value,
                code="upload_failed",
            )

        return image_urls

    async def _get_parent_post_id(self, token: str, image_urls: List[str]) -> str:
        """Create media posts for ALL uploaded images and return the first post ID."""
        parent_post_id = None
        media_service = VideoService()

        for idx, url in enumerate(image_urls):
            try:
                post_id = await media_service.create_image_post(token, url)
                logger.debug(
                    f"Created image post {idx + 1}/{len(image_urls)}: {post_id}"
                )
                if not parent_post_id:
                    parent_post_id = post_id
            except Exception as e:
                logger.warning(
                    f"Create image post failed for image {idx + 1}/{len(image_urls)} "
                    f"({url[:80]}): {e}"
                )

        if parent_post_id:
            return parent_post_id

        # Fallback: try to extract post ID from the URL pattern
        for url in image_urls:
            match = re.search(r"/generated/([a-f0-9-]+)/", url)
            if match:
                parent_post_id = match.group(1)
                break
            match = re.search(r"/users/[^/]+/([a-f0-9-]+)/content", url)
            if match:
                parent_post_id = match.group(1)
                break

        return parent_post_id or ""

    async def _collect_images(
        self,
        *,
        token: str,
        prompt: str,
        model_info: Any,
        n: int,
        response_format: str,
        tool_overrides: dict,
        model_config_override: dict,
    ) -> List[str]:
        calls_needed = (n + 1) // 2

        async def _call_edit():
            response = await GrokChatService().chat(
                token=token,
                message=prompt,
                model=model_info.grok_model,
                mode=None,
                stream=True,
                tool_overrides=tool_overrides,
                model_config_override=model_config_override,
            )
            processor = ImageCollectProcessor(
                model_info.model_id, token, response_format=response_format
            )
            return await processor.process(response)

        last_error: Exception | None = None
        rate_limit_error: Exception | None = None

        if calls_needed == 1:
            all_images = await _call_edit()
        else:
            tasks = [_call_edit() for _ in range(calls_needed)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_images: List[str] = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Concurrent call failed: {result}")
                    last_error = result
                    if rate_limited(result):
                        rate_limit_error = result
                elif isinstance(result, list):
                    all_images.extend(result)

        if not all_images:
            if rate_limit_error:
                raise rate_limit_error
            if last_error:
                raise last_error
            raise UpstreamException(
                "Image edit returned no results", details={"error": "empty_result"}
            )

        if len(all_images) >= n:
            return all_images[:n]

        selected_images = all_images.copy()
        while len(selected_images) < n:
            selected_images.append("error")
        return selected_images


def _truncate(s: str, max_len: int = 500) -> str:
    """Truncate string for logging."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...<+{len(s) - max_len}>"


class ImageStreamProcessor(BaseProcessor):
    """HTTP image stream processor."""

    def __init__(
        self, model: str, token: str = "", n: int = 1, response_format: str = "b64_json", chat_format: bool = False
    ):
        super().__init__(model, token)
        self.partial_index = 0
        self.n = n
        self.target_index = 0 if n == 1 else None
        self.response_format = response_format
        self.chat_format = chat_format
        self._id_generated = False
        self._response_id = ""
        if response_format == "url":
            self.response_field = "url"
        elif response_format == "base64":
            self.response_field = "base64"
        else:
            self.response_field = "b64_json"

    def _sse(self, event: str, data: dict) -> str:
        """Build SSE response."""
        return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"

    async def process(
        self, response: AsyncIterable[bytes]
    ) -> AsyncGenerator[str, None]:
        """Process stream response."""
        final_images = []
        emitted_chat_chunk = False
        idle_timeout = get_config("image.stream_timeout")
        line_count = 0

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_line(line)
                if not line:
                    continue

                line_count += 1

                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    logger.warning(
                        "[image_edit_stream] line#{} JSON decode failed: {}",
                        line_count, _truncate(line),
                    )
                    continue

                resp = data.get("result", {}).get("response", {})

                # Log every upstream response chunk (keys only to avoid huge logs)
                resp_keys = list(resp.keys()) if resp else []
                if line_count <= 5 or resp_keys:
                    logger.info(
                        "[image_edit_stream] line#{} resp_keys={} has_result={}",
                        line_count, resp_keys, "result" in data,
                    )

                # Log the full first response and any modelResponse for debugging
                if line_count == 1:
                    logger.info(
                        "[image_edit_stream] first line (full): {}",
                        _truncate(line, 2000),
                    )

                # Image generation progress
                if img := resp.get("streamingImageGenerationResponse"):
                    image_index = img.get("imageIndex", 0)
                    progress = img.get("progress", 0)
                    logger.debug(
                        "[image_edit_stream] progress: index={} progress={}%",
                        image_index, progress,
                    )

                    if self.n == 1 and image_index != self.target_index:
                        continue

                    out_index = 0 if self.n == 1 else image_index

                    if not self.chat_format:
                        yield self._sse(
                            "image_generation.partial_image",
                            {
                                "type": "image_generation.partial_image",
                                self.response_field: "",
                                "index": out_index,
                                "progress": progress,
                            },
                        )
                    continue

                # modelResponse — this is where grok returns generated images
                if mr := resp.get("modelResponse"):
                    # Log the full modelResponse structure (without huge blobs)
                    mr_keys = list(mr.keys()) if isinstance(mr, dict) else type(mr).__name__
                    logger.info(
                        "[image_edit_stream] modelResponse keys={}", mr_keys,
                    )

                    # Log message field if present (grok sometimes returns text instead of images)
                    if msg := mr.get("message"):
                        logger.info(
                            "[image_edit_stream] modelResponse.message={}",
                            _truncate(str(msg), 1000),
                        )

                    # Log generatedImageUrls / imageUrls etc.
                    for url_key in ("generatedImageUrls", "imageUrls", "imageURLs"):
                        if url_key in mr:
                            logger.info(
                                "[image_edit_stream] modelResponse.{}={}",
                                url_key, mr[url_key],
                            )

                    urls = _collect_images(mr)
                    logger.info(
                        "[image_edit_stream] collected {} image URL(s): {}",
                        len(urls),
                        [u[:100] for u in urls] if urls else [],
                    )

                    if urls:
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    final_images.append(processed)
                                continue
                            try:
                                dl_service = self._get_dl()
                                base64_data = await dl_service.parse_b64(
                                    url, self.token, "image"
                                )
                                if base64_data:
                                    if "," in base64_data:
                                        b64 = base64_data.split(",", 1)[1]
                                    else:
                                        b64 = base64_data
                                    final_images.append(b64)
                                    logger.info(
                                        "[image_edit_stream] downloaded image: url={} b64_len={}",
                                        url[:80], len(b64),
                                    )
                                else:
                                    logger.warning(
                                        "[image_edit_stream] parse_b64 returned empty for url={}",
                                        url[:80],
                                    )
                            except Exception as e:
                                logger.warning(
                                    "[image_edit_stream] b64 download failed (url={}): {}, falling back to URL",
                                    url[:80], e,
                                )
                                processed = await self.process_url(url, "image")
                                if processed:
                                    final_images.append(processed)
                    else:
                        # No image URLs found in modelResponse — log the full response
                        # to help diagnose unexpected grok return format
                        logger.warning(
                            "[image_edit_stream] modelResponse has NO image URLs! full mr={}",
                            _truncate(orjson.dumps(mr).decode(), 3000),
                        )
                    continue

                # Log any unexpected response keys
                unexpected_keys = [k for k in resp_keys if k not in (
                    "streamingImageGenerationResponse", "modelResponse",
                    "token", "llmInfo", "responseId", "rolloutId",
                    "isThinking", "cardAttachment",
                )]
                if unexpected_keys:
                    logger.info(
                        "[image_edit_stream] line#{} unexpected keys: {} data={}",
                        line_count, unexpected_keys,
                        _truncate(line, 1000),
                    )

            logger.info(
                "[image_edit_stream] stream ended: total_lines={} final_images={}",
                line_count, len(final_images),
            )

            for index, img_data in enumerate(final_images):
                if self.n == 1:
                    if index != self.target_index:
                        continue
                    out_index = 0
                else:
                    out_index = index

                # Wrap in markdown format for chat
                output = img_data
                if self.chat_format and output:
                    output = wrap_image_content(output, self.response_format)

                if not self._id_generated:
                    self._response_id = make_response_id()
                    self._id_generated = True

                if self.chat_format:
                    emitted_chat_chunk = True
                    yield self._sse(
                        "chat.completion.chunk",
                        make_chat_chunk(
                            self._response_id,
                            self.model,
                            output,
                            index=out_index,
                            is_final=True,
                        ),
                    )
                else:
                    yield self._sse(
                        "image_generation.completed",
                        {
                            "type": "image_generation.completed",
                            self.response_field: img_data,
                            "index": out_index,
                            "usage": {
                                "total_tokens": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "input_tokens_details": {
                                    "text_tokens": 0,
                                    "image_tokens": 0,
                                },
                            },
                        },
                    )

            if self.chat_format:
                if not self._id_generated:
                    self._response_id = make_response_id()
                    self._id_generated = True
                if not emitted_chat_chunk:
                    yield self._sse(
                        "chat.completion.chunk",
                        make_chat_chunk(
                            self._response_id,
                            self.model,
                            "",
                            index=0,
                            is_final=True,
                        ),
                    )
                yield "data: [DONE]\n\n"

            if not final_images:
                logger.error(
                    "[image_edit_stream] FAILED: no images produced after {} lines from grok",
                    line_count,
                )

        except asyncio.CancelledError:
            logger.debug("Image stream cancelled by client")
        except StreamIdleTimeoutError as e:
            raise UpstreamException(
                message=f"Image stream idle timeout after {e.idle_seconds}s",
                status_code=504,
                details={
                    "error": str(e),
                    "type": "stream_idle_timeout",
                    "idle_seconds": e.idle_seconds,
                },
            )
        except RequestsError as e:
            if _is_http2_error(e):
                logger.warning(f"HTTP/2 stream error in image: {e}")
                raise UpstreamException(
                    message="Upstream connection closed unexpectedly",
                    status_code=502,
                    details={"error": str(e), "type": "http2_stream_error"},
                )
            logger.error(f"Image stream request error: {e}")
            raise UpstreamException(
                message=f"Upstream request failed: {e}",
                status_code=502,
                details={"error": str(e)},
            )
        except Exception as e:
            logger.error(
                f"Image stream processing error: {e}",
                extra={"error_type": type(e).__name__},
            )
            raise
        finally:
            await self.close()


class ImageCollectProcessor(BaseProcessor):
    """HTTP image non-stream processor."""

    def __init__(self, model: str, token: str = "", response_format: str = "b64_json"):
        if response_format == "base64":
            response_format = "b64_json"
        super().__init__(model, token)
        self.response_format = response_format

    async def process(self, response: AsyncIterable[bytes]) -> List[str]:
        """Process and collect images."""
        images = []
        idle_timeout = get_config("image.stream_timeout")
        line_count = 0

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_line(line)
                if not line:
                    continue

                line_count += 1

                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    logger.warning(
                        "[image_edit_collect] line#{} JSON decode failed: {}",
                        line_count, _truncate(line),
                    )
                    continue

                resp = data.get("result", {}).get("response", {})

                if mr := resp.get("modelResponse"):
                    mr_keys = list(mr.keys()) if isinstance(mr, dict) else type(mr).__name__
                    logger.info(
                        "[image_edit_collect] line#{} modelResponse keys={}",
                        line_count, mr_keys,
                    )

                    # Log message if present
                    if msg := mr.get("message"):
                        logger.info(
                            "[image_edit_collect] modelResponse.message={}",
                            _truncate(str(msg), 1000),
                        )

                    urls = _collect_images(mr)
                    logger.info(
                        "[image_edit_collect] collected {} image URL(s): {}",
                        len(urls),
                        [u[:100] for u in urls] if urls else [],
                    )

                    if not urls:
                        logger.warning(
                            "[image_edit_collect] modelResponse has NO image URLs! full mr={}",
                            _truncate(orjson.dumps(mr).decode(), 3000),
                        )

                    if urls:
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    images.append(processed)
                                continue
                            try:
                                dl_service = self._get_dl()
                                base64_data = await dl_service.parse_b64(
                                    url, self.token, "image"
                                )
                                if base64_data:
                                    if "," in base64_data:
                                        b64 = base64_data.split(",", 1)[1]
                                    else:
                                        b64 = base64_data
                                    images.append(b64)
                                    logger.info(
                                        "[image_edit_collect] downloaded image: b64_len={}",
                                        len(b64),
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to convert image to base64, falling back to URL: {e}"
                                )
                                processed = await self.process_url(url, "image")
                                if processed:
                                    images.append(processed)

            logger.info(
                "[image_edit_collect] done: total_lines={} images={}",
                line_count, len(images),
            )

        except asyncio.CancelledError:
            logger.debug("Image collect cancelled by client")
        except StreamIdleTimeoutError as e:
            logger.warning(f"Image collect idle timeout: {e}")
        except RequestsError as e:
            if _is_http2_error(e):
                logger.warning(f"HTTP/2 stream error in image collect: {e}")
            else:
                logger.error(f"Image collect request error: {e}")
        except Exception as e:
            logger.error(
                f"Image collect processing error: {e}",
                extra={"error_type": type(e).__name__},
            )
        finally:
            await self.close()

        return images


__all__ = ["ImageEditService", "ImageEditResult"]
