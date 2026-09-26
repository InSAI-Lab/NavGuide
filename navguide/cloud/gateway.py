"""Run with: uvicorn cloud.gateway:app --host 127.0.0.1 --port 8090."""
import asyncio
import base64
import binascii
from dataclasses import dataclass, field
import hmac
import json
import os
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import CloudSettings
from .provider import QwenProvider, normalize_label, upstream_failure


@dataclass(frozen=True)
class GatewaySettings:
    provider: CloudSettings = field(default_factory=CloudSettings)
    token: str = field(default="", repr=False)
    max_request_bytes: int = 8 * 1024 * 1024
    max_concurrent_requests: int = 4

    @classmethod
    def from_env(cls):
        size = int(os.getenv("CLOUD_MAX_REQUEST_BYTES", str(8 * 1024 * 1024)))
        concurrent = int(os.getenv("CLOUD_MAX_CONCURRENT_REQUESTS", "4"))
        if not 1024 <= size <= 32 * 1024 * 1024:
            raise ValueError("CLOUD_MAX_REQUEST_BYTES must be between 1024 and 33554432")
        if not 1 <= concurrent <= 64:
            raise ValueError("CLOUD_MAX_CONCURRENT_REQUESTS must be between 1 and 64")
        return cls(provider=CloudSettings.from_env(), token=os.getenv("NAVGUIDE_GATEWAY_TOKEN", "").strip(),
                   max_request_bytes=size, max_concurrent_requests=concurrent)

    @property
    def ready(self):
        return len(self.token) >= 32 and bool(self.provider.api_key)


class GatewayBoundary:
    """Authenticate before parsing; bound chunked bodies as well as Content-Length."""
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {"/healthz", "/readyz"}:
            await self.app(scope, receive, send)
            return

        async def reject(status, code, headers=None):
            await JSONResponse({"error": code}, status_code=status, headers=headers)(scope, receive, send)

        if not self.settings.ready:
            await reject(503, "gateway_not_configured")
            return
        headers = dict(scope.get("headers", []))
        expected = ("Bearer " + self.settings.token).encode("utf-8")
        if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
            await reject(401, "unauthorized", {"WWW-Authenticate": "Bearer"})
            return
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await reject(400, "invalid_content_length")
            return
        if declared < 0 or declared > self.settings.max_request_bytes:
            await reject(413, "request_too_large")
            return

        chunks = bytearray()
        deadline = asyncio.get_running_loop().time() + self.settings.provider.timeout_seconds
        while True:
            try:
                event = await asyncio.wait_for(receive(), max(0, deadline - asyncio.get_running_loop().time()))
            except asyncio.TimeoutError:
                await reject(408, "request_timeout")
                return
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            if len(chunks) + len(chunk) > self.settings.max_request_bytes:
                await reject(413, "request_too_large")
                return
            chunks.extend(chunk)
            if not event.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class DescriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: List[Dict[str, Any]] = Field(min_length=1, max_length=8)
    voice: str = Field(default="Cherry", min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    audio_format: Literal["wav"] = "wav"

    @field_validator("content")
    @classmethod
    def validate_content(cls, content):
        text_found = False
        for item in content:
            if item.get("type") == "text":
                text = item.get("text")
                if set(item) != {"type", "text"} or not isinstance(text, str) or not 1 <= len(text.strip()) <= 4096:
                    raise ValueError("Expected a short text prompt")
                text_found = True
            elif item.get("type") == "image_url":
                image = item.get("image_url")
                if set(item) != {"type", "image_url"} or not isinstance(image, dict) or set(image) != {"url"}:
                    raise ValueError("Expected inline image data")
                url = image["url"]
                if not isinstance(url, str) or not url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,")):
                    raise ValueError("Remote image URLs are not accepted")
                try:
                    if not base64.b64decode(url.split(",", 1)[1], validate=True):
                        raise ValueError("Empty image data")
                except (binascii.Error, ValueError):
                    raise ValueError("Invalid image encoding") from None
            else:
                raise ValueError("Only text and inline images are supported")
        if not text_found:
            raise ValueError("A text prompt is required")
        return content


class LabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=512)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("A query is required")
        return value


def _line(event):
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"


def create_app(settings=None, provider=None):
    settings = settings or GatewaySettings.from_env()
    provider = provider or QwenProvider(settings.provider)
    application = FastAPI(title="NAVGuide Optional Cloud Gateway", docs_url=None, redoc_url=None, openapi_url=None)
    application.add_middleware(GatewayBoundary, settings=settings)
    application.state.capacity = asyncio.Semaphore(settings.max_concurrent_requests)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Do not echo images, user text or secrets in validation failures.
        return JSONResponse({"error": "invalid_request"}, status_code=422)

    @application.get("/healthz")
    async def health():
        return {"status": "ok"}

    @application.get("/readyz")
    async def ready():
        # Readiness checks configuration only, without a paid provider call.
        return JSONResponse({"status": "ready" if settings.ready else "not_configured"},
                            status_code=200 if settings.ready else 503)

    async def acquire():
        if application.state.capacity.locked():
            raise HTTPException(429, "gateway_busy", headers={"Retry-After": "1"})
        await application.state.capacity.acquire()

    @application.post("/v1/label")
    async def label(body: LabelRequest):
        await acquire()
        try:
            result = await asyncio.wait_for(provider.label(body.query), timeout=settings.provider.timeout_seconds)
            return {"label": normalize_label(result)}
        except Exception as exc:
            failure = upstream_failure(exc)
            return JSONResponse({"error": failure.code}, status_code=failure.status_code)
        finally:
            application.state.capacity.release()

    @application.post("/v1/describe")
    async def describe(body: DescriptionRequest):
        await acquire()
        source = provider.stream(body.content, body.voice, body.audio_format)
        deadline = asyncio.get_running_loop().time() + settings.provider.timeout_seconds

        async def next_piece():
            return await asyncio.wait_for(source.__anext__(), max(0, deadline - asyncio.get_running_loop().time()))

        try:
            # Obtain the first useful delta before committing an HTTP 200 response.
            first = await next_piece()
        except BaseException as exc:
            application.state.capacity.release()
            await source.aclose()
            if not isinstance(exc, Exception):
                raise
            failure = upstream_failure(exc)
            return JSONResponse({"error": failure.code}, status_code=failure.status_code)

        async def events():
            try:
                piece = first
                while True:
                    yield _line({"type": "delta", "text_delta": piece.text_delta, "audio_b64": piece.audio_b64})
                    try:
                        piece = await next_piece()
                    except StopAsyncIteration:
                        yield _line({"type": "done"})
                        break
            except Exception as exc:
                yield _line({"type": "error", "code": upstream_failure(exc).code})
            finally:
                application.state.capacity.release()
                await source.aclose()

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return application


app = create_app()
