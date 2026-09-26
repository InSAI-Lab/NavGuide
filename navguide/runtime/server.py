"""Single-device service for NavGuide."""

from __future__ import annotations
from navguide.i18n import SENTENCE_ENDINGS, terms

import asyncio
import contextlib
import json
import time
from dataclasses import asdict
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from navguide.runtime.access import DeviceAccess
from navguide.runtime.imu import IMUState, token_matches
from navguide.core.context import TaskMode, SceneType
from navguide.runtime.speech import LocalSpeech, wav_stream_header
from navguide.core.pipeline import NavGuidePipeline
from navguide.runtime.config import ROOT, Settings
from navguide.core.selection import DetectionCandidate

load_dotenv(ROOT / ".env")
Finite = Annotated[float, Field(allow_inf_nan=False)]


class CandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1, max_length=100)
    confidence: Finite = Field(ge=0, le=1)
    bbox: tuple[Finite, Finite, Finite, Finite]
    track_id: int | None = None
    is_hazard: bool = False
    is_moving: bool = False
    urgency_override: bool | None = None
    signal_color: Literal["red", "yellow", "green", "unknown"] | None = None


class ObservationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detections: list[CandidateInput] = Field(max_length=256)
    frame_width: int = Field(default=640, ge=1, le=8192)
    frame_height: int = Field(default=480, ge=1, le=8192)
    yaw_rate_dps: Finite | None = Field(default=None, ge=-4000, le=4000)
    scene: SceneType | None = None

    @model_validator(mode="after")
    def validate_boxes(self):
        for item in self.detections:
            x1, y1, x2, y2 = item.bbox
            if not (0 <= x1 < x2 <= self.frame_width and 0 <= y1 < y2 <= self.frame_height):
                raise ValueError("bbox must be positive and within the frame")
            if not item.category.strip():
                raise ValueError("category must not be blank")
        return self


class ContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_mode: TaskMode
    target_query: str | None = Field(default=None, max_length=100)
    scene: SceneType | None = None
    user_weights: dict[str, Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]] = Field(
        default_factory=dict, max_length=128
    )

    @model_validator(mode="after")
    def target_required(self):
        if self.task_mode == TaskMode.TARGET_SEARCH and not (self.target_query or "").strip():
            raise ValueError("target_search requires target_query")
        return self


class DescriptionInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    include_latest_image: bool = False


class Runtime:
    def __init__(self, settings):
        self.settings = settings
        self.pipeline = NavGuidePipeline()
        self.imu = IMUState(settings.imu_timeout_seconds)
        self.speech = LocalSpeech(
            settings.speech_command,
            settings.speech_voice,
            settings.speech_enabled,
            max_pending_age=settings.observation_timeout_seconds,
        )
        self.frontend = None
        self.frontend_error = None
        self.lock = asyncio.Lock()
        self.events = set()
        self.camera = None
        self.latest_frame = None
        self.frames_dropped = 0
        self.invalid_imu_packets = 0
        self.receive_to_audio_trigger_ms = None
        self.generation = 0
        self.guidance_enabled = True
        self.current_signatures = set()
        self.last_observation_at = None
        self.last_observation_yaw = None

    async def run_model(self, function, *args):
        """Keep the caller's model lock until a cancelled native call has ended."""
        operation = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.shield(operation)
            raise

    async def stop_guidance(self):
        self.guidance_enabled = False
        self.generation += 1
        self.current_signatures.clear()
        self.last_observation_at = None
        self.pipeline.smp.reset_history()
        await self.speech.stop()

    def current_yaw(self, now=None):
        now = time.monotonic() if now is None else now
        if self.imu.fresh(now) and (
            self.last_observation_yaw is None
            or self.last_observation_at is None
            or self.imu.received_at >= self.last_observation_at
        ):
            return self.imu.yaw_rate(now)
        if self.last_observation_yaw is not None and self.last_observation_at is not None:
            if 0 <= now - self.last_observation_at <= self.settings.imu_timeout_seconds:
                return self.last_observation_yaw
        return 25.0

    async def broadcast(self, payload):
        for queue in list(self.events):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(payload)

    async def process(self, detections, width, height, yaw=None, scene=None, received=None):
        received = time.monotonic() if received is None else received
        if not 0 <= time.monotonic() - received <= self.settings.observation_timeout_seconds:
            self.frames_dropped += 1
            self.current_signatures.clear()
            await self.speech.stop()
            return {
                "type": "guidance",
                "guidance_enabled": self.guidance_enabled,
                "should_speak": False,
                "speech_text": "",
                "eligible_cues": [],
                "deferred_cues": [],
                "audio_requested": False,
                "reason": "stale_observation",
            }
        if not self.guidance_enabled:
            payload = {
                "type": "guidance",
                "guidance_enabled": False,
                "should_speak": False,
                "speech_text": "",
                "eligible_cues": [],
                "deferred_cues": [],
                "audio_requested": False,
                "imu_fresh": self.imu.fresh(),
            }
            await self.broadcast(payload)
            return payload
        self.last_observation_at = received
        self.last_observation_yaw = yaw
        actual_yaw = self.current_yaw()
        result = self.pipeline.process(
            detections,
            yaw_rate_dps=actual_yaw,
            raw_scene=scene,
            frame_width=width,
            frame_height=height,
        )
        payload = jsonable_encoder(asdict(result))
        inline_imu_fresh = (
            yaw is not None and time.monotonic() - received <= self.settings.imu_timeout_seconds
        )
        payload.update(type="guidance", imu_fresh=inline_imu_fresh or self.imu.fresh())
        self.current_signatures = {cue.semantic_signature for cue in result.selected_cues}
        self.last_observation_at = received
        self.last_observation_yaw = yaw
        generation = self.generation

        def still_current():
            now = time.monotonic()
            if not self.guidance_enabled or generation != self.generation:
                return False
            if (
                self.last_observation_at is None
                or not 0
                <= now - self.last_observation_at
                <= self.settings.observation_timeout_seconds
            ):
                return False
            current_yaw = self.current_yaw(now)
            # Each chunk must still match observed content. The current IMU is
            # rechecked while streaming, so a later turn invalidates ordinary PCM.
            return all(
                cue.semantic_signature in self.current_signatures
                and (abs(current_yaw) < 25 or cue.urgency_flag or cue.requested_target_flag)
                for cue in result.eligible_cues
            )

        def triggered(at):
            self.receive_to_audio_trigger_ms = max(0, (at - received) * 1000)

        payload["guidance_enabled"] = True
        payload["audio_requested"] = self.speech.submit(
            result.speech_text, triggered, still_current
        )
        await self.broadcast(payload)
        return payload


class IMUProtocol(asyncio.DatagramProtocol):
    def __init__(self, runtime):
        self.runtime = runtime

    def datagram_received(self, data, addr):
        try:
            if len(data) > 2048:
                raise ValueError("Packet too large")
            packet = json.loads(data)
            if not isinstance(packet, dict) or not token_matches(
                packet.get("token", ""), self.runtime.settings.device_token
            ):
                raise ValueError("Invalid device authentication")
            sanitized = self.runtime.imu.update(packet)
            # Drop visual telemetry under load; the current sensor state is always retained.
            for queue in list(self.runtime.events):
                if not queue.full():
                    queue.put_nowait({"type": "imu", **sanitized})
        except (ValueError, TypeError, UnicodeError):
            self.runtime.invalid_imu_packets += 1


def create_app(settings=None, frontend=None):
    settings = settings or Settings.from_env()
    runtime = Runtime(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        transport = None
        try:
            runtime.speech.start()
            runtime.frontend = frontend
            if settings.frontend == "yoloe" and frontend is None:
                try:
                    from navguide.runtime.vision import YOLOEFrontend

                    runtime.frontend = await asyncio.to_thread(YOLOEFrontend)
                except Exception as exc:
                    runtime.frontend_error = type(exc).__name__
                    import logging

                    logging.getLogger(__name__).exception("Local frontend failed to initialize")
            if settings.udp_enabled:
                transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
                    lambda: IMUProtocol(runtime), local_addr=(settings.udp_host, settings.udp_port)
                )
            yield
        finally:
            if transport:
                transport.close()
            await runtime.speech.close()

    app = FastAPI(
        title="NavGuide device service",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.runtime = runtime
    app.add_middleware(DeviceAccess, settings=settings)

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/static/localization.js", include_in_schema=False)
    async def language_script():
        return FileResponse(
            ROOT / "web" / "static" / "localization.js", media_type="text/javascript"
        )

    @app.get("/static/locales/zh-CN.json", include_in_schema=False)
    async def language_resources():
        return FileResponse(
            ROOT / "web" / "static" / "locales" / "zh-CN.json", media_type="application/json"
        )

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "navguide-local"}

    @app.get("/api/ready")
    async def ready():
        ok = settings.frontend == "observations" or runtime.frontend is not None
        return JSONResponse(
            {
                "ready": ok,
                "frontend": settings.frontend,
                "camera_inference": runtime.frontend is not None,
            },
            200 if ok else 503,
        )

    @app.get("/api/status")
    async def status():
        return {
            "frontend": settings.frontend,
            "frontend_error": runtime.frontend_error,
            "context": jsonable_encoder(runtime.pipeline.context),
            "imu_fresh": runtime.imu.fresh(),
            "camera_connected": runtime.camera is not None,
            "frames_dropped": runtime.frames_dropped,
            "invalid_imu_packets": runtime.invalid_imu_packets,
            "speech_enabled": settings.speech_enabled,
            "audio_clients": len(runtime.speech.clients),
            "audio_trigger_count": runtime.speech.trigger_count,
            "speech_error": runtime.speech.last_error,
            "receive_to_audio_trigger_ms": runtime.receive_to_audio_trigger_ms,
            "guidance_enabled": runtime.guidance_enabled,
        }

    async def apply_context(body):
        runtime.generation += 1
        generation = runtime.generation
        runtime.guidance_enabled = False
        runtime.current_signatures.clear()
        await runtime.speech.stop()
        async with runtime.lock:
            if generation != runtime.generation:
                return jsonable_encoder(runtime.pipeline.context)
            if runtime.frontend:
                await runtime.run_model(runtime.frontend.set_target, body.target_query)
            if generation != runtime.generation:
                return jsonable_encoder(runtime.pipeline.context)
            runtime.pipeline.set_task_mode(body.task_mode, body.target_query)
            runtime.pipeline.smp.reset_history()
            runtime.pipeline.context.user_weights.clear()
            runtime.pipeline.set_user_weights(body.user_weights)
            # An explicit task change must not retain a scene from the old task.
            scene = body.scene if body.scene is not None else SceneType.UNKNOWN
            runtime.pipeline.smoother.reset(scene)
            runtime.pipeline.context.scene_type = scene
            runtime.guidance_enabled = True
        return jsonable_encoder(runtime.pipeline.context)

    @app.post("/api/context")
    async def context(body: ContextInput):
        return await apply_context(body)

    @app.post("/api/stop")
    async def stop():
        await runtime.stop_guidance()
        return {"status": "stopped", "guidance_enabled": runtime.guidance_enabled}

    @app.post("/api/observations")
    async def observations(body: ObservationInput):
        received = time.monotonic()
        if runtime.camera is not None:
            raise HTTPException(409, "Disconnect the camera before submitting observations")
        async with runtime.lock:
            if runtime.camera is not None:
                raise HTTPException(409, "Disconnect the camera before submitting observations")
            return await runtime.process(
                [DetectionCandidate(**item.model_dump()) for item in body.detections],
                body.frame_width,
                body.frame_height,
                body.yaw_rate_dps,
                body.scene,
                received=received,
            )

    @app.post("/api/imu")
    async def imu(request: Request):
        try:
            return runtime.imu.update(await request.json())
        except (ValueError, TypeError):
            raise HTTPException(422, "Invalid IMU packet")

    @app.post("/api/describe")
    async def describe(body: DescriptionInput):
        from navguide.cloud.descriptions import cloud_available, stream_chat

        if not cloud_available():
            raise HTTPException(503, "Optional cloud description is disabled or unconfigured")
        content = [{"type": "text", "text": body.prompt}]
        if body.include_latest_image and runtime.latest_frame:
            import base64

            content.insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(runtime.latest_frame).decode("ascii")
                    },
                },
            )

        async def generate():
            try:
                async for piece in stream_chat(content):
                    if piece.text_delta:
                        yield json.dumps({"text": piece.text_delta}, ensure_ascii=False) + "\n"
            except Exception:
                yield json.dumps({"error": "Cloud description unavailable"}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    async def accept(ws):
        await ws.accept(
            subprotocol="navguide" if "navguide" in ws.scope.get("subprotocols", []) else None
        )

    @app.websocket("/ws/events")
    async def events(ws: WebSocket):
        await accept(ws)
        queue = asyncio.Queue(maxsize=4)
        runtime.events.add(queue)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), 5)
                except asyncio.TimeoutError:
                    payload = {"type": "heartbeat"}
                await ws.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            runtime.events.discard(queue)

    @app.websocket("/ws/camera")
    async def camera(ws: WebSocket):
        if runtime.camera is not None or runtime.frontend is None:
            await ws.close(code=1013)
            return
        runtime.camera = ws
        try:
            await accept(ws)
        except BaseException:
            runtime.camera = None
            raise
        pending = asyncio.Queue(maxsize=1)

        async def consume():
            while True:
                jpeg, received, generation = await pending.get()
                try:
                    async with runtime.lock:
                        if not runtime.guidance_enabled or generation != runtime.generation:
                            runtime.frames_dropped += 1
                            continue
                        if time.monotonic() - received > settings.observation_timeout_seconds:
                            runtime.frames_dropped += 1
                            continue
                        detections, width, height = await runtime.run_model(
                            runtime.frontend.detect, jpeg, settings.max_frame_pixels
                        )
                        if (
                            generation != runtime.generation
                            or not runtime.guidance_enabled
                            or time.monotonic() - received > settings.observation_timeout_seconds
                        ):
                            runtime.frames_dropped += 1
                            continue
                        runtime.latest_frame = jpeg
                        await runtime.process(detections, width, height, received=received)
                except Exception:
                    runtime.current_signatures.clear()
                    await runtime.speech.stop()
                    await runtime.broadcast({"type": "error", "detail": "Camera inference failed"})

        worker = asyncio.create_task(consume())
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                jpeg = message.get("bytes")
                if jpeg is None:
                    # Firmware sends bounded capture acknowledgements alongside
                    # JPEGs. They are telemetry, never image bytes or commands.
                    if len(message.get("text", "")) > 512:
                        await ws.close(code=1009)
                        break
                    continue
                if not jpeg.startswith(b"\xff\xd8") or len(jpeg) > settings.max_frame_bytes:
                    await ws.close(code=1009)
                    break
                if pending.full():
                    pending.get_nowait()
                    runtime.frames_dropped += 1
                pending.put_nowait((jpeg, time.monotonic(), runtime.generation))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            runtime.camera = None
            runtime.latest_frame = None
            runtime.current_signatures.clear()
            runtime.last_observation_at = None
            await runtime.speech.stop()

    @app.websocket("/ws_audio")
    async def audio_commands(ws: WebSocket):
        from navguide.cloud.asr import ASRSession, asr_available

        await accept(ws)
        session = None
        send_lock = asyncio.Lock()

        async def reply(text):
            async with send_lock:
                await ws.send_text(text)

        async def command(text):
            text = text.strip().rstrip(SENTENCE_ENDINGS)
            if text in terms("commands.start_navigation"):
                await apply_context(ContextInput(task_mode=TaskMode.PATH_NAVIGATION))
            elif text in terms("commands.stop_navigation"):
                await runtime.stop_guidance()
            elif text in terms("commands.explore"):
                await apply_context(ContextInput(task_mode=TaskMode.SCENE_EXPLORATION))
            elif text.startswith(terms("commands.search_prefixes")) and text[3:].strip():
                from navguide.cloud.labels import async_extract_english_label

                target, _ = await async_extract_english_label(text[3:].strip())
                if not target:
                    await reply("ERR:UNKNOWN_TARGET")
                    return
                await apply_context(
                    ContextInput(task_mode=TaskMode.TARGET_SEARCH, target_query=target)
                )
            else:
                await reply("ERR:UNKNOWN_COMMAND")
                return
            await reply("OK:PROMPT_ACCEPTED")

        async def on_final(text):
            await runtime.broadcast({"type": "asr_final", "text": text})
            await command(text)

        async def on_partial(text):
            await runtime.broadcast({"type": "asr_partial", "text": text})

        async def on_error(code):
            await reply("ERR:ASR_UNAVAILABLE")

        async def stop_asr():
            nonlocal session
            if session:
                current, session = session, None
                await current.stop(cancel=True)

        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text") or ""
                data = message.get("bytes")
                if len(text) > 512 or (data is not None and len(data) > 16384):
                    await ws.close(code=1009)
                    break
                if text == "START":
                    await stop_asr()
                    if not asr_available():
                        await reply("ERR:ASR_DISABLED")
                        continue
                    session = ASRSession(
                        on_final=on_final, on_error=on_error, on_partial=on_partial
                    )
                    try:
                        await session.start()
                        await reply("OK:STARTED")
                    except Exception:
                        await stop_asr()
                        await reply("ERR:ASR_UNAVAILABLE")
                elif text == "STOP":
                    await stop_asr()
                    await reply("OK:STOPPED")
                elif text.startswith("PROMPT:"):
                    await command(text[7:])
                elif data is not None and session and not session.feed(data):
                    await stop_asr()
                    await reply("ERR:ASR_AUDIO_REJECTED")
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await stop_asr()

    @app.get("/stream.wav")
    async def stream_wav(request: Request):
        if len(runtime.speech.clients) >= 4:
            raise HTTPException(429, "Too many audio clients")
        queue = asyncio.Queue(maxsize=8)
        runtime.speech.clients.add(queue)

        async def generate():
            try:
                yield wav_stream_header()
                while not runtime.speech.closed and not await request.is_disconnected():
                    try:
                        yield await runtime.speech.next_chunk(queue, 0.5)
                    except asyncio.TimeoutError:
                        yield bytes(320)
            finally:
                runtime.speech.clients.discard(queue)

        return StreamingResponse(
            generate(), media_type="audio/wav", headers={"Cache-Control": "no-store"}
        )

    return app


app = create_app()


def main():
    import uvicorn

    settings = app.state.runtime.settings
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
        ws_max_size=settings.max_frame_bytes,
        ws_max_queue=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
