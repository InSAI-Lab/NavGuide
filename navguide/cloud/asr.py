"""Optional Paraformer session with bounded audio buffering and a socket worker.

The official WebSocket protocol avoids the SDK's unbounded internal audio queue
and blocking stop(). All user callbacks run on the caller's asyncio event loop.
"""
import asyncio
from dataclasses import dataclass, field
import inspect
import json
import os
import queue
import threading
import time
from urllib.parse import urlsplit
import uuid

from .config import CloudUnavailable, _positive_float


@dataclass(frozen=True)
class ASRSettings:
    enabled: bool = False
    api_key: str = field(default="", repr=False)
    url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    model: str = "paraformer-realtime-v2"
    connect_timeout: float = 10.0
    max_session_seconds: float = 60.0
    queue_frames: int = 64
    max_frame_bytes: int = 16384

    @classmethod
    def from_env(cls):
        url = os.getenv("DASHSCOPE_ASR_URL", cls.url).strip()
        parsed = urlsplit(url)
        if parsed.scheme != "wss" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("DASHSCOPE_ASR_URL must be a WSS URL without embedded credentials")
        return cls(enabled=os.getenv("NAVGUIDE_ASR_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
                   api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(), url=url,
                   model=os.getenv("ASR_MODEL", cls.model),
                   connect_timeout=_positive_float("ASR_CONNECT_TIMEOUT_SECONDS", 10),
                   max_session_seconds=_positive_float("ASR_MAX_SESSION_SECONDS", 60))


def asr_available():
    try:
        settings = ASRSettings.from_env()
        return settings.enabled and bool(settings.api_key)
    except ValueError:
        return False


class ASRSession:
    """One START/STOP session. feed() accepts 16 kHz mono little-endian PCM16.

    on_final(text), on_partial(text), and on_error(code) may be sync or async.
    Rejected frames return False. An overflow cancels this session rather than
    producing recognition commands from an incomplete recording.
    """
    def __init__(self, on_final, on_error, on_partial=None, settings=None, socket_factory=None):
        self.settings = settings or ASRSettings.from_env()
        self.on_final, self.on_error, self.on_partial = on_final, on_error, on_partial
        self.socket_factory = socket_factory
        self.audio = queue.Queue(maxsize=self.settings.queue_frames)
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._socket = None
        self._thread = None
        self._loop = None
        self._ready = None
        self._callbacks = None
        self._callback_task = None

    def _schedule(self, callback, text):
        if callback is None or self._loop is None or self._loop.is_closed():
            return

        def put():
            if self._cancel.is_set():
                return
            try:
                self._callbacks.put_nowait((callback, text))
            except asyncio.QueueFull:
                # Backpressure prevents unbounded callback tasks on the local loop.
                self._cancel.set()

        self._loop.call_soon_threadsafe(put)

    async def _dispatch(self):
        while True:
            callback, text = await self._callbacks.get()
            try:
                if not self._cancel.is_set() or callback == self.on_error:
                    result = callback(text)
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                # A UI callback failure must not tear down the local navigation loop.
                pass
            finally:
                self._callbacks.task_done()
            if self._callback_task is None:
                return

    def _set_ready(self, error=None):
        def resolve():
            if not self._ready.done():
                if error:
                    self._ready.set_exception(CloudUnavailable(error))
                else:
                    self._ready.set_result(None)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(resolve)

    async def start(self):
        if self._thread is not None:
            raise RuntimeError("Create a new ASRSession for each START")
        if not self.settings.enabled or not self.settings.api_key:
            raise CloudUnavailable("Optional speech recognition is disabled or not configured")
        self._loop = asyncio.get_running_loop()
        self._ready = self._loop.create_future()
        self._callbacks = asyncio.Queue(maxsize=32)
        self._callback_task = asyncio.create_task(self._dispatch())
        self._thread = threading.Thread(target=self._worker, name="navguide-asr", daemon=True)
        self._thread.start()
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), self.settings.connect_timeout + 0.5)
        except BaseException:
            await self.stop(cancel=True)
            if self._ready.done() and not self._ready.cancelled():
                self._ready.exception()
            else:
                self._ready.cancel()
            raise

    def feed(self, pcm):
        if (self._thread is None or self._ready is None or not self._ready.done()
                or self._stop.is_set() or self._cancel.is_set() or self._done.is_set()):
            return False
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2 or len(pcm) > self.settings.max_frame_bytes:
            return False
        try:
            self.audio.put_nowait(pcm)
            return True
        except queue.Full:
            # Notify before cancelling; _worker also closes the socket promptly.
            self._loop.call_soon_threadsafe(self._notify_overflow)
            self._cancel.set()
            return False

    def _notify_overflow(self):
        # An overflow notification is allowed even after the session is cancelled.
        try:
            self._callbacks.put_nowait((self.on_error, "asr_audio_queue_full"))
        except asyncio.QueueFull:
            pass

    async def stop(self, cancel=False):
        self._stop.set()
        if cancel:
            self._cancel.set()
            if self._socket is not None:
                self._socket.abort()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 3.5 if not cancel else 0.5)
            if self._thread.is_alive():
                self._cancel.set()
                if self._socket is not None:
                    self._socket.abort()
        if self._callback_task is not None:
            if asyncio.current_task() is self._callback_task:
                self._callback_task = None
                return
            if not cancel:
                try:
                    await asyncio.wait_for(self._callbacks.join(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
            self._callback_task.cancel()
            try:
                await self._callback_task
            except asyncio.CancelledError:
                pass
            self._callback_task = None

    def _worker(self):
        task_id = str(uuid.uuid4())
        started = False
        finishing = False
        deadline = time.monotonic() + self.settings.connect_timeout
        try:
            import websocket
            factory = self.socket_factory or websocket.WebSocket
            self._socket = factory(enable_multithread=True)
            self._socket.connect(self.settings.url,
                                 header=["Authorization: Bearer " + self.settings.api_key],
                                 timeout=self.settings.connect_timeout)
            if self._cancel.is_set():
                return
            self._socket.send(json.dumps({"header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {"task_group": "audio", "task": "asr", "function": "recognition",
                            "model": self.settings.model, "parameters": {"format": "pcm", "sample_rate": 16000,
                            "semantic_punctuation_enabled": False}, "input": {}}}))
            self._socket.settimeout(0.02)
            while not self._cancel.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError("asr_timeout")
                if started and not finishing:
                    try:
                        frame = self.audio.get_nowait()
                    except queue.Empty:
                        frame = None
                    if frame is not None:
                        self._socket.send_binary(frame)
                    elif self._stop.is_set():
                        self._socket.send(json.dumps({"header": {"action": "finish-task", "task_id": task_id,
                                                                "streaming": "duplex"}, "payload": {"input": {}}}))
                        finishing = True
                        deadline = time.monotonic() + 3
                try:
                    raw = self._socket.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not raw:
                    raise RuntimeError("asr_disconnected")
                if len(raw) > 65536:
                    raise ValueError("asr_invalid_response")
                event = json.loads(raw)
                header = event.get("header", {})
                if header.get("task_id") != task_id:
                    continue
                kind = header.get("event")
                if kind == "task-started":
                    started = True
                    deadline = time.monotonic() + self.settings.max_session_seconds
                    self._set_ready()
                elif kind == "result-generated":
                    sentence = event.get("payload", {}).get("output", {}).get("sentence", {})
                    text = sentence.get("text", "")
                    if text and not sentence.get("heartbeat"):
                        self._schedule(self.on_final if sentence.get("sentence_end") else self.on_partial, text)
                elif kind == "task-finished":
                    return
                elif kind == "task-failed":
                    raise RuntimeError("asr_upstream_failed")
        except Exception:
            if not self._cancel.is_set():
                self._schedule(self.on_error, "asr_unavailable")
            if not started:
                self._set_ready("Optional speech recognition is unavailable")
        finally:
            if self._socket is not None:
                try:
                    self._socket.close(timeout=0.1)
                except Exception:
                    pass
            self._done.set()
