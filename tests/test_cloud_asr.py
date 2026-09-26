from tests.language_data import text as localized_text
import asyncio
import json
import queue
import threading
import time

import pytest
import websocket

from navguide.cloud.asr import ASRSession, ASRSettings, asr_available
from navguide.cloud.config import CloudUnavailable


class Socket:
    def __init__(self, **kwargs):
        self.events = queue.Queue()
        self.commands = []
        self.frames = []
        self.closed = False
        self.aborted = threading.Event()
        self.connect_thread = None

    def connect(self, url, header, timeout):
        self.connect_thread = threading.get_ident()

    def settimeout(self, timeout):
        self.timeout = timeout

    def emit(self, event, sentence=None):
        self.events.put(
            json.dumps(
                {
                    "header": {"event": event, "task_id": self.task_id},
                    "payload": {"output": {"sentence": sentence or {}}},
                }
            )
        )

    def send(self, value):
        data = json.loads(value)
        self.commands.append(data)
        self.task_id = data["header"]["task_id"]
        if data["header"]["action"] == "run-task":
            self.emit("task-started")
        else:
            self.emit(
                "result-generated",
                {"text": localized_text("command.start_navigation"), "sentence_end": True},
            )
            self.emit("task-finished")

    def send_binary(self, data):
        self.frames.append(data)
        self.emit(
            "result-generated",
            {"text": localized_text("command.partial_start"), "sentence_end": False},
        )

    def recv(self):
        if self.aborted.is_set():
            return ""
        try:
            return self.events.get(timeout=self.timeout)
        except queue.Empty:
            raise websocket.WebSocketTimeoutException()

    def abort(self):
        self.aborted.set()

    def close(self, timeout=0.1):
        self.closed = True


def test_asr_is_opt_in(monkeypatch):
    monkeypatch.delenv("NAVGUIDE_ASR_ENABLED", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "mock")
    assert not asr_available()

    async def scenario():
        session = ASRSession(lambda _: None, lambda _: None)
        with pytest.raises(CloudUnavailable):
            await session.start()

    asyncio.run(scenario())


def test_asr_pcm_and_protocol_with_event_loop_callbacks():
    async def scenario():
        sock = Socket()
        finals, partials, errors = [], [], []
        loop_thread = threading.get_ident()

        async def final(text):
            assert threading.get_ident() == loop_thread
            finals.append(text)

        session = ASRSession(
            final,
            errors.append,
            partials.append,
            settings=ASRSettings(enabled=True, api_key="mock"),
            socket_factory=lambda **_: sock,
        )
        await session.start()
        assert sock.connect_thread != loop_thread
        assert not session.feed(b"odd")
        assert session.feed(bytes(3200))
        await asyncio.sleep(0.03)
        await session.stop()
        assert (
            finals == [localized_text("command.start_navigation")]
            and partials == [localized_text("command.partial_start")]
            and not errors
        )
        assert sock.closed and not session._thread.is_alive()
        assert sock.commands[0]["payload"]["parameters"]["sample_rate"] == 16000
        assert sock.commands[-1]["header"]["action"] == "finish-task"
        assert not session.feed(bytes(3200))

    asyncio.run(scenario())


def test_asr_queue_is_bounded_and_cancellation_closes_socket():
    async def scenario():
        class SlowSocket(Socket):
            def send_binary(self, data):
                self.aborted.wait(0.2)

        sock = SlowSocket()
        errors = []
        session = ASRSession(
            lambda _: None,
            errors.append,
            settings=ASRSettings(enabled=True, api_key="mock", queue_frames=1),
            socket_factory=lambda **_: sock,
        )
        await session.start()
        assert session.feed(bytes(3200))
        assert not session.feed(bytes(3200))
        assert session.audio.qsize() <= 1
        await asyncio.sleep(0.01)
        start = time.monotonic()
        await session.stop(cancel=True)
        assert time.monotonic() - start < 1
        assert sock.closed and not session._thread.is_alive()
        assert errors == ["asr_audio_queue_full"]

    asyncio.run(scenario())


def test_asr_connection_failure_is_redacted():
    async def scenario():
        class FailedSocket(Socket):
            def connect(self, *args, **kwargs):
                raise RuntimeError("secret upstream credentials")

        sock = FailedSocket()
        session = ASRSession(
            lambda _: None,
            lambda _: None,
            settings=ASRSettings(enabled=True, api_key="mock"),
            socket_factory=lambda **_: sock,
        )
        with pytest.raises(CloudUnavailable) as failure:
            await session.start()
        assert "secret" not in str(failure.value)
        assert sock.closed and not session._thread.is_alive()

    asyncio.run(scenario())


def test_asr_server_failure_does_not_emit_navigation_command():
    async def scenario():
        sock = Socket()
        finals, errors = [], []
        session = ASRSession(
            finals.append,
            errors.append,
            settings=ASRSettings(enabled=True, api_key="mock"),
            socket_factory=lambda **_: sock,
        )
        await session.start()
        sock.emit("task-failed")
        await asyncio.sleep(0.03)
        await session.stop()
        assert finals == [] and errors == ["asr_unavailable"]
        assert sock.closed

    asyncio.run(scenario())
