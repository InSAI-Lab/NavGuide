from tests.language_data import text as localized_text
import asyncio

from fastapi.testclient import TestClient

from navguide.runtime.server import create_app
from navguide.runtime.config import Settings


def test_microphone_final_command_stops_guidance_and_disconnect_cancels_session(monkeypatch):
    from navguide import cloud
    import navguide.cloud.asr

    sessions = []

    class FakeSession:
        def __init__(self, on_final, on_error, on_partial):
            self.on_final = on_final
            self.closed = False
            sessions.append(self)

        async def start(self):
            pass

        def feed(self, data):
            assert data == b"\x00\x00" * 320
            asyncio.create_task(self.on_final(localized_text("command.stop_navigation_sentence")))
            return True

        async def stop(self, cancel=False):
            self.closed = cancel

    monkeypatch.setattr(cloud.asr, "ASRSession", FakeSession)
    monkeypatch.setattr(cloud.asr, "asr_available", lambda: True)
    with TestClient(create_app(Settings()), base_url="http://localhost") as client:
        with client.websocket_connect("ws://localhost/ws_audio") as ws:
            ws.send_text("START")
            assert ws.receive_text() == "OK:STARTED"
            ws.send_bytes(b"\x00\x00" * 320)
            assert ws.receive_text() == "OK:PROMPT_ACCEPTED"
            assert client.get("/api/status").json()["guidance_enabled"] is False
        # Client disconnect cleanup completes as the context exits.
        assert sessions[0].closed


def test_asr_failure_keeps_navigation_running(monkeypatch):
    from navguide import cloud
    import navguide.cloud.asr

    class FailingSession:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            raise TimeoutError("simulated upstream failure")

        async def stop(self, cancel=False):
            pass

    monkeypatch.setattr(cloud.asr, "ASRSession", FailingSession)
    monkeypatch.setattr(cloud.asr, "asr_available", lambda: True)
    with TestClient(create_app(Settings()), base_url="http://localhost") as client:
        with client.websocket_connect("ws://localhost/ws_audio") as ws:
            ws.send_text("START")
            assert ws.receive_text() == "ERR:ASR_UNAVAILABLE"
            assert client.get("/api/status").json()["guidance_enabled"] is True
