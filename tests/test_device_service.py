from tests.language_data import text as localized_text
import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from navguide.runtime.imu import IMUState, parse_imu
from navguide.runtime.server import IMUProtocol, create_app
from navguide.runtime.config import Settings

TOKEN = "test-token-with-at-least-24-characters"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def observation(yaw=0):
    return {
        "yaw_rate_dps": yaw,
        "detections": [
            {"category": "person", "confidence": 0.8, "bbox": [250, 40, 390, 430], "track_id": 1}
        ],
    }


def packet(yaw=0):
    return {
        "schema_version": 1,
        "ts": 20,
        "yaw_rate_dps": yaw,
        "accel": {"x": 0, "y": 0, "z": 9.81},
        "gyro": {"x": 0, "y": 0, "z": yaw},
    }


def test_authentication_and_paper_delivery_contract():
    with TestClient(create_app(Settings(device_token=TOKEN))) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/status").status_code == 401
        assert (
            client.get(
                "/api/status", headers={**HEADERS, "Origin": "https://untrusted.example"}
            ).status_code
            == 401
        )
        # Gate history must not consume a cue while the torso is turning.
        first = client.post("/api/observations", headers=HEADERS, json=observation(25)).json()
        assert not first["should_speak"]
        second = client.post("/api/observations", headers=HEADERS, json=observation(0)).json()
        assert second["should_speak"]
        assert second["capture_to_trigger_latency_ms"] is None
        assert second["audio_requested"] is False
        third = client.post("/api/observations", headers=HEADERS, json=observation(0)).json()
        assert not third["should_speak"]


def test_invalid_input_and_bounded_body():
    with TestClient(create_app(Settings(device_token=TOKEN, max_frame_bytes=1024))) as client:
        data = observation()
        data["detections"][0]["bbox"] = [390, 40, 250, 430]
        assert client.post("/api/observations", headers=HEADERS, json=data).status_code == 422
        assert client.post("/api/imu", headers=HEADERS, json=[]).status_code == 422
        assert (
            client.post("/api/observations", headers=HEADERS, content=b"x" * 1025).status_code
            == 413
        )
        assert (
            client.post(
                "/api/context", headers=HEADERS, json={"task_mode": "target_search"}
            ).status_code
            == 422
        )


def test_missing_imu_gates_ordinary_cues_and_explicit_sample_releases():
    with TestClient(create_app(Settings()), base_url="http://localhost") as client:
        sample = observation()
        sample.pop("yaw_rate_dps")
        result = client.post("/api/observations", json=sample).json()
        assert not result["should_speak"] and not result["imu_fresh"]
        assert client.post("/api/imu", json=packet()).status_code == 200
        assert client.post("/api/observations", json=sample).json()["should_speak"]


def test_imu_freshness_units_and_credential_removal():
    state = IMUState(0.5)
    assert state.yaw_rate(now=1) == 25
    state.update({**packet(26), "token": TOKEN}, now=1)
    assert state.yaw_rate(now=1.2) == 26
    assert "token" not in state.packet
    assert not state.fresh(now=1.6)
    sample = packet(30)
    sample.pop("yaw_rate_dps")
    assert parse_imu(sample)["yaw_rate_dps"] == 30
    sample["gyro"]["x"] = float("nan")
    with pytest.raises(ValueError):
        parse_imu(sample)


def test_udp_requires_token_and_never_rebroadcasts_it():
    app = create_app(Settings(device_token=TOKEN))
    runtime = app.state.runtime
    protocol = IMUProtocol(runtime)
    protocol.datagram_received(json.dumps(packet()).encode(), ("127.0.0.1", 1000))
    assert runtime.invalid_imu_packets == 1
    queue = asyncio.Queue(maxsize=1)
    runtime.events.add(queue)
    protocol.datagram_received(
        json.dumps({**packet(42), "token": TOKEN}).encode(), ("127.0.0.1", 1000)
    )
    assert runtime.imu.yaw_rate() == 42
    assert "token" not in queue.get_nowait()


def test_websocket_authentication_and_offline_commands():
    with TestClient(create_app(Settings(device_token=TOKEN))) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws_audio"):
                pass
        with client.websocket_connect("/ws_audio", headers=HEADERS) as ws:
            ws.send_text("START")
            assert ws.receive_text() == "ERR:ASR_DISABLED"
            ws.send_text(localized_text("protocol.start_navigation"))
            assert ws.receive_text() == "OK:PROMPT_ACCEPTED"
        with client.websocket_connect(
            "/ws/events", subprotocols=["navguide", "auth." + TOKEN]
        ) as ws:
            client.post("/api/observations", headers=HEADERS, json=observation())
            assert ws.receive_json()["type"] == "guidance"


def test_network_exposure_requires_token():
    with pytest.raises(ValueError):
        Settings(host="0.0.0.0")
    with pytest.raises(ValueError):
        Settings(udp_enabled=True, udp_host="0.0.0.0")


def test_stop_endpoint_stops_guidance_and_context_restarts_without_old_scene():
    with TestClient(create_app(Settings(device_token=TOKEN))) as client:
        assert client.post("/api/stop").status_code == 401
        assert (
            client.post(
                "/api/context",
                headers=HEADERS,
                json={"task_mode": "path_navigation", "scene": "intersection"},
            ).status_code
            == 200
        )
        assert client.post("/api/stop", headers=HEADERS).json()["status"] == "stopped"
        assert not client.get("/api/status", headers=HEADERS).json()["guidance_enabled"]
        stopped = client.post("/api/observations", headers=HEADERS, json=observation()).json()
        assert not stopped["should_speak"] and not stopped["audio_requested"]
        context = client.post(
            "/api/context", headers=HEADERS, json={"task_mode": "scene_exploration"}
        ).json()
        assert context["scene_type"] == "unknown"
        assert client.post("/api/observations", headers=HEADERS, json=observation()).json()[
            "should_speak"
        ]


def test_invalid_imu_ranges_and_schema_cannot_establish_stability():
    sample = packet()
    sample["schema_version"] = True
    with pytest.raises(ValueError):
        parse_imu(sample)
    sample = packet()
    sample.pop("yaw_rate_dps")
    sample["accel"]["x"] = 1e300
    with pytest.raises(ValueError):
        parse_imu(sample)
    with pytest.raises(ValueError):
        IMUState(float("nan"))


def test_loopback_origin_and_bearer_scheme_are_validated():
    with TestClient(create_app(Settings()), base_url="http://localhost") as client:
        assert (
            client.post(
                "/api/stop",
                headers={"Host": "attacker.example", "Origin": "http://attacker.example"},
            ).status_code
            == 401
        )
        assert client.get("/api/status", headers={"Host": "attacker.example"}).status_code == 401
    with TestClient(create_app(Settings(device_token=TOKEN))) as client:
        assert client.get("/api/status", headers={"Authorization": TOKEN}).status_code == 401
        assert (
            client.get("/api/status", headers={"Authorization": b"Bearer \xff"}).status_code == 401
        )


def test_camera_keeps_latest_pending_frame_without_starving_current_result():
    import threading
    import time
    from navguide.core.selection import DetectionCandidate

    class Frontend:
        def __init__(self):
            self.entered, self.release = threading.Event(), threading.Event()
            self.calls = []

        def detect(self, jpeg, max_pixels):
            self.calls.append(jpeg)
            if len(self.calls) == 1:
                self.entered.set()
                assert self.release.wait(2)
            category = "chair" if jpeg.endswith(b"first") else "table"
            return [DetectionCandidate(category, 0.9, (100, 100, 180, 210))], 640, 480

    frontend = Frontend()
    app = create_app(Settings(observation_timeout_seconds=3), frontend=frontend)
    with TestClient(app, base_url="http://localhost") as client:
        with client.websocket_connect("ws://localhost/ws/events") as events:
            with client.websocket_connect("ws://localhost/ws/camera") as camera:
                camera.send_text("SNAP:BEGIN")
                camera.send_bytes(b"\xff\xd8first")
                assert frontend.entered.wait(1)
                camera.send_bytes(b"\xff\xd8middle")
                camera.send_bytes(b"\xff\xd8last")
                camera.send_text("SNAP:END")
                deadline = time.monotonic() + 1
                while client.get("/api/status").json()["frames_dropped"] < 1:
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
                frontend.release.set()
                first, last = events.receive_json(), events.receive_json()
                assert first["selected_cues"][0]["category"] == "chair"
                assert last["selected_cues"][0]["category"] == "table"
                assert frontend.calls == [b"\xff\xd8first", b"\xff\xd8last"]


def test_cancelled_model_call_retains_runtime_lock_until_native_completion():
    import contextlib
    import threading
    from navguide.runtime.server import Runtime

    async def scenario():
        runtime = Runtime(Settings())
        entered, release = threading.Event(), threading.Event()
        completed = False
        second_entered = False

        def native():
            nonlocal completed
            entered.set()
            assert release.wait(2)
            completed = True

        async def first():
            async with runtime.lock:
                await runtime.run_model(native)

        async def second():
            nonlocal second_entered
            async with runtime.lock:
                assert completed
                second_entered = True

        first_task = asyncio.create_task(first())
        assert await asyncio.to_thread(entered.wait, 1)
        first_task.cancel()
        second_task = asyncio.create_task(second())
        try:
            await asyncio.sleep(0.01)
            assert not second_entered
        finally:
            release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await first_task
        await second_task
        assert second_entered

    asyncio.run(scenario())


def test_runtime_rechecks_imu_scene_and_frame_freshness_during_audio():
    import time
    from navguide.runtime.server import Runtime
    from navguide.core.selection import DetectionCandidate

    async def scenario():
        runtime = Runtime(Settings(imu_timeout_seconds=0.1))
        callbacks = []
        runtime.speech.submit = lambda text, trigger, valid: callbacks.append(valid) or True
        item = DetectionCandidate("chair", 0.9, (100, 100, 180, 210))
        runtime.imu.update(packet(0))
        await runtime.process([item], 640, 480)
        valid = callbacks[-1]
        assert valid()
        runtime.imu.received_at -= 1
        assert not valid()
        runtime.imu.update(packet(0))
        assert valid()
        runtime.imu.update(packet(35))
        assert not valid()
        runtime.imu.update(packet(0))
        await runtime.process([], 640, 480)
        assert not valid()
        stale = await runtime.process([item], 640, 480, yaw=0, received=time.monotonic() - 2)
        assert not stale["should_speak"] and stale["reason"] == "stale_observation"
        await runtime.stop_guidance()
        assert not valid()

    asyncio.run(scenario())


def test_inline_yaw_sample_expires_and_newer_imu_takes_precedence():
    import time
    from navguide.runtime.server import Runtime

    runtime = Runtime(Settings(imu_timeout_seconds=0.1))
    runtime.last_observation_at = time.monotonic() - 1
    runtime.last_observation_yaw = 0
    assert runtime.current_yaw() == 25
    runtime.last_observation_at = time.monotonic()
    assert runtime.current_yaw() == 0
    runtime.imu.update(packet(35))
    assert runtime.current_yaw() == 35
