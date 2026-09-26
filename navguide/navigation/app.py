from navguide.i18n import terms, text as localized_text

# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, audioop
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re
from navguide.runtime.config import asset_path, env_bool
from navguide.runtime.imu import parse_imu, token_matches
from dotenv import load_dotenv

load_dotenv(asset_path("NAVGUIDE_ENV_FILE", ".env"))

from navguide.cloud.labels import async_extract_english_label
from navguide.navigation.coordinator import NavigationMaster, OrchestratorResult
from navguide.navigation.path_following import BlindPathNavigator
from navguide.navigation.street_crossing import CrossStreetNavigator
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from navguide.perception.obstacles import ObstacleDetectorClient

import mediapipe as mp
from navguide.io import frame_buffer
import threading
from navguide.perception import item_search

# Use the selector event loop on Windows.
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from dashscope import audio as dash_audio

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

MODEL = "paraformer-realtime-v2"
SAMPLE_RATE = 16000
AUDIO_FMT = "pcm"
CHUNK_MS = 20
BYTES_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000 * 2
SILENCE_20MS = bytes(BYTES_CHUNK)

from navguide.audio.stream import (
    register_stream_route,
    broadcast_pcm16_realtime,  # Broadcast PCM to active playback connections.
    hard_reset_audio,
    BYTES_PER_20MS,
    is_playing_now,
)
from navguide.cloud.descriptions import stream_chat, StreamPiece, cloud_available
from navguide.audio.recognition import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
)
from navguide.audio.player import initialize_audio_system, play_voice_text

from navguide.io import recorder
import signal
import atexit

# IMU UDP
UDP_IP = os.getenv("NAVGUIDE_UDP_HOST", "127.0.0.1")
UDP_PORT = int(os.getenv("NAVGUIDE_UDP_PORT", "12345"))

app = FastAPI()
from navguide.runtime.access import DeviceAccess
from navguide.runtime.config import Settings

app.add_middleware(DeviceAccess, settings=Settings.from_env())

app.mount(
    "/static", StaticFiles(directory=asset_path("NAVGUIDE_STATIC_DIR", "web/static")), name="static"
)

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None

blind_path_navigator = None
navigation_active = False
yolo_seg_model = None
obstacle_detector = None

cross_street_navigator = None
cross_street_active = False
orchestrator = None

omni_conversation_active = False
omni_previous_nav_state = None  # Navigation state to restore after the scene response.


def load_navigation_models():
    """Load the models used by tactile paving navigation."""
    global yolo_seg_model, obstacle_detector

    try:
        seg_model_path = asset_path("BLIND_PATH_MODEL", "model/yolo-seg.pt")

        if os.path.exists(seg_model_path):
            print(f"[NAVIGATION] Model file found, loading...")
            yolo_seg_model = YOLO(seg_model_path)

            if torch.cuda.is_available():
                yolo_seg_model.to("cuda")
                print(
                    f"[NAVIGATION] Tactile paving segmentation model loaded on GPU: {yolo_seg_model.device}"
                )
            else:
                print("[NAVIGATION] CUDA unavailable, using CPU")

            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                results = yolo_seg_model.predict(
                    test_img, device="cuda" if torch.cuda.is_available() else "cpu", verbose=False
                )
                print(
                    f"[NAVIGATION] Model check passed, class count: {(len(yolo_seg_model.names) if hasattr(yolo_seg_model, 'names') else 'unknown')}"
                )
                if hasattr(yolo_seg_model, "names"):
                    print(f"[NAVIGATION] Model classes: {yolo_seg_model.names}")
            except Exception as e:
                print(f"[NAVIGATION] Model check failed: {e}")
                yolo_seg_model = None
        else:
            print(f"[NAVIGATION] Error: model file not found: {seg_model_path}")
            print(f"[NAVIGATION] Working directory: {os.getcwd()}")
            print(f"[NAVIGATION] Check the configured file path")

        obstacle_model_path = asset_path("OBSTACLE_MODEL", "model/yoloe-11l-seg.pt")
        print(f"[NAVIGATION] Loading obstacle detector: {obstacle_model_path}")

        if os.path.exists(obstacle_model_path):
            print(f"[NAVIGATION] Obstacle model file found, loading...")
            try:
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                print(f"[NAVIGATION] YOLO-E obstacle detector loaded")

                if hasattr(obstacle_detector, "model") and obstacle_detector.model is not None:
                    print(f"[NAVIGATION] YOLO-E model initialized")
                    print(
                        f"[NAVIGATION] Model device: {next(obstacle_detector.model.parameters()).device}"
                    )
                else:
                    print(f"[NAVIGATION] Warning: YOLO-E model initialization incomplete")

                if hasattr(obstacle_detector, "WHITELIST_CLASSES"):
                    print(
                        f"[NAVIGATION] Allowed class count: {len(obstacle_detector.WHITELIST_CLASSES)}"
                    )
                    print(
                        f"[NAVIGATION] First 10 allowed classes: {', '.join(obstacle_detector.WHITELIST_CLASSES[:10])}"
                    )
                else:
                    print(f"[NAVIGATION] Warning: allowed classes are undefined")

                if (
                    hasattr(obstacle_detector, "whitelist_embeddings")
                    and obstacle_detector.whitelist_embeddings is not None
                ):
                    print(f"[NAVIGATION] YOLO-E text embeddings prepared")
                    print(
                        f"[NAVIGATION] Text embedding shape: {(obstacle_detector.whitelist_embeddings.shape if hasattr(obstacle_detector.whitelist_embeddings, 'shape') else 'unknown')}"
                    )
                else:
                    print(f"[NAVIGATION] Warning: YOLO-E text embeddings unavailable")

                print(f"[NAVIGATION] Checking YOLO-E detection...")
                try:
                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    cv2.rectangle(test_img, (200, 200), (400, 400), (255, 255, 255), -1)

                    test_results = obstacle_detector.detect(test_img)
                    print(f"[NAVIGATION] YOLO-E detection check passed")
                    print(f"[NAVIGATION] Test detection count: {len(test_results)}")

                    if len(test_results) > 0:
                        print(f"[NAVIGATION] Test detections:")
                        for i, obj in enumerate(test_results):
                            print(
                                f"  Object {i + 1}: {obj.get('name', 'unknown')}, area ratio: {obj.get('area_ratio', 0):.3f}, position: ({obj.get('center_x', 0):.0f}, {obj.get('center_y', 0):.0f})"
                            )
                except Exception as e:
                    print(f"[NAVIGATION] YOLO-E detection check failed: {e}")
                    import traceback

                    traceback.print_exc()

                print(f"[NAVIGATION] YOLO-E obstacle detector initialization complete")

            except Exception as e:
                print(f"[NAVIGATION] Obstacle detector loading failed: {e}")
                import traceback

                traceback.print_exc()
                obstacle_detector = None
        else:
            print(f"[NAVIGATION] Warning: obstacle model file not found: {obstacle_model_path}")

    except Exception as e:
        print(f"[NAVIGATION] Model loading failed: {e}")
        import traceback

        traceback.print_exc()


# Persist recording files when interrupted.
def cleanup_on_exit():
    """Close the recorder before the application exits."""
    print("\n[SYSTEM] Closing recorder...")
    try:
        recorder.stop_recording()
        print("[SYSTEM] Recordings saved")
    except Exception as e:
        print(f"[SYSTEM] Failed to close recorder: {e}")


def signal_handler(sig, frame):
    """Handle the interrupt signal and release recording resources."""
    print("\n[SYSTEM] Interrupt received, shutting down...")
    cleanup_on_exit()
    import sys

    sys.exit(0)


@app.on_event("startup")
async def initialize_models_and_recording():
    await asyncio.to_thread(load_navigation_models)
    if env_bool("NAVGUIDE_RECORDING_ENABLED"):
        recorder.start_recording()


interrupt_lock = asyncio.Lock()

item_search_thread: Optional[threading.Thread] = None
item_search_stop_event = threading.Event()
item_search_running = False
item_search_sending_frames = False

ITEM_TO_CLASS_MAP = {
    localized_text("object.red_bull"): "Red_Bull",
    localized_text("object.ad_milk"): "AD_milk",
    localized_text("object.ad_milk_lower"): "AD_milk",
    localized_text("object.calcium_milk"): "AD_milk",
}


async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)


async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[SPEECH FINAL] {text}", flush=True)


async def full_system_reset(reason: str = ""):
    """Cancel playback, stop recognition, clear interface and camera state,
    and notify the ESP32 to reset its audio session."""
    await hard_reset_audio(reason or "full_system_reset")

    # 2) ASR
    await stop_current_recognition()

    # 3) UI
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []

    try:
        last_frames.clear()
    except Exception:
        pass

    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass

    print("[SYSTEM] full reset done.", flush=True)


def start_item_search_with_target(target_name: str):
    """Start the item search worker for the requested target."""
    global item_search_thread, item_search_stop_event, item_search_running, item_search_sending_frames

    if item_search_running:
        stop_item_search()

    yolo_class = ITEM_TO_CLASS_MAP.get(target_name, target_name)
    print(
        f"[ITEM_SEARCH] Starting with target: {target_name} -> YOLO class: {yolo_class}", flush=True
    )
    print(f"[ITEM_SEARCH] Available mappings: {ITEM_TO_CLASS_MAP}", flush=True)

    item_search_stop_event.clear()
    item_search_running = True
    item_search_sending_frames = False

    def _run():
        try:
            item_search.main(
                headless=True, prompt_name=yolo_class, stop_event=item_search_stop_event
            )
        except Exception as e:
            print(f"[ITEM_SEARCH] worker stopped: {e}", flush=True)
        finally:
            global item_search_running, item_search_sending_frames
            item_search_running = False
            item_search_sending_frames = False

    item_search_thread = threading.Thread(target=_run, daemon=True)
    item_search_thread.start()
    print(
        f"[ITEM_SEARCH] background worker started for: {yolo_class} (initializing; displaying raw frames)",
        flush=True,
    )


def stop_item_search():
    """Stop the item search worker."""
    global item_search_thread, item_search_stop_event, item_search_running, item_search_sending_frames

    if item_search_running:
        print("[ITEM_SEARCH] Stopping worker...", flush=True)
        item_search_stop_event.set()

        # Wait up to five seconds for the worker to exit.
        if item_search_thread and item_search_thread.is_alive():
            item_search_thread.join(timeout=5.0)

        item_search_running = False
        item_search_sending_frames = False

        # Navigation resumes only through an explicit state transition.
        print("[ITEM_SEARCH] Worker stopped; awaiting a state change.", flush=True)


async def handle_voice_command(user_text: str):
    """Dispatch a voice command to the corresponding feature."""
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator

    # During navigation, only designated scene queries start a conversation.
    if orchestrator:
        current_state = orchestrator.get_state()
        if current_state not in ["CHAT", "IDLE"]:
            allowed_keywords = terms("app.scene_query_commands")
            is_allowed_query = any(keyword in user_text for keyword in allowed_keywords)

            nav_control_keywords = terms("app.mode_commands")
            is_nav_control = any(keyword in user_text for keyword in nav_control_keywords)

            if not is_allowed_query and not is_nav_control:
                mode_name = (
                    "traffic signal detection"
                    if current_state == "TRAFFIC_LIGHT_DETECTION"
                    else "navigation"
                )
                print(f"[{mode_name} mode] Ignoring unrelated speech: {user_text}")
                return

    if any(keyword in user_text for keyword in terms("app.crossing_start_commands")):
        if item_search_running:
            stop_item_search()
            print("[ITEM_SEARCH] Switching to street crossing")

        if orchestrator:
            orchestrator.start_crossing()
            print(f"[CROSS_STREET] Crossing started, state: {orchestrator.get_state()}")
            play_voice_text(localized_text("navigation.crossing_started"))
            await ui_broadcast_final(localized_text("app.crossing_started"))
        else:
            print("[CROSS_STREET] Warning: navigation coordinator is not initialized")
            play_voice_text(localized_text("navigation.crossing_start_failed"))
            await ui_broadcast_final(localized_text("app.navigation_not_ready"))
        return

    if any(keyword in user_text for keyword in terms("app.crossing_stop_commands")):
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[CROSS_STREET] Navigation stopped, state: {orchestrator.get_state()}")
            play_voice_text(localized_text("navigation.stopped"))
            await ui_broadcast_final(localized_text("app.crossing_stopped"))
        else:
            await ui_broadcast_final(localized_text("app.navigation_inactive"))
        return

    if any(keyword in user_text for keyword in terms("app.traffic_start_commands")):
        try:
            from navguide.perception import traffic_lights

            if orchestrator:
                orchestrator.start_traffic_light_detection()
                print(
                    f"[TRAFFIC] Traffic signal detection started, state: {orchestrator.get_state()}"
                )

            # Process traffic detection on the camera path to avoid competing frame consumers.
            success = traffic_lights.init_model()
            traffic_lights.reset_detection_state()

            if success:
                await ui_broadcast_final(localized_text("app.traffic_detection_started"))
            else:
                await ui_broadcast_final(localized_text("app.traffic_model_failed"))
        except Exception as e:
            print(f"[TRAFFIC] Failed to start traffic signal detection: {e}")
            await ui_broadcast_final(localized_text("app.start_failed").format(e=f"{e}"))
        return

    if any(keyword in user_text for keyword in terms("app.traffic_stop_commands")):
        try:
            if orchestrator:
                orchestrator.stop_navigation()
                print(
                    f"[TRAFFIC] Traffic signal detection stopped, restoring {orchestrator.get_state()} mode"
                )

            await ui_broadcast_final(localized_text("app.traffic_detection_stopped"))
        except Exception as e:
            print(f"[TRAFFIC] Failed to stop traffic signal detection: {e}")
            await ui_broadcast_final(localized_text("app.stop_failed").format(e=f"{e}"))
        return

    if any(keyword in user_text for keyword in terms("app.navigation_start_commands")):
        if yolo_seg_model is None:
            await ui_broadcast_final(localized_text("path.detection_unavailable"))
            return
        if item_search_running:
            stop_item_search()
            print("[ITEM_SEARCH] Switching to tactile paving navigation")

        if orchestrator:
            orchestrator.start_blind_path_navigation()
            print(
                f"[NAVIGATION] Tactile paving navigation started, state: {orchestrator.get_state()}"
            )
            await ui_broadcast_final(localized_text("app.path_navigation_started"))
        else:
            print("[NAVIGATION] Warning: navigation coordinator is not initialized")
            await ui_broadcast_final(localized_text("app.navigation_not_ready"))
        return

    if any(keyword in user_text for keyword in terms("app.navigation_stop_commands")):
        if orchestrator:
            orchestrator.stop_navigation()
            print(f"[NAVIGATION] Navigation stopped, state: {orchestrator.get_state()}")
            await ui_broadcast_final(localized_text("app.path_navigation_stopped"))
        else:
            await ui_broadcast_final(localized_text("app.navigation_inactive"))
        return

    nav_cmd_keywords = terms("app.navigation_commands")
    if any(k in user_text for k in nav_cmd_keywords):
        if orchestrator:
            orchestrator.on_voice_command(user_text)
            await ui_broadcast_final(localized_text("app.navigation_mode_updated"))
        else:
            await ui_broadcast_final(localized_text("app.coordinator_not_ready"))
        return

    find_pattern = localized_text("search.request_pattern")
    match = re.search(find_pattern, user_text)

    if match:
        item_cn = match.group(1).strip()
        if item_cn:
            # Resolve a detector label locally, then through the optional cloud provider.
            label_en, src = await async_extract_english_label(item_cn)
            if not label_en:
                await ui_broadcast_final(localized_text("app.search_target_unknown"))
                return
            print(f"[COMMAND] Finder request: '{item_cn}' -> '{label_en}' (src={src})", flush=True)

            # Pause navigation and set the target query before starting item search.
            if orchestrator:
                orchestrator.start_item_search(target_query=label_en)
                print(
                    f"[ITEM_SEARCH] Search started, state: {orchestrator.get_state()}, target: {label_en}"
                )

            start_item_search_with_target(label_en)

            try:
                await ui_broadcast_final(
                    localized_text("app.searching_for").format(item_cn=f"{item_cn}")
                )
            except Exception:
                pass

            return

    if any(keyword in user_text for keyword in terms("app.search_done_commands")):
        print("[COMMAND] Found command detected", flush=True)
        stop_item_search()

        if orchestrator:
            orchestrator.stop_item_search(restore_nav=True)
            current_state = orchestrator.get_state()
            print(f"[ITEM_SEARCH] Search complete, state: {current_state}")

            if current_state in [
                "BLINDPATH_NAV",
                "SEEKING_CROSSWALK",
                "WAIT_TRAFFIC_LIGHT",
                "CROSSING",
                "SEEKING_NEXT_BLINDPATH",
            ]:
                await ui_broadcast_final(localized_text("app.search_found_resume"))
            else:
                await ui_broadcast_final(localized_text("app.search_found"))
        else:
            await ui_broadcast_final(localized_text("app.search_found"))

        return

    if not cloud_available():
        await ui_broadcast_final(localized_text("app.cloud_disabled"))
        return

    global omni_conversation_active, omni_previous_nav_state
    omni_conversation_active = True

    if orchestrator:
        current_state = orchestrator.get_state()
        # Save the previous state only when a navigation workflow is active.
        if current_state not in ["CHAT", "IDLE"]:
            omni_previous_nav_state = current_state
            orchestrator.force_state("CHAT")
            print(f"[OMNI] Conversation started; switching from {current_state} to CHAT mode")
        else:
            omni_previous_nav_state = None
            print(f"[OMNI] Conversation started (already in {current_state} mode)")

    # Item search owns the camera while its worker is active.
    if item_search_running:
        print("[SCENE] Item search is running, skipping scene response", flush=True)
        return

    await start_scene_response(user_text)


async def start_scene_response(user_text: str):
    """Reset audio state and start a spoken scene response."""

    async def _runner():
        txt_buf: List[str] = []
        rate_state = None

        content_list = []
        if last_frames:
            try:
                _, jpeg_bytes = last_frames[-1]
                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                )
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})

        try:
            async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial(
                            localized_text("app.scene_prefix") + "".join(txt_buf)
                        )
                    except Exception:
                        pass

                # Omni returns base64 PCM16 at 24 kHz; the device stream requires 8 kHz.
                if piece.audio_b64:
                    try:
                        pcm24 = base64.b64decode(piece.audio_b64)
                    except Exception:
                        pcm24 = b""
                    if pcm24:
                        # Resample 24 kHz audio to 8 kHz without changing pitch or duration.
                        pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                        pcm8k = audioop.mul(pcm8k, 2, 0.60)
                        if pcm8k:
                            await broadcast_pcm16_realtime(pcm8k)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            try:
                await ui_broadcast_final(localized_text("app.scene_error").format(e=f"{e}"))
            except Exception:
                pass
        finally:
            global omni_conversation_active, omni_previous_nav_state
            omni_conversation_active = False

            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                print(f"[OMNI] Conversation ended, restoring {omni_previous_nav_state} mode")
                omni_previous_nav_state = None
            else:
                print(f"[OMNI] Conversation ended; no navigation state to restore")

            # Signal completion to existing listeners when the response ends normally.
            from navguide.audio.stream import (
                stream_clients,
            )  # Import locally to avoid a circular dependency.

            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try:
                        sc.q.put_nowait(b"\x00" * BYTES_PER_20MS)
                    except Exception:
                        pass
                    try:
                        sc.q.put_nowait(None)
                    except Exception:
                        pass

            final_text = ("".join(txt_buf)).strip() or localized_text("app.empty_response")
            try:
                await ui_broadcast_final(localized_text("app.scene_prefix") + final_text)
            except Exception:
                pass

    await hard_reset_audio("start_scene_response")
    loop = asyncio.get_running_loop()
    from navguide.audio import stream

    stream.current_playback_task = loop.create_task(_runner())


@app.get("/", response_class=HTMLResponse)
def root():
    with open(
        asset_path("NAVGUIDE_TEMPLATE", "web/templates/navigation.html"), "r", encoding="utf-8"
    ) as f:
        return HTMLResponse(f.read())


@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


register_stream_route(app)


@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)


@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] client connected")
    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None

    async def stop_rec(send_notice: Optional[str] = None):
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try:
                await keepalive_task
            except Exception:
                pass
        keepalive_task = None
        if recognition:
            try:
                recognition.stop()
            except Exception:
                pass
            recognition = None
        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try:
                await ws.send_text(send_notice)
            except Exception:
                pass

    async def on_sdk_error(_msg: str):
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop():
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("keepalive send failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if 'Cannot call "receive"' in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    if not API_KEY or not env_bool("NAVGUIDE_ASR_ENABLED"):
                        await ws.send_text("ERR:ASR_DISABLED")
                        continue
                    print("[AUDIO] START received")
                    await stop_rec()
                    loop = asyncio.get_running_loop()

                    def post(coro):
                        asyncio.run_coroutine_threadsafe(coro, loop)

                    cb = ASRCallback(
                        on_sdk_error=lambda s: post(on_sdk_error(s)),
                        post=post,
                        ui_broadcast_partial=ui_broadcast_partial,
                        ui_broadcast_final=ui_broadcast_final,
                        is_playing_now_fn=is_playing_now,
                        start_response_fn=handle_voice_command,
                        full_system_reset_fn=full_system_reset,
                        interrupt_lock=interrupt_lock,
                    )

                    recognition = dash_audio.asr.Recognition(
                        api_key=API_KEY,
                        model=MODEL,
                        format=AUDIO_FMT,
                        sample_rate=SAMPLE_RATE,
                        callback=cb,
                    )
                    await asyncio.to_thread(recognition.start)
                    await set_current_recognition(recognition)
                    streaming = True
                    last_ts = time.monotonic()
                    keepalive_task = asyncio.create_task(keepalive_loop())
                    await ui_broadcast_partial(localized_text("app.audio_receiving"))
                    await ws.send_text("OK:STARTED")

                elif cmd == "STOP":
                    if recognition:
                        for _ in range(15):
                            try:
                                recognition.send_audio_frame(SILENCE_20MS)
                            except Exception:
                                break
                    await stop_rec(send_notice="OK:STOPPED")

                elif raw.startswith("PROMPT:"):
                    # Reset the prior audio session before a device-initiated request.
                    text = raw[len("PROMPT:") :].strip()
                    if text:
                        async with interrupt_lock:
                            await handle_voice_command(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and recognition:
                    try:
                        recognition.send_audio_frame(msg["bytes"])
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("send_audio_frame failed")

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[WS] connection closed")


@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected")

    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        print("[NAVIGATION] Tactile paving navigator initialized")
    else:
        if blind_path_navigator is not None:
            print("[NAVIGATION] Navigator already initialized")
        elif yolo_seg_model is None:
            print("[NAVIGATION] Warning: YOLO model required to initialize navigator")

    if cross_street_navigator is None:
        if yolo_seg_model:
            cross_street_navigator = CrossStreetNavigator(
                seg_model=yolo_seg_model,
                coco_model=None,
                obs_model=None,  # Obstacle detection is disabled for this crossing configuration.
            )
            print("[CROSS_STREET] Crossing navigator initialized with crosswalk detection")
        else:
            print(
                "[CROSS_STREET] Error: segmentation model required to initialize crossing navigator"
            )

            if not yolo_seg_model:
                print("[CROSS_STREET] Missing segmentation model (yolo_seg_model)")
            if not obstacle_detector:
                print("[CROSS_STREET] Missing obstacle detector (obstacle_detector)")

    if (
        orchestrator is None
        and blind_path_navigator is not None
        and cross_street_navigator is not None
    ):
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[NAV MASTER] Navigation coordinator initialized")
    frame_counter = 0
    last_detection_notice = None

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frame_counter += 1

                try:
                    recorder.record_frame(data)
                except Exception as e:
                    if frame_counter % 100 == 0:
                        print(f"[RECORDER] Frame recording failed: {e}")

                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass

                frame_buffer.push_raw_jpeg(data)

                if frame_counter % 30 == 0:
                    state_dbg = orchestrator.get_state() if orchestrator else "N/A"
                    print(
                        f"[NAVIGATION DEBUG] Frame: {frame_counter}, state={state_dbg}, item_search_running={item_search_running}"
                    )

                try:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None or bgr.size == 0:
                        if frame_counter % 30 == 0:
                            print(f"[JPEG] Decode failed, data length={len(data)}")
                        bgr = None
                except Exception as e:
                    if frame_counter % 30 == 0:
                        print(f"[JPEG] Decode error: {e}")
                    bgr = None

                if orchestrator and not item_search_running and bgr is not None:
                    current_state = orchestrator.get_state()

                    if current_state == "ITEM_SEARCH":
                        # Show raw camera frames until the item search worker produces processed frames.
                        if not item_search_sending_frames and camera_viewers:
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok:
                                jpeg_data = enc.tobytes()
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for d in dead:
                                    camera_viewers.discard(d)
                        continue

                    out_img = bgr
                    detection_notice = None
                    try:
                        if current_state == "TRAFFIC_LIGHT_DETECTION":
                            from navguide.perception import traffic_lights

                            result = traffic_lights.process_single_frame(
                                bgr, ui_broadcast_callback=ui_broadcast_final
                            )
                            if not result.get("available", False):
                                detection_notice = "traffic.detection_unavailable"
                            out_img = (
                                result["vis_image"] if result["vis_image"] is not None else bgr
                            )
                        else:
                            res = orchestrator.process_frame(bgr)
                            if res.extras.get("detection_available") is False:
                                detection_notice = "path.detection_unavailable"
                            elif res.extras.get("signal_detection", {}).get("available") is False:
                                detection_notice = "traffic.detection_unavailable"

                            # Scene responses run in CHAT mode and suppress navigation cues.
                            if res.guidance_text:
                                try:
                                    play_voice_text(res.guidance_text)
                                    await ui_broadcast_final(
                                        localized_text("app.navigation_cue").format(
                                            guidance_text=f"{res.guidance_text}",
                                        ),
                                    )
                                except Exception:
                                    pass

                            out_img = (
                                res.annotated_image if res.annotated_image is not None else bgr
                            )
                        if detection_notice and detection_notice != last_detection_notice:
                            await ui_broadcast_final(localized_text(detection_notice))
                        last_detection_notice = detection_notice
                    except Exception as e:
                        if frame_counter % 100 == 0:
                            print(f"[NAV MASTER] Frame processing failed: {e}")

                    if camera_viewers and out_img is not None:
                        ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try:
                                    await viewer_ws.send_bytes(jpeg_data)
                                except Exception:
                                    dead.append(viewer_ws)
                            for d in dead:
                                camera_viewers.discard(d)
                    continue

                if not item_search_sending_frames and camera_viewers:
                    try:
                        if bgr is None:
                            arr = np.frombuffer(data, dtype=np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok:
                                jpeg_data = enc.tobytes()
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try:
                                        await viewer_ws.send_bytes(jpeg_data)
                                    except Exception:
                                        dead.append(viewer_ws)
                                for ws in dead:
                                    camera_viewers.discard(ws)
                    except Exception as e:
                        print(f"[CAMERA] Broadcast error: {e}")

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        print("[CAMERA] ESP32 disconnected")

        if blind_path_navigator:
            blind_path_navigator.reset()
        if cross_street_navigator:
            cross_street_navigator.reset()
        if orchestrator:
            orchestrator.reset()
            print("[NAV MASTER] Coordinator reset")


@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try:
            camera_viewers.remove(ws)
        except Exception:
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)


@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)


async def imu_broadcast(msg: str):
    if not imu_ws_clients:
        return
    dead = []
    for ws in list(imu_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        imu_ws_clients.discard(ws)


from math import atan2, hypot, pi

GRAV_BETA = 0.98
STILL_W = 0.4
YAW_DB = 0.08
YAW_LEAK = 0.2
ANG_EMA = 0.15
AUTO_REZERO = True
USE_PROJ = True
FREEZE_STILL = True
G = 9.807
A_TOL = 0.08 * G
gLP = {"x": 0.0, "y": 0.0, "z": 0.0}
gOff = {"x": 0.0, "y": 0.0, "z": 0.0}
BIAS_ALPHA = 0.002
yaw = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
holdStart = 0.0
isStill = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store = deque(maxlen=600)


def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0:
        a -= 360.0
    if a < -180.0:
        a += 360.0
    return a


def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall

    d = parse_imu(d)
    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0:
        t_ms = now_wall * 1000.0
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0:
        dt = 0.02
    else:
        dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms

    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro") or {}).get("x", 0.0)))
    wy = float(((d.get("gyro") or {}).get("y", 0.0)))
    wz = float(((d.get("gyro") or {}).get("z", 0.0)))

    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"] / gmag, "y": gLP["y"] / gmag, "z": gLP["z"] / gmag}

    roll = atan2(az, ay) * 180.0 / pi
    pitch = atan2(-ax, ay) * 180.0 / pi

    aNorm = hypot(ax, ay, az)
    wNorm = hypot(wx, wy, wz)
    nearFlat = abs(roll) < 2.0 and abs(pitch) < 2.0
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)

    if stillCond:
        if holdStart <= 0.0:
            holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA) * gOff["x"] + BIAS_ALPHA * wx
        gOff["y"] = (1.0 - BIAS_ALPHA) * gOff["y"] + BIAS_ALPHA * wy
        gOff["z"] = (1.0 - BIAS_ALPHA) * gOff["z"] + BIAS_ALPHA * wz
    else:
        holdStart = 0.0
        isStill = False

    if USE_PROJ:
        yawdot = (
            (wx - gOff["x"]) * gHat["x"]
            + (wy - gOff["y"]) * gHat["y"]
            + (wz - gOff["z"]) * gHat["z"]
        )
    else:
        yawdot = wy - gOff["y"]

    yawdot = d["yaw_rate_dps"]
    if abs(yawdot) < YAW_DB:
        yawdot = 0.0
    if FREEZE_STILL and stillCond:
        yawdot = 0.0

    # Forward torso angular velocity in degrees per second to inertial gating.
    global orchestrator
    if orchestrator is not None and hasattr(orchestrator, "update_imu_motion"):
        try:
            orchestrator.update_imu_motion(yawdot)
        except Exception:
            pass

    yaw = _wrap180(yaw + yawdot * dt)

    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step):
            yaw = 0.0
        else:
            yaw += step

    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw + (1.0 - ANG_EMA) * Yf

    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0:
            holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True

    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])

    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {
            "ts": t_ms / 1000.0,
            "angles": {"roll": R, "pitch": P, "yaw": Y},
            "accel": {"x": ax, "y": ay, "z": az},
            "gyro": {"x": wx, "y": wy, "z": wz},
        }
        imu_store.append(item)


class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")

    def datagram_received(self, data, addr):
        try:
            if len(data) > 2048:
                return
            s = data.decode("utf-8").strip()
            d = json.loads(s)
            if not isinstance(d, dict) or not token_matches(
                d.pop("token", ""), os.getenv("NAVGUIDE_DEVICE_TOKEN", "")
            ):
                return
            if "ts" not in d and "timestamp_ms" in d:
                d["ts"] = d.pop("timestamp_ms")
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass


@app.on_event("startup")
async def on_startup_register_bridge_sender():
    main_loop = asyncio.get_event_loop()

    def _sender(jpeg_bytes: bytes):
        # This callback may run in a worker thread; schedule sends on the main event loop.
        try:
            # Avoid scheduling sends after the event loop closes.
            if main_loop.is_closed():
                return

            global item_search_sending_frames
            if not item_search_sending_frames:
                item_search_sending_frames = True
                print(
                    "[ITEM_SEARCH] Processed frames available; switching to detection view",
                    flush=True,
                )

            async def _broadcast():
                if not camera_viewers:
                    return
                dead = []
                for ws in list(camera_viewers):
                    try:
                        await ws.send_bytes(jpeg_bytes)
                    except Exception as e:
                        dead.append(ws)
                for ws in dead:
                    try:
                        camera_viewers.remove(ws)
                    except Exception:
                        pass

            future = asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
            # Do not block the frame producer while broadcasting.
        except Exception as e:
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)

    frame_buffer.set_sender(_sender)


@app.on_event("startup")
async def on_startup_init_audio():
    """Initialize the audio system during startup."""

    # Initialize audio in a worker thread to keep startup responsive.
    def _init():
        try:
            initialize_audio_system()
        except Exception as e:
            print(f"[AUDIO] Initialization failed: {e}")

    threading.Thread(target=_init, daemon=True).start()


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    if env_bool("NAVGUIDE_UDP_ENABLED"):
        app.state.udp_transport, _ = await loop.create_datagram_endpoint(
            lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT)
        )


@app.on_event("shutdown")
async def on_shutdown():
    """Release application resources during shutdown."""
    print("[SHUTDOWN] Releasing resources...")
    cleanup_on_exit()
    transport = getattr(app.state, "udp_transport", None)
    if transport:
        transport.close()

    stop_item_search()

    await hard_reset_audio("shutdown")

    print("[SHUTDOWN] Resources released")


def get_last_frames():
    return last_frames


def get_camera_ws():
    return esp32_camera_ws


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("NAVGUIDE_HOST", "127.0.0.1"),
        port=int(os.getenv("NAVGUIDE_PORT", "8081")),
        log_level="warning",
        access_log=False,
        loop="asyncio",
        workers=1,
        reload=False,
    )
