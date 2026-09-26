# -*- coding: utf-8 -*-
"""Open-vocabulary item segmentation, hand landmarks, and polygon tracking.

The object-to-hand area ratio controls proximity feedback:
range_score = 1 - clamp(|ratio - 1| / RATIO_TOL, 0..1).
Tracking points are sampled from the mask interior and corrected by detection."""

from navguide.i18n import text as localized_text

import os
from navguide.runtime.config import asset_path
import time
import threading
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
from navguide.io import frame_buffer
import pygame

from navguide.audio.player import play_audio_threadsafe

PERF_DEBUG = False
HAND_DOWNSCALE = 0.8  # Scale both input dimensions before hand detection.
HAND_FPS_DIV = 1  # Run hand detection every N frames.


# BGR colors and top-right text layout.
FRONTEND_COLORS = {
    "text": (230, 237, 243),  # --text: #e6edf3
    "muted": (159, 176, 195),  # --muted: #9fb0c3
    "ok": (126, 231, 135),  # --ok: #7ee787
    "err": (128, 128, 255),  # --err: #ff8080 (BGR)
    "accent": (251, 218, 97),
}


CURRENT_COMMAND_TEXT = ":"

_UI_LINE = 0
_UI_H = 0
_UI_TR_LINE = 0
_UI_TOP_MARGIN = 12
_UI_RIGHT_MARGIN = 12
UNIFIED_FONT_PX = 12


def ui_reset_overlay(img_h: int):
    """Reset the text overlay row counters for the current frame."""
    global _UI_LINE, _UI_H, _UI_TR_LINE
    _UI_LINE = 0
    _UI_TR_LINE = 0
    _UI_H = int(img_h)


def _ui_next_y_top(font_size: int) -> int:
    """Return the next top-aligned text row and advance the row counter."""
    global _UI_TR_LINE
    line_gap = max(4, int(font_size * 0.25))
    y_top = _UI_TOP_MARGIN + (_UI_TR_LINE * (font_size + line_gap))
    _UI_TR_LINE += 1
    return y_top


def set_current_command(text: str):
    global CURRENT_COMMAND_TEXT
    try:
        CURRENT_COMMAND_TEXT = str(text) if text else ":"
    except Exception:
        CURRENT_COMMAND_TEXT = ":"


def draw_command_pill(img_bgr: np.ndarray, label: str):
    """Draw the current instruction in the top-right overlay."""
    text_prefix = localized_text("search.command_prefix")
    full_text = f"{text_prefix}{label if label else ':'}"

    draw_text_cn(
        img_bgr, full_text, (0, 0), font_size=UNIFIED_FONT_PX, color=(255, 255, 255), ui_hint=True
    )


try:
    from navguide.perception.open_vocabulary import YoloEBackend

    _YOLOE_READY = True
except Exception as e:
    _YOLOE_READY = False
    print(f"[DETECTOR] YOLOE backend not ready: {e}", flush=True)


YOLO_MODEL_PATH = asset_path("ITEM_SEARCH_MODEL", "model/shoppingbest5.pt")
HAND_TASK_PATH = asset_path("HAND_TASK_PATH", "model/hand_landmarker.task")


CAM_INDEX = 0
INPUT_W, INPUT_H = 600, 480


STROKE_WIDTH = 5
MASK_ALPHA = 0.45
CONF_THRESHOLD = 0.20


PROMPT_NAME = "AD_milk"
PROMPT_STRICT = True


ALIGN_LOOSE_PCT = 0.12  # Distance threshold normalized by the image diagonal.


RATIO_IDEAL = 1.0  # Target object-to-hand area ratio.
RATIO_TOL = 0.25  # Allow a 25% deviation from the target ratio.


TTS_INTERVAL_SEC = 1.0
ENABLE_TTS = True


LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
)
FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.001, minDistance=5, blockSize=7)


INNER_OFFSET_PX_LOCK = 5  # Erode the mask before selecting tracking points.
EDGE_DILATE_PX = 2  # Dilate the inner boundary before feature sampling.
PERI_MONITOR_PX = 40  # Monitor a 40-pixel band outside the tracked polygon.
PERI_CHECK_EVERY = 5  # Check peripheral segmentation every N frames.


CONTOUR_EPSILON_FACTOR = 0.002  # Douglas-Peucker tolerance relative to contour length.
TRACK_EPSILON_FACTOR = 0.003


YOLO_CORRECTION_IOU_THRESHOLD = 0.2  # IoU threshold for replacing the tracked mask.
YOLO_CORRECTION_CONF_THRESHOLD = 0.15


AUDIO_DIR = asset_path("AUDIO_BASE_DIR", "music")
AUDIO_FILES = {
    localized_text("direction.up"): os.path.join(AUDIO_DIR, "up.wav"),
    localized_text("direction.down"): os.path.join(AUDIO_DIR, "down.wav"),
    localized_text("direction.left"): os.path.join(AUDIO_DIR, "left.wav"),
    localized_text("direction.right"): os.path.join(AUDIO_DIR, "right.wav"),
    localized_text("direction.front"): os.path.join(AUDIO_DIR, "forward.wav"),
    localized_text("direction.back"): os.path.join(AUDIO_DIR, "backward.wav"),
    "OK": os.path.join(AUDIO_DIR, "ok.wav"),
}
GUIDANCE_INTERVAL_SEC = 1.5


pygame.mixer.init()


WINDOW = "YOLO Seg + Flow Polygon (Peri-Relock) (Grab Guidance)"


BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Latest asynchronous hand detection result.
_last_result = None  # (result, timestamp_ms)


def on_result(
    result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global _last_result
    _last_result = (result, timestamp_ms)


def _to_proto(hand_lms) -> landmark_pb2.NormalizedLandmarkList:
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in hand_lms])
    return proto


def draw_hands_mono(img_bgr, hand_lms, color=(0, 255, 255), r=2, t=2):
    mp_drawing = mp.solutions.drawing_utils
    landmark_spec = mp_drawing.DrawingSpec(color=color, thickness=-1, circle_radius=r)
    connection_spec = mp_drawing.DrawingSpec(color=color, thickness=t, circle_radius=r)
    if hasattr(hand_lms, "landmark"):
        proto = hand_lms
    else:
        proto = _to_proto(hand_lms)
    mp_drawing.draw_landmarks(
        img_bgr,
        landmark_list=proto,
        connections=HAND_CONNECTIONS,
        landmark_drawing_spec=landmark_spec,
        connection_drawing_spec=connection_spec,
    )


def norm_name(s: str) -> str:
    return "".join(str(s).lower().split())


# TTS (pyttsx3)
class Speaker:
    def __init__(self, enable=True):
        self.enable = enable
        self._engine = None
        self._lock = threading.Lock()
        if enable:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 190)
                self._engine.setProperty("volume", 1.0)
            except Exception:
                self._engine = None
                self.enable = False

    def say_async(self, text: str):
        if not self.enable or not text:
            return

        def _run():
            try:
                with self._lock:
                    self._engine.stop()
                    self._engine.say(text)
                    self._engine.iterate()
                    t0 = time.time()
                    while self._engine.isBusy() and (time.time() - t0) < 1.2:
                        self._engine.iterate()
                        time.sleep(0.01)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()


# Use Pillow for localized text rendering when available.
_PIL_OK = False
_FONT_PATH = None


def _init_font():
    global _PIL_OK, _FONT_PATH
    try:
        from PIL import ImageFont  # noqa

        _PIL_OK = True
    except Exception:
        _PIL_OK = False
        return
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyh.ttf",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simfang.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        r"C:\\Windows\\Fonts\\simsunb.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            _FONT_PATH = p
            return
    _PIL_OK = False


_init_font()


def draw_text_cn(img_bgr, text, xy, font_size=20, color=(255, 255, 255), stroke=None, ui_hint=True):
    """Draw localized text in a top-right stack using the configured font."""

    color = (255, 255, 255)
    font_size = int(UNIFIED_FONT_PX)

    H, W = img_bgr.shape[:2]
    # Stack text at the top right using measured text width.
    y_top = _ui_next_y_top(font_size) if ui_hint else _ui_next_y_top(font_size)

    tw = th = 0
    font_obj = None

    if _PIL_OK and _FONT_PATH:
        try:
            from PIL import Image, ImageDraw, ImageFont

            font_obj = ImageFont.truetype(_FONT_PATH, font_size)

            bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font_obj)
            tw = max(1, bbox[2] - bbox[0])
            th = max(1, bbox[3] - bbox[1])
        except Exception:
            pass
    if _PIL_OK and _FONT_PATH and font_obj is not None:
        try:
            from PIL import Image, ImageDraw

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            x = max(8, W - _UI_RIGHT_MARGIN - tw)
            y = y_top
            draw.text((x, y), text, fill=(255, 255, 255), font=font_obj)
            img_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    # Estimate text width for the OpenCV fallback.
    if tw <= 0 or th <= 0:
        scale = font_size / 24.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(8, W - _UI_RIGHT_MARGIN - int(tw))
    y_baseline = int(y_top + th)
    cv2.putText(
        img_bgr,
        text,
        (x, y_baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size / 24.0,
        color,
        2,
        cv2.LINE_AA,
    )


def clamp01(x):
    return max(0.0, min(1.0, x))


def draw_progress_bars(vis, align_score, range_score):
    """Draw alignment and proximity scores."""
    H, W = vis.shape[:2]
    bar_w = int(W * 0.28)
    bar_h = 12
    gap = 8
    x0 = 12
    y0 = H - 2 * bar_h - gap - 12

    cv2.rectangle(vis, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(vis, (x0, y0 + bar_h + gap), (x0 + bar_w, y0 + 2 * bar_h + gap), (50, 50, 50), -1)

    cv2.rectangle(
        vis, (x0, y0), (x0 + int(bar_w * clamp01(align_score)), y0 + bar_h), (0, 220, 0), -1
    )
    cv2.rectangle(
        vis,
        (x0, y0 + bar_h + gap),
        (x0 + int(bar_w * clamp01(range_score)), y0 + 2 * bar_h + gap),
        (0, 180, 255),
        -1,
    )
    draw_text_cn(
        vis, localized_text("search.alignment"), (x0, y0 - 18), font_size=18, color=(180, 180, 180)
    )
    draw_text_cn(
        vis,
        localized_text("search.distance_ratio"),
        (x0, y0 + bar_h + gap - 18),
        font_size=18,
        color=(180, 180, 180),
    )


def polygon_center_and_area(poly):
    if poly is None or len(poly) < 3:
        return None, 0.0
    poly = np.array(poly, dtype=np.float32)
    M = cv2.moments(poly)
    if abs(M["m00"]) < 1e-6:
        c = np.mean(poly, axis=0)
        return (float(c[0]), float(c[1])), 0.0
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    area = float(cv2.contourArea(poly.astype(np.int32)))
    return (cx, cy), area


def hand_bbox_and_area(lms, W, H):
    xs = [int(p.x * W) for p in lms]
    ys = [int(p.y * H) for p in lms]
    if not xs or not ys:
        return None, 0.0
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    area = float(w * h)
    return (x0, y0, w, h), area


THUMB_INDEX_CLOSE = 0.34
FINGERTIP_NEAR = 0.44
MIN_CURLED_COUNT = 1


def detect_grasp(hand_lms, W, H):
    box, _ = hand_bbox_and_area(hand_lms, W, H)
    if not box:
        return False, 0.0
    x0, y0, w0, h0 = box
    hand_diag = float(np.hypot(w0, h0)) + 1e-6
    palm_idx = [0, 5, 9, 13, 17]
    px = np.mean([hand_lms[i].x * W for i in palm_idx])
    py = np.mean([hand_lms[i].y * H for i in palm_idx])
    palm = np.array([px, py], dtype=np.float32)
    t4 = np.array([hand_lms[4].x * W, hand_lms[4].y * H], dtype=np.float32)
    t8 = np.array([hand_lms[8].x * W, hand_lms[8].y * H], dtype=np.float32)
    thumb_index_dist = float(np.linalg.norm(t4 - t8)) / hand_diag
    tips = [12, 16, 20]
    dists = []
    for i in tips:
        ti = np.array([hand_lms[i].x * W, hand_lms[i].y * H], dtype=np.float32)
        dists.append(float(np.linalg.norm(ti - palm)) / hand_diag)
    curled_cnt = sum(1 for d in dists if d < FINGERTIP_NEAR)
    cond1 = thumb_index_dist < THUMB_INDEX_CLOSE
    cond2 = curled_cnt >= MIN_CURLED_COUNT
    score = 0.5 * (1.0 - min(thumb_index_dist / THUMB_INDEX_CLOSE, 1.0)) + 0.5 * min(
        curled_cnt / 3.0, 1.0
    )
    return (cond1 and cond2), score


# Sample features from the eroded mask boundary.
def inner_offset_edge(mask_bin, offset_px=5, edge_dilate_px=2):
    if offset_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * offset_px + 1, 2 * offset_px + 1))
        eroded = cv2.erode(mask_bin.astype(np.uint8), k, iterations=1)
    else:
        eroded = mask_bin.astype(np.uint8)
    edges = cv2.Canny(eroded * 255, 50, 150)
    if edge_dilate_px > 0:
        k2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_dilate_px + 1, 2 * edge_dilate_px + 1)
        )
        edges = cv2.dilate(edges, k2, iterations=1)
    return edges  # uint8 0/255


# Select the best segmentation mask in the frame or ROI.
def find_best_mask(frame_bgr, yolo, W, H, target_cls_id, conf_thr=0.10, roi_rect=None):
    results = yolo(frame_bgr, verbose=False)
    best_mask = None
    best_score = 0.0
    if results and results[0].masks is not None:
        r0 = results[0]
        for mask_t, conf_t, cls_t in zip(r0.masks.data, r0.boxes.conf, r0.boxes.cls):
            cls_id = int(cls_t.item())
            conf_value = float(conf_t.item())
            if target_cls_id is not None and cls_id != target_cls_id:
                continue
            if conf_value < conf_thr:
                continue
            mask_np = mask_t.detach().cpu().numpy()
            mask_rz = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_rz > 0.5).astype(np.uint8)

            if roi_rect is not None:
                x0, y0, x1, y1 = roi_rect
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W - 1, x1), min(H - 1, y1)
                roi = np.zeros_like(mask_bin, dtype=np.uint8)
                roi[y0 : y1 + 1, x0 : x1 + 1] = 1
                overlap = (mask_bin & roi).sum()
                score = float(overlap)
            else:
                score = float(mask_bin.sum())

            if score > best_score:
                best_score = score
                best_mask = mask_bin
    return best_mask


def draw_measure_arrow(img, p1, p2, txt=None):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))

    def end_cap(pt, size=8, color=(255, 255, 255), t=1):
        x, y = pt
        cv2.line(img, (x - size, y), (x + size, y), color, t, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), color, t, cv2.LINE_AA)

    end_cap(p1, size=7, color=(255, 255, 255), t=1)
    end_cap(p2, size=7, color=(255, 255, 255), t=1)

    cv2.arrowedLine(img, p1, p2, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.18)

    if txt is None:
        d = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        txt = f"{d}px"
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 2
    (tw, th_text), _ = cv2.getTextSize(txt, font, fs, th)
    pad = 4
    x0 = mid[0] - tw // 2 - pad
    y0 = mid[1] - th_text - 6
    x1 = mid[0] + tw // 2 + pad
    y1 = mid[1] + 6
    cv2.rectangle(img, (x0, y0), (x1, y1), (32, 32, 32), -1)
    cv2.putText(img, txt, (x0 + pad, y1 - 6), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_dashed_line(
    img, pt1, pt2, color=(255, 255, 255), thickness=2, dash_length=10, gap_length=5
):
    """Draw a dashed line."""
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    line_vec = pt2 - pt1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1:
        return

    line_vec = line_vec / line_len

    current_pos = 0
    while current_pos < line_len:
        start_pos = current_pos
        end_pos = min(current_pos + dash_length, line_len)

        start_pt = pt1 + line_vec * start_pos
        end_pt = pt1 + line_vec * end_pos

        cv2.line(img, tuple(start_pt.astype(int)), tuple(end_pt.astype(int)), color, thickness)

        current_pos += dash_length + gap_length


def draw_hand_contour(img, hand_lms, W, H, color=(255, 255, 255), thickness=1):
    """Draw the convex hull of the hand landmarks."""

    points = []
    for lm in hand_lms:
        x = int(lm.x * W)
        y = int(lm.y * H)
        points.append([x, y])

    if len(points) > 3:
        points = np.array(points, dtype=np.int32)

        hull = cv2.convexHull(points)

        cv2.polylines(img, [hull], True, color, thickness)


def check_hand_object_contact(hand_box, poly, overlap_threshold=0.15):
    """Return hand-object contact and overlap normalized by hand area."""
    if hand_box is None or poly is None or len(poly) < 3:
        return False, 0.0

    hx, hy, hw, hh = hand_box
    hand_rect = np.array(
        [[hx, hy], [hx + hw, hy], [hx + hw, hy + hh], [hx, hy + hh]], dtype=np.int32
    )

    H = int(max(hy + hh, np.max(poly[:, 1])) + 10)
    W = int(max(hx + hw, np.max(poly[:, 0])) + 10)

    hand_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(hand_mask, [hand_rect], 1)

    obj_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(obj_mask, [poly.astype(np.int32)], 1)

    intersection = np.logical_and(hand_mask, obj_mask).sum()
    hand_area = hand_mask.sum()

    # Normalize overlap by the hand area.
    overlap_ratio = intersection / max(1.0, hand_area)

    return overlap_ratio > overlap_threshold, overlap_ratio


def get_guidance_direction(
    hand_center, object_center, hand_area, object_area, hand_box=None, poly=None
):
    """Return a guidance direction and whether depth adjustment is needed."""
    if hand_center is None or object_center is None:
        return None, None

    is_touching = False
    overlap_ratio = 0.0
    if hand_box is not None and poly is not None:
        is_touching, overlap_ratio = check_hand_object_contact(
            hand_box, poly, overlap_threshold=0.1
        )

    hx, hy = hand_center
    ox, oy = object_center

    dx = ox - hx  # Positive values place the object to the right.
    dy = oy - hy  # Positive values place the object below the hand.

    # Once contact is detected, guide the hand forward.
    if is_touching:
        return (
            localized_text("direction.front"),
            localized_text("search.contact_ratio").format(overlap_ratio=f"{overlap_ratio:.1%}"),
        )

    h_threshold = 30
    v_threshold = 30

    h_dir = None
    v_dir = None

    if abs(dx) > h_threshold:
        h_dir = localized_text("direction.right") if dx > 0 else localized_text("direction.left")

    if abs(dy) > v_threshold:
        v_dir = localized_text("direction.down") if dy > 0 else localized_text("direction.up")

    if abs(dx) > abs(dy) and h_dir:

        return h_dir, v_dir
    elif v_dir:

        return v_dir, h_dir
    else:
        # Guide forward when aligned but not yet touching.
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 50:
            return localized_text("direction.front"), localized_text("search.approach_slowly")
        else:
            return localized_text("search.hold"), None


def play_guidance_audio(direction):
    """Play a directional guidance prompt."""

    play_audio_threadsafe(direction)

    try:
        if isinstance(direction, str) and direction.strip():
            set_current_command(direction.strip())
    except Exception:
        pass


def get_center_guidance(object_center, frame_center, threshold=30):
    """Return the direction toward the frame center and the centered flag."""
    if object_center is None:
        return None, False

    ox, oy = object_center
    cx, cy = frame_center

    dx = cx - ox  # Positive values require moving right.
    dy = cy - oy  # Positive values require moving down.

    distance = np.sqrt(dx**2 + dy**2)
    if distance < threshold:
        return localized_text("search.centered"), True

    # Choose the dominant axis and move toward the frame center.
    if abs(dx) > abs(dy):
        return (
            localized_text("direction.left") if dx > 0 else localized_text("direction.right")
        ), False
    else:
        return localized_text("direction.up") if dy > 0 else localized_text("direction.down"), False


def main(headless: bool = False, prompt_name: str = None, stop_event=None):

    try:
        import cv2

        cv2.setUseOptimized(True)
        cv2.setNumThreads(2)  # Use one thread on devices with limited CPU capacity.
    except Exception:
        pass

    global PROMPT_NAME
    if prompt_name:
        PROMPT_NAME = prompt_name
        print(f"[ITEM_SEARCH] Using dynamic prompt: {PROMPT_NAME}")

    speaker = Speaker(ENABLE_TTS)
    last_tts_ts = 0.0
    MODE = "SEGMENT"  # Modes: SEGMENT, FLASH, CENTER_GUIDE, TRACK.
    colors = Colors()

    FRAME_IDX = 0
    last_mask = None  # Previous target mask for IoU stabilization.
    flow_mask = None  # Mask propagated by optical flow.
    flow_grace = 0  # Consecutive detection misses covered by optical flow.
    last_seen_ts = 0.0  # Timestamp of the latest successful detection.
    locked_id = None  # Track ID of the locked object.

    REDETECT_EVERY = 5  # Refresh the mask from detections every five frames.
    FLOW_GRACE_MAX = 8  # Allow up to eight missed detections during flow tracking.
    IOU_MIN_KEEP = 0.20  # Blend masks with low IoU to limit visual flicker.

    print("[INIT] Loading detection model...")

    use_yoloe = False
    vocabulary_backend = None
    if _YOLOE_READY:
        try:
            vocabulary_backend = YoloEBackend()  # Override the model with YOLOE_MODEL_PATH.
            vocabulary_backend.set_text_classes([PROMPT_NAME])
            use_yoloe = True
            print(f"[DETECTOR] YOLOE text-prompt backend enabled for: {PROMPT_NAME}", flush=True)
        except Exception as e:
            print(f"[DETECTOR] YOLOE init failed: {e}", flush=True)
    else:
        print("[DETECTOR] YOLOE backend not ready (import failed)", flush=True)

    if use_yoloe:

        id_to_name = {0: PROMPT_NAME}
        name_to_id = {norm_name(PROMPT_NAME): 0}
        target_cls_id = 0
    else:

        id_to_name = {}
        name_to_id = {}
        target_cls_id = None

    print(f"[CLASS] target id={target_cls_id}, name={id_to_name.get(target_cls_id, 'N/A')}")
    print(f"[THRESHOLD] conf >= {CONF_THRESHOLD:.2f}")

    # Hand Landmarker
    print("[INIT] Initializing Hand Landmarker...")
    base = BaseOptions(model_asset_path=HAND_TASK_PATH)
    hand_options = HandLandmarkerOptions(
        base_options=base,
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.70,
        result_callback=on_result,
    )
    landmarker = HandLandmarker.create_from_options(hand_options)

    W = None
    H = None
    print("[Bridge] Waiting for ESP32 frames...")

    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    old_gray = None
    p0 = None
    lock_edge_debug = None
    track_frame_count = 0
    last_poly_box = None

    fps_hist = []

    auto_lock_start_time = None
    auto_lock_delay = 1.0  # Lock the target after one second.
    last_detected_mask = None

    flash_start_time = None
    flash_duration = 1.0  # Flash duration in seconds.
    flash_frequency = 1  # Flash frequency in hertz.
    flash_mask = None
    flash_color = (0, 255, 255)

    last_guidance_time = 0
    last_guidance_direction = None

    center_guide_mask = None
    center_guide_start = None
    center_threshold = 30  # Centering tolerance in pixels.
    last_center_guide_time = 0
    center_reached = False

    grasp_tracking_frames = []
    grasp_tracking_duration = 1.0  # Require one second of consistent movement.
    grasp_movement_threshold = 10
    grasp_detected = False
    grasp_start_time = None

    background_points = None
    old_background_gray = None

    try:
        while True:

            if stop_event and stop_event.is_set():
                print("[ITEM_SEARCH] Stop event detected, exiting...")
                break

            frame = frame_buffer.wait_raw_bgr(timeout_sec=0.5)
            if frame is None:

                # Yield briefly when no frame is available.
                if headless:
                    cv2.waitKey(1)
                continue

            H, W = frame.shape[:2]
            ui_reset_overlay(H)

            vis = frame.copy()
            t_now = time.time()

            # Throttle and downsample hand detection.
            if FRAME_IDX % HAND_FPS_DIV == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if HAND_DOWNSCALE and HAND_DOWNSCALE != 1.0:
                    small = cv2.resize(
                        rgb,
                        None,
                        fx=HAND_DOWNSCALE,
                        fy=HAND_DOWNSCALE,
                        interpolation=cv2.INTER_AREA,
                    )
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                landmarker.detect_async(mp_image, int(t_now * 1000))
            # Reuse the latest asynchronous hand result between detections.

            hand_center = None
            hand_area = None
            hand_box = None
            grasp_now = False
            grasp_score = 0.0
            if _last_result is not None:
                res, _ = _last_result
                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    l0 = res.hand_landmarks[0]

                    draw_hands_mono(vis, l0, color=(0, 255, 255), r=2, t=2)

                    draw_hand_contour(vis, l0, W, H, color=(255, 255, 255), thickness=1)

                    xs = [p.x * W for p in l0]
                    ys = [p.y * H for p in l0]
                    hand_center = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
                    hand_box, hand_area = hand_bbox_and_area(l0, W, H)

                    #     x0, y0, w0, h0 = hand_box

                    grasp_now, grasp_score = detect_grasp(l0, W, H)
                    draw_text_cn(
                        vis,
                        localized_text("search.grasp_score").format(
                            grasp_score=f"{grasp_score:.2f}"
                        ),
                        (10, 70),
                        font_size=18,
                        color=(0, 180, 255),
                    )

            if MODE == "SEGMENT":

                FRAME_IDX += 1
                candidate_masks = []
                detected_object = False

                if use_yoloe and vocabulary_backend is not None:
                    # Persist tracks across consecutive frames.
                    det = vocabulary_backend.segment(
                        frame, conf=0.20, iou=0.45, imgsz=640, persist=True
                    )
                    H, W = frame.shape[:2]

                    # Prefer the locked track ID, otherwise select the largest mask.
                    chosen_idx = None
                    if det["masks"]:
                        if locked_id is not None and det["ids"] and (locked_id in det["ids"]):
                            chosen_idx = det["ids"].index(locked_id)
                        else:
                            areas = [int(m.sum()) for m in det["masks"]]
                            chosen_idx = int(np.argmax(areas))

                    if chosen_idx is not None:
                        m = det["masks"][chosen_idx]
                        if m.shape[:2] != (H, W):
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

                        mask_bin = (m > 0).astype(np.uint8)
                        candidate_masks.append(
                            {
                                "mask": mask_bin,
                                "area": int(mask_bin.sum()),
                                "name": PROMPT_NAME,
                                "cls_id": 0,
                                "conf": 0.99,
                            }
                        )
                        detected_object = True

                        colored = np.zeros_like(frame, dtype=np.uint8)
                        colored[mask_bin == 1] = (0, 255, 255)
                        vis = cv2.addWeighted(vis, 1.0, colored, MASK_ALPHA, 0)
                        contours, _ = cv2.findContours(
                            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        if contours:

                            largest_contour = max(contours, key=cv2.contourArea)

                            epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(largest_contour, True)
                            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                            cv2.drawContours(
                                vis, [smoothed_contour], -1, (0, 255, 255), STROKE_WIDTH
                            )

                        # Retain the track ID to reduce target switching.
                        if (
                            det["ids"]
                            and len(det["ids"]) > chosen_idx
                            and det["ids"][chosen_idx] is not None
                        ):
                            locked_id = int(det["ids"][chosen_idx])

                else:
                    # Keep the camera stream available when detection is unavailable.
                    draw_text_cn(
                        vis,
                        localized_text("search.detector_not_ready"),
                        (10, 100),
                        font_size=22,
                        color=(0, 215, 255),
                    )

                if candidate_masks:

                    candidate_masks.sort(key=lambda x: x["area"], reverse=True)
                    largest_mask_info = candidate_masks[0]
                    last_detected_mask = largest_mask_info["mask"]

                    contours, _ = cv2.findContours(
                        last_detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    if contours:

                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            cv2.circle(vis, (cx, cy), 8, (0, 255, 0), 2)
                            cv2.circle(vis, (cx, cy), 12, (0, 255, 0), 1)

                            draw_text_cn(
                                vis,
                                localized_text("search.target"),
                                (cx + 15, cy - 5),
                                font_size=16,
                                color=FRONTEND_COLORS["ok"],
                                ui_hint=False,
                            )

                    if len(candidate_masks) > 1:
                        draw_text_cn(
                            vis,
                            localized_text("search.largest_target").format(
                                value_1=f"{len(candidate_masks)}",
                                value_2=f"{largest_mask_info['area']}",
                            ),
                            (10, H - 30),
                            font_size=16,
                            color=(255, 255, 0),
                        )

                if detected_object and last_detected_mask is not None:
                    if auto_lock_start_time is None:
                        auto_lock_start_time = t_now
                        print(
                            f"[AUTO] Selected largest object (area: {np.sum(last_detected_mask)}); starting lock countdown"
                        )

                    elapsed = t_now - auto_lock_start_time
                    remaining = auto_lock_delay - elapsed

                    if remaining > 0:

                        draw_text_cn(
                            vis,
                            localized_text("search.auto_lock_countdown").format(
                                remaining=f"{remaining:.1f}"
                            ),
                            (10, 100),
                            font_size=16,
                            color=FRONTEND_COLORS["text"],
                            stroke=(0, 0, 0),
                        )

                        if last_detected_mask is not None:
                            contours, _ = cv2.findContours(
                                last_detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                            )
                            if contours:

                                largest_contour = max(contours, key=cv2.contourArea)

                                epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(
                                    largest_contour, True
                                )
                                smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                                progress = 1.0 - (remaining / auto_lock_delay)
                                color_intensity = int(100 + 155 * progress)
                                lock_color = (0, color_intensity, color_intensity)

                                pts = smoothed_contour.reshape(-1, 2)
                                for i in range(len(pts)):
                                    pt1 = tuple(pts[i])
                                    pt2 = tuple(pts[(i + 1) % len(pts)])

                                    draw_dashed_line(
                                        vis,
                                        pt1,
                                        pt2,
                                        color=lock_color,
                                        thickness=3,
                                        dash_length=15,
                                        gap_length=8,
                                    )
                    else:

                        print("[AUTO] Starting target flash")
                        MODE = "FLASH"
                        flash_start_time = t_now
                        flash_mask = last_detected_mask.copy()
                        auto_lock_start_time = None
                        play_guidance_audio(localized_text("search.object_detected"))
                else:

                    if auto_lock_start_time is not None:
                        print("[AUTO] Target lost; resetting countdown")
                    auto_lock_start_time = None
                    last_detected_mask = None
                    draw_text_cn(
                        vis,
                        localized_text("search.awaiting_detection"),
                        (10, 100),
                        font_size=16,
                        color=FRONTEND_COLORS["muted"],
                    )

            elif MODE == "FLASH":

                if flash_start_time is not None and flash_mask is not None:
                    elapsed = t_now - flash_start_time

                    if elapsed < flash_duration:

                        # Fade in for 0.3 s, hold for 0.4 s, then fade out for 0.3 s.
                        if elapsed < 0.3:

                            alpha = elapsed / 0.3 * 0.8
                        elif elapsed < 0.7:

                            alpha = 0.8
                        else:

                            alpha = (1.0 - elapsed) / 0.3 * 0.8

                        colored = np.zeros_like(frame, dtype=np.uint8)
                        colored[flash_mask == 1] = flash_color
                        vis = cv2.addWeighted(vis, 1.0 - alpha, colored, alpha, 0)

                        contours, _ = cv2.findContours(
                            flash_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        if contours:

                            contour_color = tuple(int(c * (0.5 + alpha * 0.5)) for c in flash_color)
                            cv2.drawContours(vis, contours, -1, contour_color, STROKE_WIDTH + 1)

                        draw_text_cn(
                            vis,
                            localized_text("search.locking_target"),
                            (10, 100),
                            font_size=18,
                            color=FRONTEND_COLORS["accent"],
                        )
                    else:

                        print("[AUTO] Flash complete; initializing optical flow")
                        edge_mask = inner_offset_edge(
                            flash_mask,
                            offset_px=INNER_OFFSET_PX_LOCK,
                            edge_dilate_px=EDGE_DILATE_PX,
                        )
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)

                        if pts is not None and len(pts) >= 8:
                            p0 = pts
                            old_gray = gray
                            MODE = "CENTER_GUIDE"
                            lock_edge_debug = edge_mask.copy()
                            track_frame_count = 0
                            center_guide_start = t_now
                            center_reached = False
                            flash_start_time = None
                            flash_mask = None
                            last_detected_mask = None
                            print(f"[LOCK] Inner-boundary points={len(p0)}; entering CENTER_GUIDE")
                        else:
                            print("[LOCK] Insufficient boundary features; returning to detection")
                            MODE = "SEGMENT"
                            flash_start_time = None
                            flash_mask = None
                            last_detected_mask = None

            elif MODE == "CENTER_GUIDE":

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                poly_center = None
                poly_area = 0.0

                if old_gray is not None and p0 is not None and len(p0) >= 5:

                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **LK_PARAMS)
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:
                            p0 = good_new.reshape(-1, 1, 2)
                            hull = cv2.convexHull(good_new.reshape(-1, 1, 2))
                            poly = hull.reshape(-1, 2)

                            if len(poly) >= 3:
                                H, W = frame.shape[:2]

                                # Rasterize the tracked polygon for mask IoU comparison.
                                poly_mask = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(poly_mask, [poly.astype(np.int32)], 1)

                                # Refresh detections every three frames and track between them.
                                need_reseed = False
                                new_det_mask = None

                                if (
                                    use_yoloe
                                    and vocabulary_backend is not None
                                    and (FRAME_IDX % 3 == 0)
                                ):

                                    if FRAME_IDX % 30 == 0:
                                        print(f"[YOLOE] Detecting frame {FRAME_IDX}")
                                    det = vocabulary_backend.segment(
                                        frame, conf=0.20, iou=0.45, imgsz=640, persist=True
                                    )
                                    if det["masks"]:

                                        areas = [int(m.sum()) for m in det["masks"]]
                                        j = int(np.argmax(areas))
                                        m = det["masks"][j]
                                        if m.shape[:2] != (H, W):
                                            m = cv2.resize(
                                                m, (W, H), interpolation=cv2.INTER_NEAREST
                                            )
                                        new_det_mask = (m > 0).astype(np.uint8)

                                        inter = np.logical_and(new_det_mask, poly_mask).sum()
                                        union = np.logical_or(new_det_mask, poly_mask).sum() + 1e-6
                                        iou = inter / union

                                        # Reseed optical flow from detections when mask IoU indicates drift.

                                        if iou < 0.5:
                                            need_reseed = True

                                            edge_mask = inner_offset_edge(
                                                new_det_mask,
                                                offset_px=INNER_OFFSET_PX_LOCK,
                                                edge_dilate_px=EDGE_DILATE_PX,
                                            )
                                            gray2 = gray
                                            pts = cv2.goodFeaturesToTrack(
                                                gray2, mask=edge_mask, **FEATURE_PARAMS
                                            )
                                            if pts is not None and len(pts) >= 8:
                                                p0 = pts
                                                old_gray = gray2

                                                last_mask = new_det_mask.copy()
                                                last_seen_ts = time.time()
                                                flow_grace = 0
                                                print(
                                                    "[RESEED] Low detection IoU triggered optical flow reseeding"
                                                )

                                # Blend nearby masks when reseeding is unnecessary.
                                if (not need_reseed) and (new_det_mask is not None):
                                    inter = np.logical_and(new_det_mask, poly_mask).sum()
                                    union = np.logical_or(new_det_mask, poly_mask).sum() + 1e-6
                                    iou = inter / union

                                    if iou < 0.95:
                                        # Blend detection and tracked masks with weights 0.8 and 0.2.
                                        poly_mask = (
                                            (0.8 * new_det_mask + 0.2 * poly_mask) > 0.5
                                        ).astype(np.uint8)

                                        last_mask = poly_mask.copy()

                                        contours, _ = cv2.findContours(
                                            poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                                        )
                                        if contours:

                                            largest_contour = max(contours, key=cv2.contourArea)

                                            epsilon = TRACK_EPSILON_FACTOR * cv2.arcLength(
                                                largest_contour, True
                                            )
                                            poly = cv2.approxPolyDP(
                                                largest_contour, epsilon, True
                                            ).reshape(-1, 2)

                                            edge_mask = inner_offset_edge(
                                                poly_mask,
                                                offset_px=INNER_OFFSET_PX_LOCK,
                                                edge_dilate_px=EDGE_DILATE_PX,
                                            )
                                            pts = cv2.goodFeaturesToTrack(
                                                gray, mask=edge_mask, **FEATURE_PARAMS
                                            )
                                            if pts is not None and len(pts) >= 5:
                                                p0 = pts

                                cv2.polylines(
                                    vis,
                                    [poly.astype(np.int32)],
                                    isClosed=True,
                                    color=(0, 255, 255),
                                    thickness=STROKE_WIDTH,
                                )

                                poly_center, poly_area = polygon_center_and_area(poly)

                                if poly_center:
                                    object_center = (int(poly_center[0]), int(poly_center[1]))

                                    frame_center = (W // 2, H // 2)

                                    cv2.circle(vis, object_center, 8, (0, 255, 0), -1)
                                    cv2.circle(vis, object_center, 12, (0, 255, 0), 2)

                                    cv2.line(
                                        vis,
                                        (frame_center[0] - 20, frame_center[1]),
                                        (frame_center[0] + 20, frame_center[1]),
                                        (255, 255, 255),
                                        2,
                                    )
                                    cv2.line(
                                        vis,
                                        (frame_center[0], frame_center[1] - 20),
                                        (frame_center[0], frame_center[1] + 20),
                                        (255, 255, 255),
                                        2,
                                    )

                                    draw_dashed_line(
                                        vis,
                                        object_center,
                                        frame_center,
                                        color=(255, 255, 0),
                                        thickness=2,
                                        dash_length=10,
                                        gap_length=5,
                                    )

                                    direction, is_centered = get_center_guidance(
                                        object_center, frame_center, center_threshold
                                    )

                                    if not center_reached:
                                        if is_centered:

                                            center_reached = True
                                            last_center_guide_time = t_now
                                            play_guidance_audio("OK")
                                            try:
                                                frame_buffer.send_ui_final(
                                                    localized_text("search.item_centered")
                                                )
                                            except Exception:
                                                pass
                                            draw_text_cn(
                                                vis,
                                                localized_text("search.item_centered"),
                                                (10, 60),
                                                font_size=18,
                                                color=FRONTEND_COLORS["ok"],
                                            )
                                        else:

                                            msg = localized_text("search.move_to_center").format(
                                                direction=f"{direction}",
                                            )
                                            try:
                                                # Send one final UI message per spoken instruction.
                                                if (
                                                    t_now - last_center_guide_time
                                                    > GUIDANCE_INTERVAL_SEC
                                                ):
                                                    frame_buffer.send_ui_final(msg)
                                            except Exception:
                                                pass
                                            draw_text_cn(
                                                vis,
                                                msg,
                                                (10, 40),
                                                font_size=18,
                                                color=FRONTEND_COLORS["text"],
                                            )

                                            dx = frame_center[0] - object_center[0]
                                            dy = frame_center[1] - object_center[1]
                                            distance = int(np.sqrt(dx**2 + dy**2))
                                            draw_text_cn(
                                                vis,
                                                localized_text("search.pixel_distance").format(
                                                    distance=f"{distance}"
                                                ),
                                                (10, 60),
                                                font_size=16,
                                                color=FRONTEND_COLORS["muted"],
                                            )

                                            if (
                                                t_now - last_center_guide_time
                                                > GUIDANCE_INTERVAL_SEC
                                            ):
                                                play_guidance_audio(direction)
                                                last_center_guide_time = t_now
                                    else:

                                        try:
                                            frame_buffer.send_ui_final(
                                                localized_text("search.centering_complete")
                                            )
                                        except Exception:
                                            pass
                                        draw_text_cn(
                                            vis,
                                            localized_text("search.centering_complete"),
                                            (10, 60),
                                            font_size=18,
                                            color=FRONTEND_COLORS["ok"],
                                        )

                                        # Wait one second before switching to hand tracking.
                                        if t_now - last_center_guide_time > 1.0:
                                            print("[CENTER] Entering hand tracking mode")
                                            try:
                                                frame_buffer.send_ui_final(
                                                    localized_text("search.hand_tracking_started"),
                                                )
                                            except Exception:
                                                pass
                                            MODE = "TRACK"

                                else:

                                    draw_text_cn(
                                        vis,
                                        localized_text("search.tracking_target"),
                                        (10, 100),
                                        font_size=20,
                                        color=(255, 255, 0),
                                    )
                        else:

                            MODE = "SEGMENT"
                            old_gray = None
                            p0 = None
                            print("[CENTER] Optical flow failed; returning to detection")

                old_gray = gray

            else:

                align_score = 0.0
                range_score = 0.0
                ratio = None

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                track_frame_count += 1

                relock_done = False
                poly_center = None
                poly_area = 0.0

                camera_movement = np.array([0.0, 0.0])

                # Sample background features outside the object polygon.
                if background_points is None or track_frame_count % 30 == 0:

                    mask_for_bg = np.ones((H, W), dtype=np.uint8) * 255
                    if last_poly_box:
                        x, y, w, h = last_poly_box
                        # Exclude both the object and hand from background features.
                        expand = 100
                        x1 = max(0, x - expand)
                        y1 = max(0, y - expand)
                        x2 = min(W, x + w + expand)
                        y2 = min(H, y + h + expand)
                        mask_for_bg[y1:y2, x1:x2] = 0

                    try:
                        bg_pts = cv2.goodFeaturesToTrack(
                            gray, maxCorners=20, qualityLevel=0.1, minDistance=30, mask=mask_for_bg
                        )
                        if bg_pts is not None and len(bg_pts) >= 5:
                            background_points = bg_pts
                            old_background_gray = gray.copy()
                    except Exception as e:

                        background_points = None

                # Estimate camera motion from background optical flow.
                if (
                    old_background_gray is not None
                    and background_points is not None
                    and len(background_points) > 0
                ):
                    try:
                        bg_p1, bg_st, _ = cv2.calcOpticalFlowPyrLK(
                            old_background_gray, gray, background_points, None, **LK_PARAMS
                        )
                        if bg_p1 is not None and bg_st is not None:
                            good_bg_old = background_points[bg_st == 1]
                            good_bg_new = bg_p1[bg_st == 1]
                            if len(good_bg_new) >= 3 and len(good_bg_old) >= 3:
                                # Average background displacement across valid tracks.
                                bg_movement = np.mean(good_bg_new - good_bg_old, axis=0)
                                camera_movement = bg_movement.reshape(2)
                                background_points = good_bg_new.reshape(-1, 1, 2)
                                old_background_gray = gray.copy()
                    except Exception as e:
                        print(f"[TRACK] Background optical flow failed: {e}")
                        camera_movement = np.array([0.0, 0.0])

                if old_gray is not None and p0 is not None and len(p0) >= 5:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **LK_PARAMS)
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:
                            p0 = good_new.reshape(-1, 1, 2)
                            hull = cv2.convexHull(good_new.reshape(-1, 1, 2))
                            poly = hull.reshape(-1, 2)

                            if len(poly) >= 3:

                                latest_det_mask = None
                                if use_yoloe and vocabulary_backend is not None:

                                    if track_frame_count % 30 == 0:
                                        print(
                                            f"[YOLOE] Detecting tracking frame {track_frame_count}"
                                        )

                                    det = vocabulary_backend.segment(
                                        frame,
                                        conf=YOLO_CORRECTION_CONF_THRESHOLD,
                                        iou=0.45,
                                        imgsz=640,
                                        persist=True,
                                    )
                                    if det["masks"]:

                                        areas = [int(m.sum()) for m in det["masks"]]
                                        j = int(np.argmax(areas))
                                        m = det["masks"][j]
                                        if m.shape[:2] != (H, W):
                                            m = cv2.resize(
                                                m, (W, H), interpolation=cv2.INTER_NEAREST
                                            )
                                        latest_det_mask = (m > 0).astype(np.uint8)

                                        poly_mask = np.zeros((H, W), dtype=np.uint8)
                                        cv2.fillPoly(poly_mask, [poly.astype(np.int32)], 1)
                                        inter = np.logical_and(latest_det_mask, poly_mask).sum()
                                        union = (
                                            np.logical_or(latest_det_mask, poly_mask).sum() + 1e-6
                                        )
                                        iou = inter / union

                                        if iou > YOLO_CORRECTION_IOU_THRESHOLD:

                                            contours, _ = cv2.findContours(
                                                latest_det_mask,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE,
                                            )
                                            if contours:
                                                largest_contour = max(contours, key=cv2.contourArea)

                                                epsilon = TRACK_EPSILON_FACTOR * cv2.arcLength(
                                                    largest_contour, True
                                                )
                                                poly = cv2.approxPolyDP(
                                                    largest_contour, epsilon, True
                                                ).reshape(-1, 2)

                                                edge_mask = inner_offset_edge(
                                                    latest_det_mask,
                                                    offset_px=INNER_OFFSET_PX_LOCK,
                                                    edge_dilate_px=EDGE_DILATE_PX,
                                                )
                                                pts = cv2.goodFeaturesToTrack(
                                                    gray, mask=edge_mask, **FEATURE_PARAMS
                                                )
                                                if pts is not None and len(pts) >= 5:
                                                    p0 = pts

                                is_touching = False
                                overlap_ratio = 0.0
                                if hand_box is not None and poly is not None:
                                    is_touching, overlap_ratio = check_hand_object_contact(
                                        hand_box, poly, overlap_threshold=0.1
                                    )

                                if is_touching:

                                    poly_color = (0, 255, 127)

                                    cv2.polylines(
                                        vis,
                                        [poly.astype(np.int32)],
                                        isClosed=True,
                                        color=(127, 255, 127),
                                        thickness=STROKE_WIDTH + 4,
                                    )

                                    overlay = vis.copy()
                                    cv2.fillPoly(overlay, [poly.astype(np.int32)], (0, 255, 0))
                                    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
                                else:

                                    poly_color = (0, 255, 0)
                                cv2.polylines(
                                    vis,
                                    [poly.astype(np.int32)],
                                    isClosed=True,
                                    color=poly_color,
                                    thickness=STROKE_WIDTH,
                                )

                                poly_center, poly_area = polygon_center_and_area(poly)
                                if poly_center:
                                    pc = (int(poly_center[0]), int(poly_center[1]))
                                    cv2.circle(vis, pc, 6, (0, 255, 0), -1)

                                x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                                last_poly_box = (x, y, w, h)

                                if hand_center and poly_center:
                                    hc = np.array(hand_center, dtype=np.float32)
                                    oc = np.array(poly_center, dtype=np.float32)
                                    dist = float(np.linalg.norm(oc - hc))
                                    diag = float(np.linalg.norm([W, H]))
                                    align_score = 1.0 - min(
                                        dist / (ALIGN_LOOSE_PCT * diag + 1e-6), 1.0
                                    )

                                    draw_dashed_line(
                                        vis,
                                        (hc[0], hc[1]),
                                        (oc[0], oc[1]),
                                        color=(255, 255, 0),
                                        thickness=2,
                                        dash_length=15,
                                        gap_length=10,
                                    )

                                    direction, secondary = get_guidance_direction(
                                        hand_center,
                                        poly_center,
                                        hand_area,
                                        poly_area,
                                        hand_box,
                                        poly,
                                    )

                                    if direction and direction != localized_text("search.hold"):

                                        if direction == localized_text("direction.front"):

                                            guide_color = (0, 255, 0)
                                            draw_text_cn(
                                                vis,
                                                localized_text("search.reach_guidance").format(
                                                    direction=f"{direction}",
                                                ),
                                                (W // 2 - 80, 40),
                                                font_size=24,
                                                color=guide_color,
                                                stroke=(0, 0, 0),
                                            )
                                        else:

                                            guide_color = (0, 255, 255)
                                            draw_text_cn(
                                                vis,
                                                localized_text("search.direction_guidance").format(
                                                    direction=f"{direction}",
                                                ),
                                                (W // 2 - 60, 40),
                                                font_size=24,
                                                color=guide_color,
                                                stroke=(0, 0, 0),
                                            )

                                        if secondary:
                                            if isinstance(secondary, str):

                                                draw_text_cn(
                                                    vis,
                                                    secondary,
                                                    (W // 2 - 60, 70),
                                                    font_size=18,
                                                    color=(0, 255, 0),
                                                )
                                            else:

                                                draw_text_cn(
                                                    vis,
                                                    localized_text(
                                                        "search.alternate_direction"
                                                    ).format(
                                                        secondary=f"{secondary}",
                                                    ),
                                                    (W // 2 - 60, 70),
                                                    font_size=18,
                                                    color=(200, 200, 200),
                                                )

                                        if t_now - last_guidance_time > GUIDANCE_INTERVAL_SEC:
                                            # Speak when direction changes or the repeat interval expires.
                                            if (
                                                direction != last_guidance_direction
                                                or t_now - last_guidance_time
                                                > GUIDANCE_INTERVAL_SEC * 2
                                            ):
                                                play_guidance_audio(direction)
                                                last_guidance_direction = direction
                                                last_guidance_time = t_now
                                                print(f"[GUIDE] Playing directional guidance")
                                else:
                                    align_score = 0.0

                                is_touching, overlap_ratio = check_hand_object_contact(
                                    hand_box, poly, overlap_threshold=0.1
                                )
                                if is_touching:
                                    draw_text_cn(
                                        vis,
                                        localized_text("search.contact_status").format(
                                            overlap_ratio=f"{overlap_ratio:.1%}",
                                        ),
                                        (10, 95),
                                        font_size=16,
                                        color=(0, 255, 0),
                                    )
                                else:

                                    if hand_center and poly_center:
                                        distance = np.sqrt(
                                            (hand_center[0] - poly_center[0]) ** 2
                                            + (hand_center[1] - poly_center[1]) ** 2
                                        )
                                        draw_text_cn(
                                            vis,
                                            localized_text("search.pixel_distance").format(
                                                distance=f"{distance:.0f}"
                                            ),
                                            (10, 95),
                                            font_size=16,
                                            color=FRONTEND_COLORS["muted"],
                                        )

                                if (
                                    _last_result
                                    and _last_result[0].hand_landmarks
                                    and len(_last_result[0].hand_landmarks) > 0
                                ):
                                    l0 = _last_result[0].hand_landmarks[0]
                                    grasp_now, grasp_score = detect_grasp(l0, W, H)
                                else:
                                    grasp_now, grasp_score = False, 0.0

                                # Reuse current detections for peripheral checks and relocking.
                                if (
                                    (track_frame_count % PERI_CHECK_EVERY == 0)
                                    and (last_poly_box is not None)
                                    and (latest_det_mask is not None)
                                ):
                                    # Reuse detections from this frame.
                                    px, py, pw, ph = last_poly_box
                                    x0 = max(0, px - PERI_MONITOR_PX)
                                    y0 = max(0, py - PERI_MONITOR_PX)
                                    x1 = min(W - 1, px + pw + PERI_MONITOR_PX)
                                    y1 = min(H - 1, py + ph + PERI_MONITOR_PX)

                                    peri_area = latest_det_mask[y0:y1, x0:x1].sum()
                                    total_area = latest_det_mask.sum()

                                    if (
                                        peri_area > total_area * 0.1
                                    ):  # Require at least 10% of detected area in the peripheral region.
                                        edge_mask = inner_offset_edge(
                                            latest_det_mask,
                                            offset_px=INNER_OFFSET_PX_LOCK,
                                            edge_dilate_px=EDGE_DILATE_PX,
                                        )
                                        pts = cv2.goodFeaturesToTrack(
                                            gray, mask=edge_mask, **FEATURE_PARAMS
                                        )
                                        if pts is not None and len(pts) >= 8:
                                            p0 = pts
                                            old_gray = gray
                                            lock_edge_debug = edge_mask.copy()

                            else:
                                MODE = "SEGMENT"
                                old_gray = None
                                p0 = None
                                lock_edge_debug = None
                        else:
                            MODE = "SEGMENT"
                            old_gray = None
                            p0 = None
                            lock_edge_debug = None
                    else:
                        MODE = "SEGMENT"
                        old_gray = None
                        p0 = None
                        lock_edge_debug = None
                else:
                    MODE = "SEGMENT"
                    old_gray = None
                    p0 = None
                    lock_edge_debug = None

                if MODE == "SEGMENT":
                    draw_text_cn(
                        vis,
                        localized_text("search.tracking_lost"),
                        (10, 100),
                        font_size=22,
                        color=(0, 0, 255),
                    )

                old_gray = gray

            if "fps_hist" not in locals():
                fps_hist = []
            fps_hist.append(t_now)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps = 0.0 if len(fps_hist) < 2 else (len(fps_hist) - 1) / (fps_hist[-1] - fps_hist[0])
            draw_text_cn(
                vis, f"FPS: {fps:.1f}", (10, 40), font_size=16, color=FRONTEND_COLORS["ok"]
            )

            if lock_edge_debug is not None:

                small = cv2.resize(
                    lock_edge_debug, (0, 0), fx=0.22, fy=0.22, interpolation=cv2.INTER_NEAREST
                )
                sh, sw = small.shape[:2]
                small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

                x1 = max(8, W - sw - 12)
                y1 = max(8, H - sh - 12)
                y2 = y1 + sh
                x2 = x1 + sw
                vis[y1:y2, x1:x2] = small_bgr

            draw_command_pill(vis, CURRENT_COMMAND_TEXT)

            # Send frames to browser clients in both display modes.
            frame_buffer.send_vis_bgr(vis)

            if not headless:
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("r"):
                    MODE = "SEGMENT"
                    old_gray = None
                    p0 = None
                    lock_edge_debug = None
                elif key == 13:
                    if MODE == "SEGMENT":

                        if use_yoloe and vocabulary_backend is not None:
                            det = vocabulary_backend.segment(
                                frame, conf=CONF_THRESHOLD, iou=0.45, imgsz=640, persist=True
                            )
                            if det["masks"]:

                                areas = [int(m.sum()) for m in det["masks"]]
                                j = int(np.argmax(areas))
                                m = det["masks"][j]
                                if m.shape[:2] != (H, W):
                                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                                best_mask = (m > 0.5).astype(np.uint8)
                            else:
                                best_mask = None
                        else:
                            best_mask = None
                        if best_mask is not None:
                            edge_mask = inner_offset_edge(
                                best_mask,
                                offset_px=INNER_OFFSET_PX_LOCK,
                                edge_dilate_px=EDGE_DILATE_PX,
                            )
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                            if pts is not None and len(pts) >= 8:
                                p0 = pts
                                old_gray = gray
                                MODE = "TRACK"
                                lock_edge_debug = edge_mask.copy()
                                track_frame_count = 0
                                print(f"[LOCK] Inner-boundary points={len(p0)}; entering TRACK")
                            else:
                                print(
                                    "[LOCK] Insufficient boundary features; adjust the camera view"
                                )
                        else:
                            print("[LOCK] No valid segmentation in the current frame")
            else:
                # Let OpenCV process pending callbacks without busy waiting.
                cv2.waitKey(1)

                if stop_event and stop_event.is_set():
                    print("[ITEM_SEARCH] Received stop signal in headless mode")
                    break

    finally:
        try:
            landmarker.close()
        except Exception:
            pass

        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
