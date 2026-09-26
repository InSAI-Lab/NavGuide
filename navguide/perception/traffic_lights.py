# -*- coding: utf-8 -*-
"""Traffic light detection with temporal confirmation and frame streaming."""

from navguide.i18n import text as localized_text

import os
from navguide.runtime.config import asset_path
import threading
import cv2
import numpy as np
from navguide.io import frame_buffer
import logging

logger = logging.getLogger(__name__)


YOLO_MODEL_PATH = asset_path("TRAFFIC_LIGHT_MODEL", "model/trafficlight.pt")


CONF_THRESHOLD = 0.25
STROKE_WIDTH = 3


_detection_thread = None
_stop_event = None
_detection_running = False


_model = None
_model_error = None
_detection_history = []


FRONTEND_COLORS = {
    "text": (230, 237, 243),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "muted": (159, 176, 195),
}


LIGHT_COLORS = {
    "stop": FRONTEND_COLORS["red"],
    "countdown_go": FRONTEND_COLORS["yellow"],
    "go": FRONTEND_COLORS["green"],
    "countdown_stop": FRONTEND_COLORS["red"],
}

SIGNAL_COLORS = {
    "stop": "red",
    "countdown_stop": "red",
    "countdown_go": "yellow",
    "go": "green",
}


# Exclude crosswalk and blank classes from signal labels.
LIGHT_NAMES = {
    "stop": localized_text("traffic.red"),
    "go": localized_text("traffic.green"),
    "countdown_go": localized_text("traffic.yellow"),  # Treat a green countdown as a yellow signal.
    "countdown_stop": localized_text("traffic.red"),
}


def main(headless: bool = True, stop_event=None):
    """Stream detections from raw camera frames until stopped."""
    global _detection_running
    try:
        if not init_model():
            return
        reset_detection_state()
        while not (stop_event and stop_event.is_set()):
            frame = frame_buffer.wait_raw_bgr(timeout_sec=2.0)
            if frame is None:
                reset_detection_state()
                continue
            result = process_single_frame(frame)
            frame_buffer.send_vis_bgr(result["vis_image"])
            if not headless:
                cv2.imshow("Traffic Light Detection", result["vis_image"])
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except Exception:
        logger.exception("[TRAFFIC] Detection stream failed")
    finally:
        reset_detection_state()
        _detection_running = False
        if not headless:
            cv2.destroyAllWindows()


def start_detection():
    """Start traffic light detection in a background thread."""
    global _detection_thread, _stop_event, _detection_running

    if _detection_running:
        print("[TRAFFIC] Detection is already running")
        return False

    _stop_event = threading.Event()
    _detection_thread = threading.Thread(
        target=main, args=(True, _stop_event), daemon=True, name="TrafficLightDetection"
    )
    _detection_running = True
    _detection_thread.start()
    print("[TRAFFIC] Background detection started")
    return True


def stop_detection():
    """Stop the detection thread."""
    global _detection_thread, _stop_event, _detection_running

    if not _detection_running:
        print("[TRAFFIC] Detection is not running")
        return False

    print("[TRAFFIC] Stopping detection...")
    if _stop_event:
        _stop_event.set()

    if _detection_thread:
        _detection_thread.join(timeout=2.0)
        _detection_thread = None

    _stop_event = None
    _detection_running = False
    print("[TRAFFIC] Detection stopped")
    return True


def is_detection_running():
    """Return whether the detection thread is running."""
    return _detection_running


def get_model_status() -> dict:
    """Return model availability and the latest initialization error."""
    return {
        "available": _model is not None,
        "reason": None if _model is not None else "model_unavailable",
        "error": _model_error,
    }


def init_model() -> bool:
    """Load configured weights; expose failure details through get_model_status()."""
    global _model, _model_error
    if _model is not None:
        return True
    try:
        if not os.path.isfile(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"Traffic light weights not found: {YOLO_MODEL_PATH}")
        from ultralytics import YOLO

        _model = YOLO(YOLO_MODEL_PATH)
        _model_error = None
        logger.info("[TRAFFIC] Model loaded: %s", YOLO_MODEL_PATH)
        return True
    except Exception as error:
        _model = None
        _model_error = str(error)
        reset_detection_state()
        logger.warning("[TRAFFIC] Model loading failed: %s", error)
        return False


def _parse_detections(results, class_names, image_shape) -> list:
    """Read recognized signal boxes from the current model result."""
    detections = []
    if results is None:
        raise ValueError("Traffic light inference returned no result object")
    if len(results) == 0:
        return detections
    result = results[0]
    names = getattr(result, "names", None) or class_names
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    height, width = image_shape[:2]
    for box in boxes:
        class_value = float(box.cls[0])
        confidence = float(box.conf[0])
        if not np.isfinite(class_value) or not class_value.is_integer():
            continue
        if not np.isfinite(confidence) or not CONF_THRESHOLD <= confidence <= 1.0:
            continue
        class_id = int(class_value)
        if isinstance(names, dict):
            name = names.get(class_id, names.get(str(class_id), ""))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            name = names[class_id]
        else:
            name = ""
        label = str(name).strip().lower()
        if label not in SIGNAL_COLORS:
            continue
        coords = [float(value) for value in box.xyxy[0]]
        if len(coords) != 4 or not all(np.isfinite(value) for value in coords):
            continue
        x1, y1, x2, y2 = coords
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "label": label,
                "name": LIGHT_NAMES[label],
                "color": SIGNAL_COLORS[label],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections


def draw_detections(image: np.ndarray, detections: list) -> None:
    """Draw current signal boxes on an existing visualization."""
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), LIGHT_COLORS[detection["label"]], STROKE_WIDTH)


def process_single_frame(image: np.ndarray, ui_broadcast_callback=None) -> dict:
    """Return current detections, fresh temporal confirmation and detector status."""
    result = {
        "available": False,
        "reason": "model_unavailable",
        "error": None,
        "vis_image": image.copy(),
        "detected_light": None,
        "detected_colors": [],
        "detections": [],
        "stable_light": None,
    }
    if _model is None and not init_model():
        reset_detection_state()
        result["error"] = _model_error
        return result
    try:
        predictions = _model(image, conf=CONF_THRESHOLD, verbose=False)
        detections = _parse_detections(predictions, getattr(_model, "names", {}), image.shape)
    except Exception as error:
        reset_detection_state()
        result.update(reason="inference_failed", error=str(error))
        logger.warning("[TRAFFIC] Frame inference failed: %s", error)
        return result

    result.update(available=True, reason=None, detections=detections)
    result["detected_colors"] = sorted({item["color"] for item in detections})
    if not detections:
        reset_detection_state()
        return result

    # A concurrent stop or countdown indication blocks a green confirmation.
    priority = {"stop": 2, "countdown_stop": 2, "countdown_go": 1, "go": 0}
    selected = max(detections, key=lambda item: (priority[item["label"]], item["confidence"]))
    detected_light = selected["label"]
    if _detection_history and _detection_history[-1] != detected_light:
        reset_detection_state()
    _detection_history.append(detected_light)
    del _detection_history[:-2]
    stable_light = detected_light if len(_detection_history) == 2 else None
    result.update(detected_light=detected_light, stable_light=stable_light)
    try:
        draw_detections(result["vis_image"], detections)
    except Exception as error:
        result["vis_image"] = image.copy()
        logger.warning("[TRAFFIC] Signal visualization failed: %s", error)
    return result


def reset_detection_state():
    """Reset temporal detection state."""
    global _detection_history
    _detection_history = []


if __name__ == "__main__":
    main(headless=False)
