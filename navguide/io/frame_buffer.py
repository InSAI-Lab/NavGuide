"""Buffer device JPEG frames and broadcast processed images and text."""
import threading
from collections import deque
import time
import cv2
import numpy as np

# Retain only the most recent JPEG frames.
_MAX_BUF = 4
_frames = deque(maxlen=_MAX_BUF)
_cond = threading.Condition()

# Register the JPEG broadcast callback during service startup.
_sender_lock = threading.Lock()
_sender_cb = None

# Register the interface text callback during service startup.
_ui_sender_lock = threading.Lock()
_ui_sender_cb = None

def set_sender(cb):
    """Register the server callback cb(jpeg_bytes) -> None."""
    global _sender_cb
    with _sender_lock:
        _sender_cb = cb

def set_ui_sender(cb):
    """Register the server callback cb(text) -> None."""
    global _ui_sender_cb
    with _ui_sender_lock:
        _ui_sender_cb = cb

def push_raw_jpeg(jpeg_bytes: bytes):
    """Cache a JPEG frame received through /ws/camera."""
    if not jpeg_bytes:
        return
    with _cond:
        _frames.append((time.time(), jpeg_bytes))
        _cond.notify_all()

def wait_raw_bgr(timeout_sec: float = 0.5):
    """Return the latest decoded BGR frame, or None after the timeout."""
    t_end = time.time() + timeout_sec
    last = None
    while time.time() < t_end:
        with _cond:
            if _frames:
                last = _frames[-1]
        if last is None:
            time.sleep(0.01)
            continue
        ts, jpeg = last
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
        time.sleep(0.01)
    return None

def send_vis_bgr(bgr, quality: int = 80):
    """Encode a processed BGR image as JPEG and broadcast it."""
    if bgr is None:
        return

    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return
    with _sender_lock:
        cb = _sender_cb
    if cb:
        try:
            cb(enc.tobytes())
        except Exception:
            pass

def send_ui_final(text: str):
    """Send final interface text through a thread-safe callback."""
    if not text:
        return
    with _ui_sender_lock:
        cb = _ui_sender_cb
    if cb:
        try:
            cb(str(text))
        except Exception:
            pass
