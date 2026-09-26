"""YOLOE detection frontend for the device service."""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path

from navguide.runtime.config import asset_path
from navguide.core.selection import DetectionCandidate

DEFAULT_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "dog",
                   "pole", "bench", "chair", "stairs", "crosswalk", "traffic light"]


class YOLOEFrontend:
    def __init__(self):
        from ultralytics import YOLO, YOLOE
        import torch

        path = Path(asset_path("YOLOE_MODEL_PATH", "model/yoloe-11l-seg.pt"))
        if not path.is_file():
            raise FileNotFoundError(f"Place authorized YOLOE weights at {path}")
        self.device = os.getenv("NAVGUIDE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLOE(str(path))
        self.model.to(self.device)
        self._model_lock = threading.RLock()
        self.classes = []
        self.set_target(None)
        self.specialists = []
        self.specialist_kinds = []
        for variable in ("BLIND_PATH_MODEL", "TRAFFIC_LIGHT_MODEL"):
            if os.getenv(variable):
                weights = Path(asset_path(variable, ""))
                if not weights.is_file():
                    raise FileNotFoundError(f"Configured {variable} does not exist")
                self.specialists.append(YOLO(str(weights)).to(self.device))
                self.specialist_kinds.append(variable)

    def set_target(self, target):
        with self._model_lock:
            classes = list(dict.fromkeys(DEFAULT_CLASSES + ([target] if target else [])))
            if classes != self.classes:
                self.model.set_classes(classes, self.model.get_text_pe(classes))
                self.classes = classes
                # Class index changes invalidate tracker state from prior targets.
                self.reset_tracking()

    def reset_tracking(self):
        with self._model_lock:
            for model in [self.model, *getattr(self, "specialists", [])]:
                predictor = getattr(model, "predictor", None)
                for tracker in getattr(predictor, "trackers", []):
                    tracker.reset()

    def detect(self, jpeg: bytes, max_pixels: int):
        # A cancelled asyncio.to_thread await cannot terminate inference. Guard
        # both native calls and text-embedding updates in the worker threads.
        with self._model_lock:
            return self._detect_locked(jpeg, max_pixels)

    def _detect_locked(self, jpeg: bytes, max_pixels: int):
        import cv2
        import numpy as np
        from PIL import Image

        # Inspect dimensions before OpenCV allocates a decoded image.
        with Image.open(io.BytesIO(jpeg)) as header:
            width, height = header.size
            if header.format != "JPEG" or width * height > max_pixels:
                raise ValueError("Expected a JPEG within the pixel limit")
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid JPEG frame")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0 or width * height > max_pixels:
            raise ValueError("Decoded frame exceeds the pixel limit")
        detections = []
        for model_index, model in enumerate([self.model, *self.specialists]):
            results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  conf=0.25, device=self.device, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes
                coordinates = boxes.xyxy.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
                for bbox, category, confidence, identity in zip(coordinates, classes, confidences, ids):
                    name = str(result.names[int(category)]).strip().lower()
                    # Convert specialist traffic-signal labels into a common
                    # category/state pair. Other specialists retain their labels.
                    color = None
                    if model_index > 0 and self.specialist_kinds[model_index - 1] == "TRAFFIC_LIGHT_MODEL" and name in {"red", "yellow", "green"}:
                        color, name = name, "traffic light"
                    elif name in {"red light", "red traffic light", "green light", "green traffic light", "yellow light", "yellow traffic light"}:
                        color, name = name.split()[0], "traffic light"
                    # Only explicit semantic red-light outputs mark urgency here.
                    # Class membership alone does not prove movement or collision risk.
                    urgent = color == "red"
                    detections.append(DetectionCandidate(
                        category=name, confidence=float(confidence), bbox=tuple(bbox),
                        track_id=None if identity is None else int(identity) + model_index * 1000000,
                        urgency_override=urgent, is_hazard=urgent, signal_color=color,
                    ))
        return detections, width, height
