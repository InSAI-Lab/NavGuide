import logging
import os
import cv2
import numpy as np
import torch
from threading import Semaphore
from contextlib import contextmanager
from ultralytics import YOLOE
from typing import List, Dict, Any, Optional
from navguide.runtime.config import asset_path

logger = logging.getLogger(__name__)

# Device and automatic mixed precision settings.
DEVICE = os.getenv("NAVGUIDE_DEVICE", "cuda:0")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    logger.warning(f"NAVGUIDE_DEVICE={DEVICE} requested CUDA, but CUDA is unavailable; using CPU")
    DEVICE = "cpu"
IS_CUDA = DEVICE.startswith("cuda")

AMP_POLICY = os.getenv("NAVGUIDE_AMP", "bf16").lower()
if AMP_POLICY not in ("bf16", "fp16", "off"):
    AMP_POLICY = "bf16"
AMP_DTYPE = torch.bfloat16 if AMP_POLICY == "bf16" else (torch.float16 if AMP_POLICY == "fp16" else None)

# Limit concurrent GPU inference.
GPU_SLOTS = int(os.getenv("NAVGUIDE_GPU_SLOTS", "2"))
_gpu_slots = Semaphore(GPU_SLOTS)

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


@contextmanager
def gpu_infer_slot():
    """Limit GPU concurrency and apply inference mode with optional autocast."""
    with _gpu_slots:
        if IS_CUDA and AMP_POLICY != "off":

            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
                yield
        else:
            with torch.inference_mode():
                yield


class ObstacleDetectorClient:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.whitelist_embeddings = None
        self.WHITELIST_CLASSES = [
            'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'animal', 'scooter', 'stroller', 'dog',
            'pole', 'post', 'column', 'pillar', 'stanchion', 'bollard', 'utility pole',
            'telegraph pole', 'light pole', 'street pole', 'signpost', 'support post',
            'vertical post', 'bench', 'chair', 'potted plant', 'hydrant', 'cone', 'stone', 'box'
        ]
        try:
            logger.info("Loading YOLOE obstacle model...")
            self.model = YOLOE(model_path or asset_path("OBSTACLE_MODEL", "model/yoloe-11l-seg.pt"))
            self.model.to(DEVICE)
            self.model.fuse()
            logger.info(f"YOLOE obstacle model loaded on device: {DEVICE}")

            logger.info("Precomputing text embeddings for obstacle classes...")
            if IS_CUDA and AMP_DTYPE is not None:
                with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    self.whitelist_embeddings = self.model.get_text_pe(self.WHITELIST_CLASSES)
            else:
                self.whitelist_embeddings = self.model.get_text_pe(self.WHITELIST_CLASSES)
            logger.info("YOLOE text embeddings are ready.")
        except Exception as e:
            logger.error(f"YOLOE model initialization failed: {e}", exc_info=True)
            raise
    def tensor_to_numpy_mask(mask_tensor):
        """Convert a tensor to a binary NumPy mask."""

        if mask_tensor.dtype in (torch.bfloat16, torch.float16):
            mask_tensor = mask_tensor.float()


        mask = mask_tensor.cpu().numpy()


        if mask.max() <= 1.0:
            mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask = mask.astype(np.uint8)

        return mask
    def detect(self, image: np.ndarray, path_mask: np.ndarray = None) -> List[Dict[str, Any]]:
        """Detect allowed obstacle classes and optionally restrict them to a path mask."""
        if self.model is None:
            return []

        H, W = image.shape[:2]
        try:
            self.model.set_classes(self.WHITELIST_CLASSES, self.whitelist_embeddings)
        except Exception as e:
            logger.error(f"Failed to set YOLOE class prompts: {e}")
            return []

        conf_thr = float(os.getenv("NAVGUIDE_OBS_CONF", "0.25"))
        with gpu_infer_slot():
            results = self.model.predict(image, verbose=False, conf=conf_thr)

        if not (results and results[0].masks):
            return []


        final_obstacles = []
        num_masks = len(results[0].masks.data)
        num_boxes = len(results[0].boxes.cls) if getattr(results[0].boxes, "cls", None) is not None else 0

        for i, mask_tensor in enumerate(results[0].masks.data):
            if i >= num_boxes: continue


            # Convert bfloat16 masks before exporting to NumPy.
            if mask_tensor.dtype == torch.bfloat16:
                mask_tensor = mask_tensor.float()


            mask = mask_tensor.cpu().numpy()

            # Accept both probability masks and binary masks.
            if mask.max() <= 1.0:

                mask = (mask > 0.5).astype(np.uint8) * 255
            else:

                mask = mask.astype(np.uint8)

            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            area = np.sum(mask > 0)

            # Reject oversized detections such as ground regions.
            if (area / (H * W)) > 0.7: continue

            # Restrict obstacles to the supplied path mask.
            if path_mask is not None:
                intersection_area = np.sum(cv2.bitwise_and(mask, path_mask) > 0)
                # Require sufficient overlap with the path.
                if intersection_area < 100 or (intersection_area / area) < 0.01:
                    continue

            cls_id = int(results[0].boxes.cls[i])
            class_names_map = results[0].names
            class_name = "Unknown"
            if isinstance(class_names_map, dict):

                class_name = class_names_map.get(cls_id, "Unknown")
            elif isinstance(class_names_map, list) and 0 <= cls_id < len(class_names_map):

                class_name = class_names_map[cls_id]


            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0: continue

            final_obstacles.append({
                'name': class_name.strip(),
                'conf': float(results[0].boxes.conf[i]),
                'bbox': tuple(float(v) for v in results[0].boxes.xyxy[i].cpu().tolist()),
                'mask': mask,
                'area': area,
                'area_ratio': area / (H * W),
                'center_x': np.mean(x_coords),
                'center_y': np.mean(y_coords),
                'bottom_y_ratio': np.max(y_coords) / H
            })

        return final_obstacles
