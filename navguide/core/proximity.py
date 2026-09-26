"""Coarse distance from pixel height, size priors and camera intrinsics.

Estimates support relevance and phrasing, not calibrated depth or collision timing.
See NavGuide paper Section 2.2.
"""

from __future__ import annotations
from navguide.i18n import text as localized_text

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class ProximityBin(str, Enum):
    """Distance categories for relevance weights and repetition signatures."""

    CRITICAL_NEAR = "critical_near"  # < 1.5 m: immediate hazard range.
    NEAR = "near"  # 1.5 to 2.5 m: near range, about two to four steps.
    MEDIUM = "medium"  # 2.5 to 4.5 m: approaching range.
    FAR = "far"  # >= 4.5 m: far range or background context.


# Physical height priors for common object categories, in meters.
DEFAULT_CLASS_HEIGHT_PRIORS: Dict[str, float] = {
    # Pedestrians and nonmotorized vehicles.
    "person": 1.70,
    "pedestrian": 1.70,
    "man": 1.70,
    "woman": 1.62,
    "child": 1.10,
    "bicycle": 1.00,
    "bike": 1.00,
    "motorcycle": 1.10,
    "motorbike": 1.10,
    "e-bike": 1.10,
    "scooter": 1.00,
    # Motor vehicles.
    "car": 1.50,
    "automobile": 1.50,
    "sedan": 1.45,
    "suv": 1.68,
    "van": 1.95,
    "bus": 3.20,
    "truck": 2.80,
    # Road and traffic infrastructure.
    "traffic light": 0.85,
    "traffic_light": 0.85,
    "stop sign": 0.75,
    "fire hydrant": 0.75,
    "pole": 2.50,
    "bollard": 0.80,
    "traffic cone": 0.70,
    "trash can": 0.90,
    "curb": 0.15,
    "stairs": 1.20,
    "step": 0.18,
    # Indoor and household objects.
    "chair": 0.85,
    "bench": 0.80,
    "table": 0.75,
    "desk": 0.75,
    "door": 2.00,
    "bottle": 0.22,
    "cup": 0.12,
    "backpack": 0.45,
    "umbrella": 0.85,
    "cell phone": 0.15,
    "keys": 0.08,
    # Animals.
    "dog": 0.50,
    "cat": 0.30,
}

DEFAULT_OBJECT_HEIGHT = 1.00  # Default height prior, in meters.


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics at a reference image resolution."""

    reference_width: int = 2048
    reference_height: int = 1536
    vertical_fov_deg: float = 60.0  # Approximately 60 degrees of vertical field of view.
    calibrated_focal_length_y: Optional[float] = None

    def __post_init__(self):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.reference_width, self.reference_height)
        ):
            raise ValueError("reference image dimensions must be positive integers")
        if not math.isfinite(self.vertical_fov_deg) or not 0 < self.vertical_fov_deg < 180:
            raise ValueError("vertical_fov_deg must be between 0 and 180")
        if self.calibrated_focal_length_y is not None:
            if (
                not math.isfinite(self.calibrated_focal_length_y)
                or self.calibrated_focal_length_y <= 0
            ):
                raise ValueError("calibrated focal length must be positive and finite")

    @property
    def focal_length_y_ref(self) -> float:
        """Vertical focal length in pixels at the reference resolution."""
        if self.calibrated_focal_length_y is not None:
            return self.calibrated_focal_length_y
        half_fov_rad = math.radians(self.vertical_fov_deg / 2.0)
        return (self.reference_height / 2.0) / math.tan(half_fov_rad)

    def get_focal_length_y(self, frame_height: int) -> float:
        """Scale focal length to current frame height."""
        if not math.isfinite(frame_height) or frame_height <= 0:
            raise ValueError("frame_height must be positive and finite")
        scale = frame_height / float(self.reference_height)
        return self.focal_length_y_ref * scale


class ProximityEstimator:
    """Estimate coarse distance using object height and pinhole projection."""

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        class_height_priors: Optional[Dict[str, float]] = None,
        step_length_m: float = 0.6,
    ):
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.class_height_priors = dict(DEFAULT_CLASS_HEIGHT_PRIORS)
        if class_height_priors:
            for category, height in class_height_priors.items():
                if not math.isfinite(height) or height <= 0:
                    raise ValueError("class height priors must be positive and finite")
                self.class_height_priors[category.strip().lower()] = height
        if not math.isfinite(step_length_m) or step_length_m <= 0:
            raise ValueError("step_length_m must be positive and finite")
        self.step_length_m = step_length_m

    def get_class_height_prior(self, class_name: str) -> float:
        """Lookup physical height prior for a given category name."""
        k = class_name.strip().lower()
        if k in self.class_height_priors:
            return self.class_height_priors[k]
        # Unknown classes use an explicit fallback, never a substring match.
        return DEFAULT_OBJECT_HEIGHT

    def estimate_distance_m(
        self,
        bbox: Tuple[float, float, float, float],
        class_name: str,
        frame_width: int,
        frame_height: int,
    ) -> float:
        """
        Estimate coarse distance in meters using pinhole projection:
            d = (f_y * H_real) / h_pixel

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.
            class_name: object class/label.
            frame_width: frame pixel width.
            frame_height: frame pixel height.

        Returns:
            Coarse distance in meters (clamped between 0.3m and 20.0m).
        """
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise ValueError("bbox must contain four finite coordinates")
        if not math.isfinite(frame_width) or frame_width <= 0:
            raise ValueError("frame_width must be positive and finite")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")
        pixel_height = float(y2 - y1)

        f_y = self.intrinsics.get_focal_length_y(frame_height)
        h_real = self.get_class_height_prior(class_name)

        d = (f_y * h_real) / pixel_height
        # Clamp to realistic bounds for indoor/outdoor wearable navigation
        return float(max(0.3, min(20.0, d)))

    def categorize_proximity(self, distance_m: float) -> ProximityBin:
        """Map metric distance to coarse proximity bin."""
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("distance_m must be finite and nonnegative")
        if distance_m < 1.5:
            return ProximityBin.CRITICAL_NEAR
        elif distance_m < 2.5:
            return ProximityBin.NEAR
        elif distance_m < 4.5:
            return ProximityBin.MEDIUM
        else:
            return ProximityBin.FAR

    def get_proximity_weight(self, prox_bin: ProximityBin) -> float:
        """
        Coarse proximity relevance factor W_proximity for SMP.
        Nearer obstacles and objects receive higher guidance priority.
        """
        weights = {
            ProximityBin.CRITICAL_NEAR: 2.5,
            ProximityBin.NEAR: 1.8,
            ProximityBin.MEDIUM: 1.2,
            ProximityBin.FAR: 0.6,
        }
        return weights.get(prox_bin, 1.0)

    def format_proximity_zh(self, distance_m: float, use_steps: bool = False) -> str:
        """Format coarse proximity into concise Chinese guidance wording."""
        if use_steps:
            steps = max(1, int(round(distance_m / self.step_length_m)))
            return localized_text("distance.steps").format(steps=steps)
        else:
            if distance_m < 1.0:
                return localized_text("distance.nearby")
            rounded_m = max(1, int(round(distance_m)))
            return localized_text("distance.meters").format(meters=rounded_m)

    def format_proximity_en(self, distance_m: float) -> str:
        """Format coarse proximity into concise English guidance wording."""
        if distance_m < 1.0:
            return "close"
        rounded_m = max(1, int(round(distance_m)))
        return f"{rounded_m}m"
