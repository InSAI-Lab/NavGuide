# -*- coding: utf-8 -*-
"""Track crosswalk area and produce speech prompts without changing navigation state."""

from navguide.i18n import text as localized_text
import time
import numpy as np
from collections import deque
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrosswalkAwarenessMonitor:
    """Crosswalk awareness and speech scheduling."""

    def __init__(self):
        # Crosswalk area thresholds normalized by frame area.
        self.THRESHOLDS = {
            "discover": 0.01,
            "approaching": 0.08,
            "near": 0.18,
            "arrival": 0.25,
        }

        # Track announced thresholds to suppress duplicates.
        self.broadcasted_thresholds = set()

        self.area_history = deque(maxlen=30)

        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0

        self.in_arrival_state = False
        self.last_position_zone = None

        self.REPEAT_INTERVALS = {
            "approaching": 6.7,
            "near": 3.3,
            "arrival": 5.3,
        }

        self.OCCLUSION_THRESHOLD = 0.30  # Treat more than 30% overlap as occlusion.

    def process_frame(self, crosswalk_mask, blind_path_mask=None) -> Optional[Dict[str, Any]]:
        """Return guidance text, priority, timing, area, position, and visualization data."""

        if crosswalk_mask is None:
            self._reset_if_needed()
            return None

        total_pixels = crosswalk_mask.size
        crosswalk_pixels = np.sum(crosswalk_mask > 0)
        area_ratio = crosswalk_pixels / total_pixels

        y_coords, x_coords = np.where(crosswalk_mask > 0)
        if len(y_coords) == 0:
            return None

        center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
        center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]

        current_time = time.time()
        self.area_history.append(
            {
                "area": area_ratio,
                "center_x": center_x_ratio,
                "center_y": center_y_ratio,
                "time": current_time,
            }
        )

        has_occlusion = self._check_occlusion(crosswalk_mask, blind_path_mask)

        return self._generate_guidance(
            area_ratio, center_x_ratio, center_y_ratio, has_occlusion, current_time
        )

    def _check_occlusion(self, crosswalk_mask, blind_path_mask) -> bool:
        """Check whether tactile paving occludes the crosswalk."""
        if blind_path_mask is None:
            return False

        crosswalk_area = crosswalk_mask > 0
        blind_path_area = blind_path_mask > 0

        overlap = np.logical_and(crosswalk_area, blind_path_area)
        overlap_ratio = np.sum(overlap) / max(np.sum(crosswalk_area), 1)

        return overlap_ratio > self.OCCLUSION_THRESHOLD

    def _get_position_description(self, center_x_ratio) -> str:
        """Classify horizontal position into three image regions."""
        if center_x_ratio < 0.40:
            return localized_text("crosswalk.position_left")
        elif center_x_ratio < 0.60:
            return localized_text("crosswalk.position_center")
        else:
            return localized_text("crosswalk.position_right")

    def _generate_guidance(
        self, area_ratio, center_x_ratio, center_y_ratio, has_occlusion, current_time
    ) -> Optional[Dict[str, Any]]:
        """Generate guidance from the current crosswalk stage."""

        if not self._is_area_stable(area_ratio):
            return None

        position_desc = self._get_position_description(center_x_ratio)

        # Detected stage: area ratio from 0.01 to 0.08.
        if (
            area_ratio >= self.THRESHOLDS["discover"]
            and area_ratio < self.THRESHOLDS["approaching"]
        ):
            if self.THRESHOLDS["discover"] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS["discover"])
                return {
                    "voice_text": localized_text("crosswalk.detected_distant").format(
                        position_desc=f"{position_desc}"
                    ),
                    "priority": 55,
                    "should_broadcast": True,
                    "area": area_ratio,
                    "position": position_desc,
                }

        # Approaching stage: area ratio from 0.08 to 0.18.
        elif area_ratio >= self.THRESHOLDS["approaching"] and area_ratio < self.THRESHOLDS["near"]:

            if self.THRESHOLDS["approaching"] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS["approaching"])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    "voice_text": localized_text("crosswalk.approaching").format(
                        position_desc=f"{position_desc}"
                    ),
                    "priority": 55,
                    "should_broadcast": True,
                    "area": area_ratio,
                    "position": position_desc,
                }

            elif (
                current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS["approaching"]
                or position_desc != self.last_position_zone
            ):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    "voice_text": localized_text("crosswalk.approaching").format(
                        position_desc=f"{position_desc}"
                    ),
                    "priority": 55,
                    "should_broadcast": True,
                    "area": area_ratio,
                    "position": position_desc,
                }

        # Near stage: area ratio from 0.18 to 0.25.
        elif area_ratio >= self.THRESHOLDS["near"] and area_ratio < self.THRESHOLDS["arrival"]:

            if self.THRESHOLDS["near"] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS["near"])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    "voice_text": localized_text("crosswalk.nearby").format(
                        position_desc=f"{position_desc}"
                    ),
                    "priority": 60,
                    "should_broadcast": True,
                    "area": area_ratio,
                    "position": position_desc,
                }

            elif (
                current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS["near"]
                or position_desc != self.last_position_zone
            ):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    "voice_text": localized_text("crosswalk.nearby").format(
                        position_desc=f"{position_desc}"
                    ),
                    "priority": 60,
                    "should_broadcast": True,
                    "area": area_ratio,
                    "position": position_desc,
                }

        # Arrival stage requires area ratio of at least 0.25 and no occlusion.
        elif area_ratio >= self.THRESHOLDS["arrival"]:
            # Only announce arrival when the crosswalk is unobstructed.
            if has_occlusion:
                # Remain in the near stage while the crosswalk is occluded.
                logger.info(
                    f"[CROSSWALK] Area ratio is {area_ratio:.2f}, but occlusion prevents arrival feedback"
                )
                return None

            if not self.in_arrival_state:
                self.in_arrival_state = True
                self.arrival_first_broadcast_time = current_time
                self.last_broadcast_time = current_time
                logger.info(f"[CROSSWALK] Arrival state: area={area_ratio:.2f}, unobstructed")
                return {
                    "voice_text": localized_text("crosswalk.ready_to_cross"),
                    "priority": 80,
                    "should_broadcast": True,
                    "area": area_ratio,
                    "position": localized_text("crosswalk.stage.arrived"),
                }

            elif current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS["arrival"]:
                self.last_broadcast_time = current_time
                return {
                    "voice_text": localized_text("crosswalk.ready_to_cross"),
                    "priority": 80,
                    "should_broadcast": True,
                    "area": area_ratio,
                    "position": localized_text("crosswalk.stage.arrived"),
                }
            # Expire the arrival state after 30 seconds.
            elif current_time - self.arrival_first_broadcast_time > 30.0:
                logger.info("[CROSSWALK] Arrival state expired after 30 seconds")
                self.in_arrival_state = False
                return None

        if self.in_arrival_state and area_ratio < 0.20:
            logger.info(
                f"[CROSSWALK] Area ratio dropped to {area_ratio:.2f}; leaving arrival state"
            )
            self.in_arrival_state = False

            self.broadcasted_thresholds.discard(self.THRESHOLDS["arrival"])

        return None

    def _is_area_stable(self, area_ratio, stability_frames=5) -> bool:
        """Check whether recent area measurements are stable."""
        if len(self.area_history) < stability_frames:
            return True

        recent_areas = [h["area"] for h in list(self.area_history)[-stability_frames:]]

        # Require recent area measurements to stay within 20%.
        for recent_area in recent_areas:
            if abs(recent_area - area_ratio) / max(area_ratio, 0.001) > 0.20:
                return False

        return True

    def _reset_if_needed(self):
        """Reset transient state after the crosswalk disappears."""
        if len(self.area_history) > 0:
            logger.info("[CROSSWALK] Crosswalk lost; resetting state")

        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_position_zone = None

    def reset(self):
        """Reset all crosswalk state."""
        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0
        self.last_position_zone = None
        logger.info("[CROSSWALK] Awareness monitor reset")

    def is_in_arrival_state(self) -> bool:
        """Return whether the monitor is in the arrival state."""
        return self.in_arrival_state

    def get_current_area(self) -> float:
        """Return the latest crosswalk area ratio."""
        if len(self.area_history) > 0:
            return self.area_history[-1]["area"]
        return 0.0

    def get_visualization_data(
        self, crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion
    ) -> Dict[str, Any]:
        """Return the current visualization parameters."""
        if crosswalk_mask is None:
            return {}

        if area_ratio >= self.THRESHOLDS["arrival"]:
            stage = localized_text("crosswalk.stage.arrived")
            stage_color = "rgba(255, 165, 0, 0.5)"
        elif area_ratio >= self.THRESHOLDS["near"]:
            stage = localized_text("crosswalk.stage.approaching")
            stage_color = "rgba(255, 165, 0, 0.45)"
        elif area_ratio >= self.THRESHOLDS["approaching"]:
            stage = localized_text("crosswalk.stage.nearby")
            stage_color = "rgba(255, 165, 0, 0.40)"
        else:
            stage = localized_text("crosswalk.stage.detected")
            stage_color = "rgba(255, 165, 0, 0.35)"

        position = self._get_position_description(center_x_ratio)

        return {
            "area_ratio": area_ratio,
            "stage": stage,
            "stage_color": stage_color,
            "position": position.replace(localized_text("crosswalk.position_prefix"), ""),
            "center_x_ratio": center_x_ratio,
            "center_y_ratio": center_y_ratio,
            "has_occlusion": has_occlusion,
            "in_arrival": self.in_arrival_state,
        }


def split_combined_voice(combined_text: str) -> list:
    """Split a combined guidance message into individual prompts."""
    if "," in combined_text:
        parts = combined_text.split(",")
        return [p.strip() for p in parts if p.strip()]
    return [combined_text]
