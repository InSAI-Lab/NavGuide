# -*- coding: utf-8 -*-
"""Tactile paving navigation, obstacle warnings and crosswalk awareness."""

from navguide.i18n import terms, text as localized_text
import os
import time
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from navguide.audio.player import play_voice_text
from navguide.perception.crosswalk import CrosswalkAwarenessMonitor, split_combined_voice

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageDraw, ImageFont = None, None, None

logger = logging.getLogger(__name__)


STATE_ONBOARDING = "ONBOARDING"
STATE_NAVIGATING = "NAVIGATING"
STATE_MANEUVERING_TURN = "MANEUVERING_TURN"
STATE_AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
STATE_LOCKING_ON = "LOCKING_ON"


ONBOARDING_STEP_ROTATION = "ROTATION"
ONBOARDING_STEP_TRANSLATION = "TRANSLATION"


MANEUVER_STEP_1_ISSUE_COMMAND = "ISSUE_COMMAND"
MANEUVER_STEP_2_WAIT_FOR_SHIFT = "WAIT_FOR_SHIFT"
MANEUVER_STEP_3_ALIGN_ON_NEW_PATH = "ALIGN_ON_NEW_PATH"

# Colors use OpenCV BGR order.
VIS_COLORS = {
    "blind_path": (0, 255, 0),
    "obstacle": (0, 0, 255),
    "crosswalk": (0, 165, 255),
    "centerline": (0, 255, 255),
    "target_point": (255, 0, 0),
    "turn_point": (128, 0, 128),
    "pulse_effect": (100, 100, 255),
}


_OBSTACLE_NAME_CN = {
    "person": localized_text("object.person_short"),
    "bicycle": localized_text("object.bicycle"),
    "car": localized_text("object.car_short"),
    "motorcycle": localized_text("object.motorcycle"),
    "bus": localized_text("object.bus"),
    "truck": localized_text("object.truck"),
    "animal": localized_text("object.animal"),
    "scooter": localized_text("object.electric_moped"),
    "stroller": localized_text("object.stroller"),
    "dog": localized_text("object.dog"),
}

# Categories treated as moving hazards.
DYNAMIC_CLASS_NAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "animal", "dog"}


@dataclass(frozen=True)
class PathDetectionResult:
    """Current-frame segmentation and detector availability."""

    available: bool
    blind_path_mask: Optional[np.ndarray] = None
    crosswalk_mask: Optional[np.ndarray] = None
    reason: Optional[str] = None


@dataclass
class ProcessingResult:
    """Navigation output, annotations and state information."""

    guidance_text: str
    visualizations: List[Dict[str, Any]]
    annotated_image: Optional[np.ndarray] = None
    state_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.state_info is None:
            self.state_info = {}


class BlindPathNavigator:
    """Guide the user along a segmented tactile paving path."""

    def __init__(self, yolo_model=None, obstacle_detector=None):
        """Initialize navigation state with optional segmentation and obstacle models."""
        self.yolo_model = yolo_model
        self.obstacle_detector = obstacle_detector

        self.current_state = STATE_ONBOARDING
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.05,
            minDistance=10,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04,
        )

        self.flow_points = {}  # {mask_type: points}
        self.flow_grace = {}  # {mask_type: grace_count}
        self.FLOW_GRACE_MAX = 3  # Retain optical flow predictions for at most three frames.

        self.centerline_history = []
        self.centerline_history_max = 5

        self.poly_coeffs_history = []
        self.poly_coeffs_history_max = 8

        self.turn_detection_tracker = {
            "direction": None,
            "consecutive_hits": 0,
            "last_seen_frame": 0,
            "corner_info": None,
        }

        self.turn_cooldown_frames = 0
        self.TURN_COOLDOWN_DURATION = 50

        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None

        self.crosswalk_tracker = {
            "stage": "not_detected",
            "consecutive_frames": 0,
            "last_area_ratio": 0.0,
            "last_bottom_y_ratio": 0.0,
            "last_center_x_ratio": 0.5,
            "position_announced": False,
            "alignment_status": "not_aligned",
            "last_seen_frame": 0,
            "last_angle": 0.0,
        }

        self.frame_counter = 0

        # Speech intervals can be configured through environment variables.
        self.guide_interval = float(
            os.getenv("NAVGUIDE_STRAIGHT_INTERVAL", "4.0")
        )  # Speech interval in seconds.
        self.last_guide_time = 0.0
        self.straight_continuous_mode = os.getenv("NAVGUIDE_STRAIGHT_CONTINUOUS", "1") == "1"
        self.straight_repeat_limit = int(os.getenv("NAVGUIDE_STRAIGHT_LIMIT", "2"))
        self.straight_repeat_count = 0

        self.direction_interval = float(
            os.getenv("NAVGUIDE_DIRECTION_INTERVAL", "3.0")
        )  # Direction speech interval in seconds.
        self.last_direction_time = 0.0
        self.last_direction_message = ""

        logger.info(
            f"[BlindPath] Straight speech interval={self.guide_interval} seconds, continuous={self.straight_continuous_mode}, repeat_limit={self.straight_repeat_limit}"
        )
        logger.info(f"[BlindPath] Direction speech interval={self.direction_interval} seconds")

        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_any_speech_time = 0

        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0

        self.pending_obstacle_voice = None

        self.CLASS_CONF_THRESHOLDS = {1: 0.20, 0: 0.30}  # blind_path  # crosswalk

        self.ONBOARDING_ALIGN_THRESHOLD_RATIO = 0.1
        self.VP_FIT_ERROR_THRESHOLD = 8.0

        self.ONBOARDING_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.NAV_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.NAV_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.CURVATURE_PROXY_THRESHOLD = 5e-5

        self.CROSSWALK_SWITCH_AREA_RATIO = 0.22
        self.CROSSWALK_SWITCH_BOTTOM_RATIO = 0.9
        self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES = 10

        self.OBSTACLE_DETECTION_INTERVAL = int(os.getenv("NAVGUIDE_OBS_INTERVAL", "15"))
        self.OBSTACLE_CACHE_DURATION_FRAMES = int(os.getenv("NAVGUIDE_OBS_CACHE_FRAMES", "10"))

        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.obstacle_speech_cooldown = 5.0  # Repeat interval for the same obstacle, in seconds.

        # Mask stabilization configuration.
        self.MASK_STAB_MIN_AREA = int(os.getenv("NAVGUIDE_MASK_MIN_AREA", "1500"))
        self.MASK_STAB_KERNEL = int(os.getenv("NAVGUIDE_MASK_MORPH", "3"))
        self.MASK_MISS_TTL = 0  # Disable extrapolation for current-frame guidance.
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0

        self.flow_iou_threshold = 0.3  # Reseed optical flow features below this IoU.

        self.BLINDPATH_DETECTION_INTERVAL = int(os.getenv("NAVGUIDE_BLINDPATH_INTERVAL", "8"))
        self.last_blindpath_detection_frame = 0
        self.last_blindpath_mask = None
        self.last_crosswalk_mask = None

        self.crosswalk_monitor = CrosswalkAwarenessMonitor()
        logger.info("[BlindPath] Crosswalk monitor initialized")
        logger.info(
            f"[BlindPath] Tactile path detection interval={self.BLINDPATH_DETECTION_INTERVAL} frames"
        )

    def _get_voice_priority(self, guidance_text):
        """Rank obstacle warnings above direction changes and straight guidance."""
        if not guidance_text:
            return 0

        # Obstacle warnings have the highest priority.
        obstacle_keywords = terms("path.obstacle_keywords")
        for keyword in obstacle_keywords:
            if keyword in guidance_text:
                return 100

        # Direction changes have medium priority.
        direction_keywords = terms("path.direction_keywords")
        for keyword in direction_keywords:
            if keyword in guidance_text:
                return 50

        # Straight guidance has the lowest priority.
        if any(keyword in guidance_text for keyword in terms("path.straight_keywords")):
            return 10

        return 30

    def process_frame(self, image: np.ndarray) -> ProcessingResult:
        """Process a BGR frame and return guidance, annotations and state."""
        self.frame_counter += 1

        if self.turn_cooldown_frames > 0:
            self.turn_cooldown_frames -= 1

        image_height, image_width = image.shape[:2]
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        frame_visualizations = []
        guidance_text = ""

        detection = self._detect_path_and_crosswalk(image)
        if not detection.available:
            frame_counter = self.frame_counter
            self.reset()
            self.frame_counter = frame_counter
            status = localized_text("path.detection_unavailable")
            return ProcessingResult(
                guidance_text="",
                visualizations=[],
                annotated_image=self._draw_command_button(image.copy(), status),
                state_info={
                    "state": self.current_state,
                    "crosswalk_stage": "not_detected",
                    "frame_count": self.frame_counter,
                    "detection_available": False,
                    "detection_reason": detection.reason,
                },
            )

        blind_path_mask = detection.blind_path_mask
        crosswalk_mask = detection.crosswalk_mask

        if self.frame_counter % 30 == 0:
            has_blind = blind_path_mask is not None and np.sum(blind_path_mask > 0) > 0
            has_cross = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(
                f"[YOLO] Frame={self.frame_counter}, tactile_path={('yes' if has_blind else 'no')}, crosswalk={('yes' if has_cross else 'no')}"
            )
            if has_cross:
                cross_area = np.sum(crosswalk_mask > 0) / crosswalk_mask.size
                logger.info(f"[YOLO] Raw crosswalk area: {cross_area * 100:.2f}%")

        # Use current segmentation without mask stabilization or extrapolation.

        crosswalk_mask_before_stabilize = crosswalk_mask

        if self.frame_counter % 30 == 0 and crosswalk_mask_before_stabilize is not None:
            after_stab = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(
                f"[Mask] Stabilized crosswalk: {('yes' if after_stab else 'no (filtered)')}"
            )

        logger.info(f"[Frame {self.frame_counter}] Starting obstacle detection")

        # Cache obstacle detections while retaining their visualization.
        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image, blind_path_mask)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
            logger.info(
                f"[Frame {self.frame_counter}] Fresh obstacle detection found {len(detected_obstacles)} obstacles"
            )
        else:
            if (
                self.frame_counter - self.last_obstacle_detection_frame
                < self.OBSTACLE_CACHE_DURATION_FRAMES
            ):
                detected_obstacles = self.last_detected_obstacles
                logger.info(
                    f"[Frame {self.frame_counter}] Using cached data: {len(detected_obstacles)} obstacles"
                )
            else:
                detected_obstacles = []
                logger.info(f"[Frame {self.frame_counter}] Obstacle cache expired")

        for i, obs in enumerate(detected_obstacles):
            logger.info(
                f"  Obstacle {i + 1}: {obs.get('name', 'unknown')}, bottom_y_ratio={obs.get('bottom_y_ratio', 0):.2f}, area_ratio={obs.get('area_ratio', 0):.3f}, position=({obs.get('center_x', 0):.0f}, {obs.get('center_y', 0):.0f})"
            )
            self._add_obstacle_visualization(obs, frame_visualizations)

        self._check_and_set_obstacle_voice(detected_obstacles)

        if crosswalk_mask is not None:
            cross_pixels = np.sum(crosswalk_mask > 0)
            if cross_pixels > 0:
                logger.info(
                    f"[Crosswalk] Monitor input: pixels={cross_pixels}, area={cross_pixels / crosswalk_mask.size * 100:.2f}%"
                )
            else:
                logger.info(f"[Crosswalk] Empty mask")
        else:
            if self.frame_counter % 30 == 0:
                logger.info(f"[Crosswalk] No mask")

        crosswalk_guidance = self.crosswalk_monitor.process_frame(crosswalk_mask, blind_path_mask)
        if crosswalk_guidance:
            logger.info(
                f"[Crosswalk] Detection: area={crosswalk_guidance.get('area', 0):.3f}, should_broadcast={crosswalk_guidance.get('should_broadcast', False)}, voice={crosswalk_guidance.get('voice_text', 'None')}"
            )
        if crosswalk_guidance and crosswalk_guidance["should_broadcast"]:

            if not hasattr(self, "pending_crosswalk_voice"):
                self.pending_crosswalk_voice = None
            self.pending_crosswalk_voice = crosswalk_guidance
            logger.info(
                f"[Crosswalk speech] Pending cue: {crosswalk_guidance['voice_text']}, priority={crosswalk_guidance['priority']}"
            )

        if crosswalk_mask is not None:

            total_pixels = crosswalk_mask.size
            crosswalk_pixels = np.sum(crosswalk_mask > 0)
            area_ratio = crosswalk_pixels / total_pixels

            y_coords, x_coords = np.where(crosswalk_mask > 0)
            if len(y_coords) > 0:
                center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
                center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]
                has_occlusion = self.crosswalk_monitor._check_occlusion(
                    crosswalk_mask, blind_path_mask
                )

                viz_data = self.crosswalk_monitor.get_visualization_data(
                    crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion
                )

                self._add_mask_visualization(
                    crosswalk_mask, frame_visualizations, "crosswalk_mask", viz_data["stage_color"]
                )

                self._add_crosswalk_info_visualization(
                    viz_data, image_height, image_width, frame_visualizations
                )

        self._add_mask_visualization(
            blind_path_mask, frame_visualizations, "blind_path_mask", "rgba(0, 255, 0, 0.4)"
        )

        current_stage = "not_detected"  # Crosswalk awareness is handled by the monitor.

        if blind_path_mask is None:
            frame_visualizations.append(
                {
                    "type": "data_panel",
                    "data": {
                        localized_text("ui.status"): localized_text("path.waiting_for_detection")
                    },
                    "position": (image_width - 180, 20),
                }
            )
        else:
            guidance_text = self._execute_state_machine(
                blind_path_mask,
                image,
                frame_visualizations,
                image_height,
                image_width,
                curr_gray,
            )

        self.prev_gray = curr_gray
        self.prev_blind_path_mask = blind_path_mask.copy() if blind_path_mask is not None else None
        self.prev_crosswalk_mask = crosswalk_mask.copy() if crosswalk_mask is not None else None

        current_time = time.time()

        voice_candidates = []

        if guidance_text:
            voice_candidates.append(
                {
                    "text": guidance_text,
                    "priority": self._get_voice_priority(guidance_text),
                    "source": "navigation",
                }
            )

        # Check obstacle warnings independently of path guidance.
        if hasattr(self, "pending_obstacle_voice"):
            if self.pending_obstacle_voice:
                voice_candidates.append(
                    {"text": self.pending_obstacle_voice, "priority": 100, "source": "obstacle"}
                )
                self.pending_obstacle_voice = None

        if hasattr(self, "pending_crosswalk_voice"):
            if self.pending_crosswalk_voice:
                voice_candidates.append(
                    {
                        "text": self.pending_crosswalk_voice["voice_text"],
                        "priority": self.pending_crosswalk_voice["priority"],
                        "source": "crosswalk",
                    }
                )
                self.pending_crosswalk_voice = None

        if voice_candidates:

            voice_candidates.sort(key=lambda x: x["priority"], reverse=True)
            selected_voice = voice_candidates[0]
            final_guidance_text = selected_voice["text"]

            # Enforce a global interval between speech outputs.
            MIN_SPEECH_INTERVAL = 1.2  # Minimum interval between any two cues, in seconds.
            if hasattr(self, "last_any_speech_time"):
                if current_time - self.last_any_speech_time < MIN_SPEECH_INTERVAL:
                    final_guidance_text = ""

            if final_guidance_text == localized_text("path.keep_straight"):
                if self.straight_continuous_mode:
                    # Continuous mode applies only the time interval.
                    if current_time - self.last_guide_time >= self.guide_interval:
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
                else:

                    if (current_time - self.last_guide_time >= self.guide_interval) and (
                        self.straight_repeat_count < self.straight_repeat_limit
                    ):
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice["source"] != "obstacle":

                # Direction commands can repeat at the configured interval.
                direction_keywords = terms("path.direction_keywords")
                is_direction = any(keyword in final_guidance_text for keyword in direction_keywords)

                if is_direction:

                    if final_guidance_text == self.last_direction_message:

                        if current_time - self.last_direction_time >= self.direction_interval:
                            self.last_direction_time = current_time
                            self.last_any_speech_time = current_time
                            self.straight_repeat_count = 0
                        else:
                            final_guidance_text = ""
                    else:

                        self.last_direction_message = final_guidance_text
                        self.last_direction_time = current_time
                        self.last_any_speech_time = current_time
                        self.straight_repeat_count = 0
                else:
                    # Other commands are spoken once per change.
                    if final_guidance_text != self.last_guidance_message:
                        self.last_guidance_message = final_guidance_text
                        self.straight_repeat_count = 0
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice["source"] == "obstacle":
                # Obstacle cues bypass the generic duplicate check.
                self.last_any_speech_time = current_time
            elif final_guidance_text and selected_voice["source"] == "crosswalk":
                # Crosswalk cues bypass the generic duplicate check.
                self.last_any_speech_time = current_time

            if final_guidance_text:
                try:
                    # Speak only the first segment to avoid stale queued guidance.
                    if selected_voice.get("source") == "crosswalk" and "," in final_guidance_text:
                        voice_parts = split_combined_voice(final_guidance_text)
                        logger.info(
                            f"[Crosswalk speech] Combined cue has {len(voice_parts)} segments; playing only the first to avoid stale speech"
                        )

                        if voice_parts:
                            play_voice_text(voice_parts[0])
                            logger.info(
                                f"[Speech] Priority={selected_voice['priority']}: {voice_parts[0]}"
                            )
                    else:
                        play_voice_text(final_guidance_text)
                        logger.info(
                            f"[Speech] Priority={selected_voice['priority']}: {final_guidance_text}"
                        )
                except Exception as e:
                    logger.error(f"[Speech] Playback failed: {e}")
        else:
            final_guidance_text = ""

        annotated_image = None

        if frame_visualizations:
            annotated_image = self._draw_visualizations(image.copy(), frame_visualizations)
        else:
            annotated_image = image.copy()

        current_instruction = (
            final_guidance_text if final_guidance_text else localized_text("ui.waiting")
        )
        annotated_image = self._draw_command_button(annotated_image, current_instruction)

        return ProcessingResult(
            guidance_text=guidance_text,
            visualizations=frame_visualizations,
            annotated_image=annotated_image,
            state_info={
                "state": self.current_state,
                "crosswalk_stage": current_stage,
                "frame_count": self.frame_counter,
                "detection_available": True,
                "detection_reason": None,
            },
        )

    def _detect_path_and_crosswalk(self, image: np.ndarray) -> PathDetectionResult:
        """Segment the frame without substituting geometry when inference is unavailable."""
        if self.yolo_model is None:
            return PathDetectionResult(available=False, reason="model_unavailable")

        blind_path_mask = None
        crosswalk_mask = None

        try:
            min_conf = min(self.CLASS_CONF_THRESHOLDS.values())
            results = self.yolo_model.predict(image, verbose=False, conf=min_conf, classes=[0, 1])
            if results is None:
                raise ValueError("The segmentation model returned no result object")

            for result in results:
                boxes = result.boxes
                masks = result.masks
                if boxes is None:
                    if masks is not None and len(masks.data):
                        raise ValueError("Segmentation masks are missing their class labels")
                    continue
                if masks is None:
                    if len(boxes.cls):
                        raise ValueError("The model returned boxes without segmentation masks")
                    continue

                for mask_tensor, conf_tensor, cls_tensor in zip(
                    masks.data, boxes.conf, boxes.cls, strict=True
                ):
                    class_id = int(cls_tensor.item())
                    confidence = float(conf_tensor.item())
                    if not np.isfinite(confidence):
                        raise ValueError("Segmentation confidence must be finite")
                    threshold = self.CLASS_CONF_THRESHOLDS.get(class_id, 1.0)
                    if confidence < threshold or class_id not in (0, 1):
                        continue
                    current_mask = self._tensor_to_mask(mask_tensor, image.shape[1], image.shape[0])
                    if not np.any(current_mask):
                        continue

                    if class_id == 1:
                        blind_path_mask = (
                            current_mask
                            if blind_path_mask is None
                            else cv2.bitwise_or(blind_path_mask, current_mask)
                        )
                    else:
                        crosswalk_mask = (
                            current_mask
                            if crosswalk_mask is None
                            else cv2.bitwise_or(crosswalk_mask, current_mask)
                        )
        except Exception:
            logger.exception("Tactile path segmentation failed")
            return PathDetectionResult(available=False, reason="inference_failed")

        return PathDetectionResult(
            available=True,
            blind_path_mask=blind_path_mask,
            crosswalk_mask=crosswalk_mask,
        )

    def _tensor_to_mask(
        self, mask_tensor, out_w: int, out_h: int, binarize: bool = True
    ) -> np.ndarray:
        """Convert a tensor mask to a binary NumPy array."""
        if isinstance(mask_tensor, np.ndarray):
            arr = mask_tensor.squeeze()
        else:
            arr = mask_tensor.detach().float().cpu().numpy().squeeze()
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError("Segmentation masks must contain a finite two-dimensional array")

        if binarize:
            mask_u8 = (arr > 0.5).astype(np.uint8) * 255
        elif arr.dtype == np.uint8:
            mask_u8 = arr
        else:
            mask_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        if mask_u8.shape != (out_h, out_w):
            mask_u8 = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return mask_u8

    def _stabilize_mask(self, prev_gray, curr_gray, raw_mask, prev_stable_mask, mask_type):
        """Stabilize a mask using Lucas-Kanade optical flow."""
        if mask_type == "blind_path":
            ttl = self.blind_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA
        else:  # crosswalk
            ttl = self.cross_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA

        stable_mask = self._stabilize_seg_mask(
            prev_gray,
            curr_gray,
            raw_mask,
            prev_stable_mask,
            (curr_gray.shape[1], curr_gray.shape[0]) if curr_gray is not None else (640, 480),
            min_area_px=min_area,
            morph_kernel=self.MASK_STAB_KERNEL,
            mask_type=mask_type,
        )

        if stable_mask is not None:

            if mask_type == "blind_path":
                self.blind_miss_ttl = self.MASK_MISS_TTL
            else:
                self.cross_miss_ttl = self.MASK_MISS_TTL
            return stable_mask
        else:

            if mask_type == "blind_path":
                self.blind_miss_ttl = max(0, self.blind_miss_ttl - 1)
            else:
                self.cross_miss_ttl = max(0, self.cross_miss_ttl - 1)
            return None

    def _stabilize_seg_mask(
        self,
        prev_gray,
        curr_gray,
        curr_mask,
        prev_stable_mask,
        image_wh,
        min_area_px=1500,
        morph_kernel=3,
        iou_high_thr=0.4,
        mask_type="",
        fast_clear=True,
    ):
        """Blend segmentation and optical flow within the tracking lifetime."""
        W, H = image_wh

        def _binarize(mask):
            if mask is None:
                return None
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        def _morph_smooth(mask, kernel_size):
            if mask is None:
                return None
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(1, kernel_size), max(1, kernel_size))
            )
            sm = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
            sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, k, iterations=1)
            return sm

        curr_mask_b = _binarize(curr_mask)
        prev_mask_b = _binarize(prev_stable_mask)

        if prev_mask_b is None or prev_gray is None or curr_gray is None:
            return _morph_smooth(curr_mask_b, morph_kernel) if curr_mask_b is not None else None

        if curr_mask_b is not None and np.sum(curr_mask_b > 0) >= min_area_px:

            if prev_mask_b is not None:
                inter = np.logical_and(curr_mask_b > 0, prev_mask_b > 0).sum()
                union = np.logical_or(curr_mask_b > 0, prev_mask_b > 0).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0

                # Accept stable detections directly.
                if iou >= iou_high_thr:
                    return _morph_smooth(curr_mask_b, morph_kernel)

                # Blend partially overlapping detections with optical flow.
                elif iou > 0.1:

                    flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                    if flow_mask is not None:
                        # Give optical flow more weight as overlap decreases.

                        w_curr = min(0.9, 0.4 + iou)
                        w_flow = 1.0 - w_curr

                        fused = w_curr * curr_mask_b.astype(np.float32) + w_flow * flow_mask.astype(
                            np.float32
                        )
                        fused_bin = (fused >= 128).astype(np.uint8) * 255

                        if iou < self.flow_iou_threshold:
                            self.flow_points["blind_path"] = None

                        return _morph_smooth(fused_bin, morph_kernel)

            return _morph_smooth(curr_mask_b, morph_kernel)

        else:
            # Limit extrapolation by the remaining tracking lifetime.
            if mask_type == "blind_path":
                ttl = self.blind_miss_ttl
            else:
                ttl = self.cross_miss_ttl

            if fast_clear and ttl <= 1:
                # Discard stale masks when the tracking lifetime expires.
                return None

            if prev_mask_b is not None and np.sum(prev_mask_b > 0) >= min_area_px and ttl > 0:

                flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                if flow_mask is not None and np.sum(flow_mask > 0) >= min_area_px * 0.5:
                    return _morph_smooth(flow_mask, morph_kernel)

            return None

    def _predict_mask_with_flow(self, prev_mask, prev_gray, curr_gray):
        """Predict the mask using Lucas-Kanade optical flow and affine fallback."""
        try:
            # Try tracked points and their convex hull first.
            if hasattr(self, "flow_points") and "blind_path" in self.flow_points:
                p0 = self.flow_points["blind_path"]
                if p0 is not None and len(p0) >= 5:

                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, p0, None, **self.lk_params
                    )

                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:

                            self.flow_points["blind_path"] = good_new.reshape(-1, 1, 2)

                            hull = cv2.convexHull(good_new.reshape(-1, 1, 2))
                            poly = hull.reshape(-1, 2)

                            if len(poly) >= 3:
                                H, W = curr_gray.shape[:2]
                                flow_mask = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(flow_mask, [poly.astype(np.int32)], 255)
                                return flow_mask

            # Fall back to boundary features and an affine transform.
            edge_mask = self._get_edge_mask(prev_mask, offset=10)

            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=edge_mask, **self.feature_params)
            if p0 is None or len(p0) < 8:
                return None

            self.flow_points["blind_path"] = p0

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)

            if p1 is None or st is None:
                return None

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 5:
                return None

            # Use RANSAC to reject inconsistent feature motions.
            M, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )

            if M is None:
                return None

            H, W = curr_gray.shape[:2]
            flow_mask = cv2.warpAffine(
                prev_mask,
                M,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            return flow_mask

        except Exception as e:
            logger.debug(f"Optical flow prediction failed: {e}")
            return None

    def _get_edge_mask(self, mask, offset=10):
        """Extract an inner mask boundary for optical flow feature selection."""
        if mask is None:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset * 2, offset * 2))
        inner = cv2.erode(mask, kernel, iterations=1)

        edge = cv2.subtract(mask, inner)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.dilate(edge, kernel_small, iterations=1)

        return edge

    def _smooth_centerline(self, centerline_data):
        """Smooth centerline positions and widths across nearby rows and frames."""
        if centerline_data is None or len(centerline_data) < 5:
            return centerline_data

        self.centerline_history.append(centerline_data.copy())
        if len(self.centerline_history) > self.centerline_history_max:
            self.centerline_history.pop(0)

        if len(self.centerline_history) < 3:

            smoothed_data = centerline_data.copy()

            window_size = 5
            for i in range(len(smoothed_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(smoothed_data), i + window_size // 2 + 1)
                window = smoothed_data[start_idx:end_idx]
                if len(window) > 0:
                    smoothed_data[i, 1] = np.mean(window[:, 1])
                    smoothed_data[i, 2] = np.mean(window[:, 2])
            return smoothed_data

        # Smooth corresponding rows with a weighted temporal average.
        smoothed_data = centerline_data.copy()

        for i, (y, x, width) in enumerate(centerline_data):
            x_values = [x]
            width_values = [width]
            weights = [1.0]

            for hist_idx, hist_data in enumerate(
                self.centerline_history[-3:-1]
            ):  # Use the two most recent frames.

                y_diffs = np.abs(hist_data[:, 0] - y)
                if len(y_diffs) > 0:
                    closest_idx = np.argmin(y_diffs)
                    if y_diffs[closest_idx] < 10:  # Match historical rows within ten pixels.
                        x_values.append(hist_data[closest_idx, 1])
                        width_values.append(hist_data[closest_idx, 2])
                        # Decrease the weight of older frames.
                        weights.append(0.5 ** (len(self.centerline_history) - hist_idx - 1))

            if len(x_values) > 1:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                smoothed_data[i, 1] = np.sum(np.array(x_values) * weights)
                smoothed_data[i, 2] = np.sum(np.array(width_values) * weights)

        # Apply a spatial moving average after temporal smoothing.
        window_size = 3
        final_data = smoothed_data.copy()
        for i in range(len(final_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(final_data), i + window_size // 2 + 1)
            window = smoothed_data[start_idx:end_idx]
            if len(window) > 0:
                final_data[i, 1] = np.mean(window[:, 1])
                final_data[i, 2] = np.mean(window[:, 2])

        return final_data

    def _estimate_affine(self, prev_gray, curr_gray, mask=None):
        """Estimate an affine transform from tracked optical flow features."""
        try:

            if mask is not None:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
            else:
                p0 = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)

            if p0 is None or len(p0) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)

            if p1 is None or st is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            good_new = p1[st == 1].reshape(-1, 2)
            good_old = p0[st == 1].reshape(-1, 2)

            if len(good_new) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)

            if M is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            return M

        except Exception as e:
            logger.debug(f"Affine estimation failed: {e}")
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    def _warp_mask(self, mask, M, output_shape):
        """Warp a mask using an affine transform."""
        try:
            W, H = output_shape
            warped = cv2.warpAffine(
                mask,
                M,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return warped
        except:
            return None

    def _add_mask_visualization(self, mask, visualizations, viz_type, color, add_outline=True):
        """Add mask fill and contour annotations."""
        if mask is None:
            return

        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                points = main_contour.squeeze(1)[::5].tolist()

                visualizations.append({"type": viz_type, "points": points, "color": color})

                # Outline crosswalk masks; tactile paving uses fill only.
                if add_outline and viz_type != "blind_path_mask":
                    visualizations.append(
                        {
                            "type": "outline",
                            "points": points,
                            "color": "rgba(255, 255, 255, 0.8)",
                            "thickness": 3,
                        }
                    )
        except:
            pass

    def _update_crosswalk_tracker(self, crosswalk_mask, image_height, image_width):
        """Update crosswalk geometry and tracking state."""
        if crosswalk_mask is not None:
            self.crosswalk_tracker["consecutive_frames"] += 1
            self.crosswalk_tracker["last_seen_frame"] = self.frame_counter

            total_area = image_height * image_width
            area_ratio = np.sum(crosswalk_mask > 0) / total_area
            y_coords, x_coords = np.where(crosswalk_mask > 0)

            if len(y_coords) > 0:
                bottom_y_ratio = np.max(y_coords) / image_height
                center_x_ratio = np.mean(x_coords) / image_width

                self.crosswalk_tracker["last_area_ratio"] = area_ratio
                self.crosswalk_tracker["last_bottom_y_ratio"] = bottom_y_ratio
                self.crosswalk_tracker["last_center_x_ratio"] = center_x_ratio

                try:
                    contours, _ = cv2.findContours(
                        crosswalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(main_contour)
                        angle = rect[-1]
                        w, h = rect[1]
                        if w < h:
                            angle += 90
                        self.crosswalk_tracker["last_angle"] = angle
                except:
                    self.crosswalk_tracker["last_angle"] = 0.0

                is_ready_to_switch = (
                    area_ratio >= self.CROSSWALK_SWITCH_AREA_RATIO
                    and bottom_y_ratio >= self.CROSSWALK_SWITCH_BOTTOM_RATIO
                    or (
                        self.crosswalk_tracker["consecutive_frames"]
                        >= self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES
                        and area_ratio > 0.18
                    )
                )

                if is_ready_to_switch and self.crosswalk_tracker["alignment_status"] == "aligned":
                    if self.crosswalk_tracker["stage"] != "ready":
                        self.crosswalk_tracker["stage"] = "ready"
                elif area_ratio > 0.07 or bottom_y_ratio > 0.75:
                    if self.crosswalk_tracker["stage"] in ["far", "not_detected"]:
                        self.crosswalk_tracker["stage"] = "approaching"
                elif area_ratio > 0.01:
                    if self.crosswalk_tracker["stage"] == "not_detected":
                        self.crosswalk_tracker["stage"] = "far"
        else:

            if self.frame_counter - self.crosswalk_tracker["last_seen_frame"] > 15:
                self.crosswalk_tracker["stage"] = "not_detected"
                self.crosswalk_tracker["consecutive_frames"] = 0
                self.crosswalk_tracker["position_announced"] = False
                self.crosswalk_tracker["alignment_status"] = "not_aligned"

                if hasattr(self, "crosswalk_ready_announced"):
                    self.crosswalk_ready_announced = False
                    self.crosswalk_ready_time = 0

    def _handle_crosswalk_approaching(self, frame_visualizations, image_height, image_width, image):
        """Align with a crosswalk while prioritizing nearby obstacles."""

        if self.obstacle_detector and self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter

        for obs in self.last_detected_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)

        # Warn only for obstacles that satisfy the near-field thresholds.
        NEAR_DISTANCE_Y_THRESHOLD = 0.75
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12
        near_obstacles = [
            obs
            for obs in self.last_detected_obstacles
            if (
                obs.get("bottom_y_ratio", 0) > NEAR_DISTANCE_Y_THRESHOLD
                or obs.get("area_ratio", 0) > NEAR_DISTANCE_AREA_THRESHOLD
            )
        ]

        if near_obstacles:
            main_obstacle = near_obstacles[0]
            obstacle_name = main_obstacle.get("name", "")
            current_time = time.time()

            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                should_announce = True
                self.last_obstacle_speech_time = current_time

            if should_announce:
                return self._speech_for_obstacle(obstacle_name)
        else:

            self.last_obstacle_speech = ""

        if self.crosswalk_tracker["alignment_status"] == "not_aligned":
            guidance_text = localized_text("path.approaching_crosswalk")
            self.crosswalk_tracker["alignment_status"] = "aligning"
        else:
            angle = self.crosswalk_tracker["last_angle"]
            center_x_ratio = self.crosswalk_tracker["last_center_x_ratio"]

            ANGLE_ALIGN_THRESHOLD = 15
            POSITION_ALIGN_THRESHOLD = 0.25

            if abs(angle) > ANGLE_ALIGN_THRESHOLD:
                guidance_text = (
                    localized_text("path.turn_right")
                    if angle < 0
                    else localized_text("path.turn_left")
                )
            elif abs(center_x_ratio - 0.5) > (POSITION_ALIGN_THRESHOLD / 2):
                guidance_text = (
                    localized_text("path.move_right")
                    if center_x_ratio < 0.5
                    else localized_text("path.move_left")
                )
            else:
                self.crosswalk_tracker["alignment_status"] = "aligned"
                guidance_text = localized_text("path.crosswalk_aligned")

        data_for_panel = {
            localized_text("ui.status"): localized_text("crossing.align_crosswalk"),
            localized_text("ui.guidance"): guidance_text,
            localized_text("ui.angle"): f"{self.crosswalk_tracker['last_angle']:.1f}°",
            localized_text(
                "ui.offset"
            ): f"{(self.crosswalk_tracker['last_center_x_ratio'] - 0.5):.2f}",
        }
        frame_visualizations.append(
            {"type": "data_panel", "data": data_for_panel, "position": (25, image_height - 75)}
        )

        return guidance_text

    def _execute_state_machine(
        self, mask, image, frame_visualizations, image_height, image_width, curr_gray
    ):
        """Dispatch guidance to the current navigation state."""
        if self.current_state == STATE_ONBOARDING:
            return self._handle_onboarding(
                mask, image, frame_visualizations, image_height, image_width
            )
        elif self.current_state == STATE_NAVIGATING:
            return self._handle_navigating(
                mask, image, frame_visualizations, image_height, image_width, curr_gray
            )
        elif self.current_state == STATE_MANEUVERING_TURN:
            return self._handle_maneuvering_turn(
                mask, image, frame_visualizations, image_height, image_width
            )
        elif self.current_state == STATE_LOCKING_ON:
            return self._handle_locking_on(frame_visualizations)
        elif self.current_state == STATE_AVOIDING_OBSTACLE:
            return self._handle_avoiding_obstacle(
                mask, image, frame_visualizations, image_height, image_width
            )

        return ""

    def _handle_onboarding(self, mask, image, frame_visualizations, image_height, image_width):
        """Guide the user onto the tactile paving path."""
        image_center_x = image_width / 2
        vp_features = self._get_vanishing_point_features(mask)

        if vp_features and vp_features["fit_error"] < self.VP_FIT_ERROR_THRESHOLD:

            VP, L_center = vp_features["VP"], vp_features["L_center"]

            if self.onboarding_step == ONBOARDING_STEP_ROTATION:
                if abs(VP[0] - image_center_x) < (
                    image_width * self.ONBOARDING_ALIGN_THRESHOLD_RATIO
                ):
                    guidance_text = localized_text("path.heading_aligned")
                    self.onboarding_step = ONBOARDING_STEP_TRANSLATION
                else:
                    guidance_text = (
                        localized_text("path.rotate_left")
                        if VP[0] < image_center_x
                        else localized_text("path.rotate_right")
                    )

                angle_error_px = VP[0] - image_center_x
                self._add_data_panel(
                    frame_visualizations,
                    {
                        localized_text("ui.status"): localized_text("path.align_heading"),
                        localized_text("ui.guidance"): guidance_text,
                        localized_text("ui.angle"): f"{angle_error_px:.1f}px",
                        localized_text("ui.offset"): localized_text("path.calibration_pending"),
                    },
                    (25, image_height - 75),
                )

            elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
                L_center_bottom_x = self._calculate_line_x_at_y(L_center, image_height - 1)

                if L_center_bottom_x:
                    center_offset_pixels = L_center_bottom_x - image_center_x
                    center_offset_ratio = abs(center_offset_pixels) / image_width

                    if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                        guidance_text = localized_text("path.onboarding_complete")
                        self.current_state = STATE_NAVIGATING
                    else:
                        guidance_text = (
                            localized_text("path.sidestep_left")
                            if L_center_bottom_x < image_center_x
                            else localized_text("path.sidestep_right")
                        )

                    self._add_data_panel(
                        frame_visualizations,
                        {
                            localized_text("ui.status"): localized_text("path.align_position"),
                            localized_text("ui.guidance"): guidance_text,
                            localized_text("ui.angle"): localized_text("path.aligned"),
                            localized_text("ui.offset"): f"{center_offset_ratio * 100:.1f}%",
                        },
                        (25, image_height - 75),
                    )
                else:
                    guidance_text = localized_text("path.move_forward_for_visibility")
        else:

            pixel_features = self._get_pixel_domain_features(mask, image.shape)
            if not pixel_features:
                return ""
            self._add_navigation_info_visualization(
                pixel_features, image_height, image_width, frame_visualizations
            )
            guidance_text = self._handle_pixel_domain_onboarding(
                pixel_features, image_height, image_width, frame_visualizations
            )

        return guidance_text

    def _handle_navigating(
        self, mask, image, frame_visualizations, image_height, image_width, curr_gray
    ):
        """Follow the path and check turns and obstacles."""
        image_center_x = image_width / 2

        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return localized_text("path.features_unavailable")
        self._add_navigation_info_visualization(
            features, image_height, image_width, frame_visualizations
        )

        if self.turn_cooldown_frames == 0:
            corner_info = self._detect_sharp_corner(features["centerline_data"])
            if corner_info:
                self._update_turn_tracker(corner_info)

                if self.turn_detection_tracker["consecutive_hits"] >= 3:
                    stable_corner_info = self.turn_detection_tracker["corner_info"]
                    corner_y = stable_corner_info["corner_point_pixel"][1]
                    turn_trigger_y_threshold = image_height * 0.65

                    if corner_y > turn_trigger_y_threshold:

                        direction_text = (
                            localized_text("direction.right_label")
                            if self.turn_detection_tracker["direction"] == "right"
                            else localized_text("direction.left_label")
                        )
                        self.current_state = STATE_MANEUVERING_TURN
                        self.maneuver_target_info = stable_corner_info
                        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                        self._reset_turn_tracker()
                        # Defer turn speech to the maneuvering state.
                        return ""
                    else:

                        pass

        # Prioritize obstacle warnings over path direction.
        obstacles = self._check_obstacles(image, mask, frame_visualizations)
        if obstacles:

            main_obstacle = obstacles[0]
            obstacle_name = main_obstacle.get("name", "")
            current_time = time.time()

            should_announce = False
            if obstacle_name != self.last_obstacle_speech:

                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:

                should_announce = True
                self.last_obstacle_speech_time = current_time

            if should_announce:
                # Schedule a warning without entering the full avoidance sequence.

                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)

        else:

            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None

        # Direction changes take priority over straight guidance.
        return self._generate_navigation_guidance(
            features, image_height, image_width, frame_visualizations
        )

    def _handle_maneuvering_turn(
        self, mask, image, frame_visualizations, image_height, image_width
    ):
        """Guide the user through a turn."""
        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return localized_text("path.lost_searching")
        self._add_navigation_info_visualization(
            features, image_height, image_width, frame_visualizations
        )
        if self.maneuver_step == MANEUVER_STEP_1_ISSUE_COMMAND:
            direction_text = (
                localized_text("direction.right_label")
                if self.maneuver_target_info["direction"] == "right"
                else localized_text("direction.left_label")
            )
            guidance_text = localized_text("path.sidestep_direction").format(
                direction_text=direction_text
            )

            poly_func = features["poly_func"]
            y_check = image_height * 0.7
            self.maneuver_target_info["old_path_center_x"] = poly_func(y_check)

            self.maneuver_step = MANEUVER_STEP_2_WAIT_FOR_SHIFT

            self._add_data_panel(
                frame_visualizations,
                {
                    localized_text("ui.status"): localized_text("path.turning"),
                    localized_text("ui.guidance"): guidance_text,
                    localized_text("ui.step"): localized_text("path.issue_instruction"),
                    localized_text("ui.direction"): direction_text,
                },
                (25, image_height - 75),
            )

            return guidance_text

        elif self.maneuver_step == MANEUVER_STEP_2_WAIT_FOR_SHIFT:
            old_path_x = self.maneuver_target_info.get("old_path_center_x")
            if old_path_x is None:
                self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                return ""

            poly_func = features["poly_func"]
            y_check = image_height * 0.7
            current_path_x = poly_func(y_check)
            shift_distance = abs(current_path_x - old_path_x)

            centerline_data = features["centerline_data"]
            width_at_check_y = self._get_width_at_y(centerline_data, y_check)

            if shift_distance > (width_at_check_y * 0.5):
                guidance_text = localized_text("path.shift_detected")
                self.maneuver_step = MANEUVER_STEP_3_ALIGN_ON_NEW_PATH
            else:
                direction_text = (
                    localized_text("direction.right_label")
                    if self.maneuver_target_info["direction"] == "right"
                    else localized_text("direction.left_label")
                )
                guidance_text = localized_text("path.continue_sidestepping").format(
                    direction_text=direction_text
                )

            self._add_data_panel(
                frame_visualizations,
                {
                    localized_text("ui.status"): localized_text("path.turning"),
                    localized_text("ui.guidance"): guidance_text,
                    localized_text("ui.step"): localized_text("path.waiting_for_shift"),
                    localized_text("ui.displacement"): f"{shift_distance:.1f}px",
                },
                (25, image_height - 75),
            )

            return guidance_text

        elif self.maneuver_step == MANEUVER_STEP_3_ALIGN_ON_NEW_PATH:
            poly_func = features["poly_func"]
            y_check = image_height * 0.5
            current_path_x_at_center = poly_func(y_check)

            pixel_error = current_path_x_at_center - image_width / 2
            center_offset_ratio = abs(pixel_error) / image_width

            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = localized_text("path.new_path_aligned")
                self.current_state = STATE_NAVIGATING
                self.maneuver_target_info = None
                self.turn_cooldown_frames = self.TURN_COOLDOWN_DURATION
            else:
                move_direction = (
                    localized_text("direction.right_label")
                    if pixel_error > 0
                    else localized_text("direction.left_label")
                )
                guidance_text = localized_text("path.adjust_to_paving").format(
                    move_direction=move_direction
                )

            self._add_data_panel(
                frame_visualizations,
                {
                    localized_text("ui.status"): localized_text("path.turning"),
                    localized_text("ui.guidance"): guidance_text,
                    localized_text("ui.step"): localized_text("path.align_new_path"),
                    localized_text("ui.error"): f"{center_offset_ratio * 100:.1f}%",
                },
                (25, image_height - 75),
            )

            return guidance_text

    def _handle_locking_on(self, frame_visualizations):
        """Confirm path alignment after a turn."""
        if not self.lock_on_data:
            self.current_state = STATE_NAVIGATING
            return ""

        main_obstacle = self.lock_on_data["main_obstacle"]

        self._add_obstacle_visualization(main_obstacle, frame_visualizations, pulse_effect=True)

        if time.time() - self.lock_on_data["start_time"] > 0.7:
            self.avoidance_plan = self.lock_on_data["avoidance_plan"]
            self.avoidance_step_index = 0
            self.current_state = STATE_AVOIDING_OBSTACLE
            self.lock_on_data = None

        return ""

    def _handle_avoiding_obstacle(
        self, mask, image, frame_visualizations, image_height, image_width
    ):
        """Advance the obstacle avoidance state."""
        if not self.avoidance_plan or self.avoidance_step_index >= len(self.avoidance_plan):
            self.current_state = STATE_NAVIGATING
            self.avoidance_plan = None
            return localized_text("path.avoidance_complete")

        step = self.avoidance_plan[self.avoidance_step_index]

        if step["type"] == "sidestep_clear":
            direction = step["direction"]

            if self.obstacle_detector:
                final_obstacles = self._detect_obstacles(image, mask)
            else:
                final_obstacles = []

            if final_obstacles:
                guidance_text = localized_text("path.blocked_sidestep").format(
                    value_1=(
                        localized_text("direction.right_label")
                        if direction == "right"
                        else localized_text("direction.left_label")
                    )
                )
            else:
                guidance_text = localized_text("path.stop_sidestepping")
                self.avoidance_step_index += 1

            self._add_data_panel(
                frame_visualizations,
                {
                    localized_text("ui.status"): localized_text("path.avoiding_obstacle"),
                    localized_text("ui.guidance"): guidance_text,
                    localized_text("ui.step"): localized_text("path.sidestep_out"),
                    localized_text("ui.direction"): direction,
                },
                (25, image_height - 75),
            )

            return guidance_text

        elif step["type"] == "forward_pass":

            self.avoidance_step_index += 1
            return localized_text("path.pass_obstacle")

        elif step["type"] == "sidestep_return":
            direction = step["direction"]
            features = self._get_pixel_domain_features(mask, image.shape)

            if not features:
                return localized_text("path.paving_missing_sidestep").format(
                    value_1=(
                        localized_text("direction.right_label")
                        if direction == "right"
                        else localized_text("direction.left_label")
                    )
                )

            poly_func = features["poly_func"]
            y_target = image_height * 0.5
            x_target = poly_func(y_target)

            center_offset_pixels = x_target - image_width / 2
            center_offset_ratio = abs(center_offset_pixels) / image_width

            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = localized_text("path.back_on_paving")
                self.avoidance_step_index += 1
            else:
                guidance_text = (
                    localized_text("path.align_paving_right")
                    if center_offset_pixels > 0
                    else localized_text("path.align_paving_left")
                )

            self._add_data_panel(
                frame_visualizations,
                {
                    localized_text("ui.status"): localized_text("path.avoiding_obstacle"),
                    localized_text("ui.guidance"): guidance_text,
                    localized_text("ui.step"): localized_text("path.return_to_paving"),
                    localized_text("ui.offset"): f"{center_offset_ratio * 100:.1f}%",
                },
                (25, image_height - 75),
            )

            return guidance_text

    def _get_vanishing_point_features(self, mask):
        """Extract path geometry using a vanishing point."""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < 5000:
                return None

            rect = cv2.minAreaRect(main_contour)
            center, _, angle = rect
            angle_rad = np.deg2rad(angle)
            R = np.array(
                [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]]
            )
            points_transformed = np.dot(main_contour.squeeze(1) - center, R)
            left_points = main_contour.squeeze(1)[points_transformed[:, 0] < 0]
            right_points = main_contour.squeeze(1)[points_transformed[:, 0] >= 0]

            if len(left_points) < 20 or len(right_points) < 20:
                return None

            [vx_l, vy_l, x_l, y_l] = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)
            [vx_r, vy_r, x_r, y_r] = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)

            a1, b1, c1 = vy_l, -vx_l, vx_l * y_l - vy_l * x_l
            a2, b2, c2 = vy_r, -vx_r, vx_r * y_r - vy_r * x_r
            determinant = a1 * b2 - a2 * b1

            if abs(determinant) < 1e-6:
                return None

            vp_x = (b1 * c2 - b2 * c1) / determinant
            vp_y = (a2 * c1 - a1 * c2) / determinant
            L_center = ((vx_l + vx_r) / 2, (vy_l + vy_r) / 2, (x_l + x_r) / 2, (y_l + y_r) / 2)

            total_dist = 0
            for pt in left_points:
                total_dist += abs((pt[0] - x_l) * vy_l - (pt[1] - y_l) * vx_l)
            for pt in right_points:
                total_dist += abs((pt[0] - x_r) * vx_r - (pt[1] - y_r) * vy_r)
            fit_error = total_dist / (len(left_points) + len(right_points))

            return {"VP": (vp_x, vp_y), "L_center": L_center, "fit_error": fit_error}
        except:
            return None

    def _get_pixel_domain_features(self, mask, image_shape):
        """Extract a smoothed centerline in image coordinates."""
        try:
            height, width = image_shape[:2]

            centerline_data = []
            for y in range(height - 1, int(height * 0.3), -5):
                row = mask[y, :]
                x_pixels = np.where(row > 0)[0]
                if x_pixels.size > 10:
                    x_min, x_max = x_pixels[0], x_pixels[-1]
                    path_width = x_max - x_min
                    center_x = (x_min + x_max) / 2
                    centerline_data.append([y, center_x, path_width])

            if len(centerline_data) < 20:
                return None

            data = np.array(centerline_data)

            data = self._smooth_centerline(data)

            sharp_turn_index = self._find_sharp_turn(data)
            if sharp_turn_index is not None:
                cutoff_index = int(sharp_turn_index * 0.6)
                if cutoff_index >= 10:
                    data = data[:cutoff_index]

            y_coords, x_coords, widths = data[:, 0], data[:, 1], data[:, 2]
            weights = widths

            coeffs_raw = np.polyfit(y_coords, x_coords, 2, w=weights)

            self.poly_coeffs_history.append(coeffs_raw.copy())
            if len(self.poly_coeffs_history) > self.poly_coeffs_history_max:
                self.poly_coeffs_history.pop(0)

            # Smooth polynomial coefficients with recent frames weighted more heavily.
            if len(self.poly_coeffs_history) >= 3:

                weights_time = np.array(
                    [
                        0.7 ** (len(self.poly_coeffs_history) - i - 1)
                        for i in range(len(self.poly_coeffs_history))
                    ]
                )
                weights_time = weights_time / np.sum(weights_time)

                coeffs = np.zeros_like(coeffs_raw)
                for i, hist_coeffs in enumerate(self.poly_coeffs_history):
                    coeffs += hist_coeffs * weights_time[i]
            else:
                coeffs = coeffs_raw

            poly_func = np.poly1d(coeffs)

            curvature_proxy = abs(coeffs[0])
            tangent_slope = 2 * coeffs[0] * height + coeffs[1]
            tangent_angle_rad = np.arctan(tangent_slope)

            return {
                "poly_func": poly_func,
                "curvature_proxy": curvature_proxy,
                "tangent_angle_rad": tangent_angle_rad,
                "centerline_data": np.array(centerline_data),
            }
        except Exception as e:
            logger.warning(f"Pixel domain feature calculation failed: {e}")
            return None

    def _find_sharp_turn(self, data):
        """Locate a sharp turn in the path centerline."""
        window_size = 5
        angle_threshold = 30

        for i in range(len(data) - 2 * window_size):
            front_window = data[i : i + window_size]
            back_window = data[i + window_size : i + 2 * window_size]

            front_dir = [
                front_window[-1, 1] - front_window[0, 1],
                front_window[-1, 0] - front_window[0, 0],
            ]
            back_dir = [
                back_window[-1, 1] - back_window[0, 1],
                back_window[-1, 0] - back_window[0, 0],
            ]

            angle1 = np.arctan2(front_dir[1], front_dir[0])
            angle2 = np.arctan2(back_dir[1], back_dir[0])
            angle_diff = abs(np.degrees(angle2 - angle1))

            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            if angle_diff > angle_threshold:
                return i + window_size

        return None

    def _detect_sharp_corner(self, centerline_data, angle_threshold_deg=45):
        """Detect a sharp corner from path geometry."""
        try:
            if len(centerline_data) < 15:
                return None
            points_in_range = np.array(centerline_data)
            num_points = len(points_in_range)

            window_size = max(5, int(num_points * 0.15))
            best_turn_info = None
            max_angle_diff = 0

            for i in range(0, num_points - 2 * window_size, 2):
                front_segment = points_in_range[i : i + window_size]
                back_segment = points_in_range[i + window_size : i + 2 * window_size]

                if len(front_segment) < 3 or len(back_segment) < 3:
                    continue

                front_y = front_segment[:, 0]
                front_x = front_segment[:, 1]
                front_coeffs = np.polyfit(front_y, front_x, 1)
                front_slope = front_coeffs[0]

                back_y = back_segment[:, 0]
                back_x = back_segment[:, 1]
                back_coeffs = np.polyfit(back_y, back_x, 1)
                back_slope = back_coeffs[0]

                front_angle = np.arctan(front_slope)
                back_angle = np.arctan(back_slope)

                angle_diff_rad = back_angle - front_angle
                angle_diff_deg = abs(np.degrees(angle_diff_rad))

                if angle_diff_deg > max_angle_diff and angle_diff_deg > angle_threshold_deg:
                    max_angle_diff = angle_diff_deg
                    corner_point_idx = i + window_size
                    corner_point = points_in_range[corner_point_idx]

                    direction = "right" if angle_diff_rad > 0 else "left"

                    post_turn_segment = points_in_range[
                        corner_point_idx : min(corner_point_idx + window_size * 2, num_points)
                    ]
                    if len(post_turn_segment) > 0:
                        post_turn_center_x = np.mean(post_turn_segment[:, 1])
                    else:
                        post_turn_center_x = corner_point[1]

                    best_turn_info = {
                        "corner_point_pixel": (corner_point[1], corner_point[0]),
                        "turn_angle": max_angle_diff,
                        "direction": direction,
                        "post_turn_center_x": post_turn_center_x,
                        "corner_point_idx": corner_point_idx,
                    }

            return best_turn_info

        except Exception as e:
            logger.warning(f"Corner detection error: {e}")
            return None

    def _update_turn_tracker(self, corner_info):
        """Update turn direction and confidence over time."""
        detected_direction = corner_info["direction"]

        if detected_direction == self.turn_detection_tracker["direction"]:
            self.turn_detection_tracker["consecutive_hits"] += 1
        else:
            self.turn_detection_tracker["direction"] = detected_direction
            self.turn_detection_tracker["consecutive_hits"] = 1

        self.turn_detection_tracker["last_seen_frame"] = self.frame_counter
        self.turn_detection_tracker["corner_info"] = corner_info

    def _reset_turn_tracker(self):
        """Clear the tracked turn."""
        self.turn_detection_tracker = {
            "direction": None,
            "consecutive_hits": 0,
            "last_seen_frame": 0,
            "corner_info": None,
        }

    def _calculate_line_x_at_y(self, line_params, y_target):
        """Return the horizontal coordinate of a line at the requested image row."""
        vx, vy, x0, y0 = line_params
        if abs(vy) < 1e-6:
            return None
        t = (y_target - y0) / vy
        x = x0 + t * vx
        return x

    def _get_width_at_y(self, centerline_data, y_target):
        """Return the path width at the requested image row."""
        ys = centerline_data[:, 0]
        ws = centerline_data[:, 2]
        idx = np.abs(ys - y_target).argmin()
        return ws[idx]

    def _detect_obstacles(self, image, path_mask=None):
        """Detect obstacles and populate the fields required by navigation."""
        logger.info(
            f"[_detect_obstacles] Starting frame={self.frame_counter}, obstacle_detector={('loaded' if self.obstacle_detector else 'unavailable')}"
        )

        if self.obstacle_detector is None:
            logger.warning("[_detect_obstacles] Obstacle detector unavailable")
            return []

        if not hasattr(self, "_classes_printed"):
            self._classes_printed = True
            if hasattr(self.obstacle_detector, "WHITELIST_CLASSES"):
                logger.info("[_detect_obstacles] Allowed classes")
                for idx, name in enumerate(self.obstacle_detector.WHITELIST_CLASSES):
                    logger.info(f"  Class {idx}: {name}")
                logger.info(
                    f"[_detect_obstacles] Total: {len(self.obstacle_detector.WHITELIST_CLASSES)} classes"
                )

        try:
            # The obstacle client handles text prompts and detector filtering.

            logger.info(
                f"[_detect_obstacles] Calling ObstacleDetectorClient.detect(); image.shape={image.shape}"
            )
            detected_obstacles = self.obstacle_detector.detect(image, path_mask=path_mask)

            logger.info(
                f"[_detect_obstacles] ObstacleDetectorClient returned {len(detected_obstacles)} objects"
            )

            # Populate optional geometry required by downstream rendering.
            H, W = image.shape[:2]
            for i, obj in enumerate(detected_obstacles):

                if "mask" in obj and obj["mask"] is not None:
                    y_coords, x_coords = np.where(obj["mask"] > 0)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                        obj["box_coords"] = (x1, y1, x2, y2)

                        if "y_position_ratio" not in obj:
                            obj["y_position_ratio"] = obj.get("center_y", 0) / H
                        if "label" not in obj:
                            obj["label"] = obj.get("name", "unknown")
                        if "center" not in obj:
                            obj["center"] = (obj.get("center_x", 0), obj.get("center_y", 0))
                        # Supply a fallback confidence when the detector omits it.
                        if "confidence" not in obj:
                            obj["confidence"] = 0.5

                logger.info(f"[_detect_obstacles] Object {i + 1}/{len(detected_obstacles)}: ")
                logger.info(f"  Class: {obj.get('name', 'unknown')}")
                logger.info(
                    f"  Area: {obj.get('area', 0)} pixels ({obj.get('area_ratio', 0):.3f} of image)"
                )
                logger.info(
                    f"  Center: ({obj.get('center_x', 0):.1f}, {obj.get('center_y', 0):.1f})"
                )
                logger.info(
                    f"  - bottom_y_ratio: {obj.get('bottom_y_ratio', 0):.3f} (nearby: {('yes' if obj.get('bottom_y_ratio', 0) > 0.7 else 'no')})"
                )
                logger.info(
                    f"  - area_ratio: {obj.get('area_ratio', 0):.3f} (oversized: {('yes' if obj.get('area_ratio', 0) > 0.1 else 'no')})"
                )

            # The detector applies size, confidence and path-overlap filters.

            logger.info(f"[_detect_obstacles] Final result: {len(detected_obstacles)} obstacles")
            for idx, obj in enumerate(detected_obstacles):
                logger.info(
                    f"  {idx + 1}. {obj.get('name', 'unknown')} position:({obj.get('center_x', 0):.0f},{obj.get('center_y', 0):.0f}) bottom_y_ratio:{obj.get('bottom_y_ratio', 0):.2f} area_ratio:{obj.get('area_ratio', 0):.3f}"
                )

            return detected_obstacles

        except Exception as e:
            logger.error(f"[_detect_obstacles] Detection failed: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _check_and_set_obstacle_voice(self, obstacles):
        """Schedule a nearby obstacle warning subject to its repeat interval."""
        if not obstacles:
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None
            return

        # A nearby obstacle must satisfy both vertical and area thresholds.
        NEAR_DISTANCE_Y_THRESHOLD = 0.75
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12

        near_obstacles = []
        for obs in obstacles:
            if (
                obs.get("bottom_y_ratio", 0) > NEAR_DISTANCE_Y_THRESHOLD
                or obs.get("area_ratio", 0) > NEAR_DISTANCE_AREA_THRESHOLD
            ):
                near_obstacles.append(obs)

        if near_obstacles:
            # Use the largest qualifying obstacle for the warning.
            main_obstacle = max(near_obstacles, key=lambda x: x.get("area_ratio", 0))
            obstacle_name = main_obstacle.get("name", "")
            current_time = time.time()

            should_announce = False
            if obstacle_name != self.last_obstacle_speech:

                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:

                should_announce = True
                self.last_obstacle_speech_time = current_time

            if should_announce:
                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)
        else:

            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None

    def _check_obstacles(self, image, mask, frame_visualizations):
        """Reuse or refresh obstacle detections and select nearby objects."""

        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            final_obstacles = self._detect_obstacles(image, mask)

            if hasattr(self, "prev_gray") and self.prev_gray is not None:
                curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                final_obstacles = self._stabilize_obstacle_list(
                    final_obstacles,
                    self.last_detected_obstacles,
                    self.prev_gray,
                    curr_gray,
                    image.shape[:2],
                )
            self.last_detected_obstacles = final_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
        else:
            if (
                self.frame_counter - self.last_obstacle_detection_frame
                < self.OBSTACLE_CACHE_DURATION_FRAMES
            ):
                final_obstacles = self.last_detected_obstacles
            else:
                final_obstacles = []

        for obs in final_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)

        NEAR_DISTANCE_Y_THRESHOLD = 0.75
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12

        near_obstacles = [
            obs
            for obs in final_obstacles
            if (
                obs.get("bottom_y_ratio", 0) > NEAR_DISTANCE_Y_THRESHOLD
                or obs.get("area_ratio", 0) > NEAR_DISTANCE_AREA_THRESHOLD
            )
        ]

        return near_obstacles

    def _plan_avoidance(self, obstacle_info, image_width):
        """Choose an avoidance direction from the obstacle and path geometry."""
        obstacle_center_x = obstacle_info["center_x"]
        image_center_x = image_width / 2

        if obstacle_center_x < image_center_x:
            turn_direction = "right"
        else:
            turn_direction = "left"

        plan = [
            {"type": "sidestep_clear", "direction": turn_direction},
            {"type": "forward_pass"},
            {
                "type": "sidestep_return",
                "direction": "left" if turn_direction == "right" else "right",
            },
        ]
        return plan

    def _generate_navigation_guidance(
        self, features, image_height, image_width, frame_visualizations
    ):
        """Choose direction or straight guidance from path features."""
        poly_func = features["poly_func"]
        is_curve = features["curvature_proxy"] > self.CURVATURE_PROXY_THRESHOLD
        lookahead_ratio = 0.6 if is_curve else 0.4
        y_target = image_height * lookahead_ratio
        x_target = poly_func(y_target)

        plot_y = np.arange(int(image_height * 0.3), image_height, 5).astype(int)
        plot_x = poly_func(plot_y).astype(int)
        centerline_points = np.vstack((plot_x, plot_y)).T.tolist()
        frame_visualizations.append(
            {"type": "polyline", "points": centerline_points, "color": "yellow", "width": 2}
        )

        frame_visualizations.append(
            {
                "type": "circle",
                "center": [int(x_target), int(y_target)],
                "radius": 10,
                "color": "red",
            }
        )

        # Prioritize turns, then lateral alignment, then straight guidance.
        center_offset_pixels = x_target - image_width / 2
        center_offset_ratio = abs(center_offset_pixels) / image_width
        orientation_error_rad = features["tangent_angle_rad"]

        if orientation_error_rad > self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = localized_text("path.turn_left")
        elif orientation_error_rad < -self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = localized_text("path.turn_right")

        elif center_offset_ratio > self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
            guidance_text = (
                localized_text("path.move_right")
                if center_offset_pixels > 0
                else localized_text("path.move_left")
            )

        else:
            guidance_text = localized_text("path.keep_straight")

        self._add_data_panel(
            frame_visualizations,
            {
                localized_text("ui.status"): localized_text("path.following"),
                localized_text("ui.guidance"): guidance_text,
                localized_text("ui.heading"): f"{np.degrees(orientation_error_rad):.1f}°",
                localized_text("ui.offset"): f"{center_offset_ratio * 100:.1f}%",
            },
            (25, image_height - 75),
        )

        return guidance_text

    def _handle_pixel_domain_onboarding(
        self, pixel_features, image_height, image_width, frame_visualizations
    ):
        """Guide the user onto the path using image coordinates."""
        image_center_x = image_width / 2
        orientation_error_rad = pixel_features["tangent_angle_rad"]
        poly_func = pixel_features["poly_func"]

        y_bottom = image_height - 1
        x_target_bottom = poly_func(y_bottom)
        center_offset_pixels = x_target_bottom - image_center_x
        center_offset_ratio = abs(center_offset_pixels) / image_width

        if self.onboarding_step == ONBOARDING_STEP_ROTATION:
            if abs(orientation_error_rad) < self.ONBOARDING_ORIENTATION_THRESHOLD_RAD:
                guidance_text = localized_text("path.heading_aligned")
                self.onboarding_step = ONBOARDING_STEP_TRANSLATION
            else:
                guidance_text = (
                    localized_text("path.rotate_left")
                    if orientation_error_rad > 0.1
                    else localized_text("path.rotate_right")
                )

            self._add_data_panel(
                frame_visualizations,
                {
                    localized_text("ui.status"): localized_text("path.align_heading"),
                    localized_text("ui.guidance"): guidance_text,
                    localized_text("ui.angle"): f"{np.degrees(orientation_error_rad):.1f}°",
                    localized_text("ui.offset"): localized_text("path.calibration_pending"),
                },
                (25, image_height - 75),
            )
            self._add_navigation_info_visualization(
                pixel_features, image_height, image_width, frame_visualizations
            )

            return guidance_text

        elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
            if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = localized_text("path.onboarding_complete")
                self.current_state = STATE_NAVIGATING
            else:
                guidance_text = (
                    localized_text("path.sidestep_right")
                    if center_offset_pixels > 0
                    else localized_text("path.sidestep_left")
                )

            self._add_data_panel(
                frame_visualizations,
                {
                    localized_text("ui.status"): localized_text("path.align_position"),
                    localized_text("ui.guidance"): guidance_text,
                    localized_text("ui.angle"): localized_text("path.aligned"),
                    localized_text("ui.offset"): f"{center_offset_ratio * 100:.1f}%",
                },
                (25, image_height - 75),
            )

        return guidance_text

    def _add_obstacle_visualization(self, obstacle, visualizations, pulse_effect=False):
        """Draw red contours for nearby obstacles and yellow contours otherwise."""
        try:

            bottom_y_ratio = obstacle.get("bottom_y_ratio", 0)
            area_ratio = obstacle.get("area_ratio", 0)

            is_near = bottom_y_ratio > 0.7 or area_ratio > 0.1

            if "mask" in obstacle and obstacle["mask"] is not None:
                mask = obstacle["mask"]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:

                    max_contour = max(contours, key=cv2.contourArea)
                    points = max_contour.squeeze(1)[::5].tolist()

                    # Nearby contours are red; other contours are yellow.
                    if is_near:
                        outline_color = "rgba(255, 0, 0, 1.0)"
                        thickness = 3
                    else:
                        outline_color = "rgba(255, 255, 0, 0.8)"
                        thickness = 2

                    visualizations.append(
                        {
                            "type": "outline",
                            "points": points,
                            "color": outline_color,
                            "thickness": thickness,
                        }
                    )
        except Exception as e:
            logger.error(f"[_add_obstacle_visualization] Drawing failed: {e}")

    def _add_navigation_info_visualization(
        self, features, image_height, image_width, frame_visualizations
    ):
        """Draw centerline, tangent, curvature and path width diagnostics."""
        if not features:
            return

        try:

            poly_func = features.get("poly_func")
            curvature_proxy = features.get("curvature_proxy", 0)
            tangent_angle_rad = features.get("tangent_angle_rad", 0)
            tangent_angle_deg = np.degrees(tangent_angle_rad)

            if poly_func:

                y_bottom = image_height - 50
                x_bottom = poly_func(y_bottom)

                tangent_length = 100
                dx = tangent_length * np.cos(tangent_angle_rad)
                dy = tangent_length * np.sin(tangent_angle_rad)

                baseline_length = 80
                frame_visualizations.append(
                    {
                        "type": "dashed_line",
                        "start": [int(x_bottom), int(y_bottom)],
                        "end": [int(x_bottom), int(y_bottom - baseline_length)],
                        "color": "rgba(255, 255, 255, 0.6)",
                        "thickness": 2,
                    }
                )

                frame_visualizations.append(
                    {
                        "type": "arrow",
                        "start": [int(x_bottom), int(y_bottom)],
                        "end": [
                            int(x_bottom + dx),
                            int(y_bottom - dy),
                        ],  # Image coordinates have a downward positive y axis.
                        "color": "rgba(0, 255, 255, 0.8)",
                        "thickness": 3,
                        "tip_length": 0.3,
                    }
                )

                arc_radius = 40
                # The vertical reference points upward at minus 90 degrees.

                start_angle = -90
                end_angle = -90 + tangent_angle_deg
                frame_visualizations.append(
                    {
                        "type": "angle_arc",
                        "center": [int(x_bottom), int(y_bottom)],
                        "radius": arc_radius,
                        "start_angle": start_angle,
                        "end_angle": end_angle,
                        "color": "rgba(255, 200, 0, 0.8)",
                        "thickness": 2,
                    }
                )

                frame_visualizations.append(
                    {
                        "type": "text_with_bg",
                        "text": localized_text("path.angle_degrees").format(
                            tangent_angle_deg=f"{tangent_angle_deg:.1f}"
                        ),
                        "position": [int(x_bottom + 10), int(y_bottom - 30)],
                        "font_scale": 0.3,
                        "color": "rgba(255, 255, 255, 1.0)",
                        "bg_color": "rgba(0, 0, 0, 0.7)",
                    }
                )

            if curvature_proxy > 0.00001:
                curve_text = (
                    localized_text("path.curve")
                    if curvature_proxy > 5e-05
                    else localized_text("path.gentle_curve")
                )
                frame_visualizations.append(
                    {
                        "type": "text_with_bg",
                        "text": f"{curve_text}: {curvature_proxy:.2e}",
                        "position": [20, 100],
                        "font_scale": 0.25,
                        "color": "rgba(255, 255, 0, 1.0)",
                        "bg_color": "rgba(0, 0, 0, 0.7)",
                    }
                )

            if "centerline_data" in features:
                centerline_data = features["centerline_data"]

                mid_idx = len(centerline_data) // 2
                if mid_idx < len(centerline_data):
                    y, x, width = centerline_data[mid_idx]

                    frame_visualizations.append(
                        {
                            "type": "double_arrow",
                            "start": [int(x - width / 2), int(y)],
                            "end": [int(x + width / 2), int(y)],
                            "color": "rgba(0, 255, 0, 0.8)",
                            "thickness": 2,
                            "tip_length": 0.15,
                        }
                    )

                    frame_visualizations.append(
                        {
                            "type": "text_with_bg",
                            "text": localized_text("path.width_pixels").format(
                                width=f"{width:.0f}"
                            ),
                            "position": [int(x - 30), int(y - 10)],
                            "font_scale": 0.25,
                            "color": "rgba(255, 255, 255, 1.0)",
                            "bg_color": "rgba(0, 0, 0, 0.7)",
                        }
                    )
        except Exception as e:
            logger.error(f"Navigation overlay failed: {e}")

    def _add_data_panel(self, visualizations, data, position):
        """Append a diagnostic text panel."""
        visualizations.append({"type": "data_panel", "data": data, "position": position})

    def _add_crosswalk_info_visualization(
        self, viz_data, image_height, image_width, visualizations
    ):
        """Draw crosswalk alignment and proximity indicators."""
        try:

            center_x = int(viz_data["center_x_ratio"] * image_width)
            center_y = int(viz_data["center_y_ratio"] * image_height)

            cross_size = 20 if viz_data["in_arrival"] else 15
            cross_color = (
                "rgba(255, 100, 0, 1.0)" if viz_data["in_arrival"] else "rgba(0, 200, 255, 0.8)"
            )

            visualizations.append(
                {
                    "type": "line",
                    "start": [center_x - cross_size, center_y],
                    "end": [center_x + cross_size, center_y],
                    "color": cross_color,
                    "thickness": 2,
                }
            )

            visualizations.append(
                {
                    "type": "line",
                    "start": [center_x, center_y - cross_size],
                    "end": [center_x, center_y + cross_size],
                    "color": cross_color,
                    "thickness": 2,
                }
            )

            screen_center_x = image_width // 2
            screen_center_y = image_height // 2

            distance = np.sqrt(
                (center_x - screen_center_x) ** 2 + (center_y - screen_center_y) ** 2
            )
            if distance > 80:
                visualizations.append(
                    {
                        "type": "arrow",
                        "start": [screen_center_x, screen_center_y],
                        "end": [center_x, center_y],
                        "color": "rgba(255, 150, 0, 0.6)",
                        "thickness": 2,
                        "tip_length": 0.15,
                    }
                )

            panel_x = image_width - 180
            panel_y = 20

            panel_data = {
                localized_text("ui.crosswalk"): viz_data["stage"],
                localized_text("ui.area"): f"{viz_data['area_ratio']*100:.1f}%",
                localized_text("ui.position"): viz_data["position"],
            }

            if viz_data["has_occlusion"]:
                panel_data[localized_text("ui.status")] = localized_text("crossing.occluded")
            elif viz_data["in_arrival"]:
                panel_data[localized_text("ui.status")] = localized_text("crossing.ready_to_cross")

            visualizations.append(
                {"type": "data_panel", "data": panel_data, "position": (panel_x, panel_y)}
            )

            bar_width = 150
            bar_height = 20
            bar_x = image_width - bar_width - 20
            bar_y = panel_y + 90

            visualizations.append(
                {
                    "type": "rectangle",
                    "top_left": (bar_x, bar_y),
                    "bottom_right": (bar_x + bar_width, bar_y + bar_height),
                    "color": "rgba(50, 50, 50, 0.7)",
                    "filled": True,
                }
            )

            # Scale progress to the arrival area threshold of 0.25.
            progress = min(viz_data["area_ratio"] / 0.25, 1.0)
            fill_width = int(bar_width * progress)

            if viz_data["in_arrival"]:
                fill_color = "rgba(0, 255, 100, 0.8)"
            elif viz_data["area_ratio"] >= 0.18:
                fill_color = "rgba(255, 200, 0, 0.8)"
            elif viz_data["area_ratio"] >= 0.08:
                fill_color = "rgba(0, 200, 255, 0.8)"
            else:
                fill_color = "rgba(100, 150, 255, 0.8)"

            visualizations.append(
                {
                    "type": "rectangle",
                    "top_left": (bar_x + 2, bar_y + 2),
                    "bottom_right": (bar_x + fill_width - 2, bar_y + bar_height - 2),
                    "color": fill_color,
                    "filled": True,
                }
            )

            visualizations.append(
                {
                    "type": "text_with_bg",
                    "text": localized_text("crossing.proximity_percent").format(
                        value_1=f"{int(progress * 100)}"
                    ),
                    "position": [bar_x, bar_y - 18],
                    "font_scale": 0.25,
                    "color": "rgba(255, 255, 255, 1.0)",
                    "bg_color": "rgba(0, 0, 0, 0.7)",
                }
            )

        except Exception as e:
            logger.error(f"Crosswalk overlay failed: {e}")

    def _to_cn_obstacle(self, name: str) -> str:
        """Resolve the localized obstacle category name."""
        try:
            key = (name or "").strip().lower()
            return _OBSTACLE_NAME_CN.get(key, localized_text("object.obstacle"))
        except:
            return localized_text("object.obstacle")

    def _speech_for_obstacle(self, name: str) -> str:
        k = (name or "").strip().lower()
        if k == "person":
            return localized_text("obstacle.person_ahead")
        if k == "car":
            return localized_text("obstacle.car_ahead")
        if k == "bicycle":
            return localized_text("obstacle.bicycle_ahead")
        if k == "motorcycle":
            return localized_text("obstacle.motorcycle_ahead")
        if k == "bus":
            return localized_text("obstacle.bus_ahead")
        if k == "truck":
            return localized_text("obstacle.truck_ahead")
        if k == "scooter":
            return localized_text("obstacle.scooter_ahead")
        if k == "stroller":
            return localized_text("obstacle.stroller_ahead")
        if k == "dog":
            return localized_text("obstacle.dog_ahead")
        if k == "animal":
            return localized_text("obstacle.animal_ahead")
        return localized_text("obstacle.ahead")

    def _draw_command_button(self, image, text):
        """Draw the current guidance text at the bottom center."""
        try:
            H, W = image.shape[:2]
            full_text = localized_text("ui.current_instruction").format(
                value_1=text if text else localized_text("ui.unavailable")
            )

            font_px = 14
            pad_x, pad_y = 14, 8
            bottom_margin = 28

            if PIL_AVAILABLE:
                try:
                    from PIL import Image as PILImage, ImageDraw, ImageFont

                    font = None
                    for font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
                        if os.path.exists(font_path):
                            try:
                                font = ImageFont.truetype(font_path, font_px)
                                break
                            except:
                                continue
                    if font:
                        bbox = ImageDraw.Draw(PILImage.new("RGB", (1, 1))).textbbox(
                            (0, 0), full_text, font=font
                        )
                        tw = max(1, bbox[2] - bbox[0])
                        th = max(1, bbox[3] - bbox[1])
                    else:
                        scale = font_px / 24.0
                        (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                except:
                    scale = font_px / 24.0
                    (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            else:
                scale = font_px / 24.0
                (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

            bw = tw + pad_x * 2
            bh = th + pad_y * 2
            radius = max(10, bh // 2)

            cx = W // 2
            left = max(8, cx - bw // 2)
            top = H - bottom_margin - bh
            right = min(W - 8, left + bw)
            bottom = top + bh

            overlay = image.copy()
            bg_color = (26, 32, 41)
            border_color = (60, 76, 102)

            cv2.rectangle(overlay, (left + radius, top), (right - radius, bottom), bg_color, -1)
            cv2.circle(overlay, (left + radius, (top + bottom) // 2), radius, bg_color, -1)
            cv2.circle(overlay, (right - radius, (top + bottom) // 2), radius, bg_color, -1)

            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

            cv2.rectangle(image, (left + radius, top), (right - radius, bottom), border_color, 1)
            cv2.circle(image, (left + radius, (top + bottom) // 2), radius, border_color, 1)
            cv2.circle(image, (right - radius, (top + bottom) // 2), radius, border_color, 1)

            text_x = left + pad_x
            text_y = top + pad_y + th

            if PIL_AVAILABLE and "font" in locals() and font:

                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((text_x, top + pad_y), full_text, font=font, fill=(255, 255, 255))
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:

                cv2.putText(
                    image,
                    full_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (255, 255, 255),
                    1,
                )

            return image
        except Exception as e:
            logger.error(f"Command button rendering failed: {e}")
            return image

    def _parse_color(self, color_str):
        """Parse a color string into OpenCV BGR components."""
        try:
            if color_str.startswith("rgba("):
                values = color_str[5:-1].split(",")
                r, g, b = int(values[0]), int(values[1]), int(values[2])
                return (b, g, r)  # OpenCV expects BGR components.
            elif color_str == "yellow":
                return (0, 255, 255)
            elif color_str == "red":
                return (0, 0, 255)
            else:
                return (0, 0, 255)
        except:
            return (0, 0, 255)

    def _draw_data_panel_no_bg(self, image, data, position=(15, 15)):
        """Draw an outlined text panel without a background."""
        if not PIL_AVAILABLE:
            return image

        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")

            env_scale = float(os.getenv("NAVGUIDE_PANEL_SCALE", "0.7"))
            base_font_size = max(10, int(round(14 * env_scale)))

            # Try installed Chinese fonts in priority order.
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]

            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                except:
                    continue

            if font is None:
                font = ImageFont.load_default()

            y_offset = position[1]
            for key, value in data.items():
                text = f"{key}: {value}"

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text(
                                (position[0] + dx, y_offset + dy),
                                text,
                                font=font,
                                fill=(0, 0, 0, 255),
                            )

                draw.text((position[0], y_offset), text, font=font, fill=(255, 255, 255, 255))
                y_offset += base_font_size + 5

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"Data panel rendering failed: {e}")
            return image

    def _draw_visualizations(self, image, viz_elements):
        """Render translucent masks, contours, guidance and diagnostic overlays."""
        if not viz_elements:
            return image

        current_time = time.time()

        panel_elements = [v for v in viz_elements if v.get("type") == "data_panel"]
        standard_elements = [v for v in viz_elements if v.get("type") != "data_panel"]

        # Render translucent fills before contours and labels.
        for element in standard_elements:
            elem_type = element.get("type")

            if elem_type in ["blind_path_mask", "obstacle_mask", "crosswalk_mask"]:
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.5)"))

                    if element.get("effect") == "pulse":
                        pulse_speed = element.get("pulse_speed", 1.0)
                        alpha = 0.3 + 0.3 * np.sin(current_time * pulse_speed * 2 * np.pi)
                    else:
                        alpha = 0.4

                    x, y, w, h = cv2.boundingRect(points)

                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)

                    if w > 0 and h > 0:

                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        local_points = points - np.array([x, y])
                        cv2.fillPoly(binary_mask, [local_points], 255)

                        # Blend only inside the mask to preserve the surrounding image.
                        local_region = image[y : y + h, x : x + w].copy()

                        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                        color_overlay[:] = color

                        for c in range(3):
                            local_region[:, :, c] = np.where(
                                binary_mask > 0,
                                (1 - alpha) * local_region[:, :, c]
                                + alpha * color_overlay[:, :, c],
                                local_region[:, :, c],
                            )

                        image[y : y + h, x : x + w] = local_region

        # Render outlines and other elements over the fills.
        for element in standard_elements:
            elem_type = element.get("type")

            if elem_type == "line":
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                cv2.line(image, start, end, color, thickness)

            elif elem_type == "outline":
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                    thickness = element.get("thickness", 3)
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

            elif elem_type == "polyline":
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 0, 1.0)"))
                    thickness = element.get("width", 2)
                    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)

            elif elem_type == "circle":
                center = tuple(element.get("center", (0, 0)))
                radius = element.get("radius", 10)
                color = self._parse_color(element.get("color", "rgba(255, 0, 0, 1.0)"))
                thickness = element.get("thickness", -1 if element.get("filled", True) else 2)
                cv2.circle(image, center, radius, color, thickness)

            elif elem_type == "rectangle":
                top_left = tuple(element.get("top_left", (0, 0)))
                bottom_right = tuple(element.get("bottom_right", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 0, 0, 0.5)"))
                thickness = -1 if element.get("filled", True) else 2
                cv2.rectangle(image, top_left, bottom_right, color, thickness)

            elif elem_type == "arrow":
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.3)
                cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip_length)

            elif elem_type == "double_arrow":
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.15)

                cv2.line(image, start, end, color, thickness)

                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx * dx + dy * dy)
                if length > 0:

                    ux, uy = dx / length, dy / length

                    arrow_len = length * tip_length

                    tip1_x = int(start[0] + arrow_len * ux)
                    tip1_y = int(start[1] + arrow_len * uy)

                    angle = np.arctan2(dy, dx)
                    arrow_angle = 30 * np.pi / 180
                    p1 = (
                        int(start[0] + arrow_len * np.cos(angle - arrow_angle)),
                        int(start[1] + arrow_len * np.sin(angle - arrow_angle)),
                    )
                    p2 = (
                        int(start[0] + arrow_len * np.cos(angle + arrow_angle)),
                        int(start[1] + arrow_len * np.sin(angle + arrow_angle)),
                    )
                    cv2.line(image, start, p1, color, thickness)
                    cv2.line(image, start, p2, color, thickness)

                    p3 = (
                        int(end[0] - arrow_len * np.cos(angle - arrow_angle)),
                        int(end[1] - arrow_len * np.sin(angle - arrow_angle)),
                    )
                    p4 = (
                        int(end[0] - arrow_len * np.cos(angle + arrow_angle)),
                        int(end[1] - arrow_len * np.sin(angle + arrow_angle)),
                    )
                    cv2.line(image, end, p3, color, thickness)
                    cv2.line(image, end, p4, color, thickness)

            elif elem_type == "dashed_line":
                start = np.array(element.get("start", (0, 0)))
                end = np.array(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.6)"))
                thickness = element.get("thickness", 2)
                dash_length = 10
                gap_length = 5

                total_vec = end - start
                total_len = np.linalg.norm(total_vec)
                if total_len > 0:
                    unit_vec = total_vec / total_len

                    current_len = 0
                    while current_len < total_len:
                        seg_start = start + unit_vec * current_len
                        seg_end = start + unit_vec * min(current_len + dash_length, total_len)
                        cv2.line(
                            image,
                            tuple(seg_start.astype(int)),
                            tuple(seg_end.astype(int)),
                            color,
                            thickness,
                        )
                        current_len += dash_length + gap_length

            elif elem_type == "angle_arc":
                center = tuple(element.get("center", (100, 100)))
                radius = element.get("radius", 40)
                start_angle = element.get("start_angle", -90)
                end_angle = element.get("end_angle", 0)
                color = self._parse_color(element.get("color", "rgba(255, 200, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                # OpenCV ellipse angles are clockwise from the positive x axis.

                # Convert mathematical angles to the OpenCV convention.
                cv2_start = -end_angle
                cv2_end = -start_angle

                if cv2_start > cv2_end:
                    cv2_start, cv2_end = cv2_end, cv2_start
                cv2.ellipse(
                    image, center, (radius, radius), 0, cv2_start, cv2_end, color, thickness
                )

            elif elem_type == "text_with_bg":
                text = element.get("text", "")
                pos = element.get("position", [10, 30])
                font_scale = element.get("font_scale", 0.6)
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))

                image = self._draw_chinese_text(
                    image,
                    text,
                    tuple(pos),
                    font_scale=font_scale,
                    color=color,
                    stroke_color=(0, 0, 0),
                    stroke_width=1,
                )

            elif elem_type == "warning_icon":
                pos = element.get("position", (100, 100))
                level = element.get("level", "info")
                text = element.get("text", "")
                flash = element.get("flash", False)

                if level == "danger":
                    icon_color = (0, 0, 255)
                    text_color = (255, 255, 255)
                elif level == "warning":
                    icon_color = (0, 165, 255)
                    text_color = (255, 255, 255)
                else:
                    icon_color = (0, 255, 255)
                    text_color = (0, 0, 0)

                if flash:
                    alpha = 0.5 + 0.5 * np.sin(current_time * 4 * np.pi)
                    icon_color = tuple(int(c * alpha) for c in icon_color)

                triangle = np.array(
                    [[pos[0], pos[1] - 20], [pos[0] - 15, pos[1]], [pos[0] + 15, pos[1]]], np.int32
                )
                cv2.fillPoly(image, [triangle], icon_color)
                cv2.polylines(image, [triangle], True, (255, 255, 255), 2)

                cv2.putText(
                    image,
                    "!",
                    (pos[0] - 5, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if text:
                    font_scale = 0.5

                    text_pos = (pos[0] - 20, pos[1] + 20)
                    image = self._draw_chinese_text(
                        image,
                        text,
                        text_pos,
                        font_scale=font_scale,
                        color=text_color,
                        stroke_color=(0, 0, 0),
                        stroke_width=1,
                    )

            elif elem_type == "text":
                text = element.get("text", "")
                pos = tuple(element.get("pos", (10, 30)))

                image = self._draw_chinese_text(
                    image,
                    text,
                    pos,
                    font_scale=0.7,
                    color=(255, 255, 255),
                    stroke_color=(0, 0, 0),
                    stroke_width=1,
                )

        if PIL_AVAILABLE:
            for panel in panel_elements:
                image = self._draw_data_panel_no_bg(image, panel["data"], panel["position"])
        else:

            for panel in panel_elements:
                y_offset = panel["position"][1]
                for key, value in panel["data"].items():
                    text = f"{key}: {value}"

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(
                                    image,
                                    text,
                                    (panel["position"][0] + dx, y_offset + dy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 0, 0),
                                    3,
                                )

                    cv2.putText(
                        image,
                        text,
                        (panel["position"][0], y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
                    y_offset += 25

        return image

    def _draw_chinese_text(
        self,
        image,
        text,
        position,
        font_scale=0.6,
        color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=1,
    ):
        """Draw Chinese text with an available font and a contrasting outline."""
        if not PIL_AVAILABLE:
            # OpenCV fallback may not support Chinese glyphs.
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            return image

        try:

            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            base_size = 24
            font_size = int(base_size * font_scale / 0.6)

            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/msyh.ttf",
                "C:/Windows/Fonts/simhei.ttf",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]

            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue

            if font is None:
                font = ImageFont.load_default()

            # Convert BGR to RGB for Pillow.
            rgb_color = (color[2], color[1], color[0])
            rgb_stroke = (stroke_color[2], stroke_color[1], stroke_color[0])

            x, y = position

            draw.text(
                (x, y),
                text,
                font=font,
                fill=rgb_stroke,
                stroke_width=stroke_width,
                stroke_fill=rgb_stroke,
            )

            draw.text((x, y), text, font=font, fill=rgb_color)

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"Localized text rendering failed: {e}")

            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            return image

    def _draw_data_panel(self, image, data, position=(15, 15)):
        """Draw a text panel with Pillow."""
        if not PIL_AVAILABLE:
            return image

        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")

            env_scale = float(os.getenv("NAVGUIDE_PANEL_SCALE", "0.65"))
            base_font_size = max(8, int(round(16 * env_scale)))
            padding = max(4, int(round(8 * env_scale)))

            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/msyh.ttf",
                "C:/Windows/Fonts/simhei.ttf",
            ]

            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                    except:
                        continue

            if font is None:
                font = ImageFont.load_default()

            text_lines = [f"{key}: {value}" for key, value in data.items()]
            text_to_draw = "\n".join(text_lines)

            bbox = draw.textbbox(position, text_to_draw, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            bg_rect = [
                (position[0] - padding, position[1] - padding),
                (position[0] + text_w + padding, position[1] + text_h + padding),
            ]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 128))
            draw.text(position, text_to_draw, font=font, fill=(255, 255, 255, 255))

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception:
            return image

    def reset(self):
        """Reset navigation, tracking and speech state."""
        self.current_state = STATE_ONBOARDING
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None
        self.turn_detection_tracker = {
            "direction": None,
            "consecutive_hits": 0,
            "last_seen_frame": 0,
            "corner_info": None,
        }
        self.turn_cooldown_frames = 0
        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None

        self.flow_points = {}
        self.flow_grace = {}
        self.centerline_history = []
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0

        self.pending_obstacle_voice = None
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0

        self.poly_coeffs_history = []
        self.crosswalk_tracker = {
            "stage": "not_detected",
            "consecutive_frames": 0,
            "last_area_ratio": 0.0,
            "last_bottom_y_ratio": 0.0,
            "last_center_x_ratio": 0.5,
            "position_announced": False,
            "alignment_status": "not_aligned",
            "last_seen_frame": 0,
            "last_angle": 0.0,
        }
        self.frame_counter = 0
        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.last_any_speech_time = 0
        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0
        self.pending_crosswalk_voice = None
        self.last_blindpath_mask = None
        self.last_crosswalk_mask = None
        self.last_blindpath_detection_frame = 0
        self.last_guide_time = 0
        self.last_direction_time = 0
        self.last_direction_message = ""
        self.straight_repeat_count = 0
        self.crosswalk_monitor.reset()

    def _stabilize_obstacle_list(
        self, obstacles, prev_obstacles, prev_gray, curr_gray, image_shape, threshold=0.5
    ):
        """Associate obstacles across frames and smooth their masks."""
        if not obstacles or prev_gray is None or curr_gray is None:
            return obstacles

        H, W = image_shape
        stabilized = []
        used_prev = set()

        for curr_obs in obstacles:
            if "mask" not in curr_obs or curr_obs["mask"] is None:
                stabilized.append(curr_obs)
                continue

            curr_mask = curr_obs["mask"]
            best_match = None
            best_iou = 0
            best_idx = -1

            if prev_obstacles:
                for idx, prev_obs in enumerate(prev_obstacles):
                    if idx in used_prev or "mask" not in prev_obs:
                        continue

                    flow_mask = self._predict_mask_with_flow(prev_obs["mask"], prev_gray, curr_gray)
                    if flow_mask is None:
                        flow_mask = prev_obs["mask"]

                    inter = np.logical_and(curr_mask > 0, flow_mask > 0).sum()
                    union = np.logical_or(curr_mask > 0, flow_mask > 0).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0

                    if iou > best_iou and iou > threshold:
                        best_iou = iou
                        best_match = flow_mask
                        best_idx = idx

            if best_match is not None and best_idx >= 0:
                used_prev.add(best_idx)
                # Blend matched detections with optical flow predictions.
                fused_mask = ((0.8 * curr_mask + 0.2 * best_match) > 128).astype(np.uint8) * 255
                curr_obs["mask"] = fused_mask

                self._update_obstacle_properties(curr_obs, H, W)

            stabilized.append(curr_obs)

        return stabilized

    def _speech_for_obstacle(self, name: str) -> str:
        k = (name or "").strip().lower()
        if k == "person":
            return localized_text("obstacle.person_ahead")
        if k == "car":
            return localized_text("obstacle.car_ahead")
        if k == "bicycle":
            return localized_text("obstacle.bicycle_ahead")
        if k == "motorcycle":
            return localized_text("obstacle.motorcycle_ahead")
        if k == "bus":
            return localized_text("obstacle.bus_ahead")
        if k == "truck":
            return localized_text("obstacle.truck_ahead")
        if k == "scooter":
            return localized_text("obstacle.scooter_ahead")
        if k == "stroller":
            return localized_text("obstacle.stroller_ahead")
        if k == "dog":
            return localized_text("obstacle.dog_ahead")
        if k == "animal":
            return localized_text("obstacle.animal_ahead")
        return localized_text("obstacle.ahead")

    def _update_obstacle_properties(self, obs, H, W):
        """Update obstacle geometry derived from its mask."""
        if "mask" not in obs or obs["mask"] is None:
            return

        mask = obs["mask"]
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) > 0:
            obs["area"] = len(y_coords)
            obs["center_x"] = float(np.mean(x_coords))
            obs["center_y"] = float(np.mean(y_coords))
            obs["y_position_ratio"] = obs["center_y"] / H
            obs["area_ratio"] = obs["area"] / (H * W)
            obs["bottom_y_ratio"] = np.max(y_coords) / H

            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            obs["box_coords"] = (x1, y1, x2, y2)
