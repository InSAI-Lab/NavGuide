"""Semantic candidate selection and repetition control, NavGuide Algorithm 1.

Deduplicate same-class boxes, rank confidence times five relevance factors,
select Top K, build cues and suppress repeated signatures.
"""

from __future__ import annotations
from navguide.i18n import text as localized_text

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from navguide.core.context import GuidanceContext, SceneType, TaskMode
from navguide.core.proximity import ProximityBin, ProximityEstimator

# Moving and hazardous categories.
DYNAMIC_HAZARD_CLASSES = {
    "car",
    "automobile",
    "suv",
    "van",
    "bus",
    "truck",
    "bicycle",
    "bike",
    "motorcycle",
    "motorbike",
    "e-bike",
    "scooter",
}

STATIC_OBSTACLE_CLASSES = {
    "pole",
    "bollard",
    "traffic cone",
    "trash can",
    "curb",
    "stairs",
    "step",
    "fire hydrant",
    "barrier",
    "construction",
    "obstacle",
    "blind_path_obstacle",
}

NAVIGATION_PATH_CLASSES = {
    "crosswalk",
    "zebra_crossing",
    "blindpath",
    "blind_path",
    "traffic light",
    "traffic_light",
    "walkable_area",
}

# Spoken labels for common detection categories.
CLASS_ZH_MAP: Dict[str, str] = {
    "person": localized_text("object.person"),
    "pedestrian": localized_text("object.person"),
    "car": localized_text("object.car"),
    "automobile": localized_text("object.car"),
    "bus": localized_text("object.bus"),
    "truck": localized_text("object.truck"),
    "bicycle": localized_text("object.bicycle"),
    "bike": localized_text("object.bicycle"),
    "motorcycle": localized_text("object.electric_bike"),
    "motorbike": localized_text("object.motorcycle"),
    "scooter": localized_text("object.scooter"),
    "traffic light": localized_text("object.traffic_light"),
    "traffic_light": localized_text("object.traffic_light"),
    "crosswalk": localized_text("object.crosswalk"),
    "blindpath": localized_text("object.blindpath"),
    "blind_path": localized_text("object.blindpath"),
    "pole": localized_text("object.pole"),
    "bollard": localized_text("object.bollard"),
    "traffic cone": localized_text("object.traffic_cone"),
    "trash can": localized_text("object.trash_can"),
    "curb": localized_text("object.curb"),
    "stairs": localized_text("object.stairs"),
    "step": localized_text("object.stairs"),
    "chair": localized_text("object.chair"),
    "bench": localized_text("object.bench"),
    "table": localized_text("object.table"),
    "door": localized_text("object.door"),
    "bottle": localized_text("object.bottle"),
    "cup": localized_text("object.cup"),
    "backpack": localized_text("object.backpack"),
    "umbrella": localized_text("object.umbrella"),
    "keys": localized_text("object.keys"),
    "obstacle": localized_text("object.obstacle"),
}


@dataclass
class DetectionCandidate:
    """Visual detection with optional tracking, hazard and signal metadata."""

    category: str  # c_i: category/class name
    confidence: float  # p_i: detection confidence in [0, 1]
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    track_id: Optional[Union[int, str]] = None
    is_hazard: bool = False  # Whether the detection represents a hazard.
    is_moving: bool = False  # Whether the object is moving.
    urgency_override: Optional[bool] = None  # Optional urgency override.
    raw_data: Optional[Dict[str, Any]] = None
    signal_color: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("category must be a nonempty string")
        self.category = self.category.strip().lower()
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and within [0, 1]")
        if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
            raise ValueError("bbox must contain four finite coordinates")
        self.bbox = tuple(float(v) for v in self.bbox)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bbox must have positive width and height")
        for flag in (self.is_hazard, self.is_moving, self.urgency_override):
            if flag is not None and not isinstance(flag, bool):
                raise ValueError("hazard, motion and urgency flags must be boolean")
        if self.track_id is not None and (
            isinstance(self.track_id, bool) or not isinstance(self.track_id, (str, int))
        ):
            raise ValueError("track_id must be an integer, string or null")
        if self.signal_color is None and self.raw_data:
            self.signal_color = self.raw_data.get("color")
        if self.signal_color is not None:
            if not isinstance(self.signal_color, str):
                raise ValueError("signal_color must be a string or null")
            self.signal_color = self.signal_color.strip().lower()

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


def compute_iou(
    box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
) -> float:
    """Compute Intersection-over-Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def same_class_dedup(
    detections: List[DetectionCandidate], iou_threshold: float = 0.6
) -> List[DetectionCandidate]:
    """Suppress lower-confidence same-class boxes whose IoU exceeds the threshold."""
    if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be within [0, 1]")
    if not detections:
        return []

    by_category: Dict[str, List[DetectionCandidate]] = {}
    for d in detections:
        cat = d.category.strip().lower()
        by_category.setdefault(cat, []).append(d)

    deduped: List[DetectionCandidate] = []

    for cat, group in by_category.items():
        group.sort(key=lambda x: x.confidence, reverse=True)
        kept_group: List[DetectionCandidate] = []

        for candidate in group:
            suppressed = False
            for kept in kept_group:
                if compute_iou(candidate.bbox, kept.bbox) > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept_group.append(candidate)

        deduped.extend(kept_group)

    return deduped


@dataclass
class ActionCue:
    """Selected object with direction, distance, action and repetition signature."""

    category: str  # Object category (English)
    category_zh: str  # Object category (Chinese)
    confidence: float
    bbox: Tuple[float, float, float, float]
    distance_m: float
    proximity_bin: ProximityBin
    clock_hour: int  # 1 to 12 clock direction
    clock_direction_zh: str  # Localized clock direction.
    clock_direction_en: str  # e.g., "12 o'clock", "1 o'clock"
    action_type: str  # "avoid", "stop", "continue", "search", "caution"
    action_zh: str  # Localized action label.
    action_en: str
    urgency_flag: bool = False  # u_c: urgency indicator
    requested_target_flag: bool = False  # q_c: eligible requested target indicator
    relevance_score: float = 0.0  # S_i priority score
    relevance_factors: Dict[str, float] = field(default_factory=dict)
    source_candidate: Optional[DetectionCandidate] = field(default=None, repr=False, compare=False)
    semantic_signature: Tuple[str, int, str] = field(init=False)

    def __post_init__(self):
        # Semantic signature combining dominant object category, clock direction, and proximity bin
        semantic_category = self.category.strip().lower()
        if self.source_candidate and semantic_category in {"traffic light", "traffic_light"}:
            color = self.source_candidate.signal_color
            if color in {"red", "green", "yellow"}:
                semantic_category = f"traffic light:{color}"
        self.semantic_signature = (semantic_category, self.clock_hour, self.proximity_bin.value)


class SemanticMaximizationPolicy:
    """Rank candidates by confidence times task, scene, user, proximity and hazard weights."""

    def __init__(
        self,
        proximity_estimator: Optional[ProximityEstimator] = None,
        top_k: int = 3,
        dedup_iou_threshold: float = 0.6,
        repetition_window_sec: float = 3.0,
    ):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not math.isfinite(repetition_window_sec) or repetition_window_sec < 0:
            raise ValueError("repetition_window_sec must be finite and nonnegative")
        if not math.isfinite(dedup_iou_threshold) or not 0 <= dedup_iou_threshold <= 1:
            raise ValueError("dedup_iou_threshold must be within [0, 1]")
        self.prox_est = proximity_estimator or ProximityEstimator()
        self.top_k = top_k
        self.dedup_iou_threshold = dedup_iou_threshold
        self.repetition_window_sec = repetition_window_sec
        # Cue history H_t: maps signature to last eligible-content timestamp.
        self.cue_history: Dict[Tuple[str, int, str], float] = {}
        self.last_selected_cues: List[ActionCue] = []
        self.last_deduplicated_count = 0
        self.last_repetition_suppressed_count = 0
        self._last_timestamp: Optional[float] = None

    def compute_task_weight(self, candidate: DetectionCandidate, context: GuidanceContext) -> float:
        """Factor 1: Task relevance W_task,i(z_t)."""
        cat = candidate.category.strip().lower()

        if context.task_mode == TaskMode.TARGET_SEARCH:
            # If matches requested search query, give decisive boost
            if context.matches_target(cat):
                return 4.5
            # Immediate hazards retain high relevance for safety
            if cat in DYNAMIC_HAZARD_CLASSES or candidate.is_hazard:
                return 1.8
            # Non-target items suppressed
            return 0.35

        elif context.task_mode == TaskMode.PATH_NAVIGATION:
            # Navigation essentials
            if cat in NAVIGATION_PATH_CLASSES or "blind" in cat or "crosswalk" in cat:
                return 2.4
            if cat in DYNAMIC_HAZARD_CLASSES:
                return 2.0
            if cat in STATIC_OBSTACLE_CLASSES or candidate.is_hazard:
                return 1.8
            if cat in {"person", "pedestrian"}:
                return 1.3
            return 0.5

        elif context.task_mode == TaskMode.SCENE_EXPLORATION:
            # Exploration balances landmarks, environment, hazards
            if cat in {"door", "stairs", "bench", "traffic light", "bus"}:
                return 1.6
            if cat in DYNAMIC_HAZARD_CLASSES:
                return 1.5
            if cat in {"person", "chair", "table", "crosswalk"}:
                return 1.2
            return 0.8

        return 1.0

    def compute_scene_weight(
        self, candidate: DetectionCandidate, context: GuidanceContext
    ) -> float:
        """Factor 2: Scene relevance W_scene,i(z_t)."""
        cat = candidate.category.strip().lower()
        scene = context.scene_type

        if scene in (SceneType.CROSSWALK, SceneType.INTERSECTION):
            if cat in DYNAMIC_HAZARD_CLASSES or cat in {
                "traffic light",
                "traffic_light",
                "crosswalk",
            }:
                return 2.0
            if cat in {"person", "curb"}:
                return 1.4
            return 0.6

        elif scene == SceneType.SIDEWALK:
            if cat in STATIC_OBSTACLE_CLASSES or "blind" in cat:
                return 1.8
            if cat in {"bicycle", "bike", "scooter", "motorcycle"}:
                return 1.7
            if cat in {"person", "dog"}:
                return 1.3
            return 0.8

        elif scene == SceneType.INDOOR:
            if cat in {"door", "stairs", "step", "chair", "table", "obstacle"}:
                return 1.8
            if cat in DYNAMIC_HAZARD_CLASSES:
                return 0.4
            return 1.0

        elif scene == SceneType.TRANSIT:
            if cat in {"bus", "train", "person", "stairs", "door"}:
                return 1.8
            return 0.8

        return 1.0

    def compute_user_weight(self, candidate: DetectionCandidate, context: GuidanceContext) -> float:
        """Factor 3: User preference relevance W_user,i(z_t)."""
        cat = candidate.category.strip().lower()
        weight = float(context.user_weights.get(cat, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("user weights must be finite and nonnegative")
        return weight

    def compute_proximity_weight(self, prox_bin: ProximityBin) -> float:
        """Factor 4: Coarse proximity relevance W_proximity,i(z_t)."""
        return self.prox_est.get_proximity_weight(prox_bin)

    def compute_hazard_weight(
        self, candidate: DetectionCandidate, context: GuidanceContext, distance_m: float
    ) -> float:
        """Factor 5: Moving-hazard salience W_hazard,i(z_t)."""
        cat = candidate.category.strip().lower()
        is_dynamic = cat in DYNAMIC_HAZARD_CLASSES or candidate.is_moving
        is_close = distance_m < 3.0

        if is_dynamic:
            # Fast/approaching vehicle in proximity has highest hazard salience
            if is_close:
                return 2.5 * context.hazard_sensitivity
            return 1.8 * context.hazard_sensitivity
        elif candidate.is_hazard or cat in STATIC_OBSTACLE_CLASSES:
            if distance_m < 2.0:
                return 1.8 * context.hazard_sensitivity
            return 1.3 * context.hazard_sensitivity

        return 1.0

    def compute_clock_direction(self, center_x: float, frame_width: int) -> Tuple[int, str, str]:
        """
        Convert horizontal box center to clock direction relative to user's heading.
        12 o'clock is directly forward.
        """
        if not math.isfinite(frame_width) or frame_width <= 0 or not math.isfinite(center_x):
            raise ValueError("clock direction requires finite coordinates and positive width")
        norm_x = (center_x - frame_width / 2.0) / (frame_width / 2.0)  # [-1.0, 1.0]

        # Horizontal FOV span approx -45 deg to +45 deg mapped to clock hours
        if abs(norm_x) < 0.18:
            hour = 12
            zh = localized_text("clock.twelve")
            en = "12 o'clock"
        elif 0.18 <= norm_x < 0.55:
            hour = 1
            zh = localized_text("clock.one")
            en = "1 o'clock"
        elif 0.55 <= norm_x < 0.85:
            hour = 2
            zh = localized_text("clock.two")
            en = "2 o'clock"
        elif norm_x >= 0.85:
            hour = 3
            zh = localized_text("clock.three")
            en = "3 o'clock"
        elif -0.55 < norm_x <= -0.18:
            hour = 11
            zh = localized_text("clock.eleven")
            en = "11 o'clock"
        elif -0.85 < norm_x <= -0.55:
            hour = 10
            zh = localized_text("clock.ten")
            en = "10 o'clock"
        else:
            hour = 9
            zh = localized_text("clock.nine")
            en = "9 o'clock"

        return hour, zh, en

    def determine_action(
        self,
        candidate: DetectionCandidate,
        distance_m: float,
        clock_hour: int,
        context: GuidanceContext,
    ) -> Tuple[str, str, str, bool]:
        """
        Determine action-first recommendation and urgency flag u_c.
        Returns: (action_type, action_zh, action_en, urgency_flag)
        """
        cat = candidate.category.strip().lower()
        is_dynamic = cat in DYNAMIC_HAZARD_CLASSES or candidate.is_moving
        is_critical = distance_m < 1.8

        if candidate.urgency_override is not None:
            urgency = candidate.urgency_override
        else:
            urgency = (is_dynamic and distance_m < 3.0) or (candidate.is_hazard and is_critical)

        # Traffic light signals
        if cat in {"red light", "red_light", "red traffic light", "traffic_light_red"} or (
            cat in {"traffic light", "traffic_light"} and candidate.signal_color == "red"
        ):
            return (
                "stop",
                localized_text("action.stop"),
                "Stop and wait",
                True if candidate.urgency_override is None else urgency,
            )
        if cat in {"green light", "green_light", "green traffic light", "traffic_light_green"} or (
            cat in {"traffic light", "traffic_light"} and candidate.signal_color == "green"
        ):
            return "caution", localized_text("action.check_crossing"), "Check the crossing", urgency

        if context.is_target_search() and context.matches_target(cat):
            return "found", localized_text("action.target_found"), "Target found", urgency

        # Collision avoidance actions
        if is_dynamic and is_critical:
            return "avoid", localized_text("action.avoid"), "Caution, yield", urgency
        elif is_critical:
            if clock_hour in (11, 12):
                return "avoid", localized_text("action.avoid"), "Avoid obstacle", urgency
            elif clock_hour in (1, 2):
                return "avoid", localized_text("action.avoid"), "Avoid obstacle", urgency
            else:
                return "avoid", localized_text("action.avoid"), "Avoid obstacle", urgency
        elif distance_m < 3.5 and (candidate.is_hazard or is_dynamic):
            return "caution", localized_text("action.slow_down"), "Slow down", urgency

        # General path walking
        if cat in {"crosswalk", "zebra_crossing", "blindpath", "blind_path", "walkable_area"}:
            return "continue", localized_text("action.walk_straight"), "Walk straight", False

        return "notice", localized_text("action.notice"), "Notice", urgency

    def build_cue(
        self,
        candidate: DetectionCandidate,
        context: GuidanceContext,
        frame_width: int,
        frame_height: int,
        score: float = 0.0,
        factors: Optional[Dict[str, float]] = None,
    ) -> ActionCue:
        """Build flags independently from relevance scoring."""
        distance = self.prox_est.estimate_distance_m(
            candidate.bbox, candidate.category, frame_width, frame_height
        )
        proximity = self.prox_est.categorize_proximity(distance)
        hour, clock_zh, clock_en = self.compute_clock_direction(candidate.center_x, frame_width)
        action, action_zh, action_en, urgent = self.determine_action(
            candidate, distance, hour, context
        )
        return ActionCue(
            category=candidate.category,
            category_zh=CLASS_ZH_MAP.get(candidate.category, candidate.category),
            confidence=candidate.confidence,
            bbox=candidate.bbox,
            distance_m=distance,
            proximity_bin=proximity,
            clock_hour=hour,
            clock_direction_zh=clock_zh,
            clock_direction_en=clock_en,
            action_type=action,
            action_zh=action_zh,
            action_en=action_en,
            urgency_flag=urgent,
            requested_target_flag=context.matches_target(candidate.category),
            relevance_score=score,
            relevance_factors=dict(factors or {}),
            source_candidate=candidate,
        )

    def select_cues(
        self,
        detections: List[DetectionCandidate],
        context: GuidanceContext,
        frame_width: int,
        frame_height: int,
    ) -> List[ActionCue]:
        """Deduplicate, rank and select cues before repetition and motion gating.

        Relevance weights use configurable defaults.
        """
        if not math.isfinite(context.hazard_sensitivity) or context.hazard_sensitivity < 0:
            raise ValueError("hazard_sensitivity must be finite and nonnegative")
        deduped = same_class_dedup(detections, self.dedup_iou_threshold)
        self.last_deduplicated_count = len(deduped)
        scored = []
        for candidate in deduped:
            distance = self.prox_est.estimate_distance_m(
                candidate.bbox, candidate.category, frame_width, frame_height
            )
            factors = {
                "task": self.compute_task_weight(candidate, context),
                "scene": self.compute_scene_weight(candidate, context),
                "user": self.compute_user_weight(candidate, context),
                "proximity": self.compute_proximity_weight(
                    self.prox_est.categorize_proximity(distance)
                ),
                "hazard": self.compute_hazard_weight(candidate, context, distance),
            }
            score = candidate.confidence * math.prod(factors.values())
            if not math.isfinite(score):
                raise ValueError("relevance score overflowed; reduce context weights")
            scored.append((score, candidate, factors))
        # Python's stable sort keeps input order for equal scores.
        scored.sort(key=lambda row: row[0], reverse=True)
        self.last_selected_cues = [
            self.build_cue(candidate, context, frame_width, frame_height, score, factors)
            for score, candidate, factors in scored[: self.top_k]
        ]
        return list(self.last_selected_cues)

    def filter_repetition(
        self, cues: List[ActionCue], now: Optional[float] = None
    ) -> List[ActionCue]:
        """Suppress repeated signatures for the shared time window.

        Urgent and target cues use the same window. History contains eligible content,
        not deferred observations or completed speech.
        """
        current = self._validate_timestamp(now)
        self._prune_history(current)
        filtered = []
        seen = set()
        for cue in cues:
            signature = cue.semantic_signature
            previous = self.cue_history.get(signature)
            if signature not in seen and (
                previous is None or current - previous >= self.repetition_window_sec
            ):
                filtered.append(cue)
                seen.add(signature)
        self.last_repetition_suppressed_count = len(cues) - len(filtered)
        return filtered

    def mark_eligible(self, cues: List[ActionCue], now: Optional[float] = None):
        """Commit only cues released by the gate. No deferred replay queue exists."""
        current = self._validate_timestamp(now)
        for cue in cues:
            self.cue_history[cue.semantic_signature] = current
        self._prune_history(current)

    def _validate_timestamp(self, now: Optional[float]) -> float:
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and current < self._last_timestamp:
            raise ValueError("timestamps must be nondecreasing; reset before a new replay")
        self._last_timestamp = current
        return current

    def select_and_build_cues(
        self,
        detections: List[DetectionCandidate],
        context: GuidanceContext,
        frame_width: int,
        frame_height: int,
        now: Optional[float] = None,
        record_history: bool = True,
    ) -> List[ActionCue]:
        """Convenience API for selection and repetition control.

        A standalone caller implicitly releases returned cues. A caller with its
        own gate sets record_history=False and calls mark_eligible afterwards.
        """
        current = self._validate_timestamp(now)
        cues = self.select_cues(detections, context, frame_width, frame_height)
        filtered = self.filter_repetition(cues, current)
        if record_history:
            self.mark_eligible(filtered, current)
        return filtered

    def reset_history(self):
        self.cue_history.clear()
        self._last_timestamp = None

    def _prune_history(self, current_ts: float):
        expired = [
            signature
            for signature, timestamp in self.cue_history.items()
            if current_ts - timestamp >= self.repetition_window_sec
        ]
        for signature in expired:
            del self.cue_history[signature]
