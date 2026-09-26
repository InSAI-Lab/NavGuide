"""NavGuide selection, scheduling and measurement stages.

P implements Algorithm 1, T emits track events and B forwards valid detections.
Remote descriptions use a separate service client.
"""

from __future__ import annotations
from navguide.i18n import text as localized_text

import copy
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from navguide.core.phrasing import ActionFirstPhraser
from navguide.core.context import ContextSmoother, GuidanceContext, SceneType, TaskMode
from navguide.core.gating import InertialMotionGate
from navguide.core.proximity import ProximityEstimator
from navguide.core.selection import (
    DYNAMIC_HAZARD_CLASSES,
    STATIC_OBSTACLE_CLASSES,
    ActionCue,
    DetectionCandidate,
    SemanticMaximizationPolicy,
    compute_iou,
    same_class_dedup,
)


class EvaluationCondition(str, Enum):
    P = "P"
    T = "T"
    B = "B"
    REMOTE = "REMOTE"


@dataclass
class NavGuideResult:
    eligible_cues: List[ActionCue]
    deferred_cues: List[ActionCue]
    guidance_phrases: List[str]
    speech_text: str
    should_speak: bool
    raw_item_count: int
    retained_item_count: int
    critical_detected_count: int
    critical_retained_count: int
    item_reduction: Optional[float]
    conditional_retention: Optional[float]
    capture_to_trigger_latency_ms: Optional[float]
    current_yaw_rate: float
    is_motion_gated: bool
    context: GuidanceContext
    selected_cues: List[ActionCue] = field(default_factory=list)
    deduplicated_item_count: int = 0
    repetition_suppressed_count: int = 0
    processing_latency_ms: float = 0.0
    capture_to_output_ready_latency_ms: Optional[float] = None


class NavGuidePipeline:
    """Algorithm 1 with separate selection, repetition and motion stages."""

    def __init__(
        self,
        condition: EvaluationCondition = EvaluationCondition.P,
        top_k: int = 3,
        yaw_rate_threshold_dps: float = 25.0,
        repetition_window_sec: float = 3.0,
        lang: str = "zh",
        use_steps: bool = False,
        proximity_estimator: Optional[ProximityEstimator] = None,
        track_expiry_sec: float = 3.0,
    ):
        self.condition = EvaluationCondition(condition)
        if self.condition == EvaluationCondition.REMOTE:
            raise ValueError(
                "REMOTE requires the cloud client and is not a local selection condition"
            )
        if not math.isfinite(track_expiry_sec) or track_expiry_sec <= 0:
            raise ValueError("track_expiry_sec must be positive and finite")
        self.context = GuidanceContext()
        self.smoother = ContextSmoother(window_size=8, switch_threshold=4)
        self.prox_est = proximity_estimator or ProximityEstimator(step_length_m=0.6)
        self.smp = SemanticMaximizationPolicy(self.prox_est, top_k, 0.6, repetition_window_sec)
        self.motion_gate = InertialMotionGate(yaw_rate_threshold_dps)
        self.phraser = ActionFirstPhraser(default_lang=lang, use_steps=use_steps)
        self.latest_yaw_rate_dps = 0.0
        self.track_expiry_sec = track_expiry_sec
        self._tracks: Dict[Any, Tuple[ActionCue, float]] = {}
        self._next_anonymous_track = 0
        self._last_timestamp: Optional[float] = None
        self.total_frames = 0
        self.total_raw_items = 0
        self.total_retained_items = 0
        self.total_crit_detected = 0
        self.total_crit_retained = 0
        self.total_eligible_items = 0
        self.total_repetition_suppressed = 0
        self.latencies_ms: List[float] = []
        self.processing_latencies_ms: List[float] = []

    def update_imu_motion(self, yaw_rate_dps: float):
        value = float(yaw_rate_dps)
        if not math.isfinite(value):
            raise ValueError("yaw_rate_dps must be finite")
        self.latest_yaw_rate_dps = value

    def set_task_mode(self, mode: TaskMode, target_query: Optional[str] = None):
        mode = TaskMode(mode)
        if target_query is not None and not isinstance(target_query, str):
            raise ValueError("target_query must be a string or null")
        if (mode, target_query) != (self.context.task_mode, self.context.target_query):
            # A new task starts a new content history but does not reset metrics.
            self.smp.reset_history()
        self.context.task_mode = mode
        self.context.target_query = target_query
        self.context.timestamp = time.time()

    def set_user_weights(self, weights: Dict[str, float]):
        self.context.user_weights.update(GuidanceContext.validate_user_weights(weights))

    def set_scene_type(self, raw_scene: SceneType) -> SceneType:
        self.context.scene_type = self.smoother.update(raw_scene)
        return self.context.scene_type

    def is_safety_critical(self, candidate: DetectionCandidate) -> bool:
        """Classify detections using category, signal and explicit hazard flags."""
        category = candidate.category
        return bool(
            candidate.is_hazard
            or candidate.urgency_override
            or category in DYNAMIC_HAZARD_CLASSES
            or category in STATIC_OBSTACLE_CLASSES
            or category in {"red light", "red_light", "red traffic light", "traffic_light_red"}
            or (category in {"traffic light", "traffic_light"} and candidate.signal_color == "red")
        )

    def _tracking_events(self, cues: List[ActionCue], now: float) -> List[ActionCue]:
        """Emit new or returning tracks and direction or proximity changes.

        Prefer frontend track IDs; otherwise use greedy same-class IoU association.
        """
        self._tracks = {
            key: value
            for key, value in self._tracks.items()
            if now - value[1] < self.track_expiry_sec
        }
        used = set()
        emitted = []
        for cue in cues:
            candidate = cue.source_candidate
            track_id = candidate.track_id if candidate else None
            if track_id is not None:
                key = ("frontend", cue.category, track_id)
            else:
                matches = [
                    (compute_iou(cue.bbox, previous.bbox), key)
                    for key, (previous, _) in self._tracks.items()
                    if key[0] == "anonymous"
                    and key not in used
                    and previous.category == cue.category
                ]
                best = max(matches, key=lambda pair: pair[0]) if matches else None
                if best is not None and best[0] >= 0.3:
                    key = best[1]
                else:
                    key = ("anonymous", self._next_anonymous_track)
                    self._next_anonymous_track += 1
            previous = self._tracks.get(key)
            if key not in used and (
                previous is None or previous[0].semantic_signature != cue.semantic_signature
            ):
                emitted.append(cue)
            self._tracks[key] = (cue, now)
            used.add(key)
        return emitted

    def process(
        self,
        detections: List[DetectionCandidate],
        yaw_rate_dps: Optional[float] = None,
        capture_timestamp: Optional[float] = None,
        raw_scene: Optional[SceneType] = None,
        frame_width: int = 640,
        frame_height: int = 480,
        now: Optional[float] = None,
    ) -> NavGuideResult:
        """Produce eligible content. This does not start or complete audio.

        ``now`` is an optional nondecreasing replay/monotonic clock in seconds.
        ``capture_timestamp`` is an optional wall-clock timestamp used solely for
        capture-to-output-ready timing, never for temporal content control.
        """
        start = time.perf_counter()
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current) or (
            self._last_timestamp is not None and current < self._last_timestamp
        ):
            raise ValueError("now must be finite and nondecreasing")
        if any(not math.isfinite(v) or v <= 0 for v in (frame_width, frame_height)):
            raise ValueError("frame dimensions must be positive and finite")
        if capture_timestamp is not None and not math.isfinite(capture_timestamp):
            raise ValueError("capture_timestamp must be finite")
        condition = EvaluationCondition(self.condition)
        if condition == EvaluationCondition.REMOTE:
            raise ValueError("REMOTE is evaluated by the cloud client, not the local pipeline")
        if yaw_rate_dps is not None:
            self.update_imu_motion(yaw_rate_dps)
        if raw_scene is not None:
            self.set_scene_type(raw_scene)
        self._last_timestamp = current
        omega = self.latest_yaw_rate_dps
        critical = [candidate for candidate in detections if self.is_safety_critical(candidate)]
        deferred: List[ActionCue] = []
        suppressed = 0
        if condition == EvaluationCondition.P:
            selected = self.smp.select_cues(detections, self.context, frame_width, frame_height)
            deduplicated_count = self.smp.last_deduplicated_count
            candidates = self.smp.filter_repetition(selected, current)
            suppressed = self.smp.last_repetition_suppressed_count
            eligible, deferred = self.motion_gate.filter_eligible_cues(candidates, omega)
            self.smp.mark_eligible(eligible, current)
        else:
            retained = (
                detections if condition == EvaluationCondition.B else same_class_dedup(detections)
            )
            deduplicated_count = len(retained)
            selected = [
                self.smp.build_cue(candidate, self.context, frame_width, frame_height)
                for candidate in retained
            ]
            # Baseline wording describes detections and does not prescribe SMP actions.
            for cue in selected:
                cue.action_type, cue.action_zh, cue.action_en = (
                    "notice",
                    localized_text("action.detected"),
                    "Detected",
                )
                cue.urgency_flag = cue.requested_target_flag = False
                cue.relevance_score = 0.0
            eligible = (
                selected
                if condition == EvaluationCondition.B
                else self._tracking_events(selected, current)
            )
            suppressed = len(selected) - len(eligible)
        phrases = self.phraser.generate_batch_phrases(eligible)
        # Keep every eligible item in the returned speech string, including all Top-3.
        speech = self.phraser.combine_phrases(eligible, max_phrases=len(eligible))
        critical_ids = {id(candidate) for candidate in critical}
        retained_critical = sum(id(cue.source_candidate) in critical_ids for cue in selected)
        n_raw, n_keep = len(detections), len(selected)
        processing_ms = (time.perf_counter() - start) * 1000.0
        capture_ready_ms = (
            None
            if capture_timestamp is None
            else max(0.0, (time.time() - capture_timestamp) * 1000.0)
        )
        self.total_frames += 1
        self.total_raw_items += n_raw
        self.total_retained_items += n_keep
        self.total_crit_detected += len(critical)
        self.total_crit_retained += retained_critical
        self.total_eligible_items += len(eligible)
        self.total_repetition_suppressed += suppressed
        self.processing_latencies_ms.append(processing_ms)
        return NavGuideResult(
            eligible_cues=eligible,
            deferred_cues=deferred,
            guidance_phrases=phrases,
            speech_text=speech,
            should_speak=bool(phrases),
            raw_item_count=n_raw,
            retained_item_count=n_keep,
            critical_detected_count=len(critical),
            critical_retained_count=retained_critical,
            item_reduction=1 - n_keep / n_raw if n_raw else None,
            conditional_retention=retained_critical / len(critical) if critical else None,
            capture_to_trigger_latency_ms=None,
            current_yaw_rate=omega,
            is_motion_gated=bool(deferred),
            context=copy.deepcopy(self.context),
            selected_cues=list(selected),
            deduplicated_item_count=deduplicated_count,
            repetition_suppressed_count=suppressed,
            processing_latency_ms=processing_ms,
            capture_to_output_ready_latency_ms=capture_ready_ms,
        )

    def record_audio_trigger(
        self,
        capture_timestamp: float,
        trigger_timestamp: Optional[float] = None,
        result: Optional[NavGuideResult] = None,
    ) -> float:
        """Record an actual audio trigger, using timestamps in the same clock domain."""
        trigger = time.time() if trigger_timestamp is None else float(trigger_timestamp)
        if (
            not math.isfinite(capture_timestamp)
            or not math.isfinite(trigger)
            or trigger < capture_timestamp
        ):
            raise ValueError("audio timestamps must be finite and trigger must follow capture")
        latency = (trigger - capture_timestamp) * 1000.0
        self.latencies_ms.append(latency)
        if result is not None:
            result.capture_to_trigger_latency_ms = latency
        return latency

    @staticmethod
    def _latency_summary(values: List[float]) -> Dict[str, Any]:
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "max": max(values) if values else None,
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Report counts, denominators and measurement stages."""
        return {
            "condition": EvaluationCondition(self.condition).value,
            "frames": self.total_frames,
            "total_raw_items": self.total_raw_items,
            "total_retained_items": self.total_retained_items,
            "total_critical_detected": self.total_crit_detected,
            "total_critical_retained": self.total_crit_retained,
            "total_eligible_items": self.total_eligible_items,
            "total_temporally_suppressed_items": self.total_repetition_suppressed,
            "item_reduction_rate": (
                (1 - self.total_retained_items / self.total_raw_items)
                if self.total_raw_items
                else None
            ),
            "conditional_retention_rate": (
                (self.total_crit_retained / self.total_crit_detected)
                if self.total_crit_detected
                else None
            ),
            "latency_ms": self._latency_summary(self.latencies_ms),
            "processing_latency_ms": self._latency_summary(self.processing_latencies_ms),
            "measurement_stages": {
                "retained_items": "after spatial deduplication and selection, before repetition and motion gating",
                "eligible_items": "after temporal control and motion gating, before audio scheduling",
                "conditional_retention": "selected raw detection identities divided by critical frontend detections",
                "latency_ms": "recorded actual capture-to-audio-trigger samples only",
                "processing_latency_ms": "local process() execution, excluding frontend and audio",
            },
            "gate_stats": {
                "total_evaluated": self.motion_gate.stats.total_evaluated,
                "eligible": self.motion_gate.stats.eligible_count,
                "deferred": self.motion_gate.stats.deferred_count,
                "urgent_bypass": self.motion_gate.stats.urgent_bypass_count,
                "target_bypass": self.motion_gate.stats.target_bypass_count,
            },
        }
