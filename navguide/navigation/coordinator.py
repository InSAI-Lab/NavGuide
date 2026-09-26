# -*- coding: utf-8 -*-
from __future__ import annotations
from navguide.i18n import terms, text as localized_text
import time

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from dataclasses import dataclass
from typing import Optional, Dict, Any, Deque, List, Tuple
from collections import deque

# Load workflows independently so unavailable backends can be handled.
try:
    from navguide.navigation.path_following import (
        BlindPathNavigator,
        ProcessingResult as BlindResult,
    )
except ImportError:
    BlindPathNavigator = Any
    BlindResult = Any

try:
    from navguide.navigation.street_crossing import (
        CrossStreetNavigator,
        CrossStreetResult as CrossResult,
    )
except ImportError:
    CrossStreetNavigator = Any
    CrossResult = Any

from navguide.core.context import GuidanceContext, TaskMode, SceneType
from navguide.core.selection import DetectionCandidate, ActionCue
from navguide.core.pipeline import NavGuidePipeline, EvaluationCondition, NavGuideResult

IDLE = "IDLE"
CHAT = "CHAT"
BLINDPATH_NAV = "BLINDPATH_NAV"
SEEKING_CROSSWALK = "SEEKING_CROSSWALK"
WAIT_TRAFFIC_LIGHT = "WAIT_TRAFFIC_LIGHT"
CROSSING = "CROSSING"
SEEKING_NEXT_BLINDPATH = "SEEKING_NEXT_BLINDPATH"
RECOVERY = "RECOVERY"
TRAFFIC_LIGHT_DETECTION = "TRAFFIC_LIGHT_DETECTION"
ITEM_SEARCH = "ITEM_SEARCH"


@dataclass
class OrchestratorResult:
    annotated_image: Optional[Any]
    guidance_text: str
    state: str
    extras: Dict[str, Any]


class MajorityFilter:
    def __init__(self, size: int = 8):
        self.buf: Deque[str] = deque(maxlen=size)

    def push(self, v: str):
        self.buf.append(v)

    def majority(self) -> str:
        if not self.buf:
            return "unknown"
        cnt = {}
        for v in self.buf:
            cnt[v] = cnt.get(v, 0) + 1
        # Prefer known colors over unknown entries in the voting window.
        items = sorted(
            cnt.items(), key=lambda x: (0 if x[0] == "unknown" else 1, x[1]), reverse=True
        )
        return items[0][0]

    def history(self) -> List[str]:
        return list(self.buf)

    def clear(self):
        self.buf.clear()


class TrafficLightDetector:
    """Adapt the configured signal model to the navigation coordinator."""

    COLORS = {
        "stop": "red",
        "countdown_stop": "red",
        "go": "green",
        "countdown_go": "yellow",
    }

    def __init__(self, backend=None):
        self.backend = backend

    def reset(self):
        if self.backend is not None:
            self.backend.reset_detection_state()

    def detect(self, bgr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """Return current signal evidence and model availability."""
        try:
            if self.backend is None:
                from navguide.perception import traffic_lights

                self.backend = traffic_lights
            result = self.backend.process_single_frame(bgr)
        except Exception as error:
            self.reset()
            return "unknown", {
                "available": False,
                "reason": "inference_failed" if self.backend is not None else "model_unavailable",
                "error": str(error),
                "stable_color": "unknown",
            }

        if not isinstance(result, dict):
            self.reset()
            return "unknown", {
                "available": False,
                "reason": "inference_failed",
                "error": "Invalid signal detector response",
                "stable_color": "unknown",
            }
        if not result.get("available", False):
            self.reset()
            return "unknown", {
                "available": False,
                "reason": result.get("reason", "model_unavailable"),
                "error": result.get("error"),
                "stable_color": "unknown",
            }

        current = result.get("detected_light")
        color = self.COLORS.get(current, "unknown")
        stable = result.get("stable_light")
        stable_color = color if current == stable else "unknown"
        if color == "unknown":
            self.reset()
        return color, {
            "available": True,
            "reason": None,
            "detected_light": current,
            "stable_light": stable,
            "stable_color": stable_color,
            "detected_colors": result.get("detected_colors", []),
        }


def _color_bgr(name: str) -> Tuple[int, int, int]:
    if name == "red":
        return (0, 0, 255)
    if name == "green":
        return (0, 255, 0)
    if name == "yellow":
        return (0, 255, 255)
    if name == "blue":
        return (255, 0, 0)
    if name == "orange":
        return (0, 165, 255)
    if name == "cyan":
        return (255, 255, 0)
    if name == "magenta":
        return (255, 0, 255)
    if name == "gray":
        return (128, 128, 128)
    if name == "white":
        return (255, 255, 255)
    return (200, 200, 200)


def _put_text(img, text, org, color=(255, 255, 255), scale=0.7, thick=2, outline=True):
    if outline:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cv2.putText(
                    img,
                    text,
                    (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (0, 0, 0),
                    thick + 1,
                )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def _draw_badge(img, text, pos=(10, 28), fg="white", bg="blue"):
    color_fg = _color_bgr(fg)
    color_bg = _color_bgr(bg)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = pos
    pad = 6
    cv2.rectangle(img, (x - 4, y - th - pad), (x + tw + 8, y + pad // 2), color_bg, -1)
    _put_text(img, text, (x, y), color=color_fg, scale=0.6, thick=2, outline=False)


def _draw_state_panel(img, kv: Dict[str, Any], pos=(10, 60)):
    x, y = pos
    line_h = 22
    for i, (k, v) in enumerate(kv.items()):
        _put_text(img, f"{k}: {v}", (x, y + i * line_h), color=(255, 255, 255), scale=0.6, thick=2)


def _draw_frame_border(img, color=(0, 255, 0), thickness=3):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)


def _draw_progress_bar(img, ratio: float, pos=(10, 90), size=(180, 10), color="cyan"):
    ratio = max(0.0, min(1.0, float(ratio)))
    x, y = pos
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.rectangle(
        img, (x + 1, y + 1), (x + 1 + int((w - 2) * ratio), y + h - 1), _color_bgr(color), -1
    )


class NavigationMaster:
    def __init__(
        self,
        blind_nav: BlindPathNavigator,
        cross_nav: CrossStreetNavigator,
        *,
        min_tts_interval: float = 1.2,
    ):
        self.blind = blind_nav
        self.cross = cross_nav
        self.state = IDLE
        self.last_guidance_ts = 0.0
        self.min_tts_interval = min_tts_interval

        self.cnt_crosswalk_seen = 0
        self.cnt_align_ready = 0
        self.cnt_cross_end = 0
        self.cnt_lost = 0

        # Use a cooldown to suppress rapid state changes.
        self.cooldown_until = 0.0

        self.prev_target_state = BLINDPATH_NAV

        self.tld = TrafficLightDetector()
        self.tl_major = MajorityFilter(size=8)
        self.tl_last_color = "unknown"

        self.FRAMES_CROSS_SEEN = 8
        self.FRAMES_ALIGN_READY = 12
        self.FRAMES_CROSS_END = 12
        self.FRAMES_NEXT_BLIND_OK = 8
        self.FRAMES_LOST_MAX = 45

        self.ANGLE_ALIGN_THR_DEG = 12.0
        self.OFFSET_ALIGN_THR = 0.15

        self.COOLDOWN_SEC = 0.6

        self.prev_nav_state_before_search = (
            None  # Navigation state to restore when item search ends.
        )

        # Apply five-factor ranking, the Top 3 budget, repetition control, and inertial gating.
        self.navguide = NavGuidePipeline(
            condition=EvaluationCondition.P,
            top_k=3,
            yaw_rate_threshold_dps=25.0,
            repetition_window_sec=3.0,
            lang="zh",
        )

    def update_imu_motion(self, yaw_rate_dps: float):
        """Update torso angular velocity in degrees per second for inertial gating."""
        if hasattr(self, "navguide") and self.navguide is not None:
            self.navguide.update_imu_motion(yaw_rate_dps)

    def set_navguide_condition(self, condition: EvaluationCondition):
        """Select the evaluation condition: P, T, B, or REMOTE."""
        if hasattr(self, "navguide") and self.navguide is not None:
            self.navguide.condition = condition

    def get_state(self) -> str:
        return self.state

    def _reset_signal_detection(self):
        self.tld.reset()
        self.tl_major.clear()
        self.tl_last_color = "unknown"

    def start_blind_path_navigation(self):
        """Start tactile paving navigation."""
        self._reset_signal_detection()
        self.state = BLINDPATH_NAV
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if hasattr(self, "navguide") and self.navguide is not None:
            self.navguide.set_task_mode(TaskMode.PATH_NAVIGATION)
            self.navguide.set_scene_type(SceneType.SIDEWALK)
        if self.blind:
            self.blind.reset()

    def stop_navigation(self):
        """Stop navigation and return to conversation mode."""
        self._reset_signal_detection()
        self.state = CHAT
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if hasattr(self, "navguide") and self.navguide is not None:
            self.navguide.set_task_mode(TaskMode.SCENE_EXPLORATION)
        if self.blind:
            self.blind.reset()

    def start_crossing(self):
        """Start the street crossing workflow."""
        self._reset_signal_detection()
        self.state = CROSSING
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if hasattr(self, "navguide") and self.navguide is not None:
            self.navguide.set_task_mode(TaskMode.PATH_NAVIGATION)
            self.navguide.set_scene_type(SceneType.CROSSWALK)
        if self.cross:
            self.cross.reset()

    def start_traffic_light_detection(self):
        """Start traffic signal detection."""
        self._reset_signal_detection()
        self.state = TRAFFIC_LIGHT_DETECTION
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if hasattr(self, "navguide") and self.navguide is not None:
            self.navguide.set_scene_type(SceneType.INTERSECTION)

    def is_in_navigation_mode(self):
        """Return whether a navigation workflow is active."""
        return self.state not in ["CHAT", "IDLE", "TRAFFIC_LIGHT_DETECTION", "ITEM_SEARCH"]

    def start_item_search(self, target_query: Optional[str] = None):
        """Pause navigation and start item search."""
        if self.state in [
            BLINDPATH_NAV,
            SEEKING_CROSSWALK,
            WAIT_TRAFFIC_LIGHT,
            CROSSING,
            SEEKING_NEXT_BLINDPATH,
        ]:
            self.prev_nav_state_before_search = self.state
            print(f"[NAV MASTER] Pausing navigation state {self.state}, switching to item search")
        else:
            self.prev_nav_state_before_search = None

        self.state = ITEM_SEARCH
        self.cooldown_until = time.time() + self.COOLDOWN_SEC
        if hasattr(self, "navguide") and self.navguide is not None:
            self.navguide.set_task_mode(TaskMode.TARGET_SEARCH, target_query=target_query)

    def stop_item_search(self, restore_nav: bool = True):
        """Stop item search and optionally resume navigation."""
        if restore_nav and self.prev_nav_state_before_search:
            self.state = self.prev_nav_state_before_search
            print(f"[NAV MASTER] Item search ended, restoring navigation state {self.state}")
            self.prev_nav_state_before_search = None
            if hasattr(self, "navguide") and self.navguide is not None:
                self.navguide.set_task_mode(TaskMode.PATH_NAVIGATION)
        else:
            self.state = CHAT
            print(f"[NAV MASTER] Item search ended, returning to conversation mode")
            if hasattr(self, "navguide") and self.navguide is not None:
                self.navguide.set_task_mode(TaskMode.SCENE_EXPLORATION)

        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def force_state(self, s: str):
        if s == WAIT_TRAFFIC_LIGHT:
            self._reset_signal_detection()
        self.state = s
        self.cooldown_until = time.time() + self.COOLDOWN_SEC

    def on_voice_command(self, text: str):
        t = (text or "").strip()
        if localized_text("navigation.command.start_crossing") in t:
            if self.state in (
                BLINDPATH_NAV,
                SEEKING_CROSSWALK,
                WAIT_TRAFFIC_LIGHT,
                IDLE,
                RECOVERY,
                SEEKING_NEXT_BLINDPATH,
            ):
                self._reset_signal_detection()
                self.state = WAIT_TRAFFIC_LIGHT
                self.cooldown_until = time.time() + self.COOLDOWN_SEC
        elif any(keyword in t for keyword in terms("navigation.cross_now_commands")):
            self.state = CROSSING
            self.cooldown_until = time.time() + self.COOLDOWN_SEC
        elif any(keyword in t for keyword in terms("navigation.stop_commands")):
            self.state = IDLE
        elif localized_text("navigation.command.resume") in t:
            if self.state == IDLE:
                self.state = BLINDPATH_NAV

    def reset(self):
        self.state = IDLE
        self.cnt_crosswalk_seen = 0
        self.cnt_align_ready = 0
        self.cnt_cross_end = 0
        self.cnt_lost = 0
        self._reset_signal_detection()
        self.prev_target_state = BLINDPATH_NAV
        self._last_wait_light_announce = 0
        try:
            self.blind.reset()
        except Exception:
            pass
        try:
            self.cross.reset()
        except Exception:
            pass

    def _collect_and_process_navguide_cues(
        self,
        bgr: np.ndarray,
        obstacles: Optional[List[Any]] = None,
        extra_candidates: Optional[List[DetectionCandidate]] = None,
    ) -> Optional[NavGuideResult]:
        """Collect visual candidates and apply content selection and inertial gating."""
        if not hasattr(self, "navguide") or self.navguide is None:
            return None

        H, W = bgr.shape[:2]
        candidates: List[DetectionCandidate] = list(extra_candidates or [])

        if obstacles:
            for obs in obstacles:
                if isinstance(obs, dict):
                    name = obs.get("name", "obstacle")
                    conf = float(obs.get("conf", 0.75))
                    box = obs.get("box") or obs.get("bbox")
                    if box and len(box) == 4:
                        candidates.append(
                            DetectionCandidate(
                                category=str(name),
                                confidence=conf,
                                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                                is_hazard=True,
                                raw_data=obs,
                            )
                        )

        # Path state alone is not a visual detection. Only actual frontend boxes enter SMP.

        return self.navguide.process(detections=candidates, frame_width=W, frame_height=H)

    def _say(self, now: float, text: str) -> str:
        if not text:
            return ""
        if now - self.last_guidance_ts >= self.min_tts_interval:
            self.last_guidance_ts = now
            return text
        return ""

    def _draw_tl_status(self, img: np.ndarray, color: str, meta: Dict[str, Any]):
        if img is None:
            return
        color_bgr = _color_bgr(color)
        cv2.circle(img, (24, 24), 10, color_bgr, -1)
        _put_text(
            img,
            localized_text("traffic.signal_status").format(color=f"{color}"),
            (40, 30),
            color=color_bgr,
            scale=0.6,
            thick=2,
            outline=False,
        )
        if meta and "bbox" in meta:
            x1, y1, x2, y2 = meta["bbox"]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)

        hist = self.tl_major.history()
        if hist:
            x0, y0 = 10, 50
            r = 6
            gap = 16
            for i, hcol in enumerate(hist[-12:]):
                cv2.circle(img, (x0 + i * gap, y0), r, _color_bgr(hcol), -1)
            _put_text(
                img,
                localized_text("traffic.signal_history"),
                (x0, y0 + 20),
                color=(255, 255, 255),
                scale=0.5,
                thick=1,
            )

    def process_frame(self, bgr: np.ndarray) -> OrchestratorResult:
        now = time.time()

        # Start in CHAT mode until navigation is requested.
        if self.state == IDLE:
            self.state = CHAT
            self.cooldown_until = now + self.COOLDOWN_SEC

        if self.state == CHAT:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="CHAT",
                extras={"mode": localized_text("navigation.mode.chat")},
            )

        if self.state == TRAFFIC_LIGHT_DETECTION:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="TRAFFIC_LIGHT_DETECTION",
                extras={"mode": localized_text("navigation.mode.traffic")},
            )

        if self.state == ITEM_SEARCH:
            return OrchestratorResult(
                annotated_image=bgr,
                guidance_text="",
                state="ITEM_SEARCH",
                extras={
                    "mode": localized_text("navigation.mode.search"),
                    "prev_nav_state": self.prev_nav_state_before_search,
                },
            )

        # Continue rendering during cooldown while suppressing state transitions.
        in_cooldown = now < self.cooldown_until

        if self.state in (BLINDPATH_NAV, SEEKING_CROSSWALK, SEEKING_NEXT_BLINDPATH, RECOVERY):
            try:
                bres: BlindResult = self.blind.process_frame(bgr)
            except Exception as e:
                self.state = RECOVERY
                self.cnt_lost += 5
                ann_err = bgr.copy()

                return OrchestratorResult(
                    ann_err, self._say(now, ""), self.state, {"error": str(e)}
                )

            ann = bres.annotated_image if bres.annotated_image is not None else bgr.copy()
            say = bres.guidance_text or ""

            state_info = bres.state_info or {}
            if state_info.get("detection_available") is False:
                self.state = RECOVERY
                self.cnt_crosswalk_seen = 0
                self.cnt_align_ready = 0
                self.cnt_cross_end = 0
                self.cnt_lost = 0
                self.tl_major.clear()
                self.tl_last_color = "unknown"
                return OrchestratorResult(
                    ann,
                    "",
                    self.state,
                    {
                        "source": "blind",
                        "detection_available": False,
                        "detection_reason": state_info.get("detection_reason"),
                    },
                )
            cross_stage = state_info.get("crosswalk_stage", "not_detected")
            blind_state = state_info.get("state", "UNKNOWN")
            angle = float(state_info.get("last_angle", 0.0))
            center_x_ratio = float(state_info.get("last_center_x_ratio", 0.5))

            if self.state == BLINDPATH_NAV:
                if cross_stage in ("approaching", "ready"):
                    self.cnt_crosswalk_seen += 1
                else:
                    self.cnt_crosswalk_seen = max(0, self.cnt_crosswalk_seen - 1)

                if self.cnt_crosswalk_seen >= self.FRAMES_CROSS_SEEN and not in_cooldown:
                    self.state = SEEKING_CROSSWALK
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = localized_text("navigation.approach_crosswalk")

            # Use the path tracker angle and offset when available to align with the crosswalk.
            elif self.state == SEEKING_CROSSWALK:
                aligned = (
                    abs(angle) <= self.ANGLE_ALIGN_THR_DEG
                    and abs(center_x_ratio - 0.5) <= self.OFFSET_ALIGN_THR
                )
                if cross_stage == "ready" and aligned:
                    self.cnt_align_ready += 1
                else:
                    self.cnt_align_ready = max(0, self.cnt_align_ready - 1)

                if self.cnt_align_ready >= self.FRAMES_ALIGN_READY and not in_cooldown:
                    self.state = WAIT_TRAFFIC_LIGHT
                    self._reset_signal_detection()
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = localized_text("navigation.crosswalk_arrived")

            elif self.state == SEEKING_NEXT_BLINDPATH:
                if blind_state == "NAVIGATING":
                    self.cnt_cross_end += 1
                else:
                    self.cnt_cross_end = max(0, self.cnt_cross_end - 1)
                if self.cnt_cross_end >= self.FRAMES_NEXT_BLIND_OK and not in_cooldown:
                    self.state = BLINDPATH_NAV
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = localized_text("navigation.continue_forward")

            # Resume path following once tactile paving is detected again.
            elif self.state == RECOVERY:
                if blind_state in ("ONBOARDING", "NAVIGATING"):
                    self.state = BLINDPATH_NAV
                    self.cooldown_until = now + self.COOLDOWN_SEC
                    say = ""
                else:
                    say = ""

            if blind_state == "UNKNOWN" and cross_stage == "not_detected":
                self.cnt_lost += 1
            else:
                self.cnt_lost = max(0, self.cnt_lost - 2)
            if self.cnt_lost >= self.FRAMES_LOST_MAX and self.state != RECOVERY:
                self.prev_target_state = self.state
                self.state = RECOVERY
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = localized_text("navigation.recovery_started")

            obs_list = getattr(self.blind, "last_detected_obstacles", None) or []
            nav_res = self._collect_and_process_navguide_cues(bgr, obstacles=obs_list)
            if nav_res and nav_res.should_speak:
                has_urgent = any(c.urgency_flag for c in nav_res.eligible_cues)
                if has_urgent or not say:
                    say = nav_res.speech_text

            extras = {
                "source": "blind",
                "cross_stage": cross_stage,
                "blind_state": blind_state,
                "detection_available": state_info.get("detection_available", True),
                "detection_reason": state_info.get("detection_reason"),
                "navguide": nav_res,
            }
            return OrchestratorResult(ann, self._say(now, say), self.state, extras)

        if self.state == WAIT_TRAFFIC_LIGHT:
            ann = bgr.copy()
            color, meta = self.tld.detect(bgr)
            if color == "unknown" or not meta.get("available", False):
                self.tl_major.clear()
            self.tl_major.push(color)
            major = meta.get("stable_color", "unknown")
            self.tl_last_color = major

            say = ""
            if meta.get("available") and color == major == "green" and not in_cooldown:
                self.state = CROSSING
                self.cooldown_until = now + self.COOLDOWN_SEC
                say = localized_text("navigation.green_crossing")
            else:
                # Throttle repeated traffic signal announcements.
                if not hasattr(self, "_last_wait_light_announce"):
                    self._last_wait_light_announce = 0
                if (
                    now - self._last_wait_light_announce > 5.0
                ):  # Repeat at most once every five seconds.
                    say = localized_text("navigation.wait_green")
                    self._last_wait_light_announce = now

            return OrchestratorResult(
                ann,
                self._say(now, say),
                self.state,
                {"traffic_light": major, "signal_detection": meta},
            )

        if self.state == CROSSING:
            try:
                cres: CrossResult = self.cross.process_frame(bgr)
            except Exception as e:
                self.state = RECOVERY
                ann_err = bgr.copy()

                return OrchestratorResult(
                    ann_err, self._say(now, ""), self.state, {"error": str(e)}
                )

            ann = cres.annotated_image if cres.annotated_image is not None else bgr.copy()
            say = cres.guidance_text or ""

            blind_path_detected = getattr(cres, "blind_path_detected", False)
            blind_path_guidance = getattr(cres, "blind_path_guidance", "")

            # Prioritize tactile paving guidance when a suitable path is detected.
            if blind_path_detected and blind_path_guidance:
                # Switch to path following when the next tactile path is close.
                if hasattr(cres, "should_switch_to_blindpath") and cres.should_switch_to_blindpath:
                    if not in_cooldown:
                        self.state = BLINDPATH_NAV
                        self.cooldown_until = now + self.COOLDOWN_SEC
                        say = localized_text("navigation.switch_to_path")
                        self.cnt_cross_end = 0
                        if hasattr(self.blind, "reset"):
                            self.blind.reset()
                else:
                    # The crossing result already contains tactile paving guidance.
                    pass

            # Finish crossing after repeated crosswalk search results.
            end_hint = False
            if localized_text("crosswalk.search_phrase") in (say or ""):
                end_hint = True
            # A path transition hint alone does not end the crossing workflow.

            self.cnt_cross_end = (
                self.cnt_cross_end + 1 if end_hint else max(0, self.cnt_cross_end - 1)
            )

            if self.cnt_cross_end >= self.FRAMES_CROSS_END and not in_cooldown:
                self.state = SEEKING_NEXT_BLINDPATH
                self.cooldown_until = now + self.COOLDOWN_SEC
            nav_res = self._collect_and_process_navguide_cues(bgr)
            if nav_res and nav_res.should_speak:
                has_urgent = any(c.urgency_flag for c in nav_res.eligible_cues)
                if has_urgent or not say:
                    say = nav_res.speech_text

            return OrchestratorResult(
                ann,
                self._say(now, say),
                self.state,
                {"source": "cross", "end_cnt": self.cnt_cross_end, "navguide": nav_res},
            )

        ann = bgr.copy()

        return OrchestratorResult(ann, "", self.state, {})
