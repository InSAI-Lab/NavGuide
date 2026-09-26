"""Task context and scene smoothing for NavGuide, paper Sections 2.1 and 2.2."""

from __future__ import annotations
from navguide.i18n import text as localized_text

import time
import math
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskMode(str, Enum):
    """User-selected navigation task."""

    PATH_NAVIGATION = (
        "path_navigation"  # Tactile paving, sidewalks, crossings, and obstacle avoidance.
    )
    TARGET_SEARCH = "target_search"  # Search for a user-selected item or landmark.
    SCENE_EXPLORATION = "scene_exploration"  # Explore surroundings and describe landmarks.


class SceneType(str, Enum):
    """Scene categories used for relevance weighting."""

    SIDEWALK = "sidewalk"  # Sidewalk or tactile paving.
    CROSSWALK = "crosswalk"  # Crosswalk or crossing area.
    INTERSECTION = "intersection"  # Intersection or complex traffic area.
    INDOOR = "indoor"  # Indoor corridor, room, or lobby.
    TRANSIT = "transit"  # Bus stop or metro station.
    UNKNOWN = "unknown"  # Unknown scene.


@dataclass
class GuidanceContext:
    """Task, smoothed scene, target query and user relevance preferences."""

    task_mode: TaskMode = TaskMode.PATH_NAVIGATION
    scene_type: SceneType = SceneType.SIDEWALK
    target_query: Optional[str] = None  # Requested target, such as cup, keys, or chair.
    user_weights: Dict[str, float] = field(default_factory=dict)  # User relevance weights.
    hazard_sensitivity: float = 1.0  # Hazard sensitivity multiplier.
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.task_mode = TaskMode(self.task_mode)
        self.scene_type = SceneType(self.scene_type)
        if not math.isfinite(self.hazard_sensitivity) or self.hazard_sensitivity < 0:
            raise ValueError("hazard_sensitivity must be finite and nonnegative")
        self.user_weights = self.validate_user_weights(self.user_weights)

    @staticmethod
    def validate_user_weights(weights: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        for category, weight in weights.items():
            if not isinstance(category, str) or not category.strip():
                raise ValueError("user weight categories must be nonempty strings")
            value = float(weight)
            if not math.isfinite(value) or value < 0:
                raise ValueError("user weights must be finite and nonnegative")
            normalized[category.strip().lower()] = value
        return normalized

    def is_target_search(self) -> bool:
        return self.task_mode == TaskMode.TARGET_SEARCH

    def matches_target(self, category_name: str) -> bool:
        """Check if an object category matches the current target search query."""
        if not self.is_target_search() or not self.target_query:
            return False
        q = self.target_query.strip().lower()
        c = category_name.strip().lower()
        # Match complete category phrases, so "car" never matches "carpet".
        normalize = lambda text: " ".join(re.sub(r"[_-]+", " ", text).split())
        q, c = normalize(q), normalize(c)
        if not q or not c:
            return False
        aliases = {
            localized_text("object.cup"): "cup",
            localized_text("alias.cup"): "cup",
            localized_text("object.chair"): "chair",
            localized_text("object.keys"): "keys",
            localized_text("object.car"): "car",
            localized_text("object.traffic_light"): "traffic light",
            localized_text("object.crosswalk"): "crosswalk",
            localized_text("object.blindpath"): "blindpath",
            localized_text("object.door"): "door",
            localized_text("object.bottle"): "bottle",
        }
        q, c = aliases.get(q, q), aliases.get(c, c)
        return q == c or f" {q} " in f" {c} " or f" {c} " in f" {q} "


class ContextSmoother:
    """Smooth scene changes using a temporal majority and switch threshold."""

    def __init__(self, window_size: int = 8, switch_threshold: int = 4):
        """Retain window_size observations and require switch_threshold votes to switch."""
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(switch_threshold, int) or not 1 <= switch_threshold <= window_size:
            raise ValueError("switch_threshold must be within the window")
        self.window_size = window_size
        self.switch_threshold = switch_threshold
        self._history: deque[SceneType] = deque(maxlen=window_size)
        self.current_scene: SceneType = SceneType.SIDEWALK

    def update(self, raw_scene: SceneType) -> SceneType:
        """Add a scene observation and return the smoothed category."""
        raw_scene = SceneType(raw_scene)
        self._history.append(raw_scene)

        counts: Dict[SceneType, int] = {}
        for s in self._history:
            counts[s] = counts.get(s, 0) + 1

        # Find most frequent scene type (excluding UNKNOWN if known types exist)
        sorted_scenes = sorted(
            counts.items(), key=lambda item: (item[0] != SceneType.UNKNOWN, item[1]), reverse=True
        )

        if sorted_scenes:
            top_scene, top_count = sorted_scenes[0]
            # Require at least switch_threshold observations to switch away from current_scene
            # Retain the current scene on ties to avoid oscillation.
            current_count = counts.get(self.current_scene, 0)
            if (
                top_scene != self.current_scene
                and top_count >= self.switch_threshold
                and top_count > current_count
            ):
                self.current_scene = top_scene
            elif self.current_scene == SceneType.UNKNOWN:
                self.current_scene = top_scene

        return self.current_scene

    def reset(self, initial_scene: SceneType = SceneType.SIDEWALK):
        self._history.clear()
        self.current_scene = SceneType(initial_scene)
