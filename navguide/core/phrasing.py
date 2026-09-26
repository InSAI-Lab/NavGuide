"""Action, clock direction and proximity phrasing, NavGuide paper Section 2.3."""

from __future__ import annotations
from navguide.i18n import PHRASE_SEPARATOR, text as localized_text

from typing import List, Optional

from navguide.core.proximity import ProximityBin
from navguide.core.selection import ActionCue


class ActionFirstPhraser:
    """Format speech from actions, clock directions, coarse distances and objects."""

    def __init__(self, default_lang: str = "zh", use_steps: bool = False):
        """
        Args:
            default_lang: Language code ('zh' for Chinese, 'en' for English).
            use_steps: If True, express distances in footsteps instead of meters.
        """
        if default_lang not in {"zh", "en"}:
            raise ValueError("default_lang must be zh or en")
        self.default_lang = default_lang
        self.use_steps = use_steps

    def format_cue_zh(self, cue: ActionCue) -> str:
        """Format a Chinese action-first cue."""
        if self.use_steps:
            steps = max(1, int(round(cue.distance_m / 0.6)))
            distance = localized_text("distance.steps").format(steps=steps)
        elif cue.distance_m < 1.0:
            distance = localized_text("distance.nearby")
        else:
            meters = max(1, int(round(cue.distance_m)))
            distance = localized_text("distance.approx_meters").format(meters=meters)

        location = {"direction": cue.clock_direction_zh, "distance": distance}
        cat_lower = cue.category.strip().lower()
        if cue.action_type == "stop":
            return localized_text("guidance.red_light").format(**location)

        color = cue.source_candidate.signal_color if cue.source_candidate else None
        green_labels = {"green light", "green_light", "green traffic light", "traffic_light_green"}
        if cue.action_type == "caution" and (color == "green" or cat_lower in green_labels):
            return localized_text("guidance.green_light").format(**location)

        if cue.requested_target_flag:
            return localized_text("guidance.target_found").format(
                category=cue.category_zh, **location
            )

        return localized_text("guidance.action").format(
            action=cue.action_zh, category=cue.category_zh, **location
        )

    def format_cue_en(self, cue: ActionCue) -> str:
        """Format an English action-first cue."""
        if self.use_steps:
            dist_str = f"about {max(1, int(round(cue.distance_m / 0.6)))} steps"
        else:
            dist_str = (
                f"about {max(1, int(round(cue.distance_m)))}m" if cue.distance_m >= 1.0 else "close"
            )

        cat_lower = cue.category.strip().lower()
        if cue.action_type == "stop":
            return f"Stop and wait, red light at {cue.clock_direction_en}, {dist_str}"
        color = cue.source_candidate.signal_color if cue.source_candidate else None
        if cue.action_type == "caution" and (
            color == "green"
            or cat_lower
            in {"green light", "green_light", "green traffic light", "traffic_light_green"}
        ):
            return f"Check the crossing, green light at {cue.clock_direction_en}, {dist_str}"

        if cue.requested_target_flag:
            return f"Target found, {cue.category} at {cue.clock_direction_en}, {dist_str}"

        return f"{cue.action_en}, {cue.category} at {cue.clock_direction_en}, {dist_str}"

    def generate_phrase(self, cue: ActionCue, lang: Optional[str] = None) -> str:
        """Format a single ActionCue into an action-first phrase."""
        l = lang or self.default_lang
        if l not in {"zh", "en"}:
            raise ValueError("language must be zh or en")
        if l == "en":
            return self.format_cue_en(cue)
        return self.format_cue_zh(cue)

    def generate_batch_phrases(
        self, cues: List[ActionCue], lang: Optional[str] = None
    ) -> List[str]:
        """Convert a list of eligible cues E_t into spoken guidance strings."""
        return [self.generate_phrase(cue, lang=lang) for cue in cues]

    def combine_phrases(
        self, cues: List[ActionCue], lang: Optional[str] = None, max_phrases: int = 3
    ) -> str:
        """Combine up to max_phrases cues, prioritizing urgency and relevance."""
        if not isinstance(max_phrases, int) or max_phrases < 0:
            raise ValueError("max_phrases must be a nonnegative integer")
        if not cues:
            return ""

        # Sort so urgent cues come first, then by relevance score
        sorted_cues = sorted(
            cues, key=lambda c: (1 if c.urgency_flag else 0, c.relevance_score), reverse=True
        )[:max_phrases]

        phrases = [self.generate_phrase(c, lang=lang) for c in sorted_cues]
        sep = PHRASE_SEPARATOR if (lang or self.default_lang) == "zh" else "; "
        return sep.join(phrases)
