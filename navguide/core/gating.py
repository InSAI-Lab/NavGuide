"""Torso yaw motion gating, NavGuide paper Section 2.3 and Equation 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

from navguide.core.selection import ActionCue


@dataclass
class InertialGatingStats:
    """Statistics tracked by the inertial motion gate."""
    total_evaluated: int = 0
    eligible_count: int = 0
    deferred_count: int = 0
    urgent_bypass_count: int = 0
    target_bypass_count: int = 0
    motion_eligible_count: int = 0

    @property
    def defer_rate(self) -> float:
        """Fraction of cues deferred by motion gating."""
        if self.total_evaluated == 0:
            return 0.0
        return float(self.deferred_count / self.total_evaluated)


class InertialMotionGate:
    """Defer ordinary cues at torso yaw rates of 25 deg/s or above."""

    def __init__(self, yaw_rate_threshold_dps: float = 25.0):
        """
        Args:
            yaw_rate_threshold_dps: Angular speed threshold omega_0 (default 25.0 deg/s).
        """
        if not math.isfinite(yaw_rate_threshold_dps) or yaw_rate_threshold_dps <= 0:
            raise ValueError("yaw_rate_threshold_dps must be positive and finite")
        self.omega_0 = yaw_rate_threshold_dps
        self.stats = InertialGatingStats()

    def evaluate_gate(self, cue: ActionCue, yaw_rate_dps: float) -> bool:
        """Allow a cue when abs(yaw_rate_dps) < omega_0, urgent, or a requested target."""
        if not math.isfinite(yaw_rate_dps):
            raise ValueError("yaw_rate_dps must be finite")
        is_stable = abs(yaw_rate_dps) < self.omega_0
        is_urgent = bool(cue.urgency_flag)
        is_target = bool(cue.requested_target_flag)

        eligible = is_stable or is_urgent or is_target

        self.stats.total_evaluated += 1
        if eligible:
            self.stats.eligible_count += 1
            if not is_stable:
                if is_urgent:
                    self.stats.urgent_bypass_count += 1
                if is_target:
                    self.stats.target_bypass_count += 1
            else:
                self.stats.motion_eligible_count += 1
        else:
            self.stats.deferred_count += 1

        return eligible

    def filter_eligible_cues(
        self,
        cues: List[ActionCue],
        yaw_rate_dps: float
    ) -> Tuple[List[ActionCue], List[ActionCue]]:
        """Return eligible and deferred cues without queuing deferred content for replay."""
        if not math.isfinite(yaw_rate_dps):
            raise ValueError("yaw_rate_dps must be finite")
        eligible_cues: List[ActionCue] = []
        deferred_cues: List[ActionCue] = []

        for cue in cues:
            if self.evaluate_gate(cue, yaw_rate_dps):
                eligible_cues.append(cue)
            else:
                # Deferred: discarded from current update without queued replay
                deferred_cues.append(cue)

        return eligible_cues, deferred_cues

    def reset_stats(self):
        self.stats = InertialGatingStats()
