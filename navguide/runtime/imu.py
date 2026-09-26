"""Versioned IMU validation and constant-time device authentication."""
from __future__ import annotations

import hmac
import math
import time
from typing import Mapping


def token_matches(actual: str, expected: str) -> bool:
    return not expected or hmac.compare_digest(str(actual).encode(), expected.encode())


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def parse_imu(packet: Mapping) -> dict:
    if not isinstance(packet, dict):
        raise ValueError("IMU packet must be an object")
    version = packet.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Unsupported IMU schema")
    vectors = {}
    for name in ("accel", "gyro"):
        vector = packet.get(name)
        if not isinstance(vector, dict):
            raise ValueError(f"Missing {name} vector")
        vectors[name] = {axis: finite_number(vector.get(axis), f"{name}.{axis}") for axis in "xyz"}
    timestamp = finite_number(packet.get("ts", packet.get("timestamp_ms", 0)), "ts")
    if timestamp < 0:
        raise ValueError("ts must be nonnegative milliseconds")
    if any(abs(v) > 4000 for v in vectors["gyro"].values()):
        raise ValueError("gyro must be degrees per second within sensor range")
    if any(abs(v) > 200 for v in vectors["accel"].values()):
        raise ValueError("accel must be meters per second squared within sensor range")
    if "yaw_rate_dps" in packet:
        yaw = finite_number(packet["yaw_rate_dps"], "yaw_rate_dps")
    else:
        accel = vectors["accel"]
        norm = math.hypot(*accel.values())
        if norm < 1e-6:
            raise ValueError("Cannot project yaw without gravity")
        yaw = sum(vectors["gyro"][axis] * accel[axis] / norm for axis in "xyz")
    if abs(yaw) > 4000:
        raise ValueError("yaw rate is outside the sensor range")
    # Deliberately omit credentials from returned/logged/broadcast data.
    return {"schema_version": 1, "ts": timestamp, **vectors, "yaw_rate_dps": yaw}


class IMUState:
    def __init__(self, timeout_seconds: float):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("IMU timeout must be positive and finite")
        self.timeout_seconds = timeout_seconds
        self.received_at = None
        self.packet = None

    def update(self, packet: dict, now=None):
        parsed = parse_imu(packet)
        received_at = time.monotonic() if now is None else finite_number(now, "received_at")
        self.packet = parsed
        self.received_at = received_at
        return self.packet

    def fresh(self, now=None):
        now = time.monotonic() if now is None else finite_number(now, "now")
        return self.received_at is not None and 0 <= now - self.received_at <= self.timeout_seconds

    def yaw_rate(self, now=None):
        # A missing/stale IMU cannot establish that the torso is stationary.
        # The gate threshold defers ordinary cues while preserving its bypass rules.
        return self.packet["yaw_rate_dps"] if self.fresh(now) else 25.0
