"""Portable configuration shared by the device service and navigation adapters."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} must be a boolean")
    return value in {"1", "true", "yes", "on"}


def asset_path(name: str, relative: str) -> str:
    value = Path(os.getenv(name, relative)).expanduser()
    return str(value if value.is_absolute() else ROOT / value)


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8081
    device_token: str = ""
    frontend: str = "observations"
    udp_enabled: bool = False
    udp_host: str = "127.0.0.1"
    udp_port: int = 12345
    imu_timeout_seconds: float = 0.5
    observation_timeout_seconds: float = 1.5
    max_frame_bytes: int = 2 * 1024 * 1024
    max_frame_pixels: int = 4096 * 3072
    speech_enabled: bool = False
    speech_command: str = "espeak-ng"
    speech_voice: str = "cmn"

    def __post_init__(self):
        if self.frontend not in {"observations", "yoloe"}:
            raise ValueError("NAVGUIDE_FRONTEND must be observations or yoloe")
        if any(isinstance(port, bool) or not isinstance(port, int) or not 0 < port <= 65535
               for port in (self.port, self.udp_port)):
            raise ValueError("Ports must be between 1 and 65535")
        if not math.isfinite(self.imu_timeout_seconds) or self.imu_timeout_seconds <= 0:
            raise ValueError("IMU timeout must be finite and positive")
        if not math.isfinite(self.observation_timeout_seconds) or self.observation_timeout_seconds <= 0:
            raise ValueError("Observation timeout must be finite and positive")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.max_frame_bytes, self.max_frame_pixels)):
            raise ValueError("Frame limits must be positive integers")
        if self.device_token and not re.fullmatch(r"[A-Za-z0-9_.-]{24,256}", self.device_token):
            raise ValueError("NAVGUIDE_DEVICE_TOKEN requires 24 to 256 safe ASCII characters")
        if (self.host not in {"127.0.0.1", "::1", "localhost"} or
                (self.udp_enabled and self.udp_host not in {"127.0.0.1", "::1", "localhost"})) and not self.device_token:
            raise ValueError("A device token is required when listening outside loopback")

    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("NAVGUIDE_HOST", "127.0.0.1"),
            port=int(os.getenv("NAVGUIDE_PORT", "8081")),
            device_token=os.getenv("NAVGUIDE_DEVICE_TOKEN", ""),
            frontend=os.getenv("NAVGUIDE_FRONTEND", "observations"),
            udp_enabled=env_bool("NAVGUIDE_UDP_ENABLED"),
            udp_host=os.getenv("NAVGUIDE_UDP_HOST", "127.0.0.1"),
            udp_port=int(os.getenv("NAVGUIDE_UDP_PORT", "12345")),
            imu_timeout_seconds=float(os.getenv("NAVGUIDE_IMU_TIMEOUT_SECONDS", "0.5")),
            observation_timeout_seconds=float(os.getenv("NAVGUIDE_OBSERVATION_TIMEOUT_SECONDS", "1.5")),
            max_frame_bytes=int(os.getenv("NAVGUIDE_MAX_FRAME_BYTES", str(2 * 1024 * 1024))),
            max_frame_pixels=int(os.getenv("NAVGUIDE_MAX_FRAME_PIXELS", str(4096 * 3072))),
            speech_enabled=env_bool("NAVGUIDE_SPEECH_ENABLED"),
            speech_command=os.getenv("NAVGUIDE_SPEECH_COMMAND", "espeak-ng"),
            speech_voice=os.getenv("NAVGUIDE_SPEECH_VOICE", "cmn"),
        )
