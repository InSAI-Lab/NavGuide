"""Read cloud configuration only when a feature is invoked."""
from dataclasses import dataclass, field
import math
import os
from urllib.parse import urlsplit


class CloudUnavailable(RuntimeError):
    """Remote descriptions are disabled, unconfigured or unavailable."""


def _url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Cloud endpoints must be HTTP(S) URLs without embedded credentials")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use HTTPS for cloud endpoints outside localhost")
    return value.rstrip("/")


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not 0 < value <= 600:
        raise ValueError(f"{name} must be greater than zero and at most 600")
    return value


@dataclass(frozen=True)
class CloudSettings:
    enabled: bool = False
    gateway_url: str = ""
    gateway_token: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    omni_model: str = "qwen-omni-turbo"
    text_model: str = "qwen-turbo"
    timeout_seconds: float = 30.0
    max_output_tokens: int = 1024

    @classmethod
    def from_env(cls):
        gateway_url = os.getenv("NAVGUIDE_CLOUD_URL", "").strip()
        max_tokens = int(os.getenv("CLOUD_MAX_OUTPUT_TOKENS", "1024"))
        if not 1 <= max_tokens <= 4096:
            raise ValueError("CLOUD_MAX_OUTPUT_TOKENS must be between 1 and 4096")
        return cls(
            enabled=os.getenv("NAVGUIDE_CLOUD_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            gateway_url=_url(gateway_url) if gateway_url else "",
            gateway_token=os.getenv("NAVGUIDE_CLOUD_TOKEN", "").strip(),
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            base_url=_url(os.getenv("DASHSCOPE_COMPAT_BASE", cls.base_url).strip()),
            omni_model=os.getenv("QWEN_OMNI_MODEL", cls.omni_model).strip(),
            text_model=os.getenv("QWEN_TEXT_MODEL", os.getenv("QWEN_MODEL", cls.text_model)).strip(),
            timeout_seconds=_positive_float("CLOUD_TIMEOUT_SECONDS", 30.0),
            max_output_tokens=max_tokens,
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.gateway_token if self.gateway_url else self.api_key)

    def require_available(self):
        if not self.available:
            raise CloudUnavailable("Optional cloud descriptions are disabled or not configured")


def cloud_available() -> bool:
    try:
        return CloudSettings.from_env().available
    except (ValueError, TypeError):
        return False
