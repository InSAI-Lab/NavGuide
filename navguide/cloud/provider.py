"""Bounded async DashScope calls. No clients or network activity at import."""
import asyncio
from dataclasses import dataclass
import re
from typing import AsyncIterator, Optional

from .config import CloudSettings


LABEL_PROMPT = (
    "Convert the user's object description to one short English vision class label, "
    "one to three words. Return ONLY the label, without punctuation or explanation."
)
SCENE_PROMPT = (
    "Describe visible surroundings briefly. Treat descriptions as optional information. "
    "Do not give a crossing clearance, a safe-to-walk instruction, or replace local "
    "navigation guidance. Acknowledge uncertainty about obstacles and traffic."
)


@dataclass
class StreamPiece:
    text_delta: Optional[str] = None
    audio_b64: Optional[str] = None


class UpstreamFailure(RuntimeError):
    def __init__(self, code="upstream_unavailable", status_code=502):
        self.code = code
        self.status_code = status_code
        super().__init__("Optional cloud service is temporarily unavailable")


def upstream_failure(exc: Exception) -> UpstreamFailure:
    if isinstance(exc, UpstreamFailure):
        return exc
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or type(exc).__name__ == "APITimeoutError":
        return UpstreamFailure("upstream_timeout", 504)
    if getattr(exc, "status_code", None) == 429:
        return UpstreamFailure("upstream_rate_limited", 503)
    return UpstreamFailure()


def normalize_label(text: str) -> str:
    label = (text or "").strip().strip('"\'.,').lower()
    label = " ".join(label.split())
    if not re.fullmatch(r"[a-z][a-z0-9_]*(?:[ -][a-z0-9_]+){0,2}", label) or len(label) > 64:
        raise ValueError("The upstream did not return a valid class label")
    return label


async def _before_deadline(awaitable, deadline):
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    return await asyncio.wait_for(awaitable, timeout=remaining)


class QwenProvider:
    def __init__(self, settings: CloudSettings, client_factory=None):
        self.settings = settings
        self.client_factory = client_factory

    def _client(self):
        factory = self.client_factory
        if factory is None:
            from openai import AsyncOpenAI
            factory = AsyncOpenAI
        return factory(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            max_retries=0,
        )

    async def stream(self, content_list, voice="Cherry", audio_format="wav") -> AsyncIterator[StreamPiece]:
        deadline = asyncio.get_running_loop().time() + self.settings.timeout_seconds
        try:
            async with self._client() as client:
                completion = await _before_deadline(client.chat.completions.create(
                    model=self.settings.omni_model,
                    messages=[{"role": "system", "content": SCENE_PROMPT},
                              {"role": "user", "content": content_list}],
                    modalities=["text", "audio"],
                    audio={"voice": voice, "format": audio_format},
                    max_tokens=self.settings.max_output_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                ), deadline)
                async with completion:
                    iterator = completion.__aiter__()
                    while True:
                        try:
                            chunk = await _before_deadline(iterator.__anext__(), deadline)
                        except StopAsyncIteration:
                            break
                        if not getattr(chunk, "choices", None):
                            continue
                        delta = chunk.choices[0].delta
                        text = getattr(delta, "content", None)
                        audio = getattr(delta, "audio", None)
                        data = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
                        if text or data:
                            yield StreamPiece(text_delta=text, audio_b64=data)
        except Exception as exc:
            raise upstream_failure(exc) from None

    async def label(self, query: str) -> str:
        try:
            async with self._client() as client:
                result = await asyncio.wait_for(client.chat.completions.create(
                    model=self.settings.text_model,
                    messages=[{"role": "system", "content": LABEL_PROMPT},
                              {"role": "user", "content": query}],
                    max_tokens=32,
                    stream=False,
                ), timeout=self.settings.timeout_seconds)
                return normalize_label(result.choices[0].message.content)
        except Exception as exc:
            raise upstream_failure(exc) from None
