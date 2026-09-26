"""Optional Qwen descriptions through a gateway or direct async SDK calls."""
import asyncio
import json
from typing import AsyncGenerator, Any, Dict, List

from navguide.cloud.config import CloudSettings, CloudUnavailable, cloud_available
from navguide.cloud.provider import QwenProvider, StreamPiece, UpstreamFailure


async def _gateway_stream(settings, content_list, voice, audio_format):
    import httpx

    try:
        async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
            async with client.stream(
                "POST", settings.gateway_url + "/v1/describe",
                headers={"Authorization": "Bearer " + settings.gateway_token},
                json={"content": content_list, "voice": voice, "audio_format": audio_format},
            ) as response:
                if response.status_code != 200:
                    raise CloudUnavailable("Optional cloud gateway is temporarily unavailable")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("type") == "error":
                        raise CloudUnavailable("Optional cloud description was interrupted")
                    if event.get("type") == "done":
                        return
                    if event.get("type") != "delta":
                        raise CloudUnavailable("Invalid cloud gateway response")
                    yield StreamPiece(text_delta=event.get("text_delta"), audio_b64=event.get("audio_b64"))
                raise CloudUnavailable("Optional cloud description ended unexpectedly")
    except (httpx.HTTPError, ValueError):
        raise CloudUnavailable("Optional cloud gateway is temporarily unavailable") from None


async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",
    audio_format: str = "wav",
) -> AsyncGenerator[StreamPiece, None]:
    """Yield text and base64 PCM16 audio without blocking the local event loop."""
    settings = CloudSettings.from_env()
    settings.require_available()
    if audio_format != "wav":
        raise ValueError("Qwen streaming playback supports the wav/PCM16 format only")
    source = (_gateway_stream(settings, content_list, voice, audio_format)
              if settings.gateway_url else QwenProvider(settings).stream(content_list, voice, audio_format))
    deadline = asyncio.get_running_loop().time() + settings.timeout_seconds
    try:
        while True:
            try:
                piece = await asyncio.wait_for(source.__anext__(), max(0, deadline - asyncio.get_running_loop().time()))
            except StopAsyncIteration:
                break
            yield piece
    except (asyncio.TimeoutError, UpstreamFailure):
        raise CloudUnavailable("Optional cloud description timed out or is unavailable") from None
    finally:
        await source.aclose()
