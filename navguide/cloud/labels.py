"""Local object aliases with an explicitly enabled optional cloud fallback."""

from navguide.i18n import text as localized_text
import asyncio
import re
from typing import Tuple

from navguide.cloud.config import CloudSettings
from navguide.cloud.provider import QwenProvider, normalize_label

LOCAL_CN2EN = {
    localized_text("object.red_bull"): "Red_Bull",
    localized_text("alias.ad_milk"): "AD_milk",
    localized_text("alias.ad_milk_spaced"): "AD_milk",
    "ad": "AD_milk",
    localized_text("alias.calcium_milk"): "AD_milk",
    localized_text("alias.mineral_water"): "bottle",
    localized_text("object.bottle"): "bottle",
    localized_text("object.coke"): "coke",
    localized_text("object.sprite"): "sprite",
}


def _local_label(query: str):
    q = (query or "").strip().lower()
    if q in LOCAL_CN2EN:
        return LOCAL_CN2EN[q]
    for key in sorted(LOCAL_CN2EN, key=len, reverse=True):
        # Match ASCII aliases as words, so "road" cannot become "AD_milk".
        if re.search(r"\b" + re.escape(key) + r"\b", q) if key.isascii() else key in q:
            return LOCAL_CN2EN[key]
    try:
        return normalize_label(q)
    except ValueError:
        return None


async def async_extract_english_label(query_cn: str) -> Tuple[str, str]:
    """Return (label, source); an unresolved target is empty, never a guessed bottle."""
    local = _local_label(query_cn)
    if local:
        return local, "local"
    if not (query_cn or "").strip() or len(query_cn) > 512:
        return "", "fallback"
    try:
        settings = CloudSettings.from_env()
        settings.require_available()
        if settings.gateway_url:
            import httpx

            async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
                response = await asyncio.wait_for(
                    client.post(
                        settings.gateway_url + "/v1/label",
                        headers={"Authorization": "Bearer " + settings.gateway_token},
                        json={"query": query_cn.strip()},
                    ),
                    timeout=settings.timeout_seconds,
                )
                response.raise_for_status()
                label = normalize_label(response.json()["label"])
        else:
            label = await QwenProvider(settings).label(query_cn.strip())
        return label, "qwen"
    except Exception:
        return "", "fallback"


def extract_english_label(query_cn: str) -> Tuple[str, str]:
    """Synchronous compatibility API. Async callers should await the async variant."""
    local = _local_label(query_cn)
    if local:
        return local, "local"
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_extract_english_label(query_cn))
    # A synchronous call inside an event loop must not stall local navigation.
    return "", "fallback"
