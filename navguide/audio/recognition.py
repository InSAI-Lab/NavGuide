"""Dispatch speech recognition events, stop keywords, and navigation requests."""

from navguide.i18n import terms
import os, json, asyncio
from typing import Any, Dict, List, Optional, Callable, Tuple

ASR_DEBUG_RAW = os.getenv("ASR_DEBUG_RAW", "0") == "1"


def _shorten(s: str, limit: int = 200) -> str:
    if not s:
        return ""
    return s if len(s) <= limit else (s[:limit] + "…")


def _safe_to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    for attr in ("to_dict", "model_dump", "__dict__"):
        try:
            v = getattr(x, attr, None)
        except Exception:
            v = None
        if callable(v):
            try:
                d = v()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        elif isinstance(v, dict):
            return v
    try:
        s = str(x)
        if s and s.lstrip().startswith("{") and s.rstrip().endswith("}"):
            return json.loads(s)
    except Exception:
        pass
    return {"_raw": str(x)}


def _extract_sentence(event_obj: Any) -> Tuple[Optional[str], Optional[bool]]:
    d = _safe_to_dict(event_obj)
    cands: List[Dict[str, Any]] = [d]
    for k in ("output", "data", "result"):
        v = d.get(k)
        if isinstance(v, dict):
            cands.append(v)
    for obj in cands:
        sent = obj.get("sentence")
        if isinstance(sent, dict):
            text = sent.get("text")
            is_end = sent.get("sentence_end")
            if is_end is not None:
                is_end = bool(is_end)
            return text, is_end
    for obj in cands:
        if "text" in obj and isinstance(obj.get("text"), str):
            return obj.get("text"), None
    return None, None


_interrupt_override = os.getenv("INTERRUPT_KEYWORDS")
INTERRUPT_KEYWORDS = set(
    _interrupt_override.split(",")
    if _interrupt_override is not None
    else terms("commands.interrupt")
)


def _normalize_cn(s: str) -> str:
    try:
        import unicodedata

        s = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in s)
        s = s.strip().lower()
    except Exception:
        s = (s or "").strip().lower()
    return s


_current_recognition: Optional[object] = None
_rec_lock = asyncio.Lock()


async def set_current_recognition(r):
    global _current_recognition
    async with _rec_lock:
        _current_recognition = r


async def stop_current_recognition():
    global _current_recognition
    async with _rec_lock:
        r = _current_recognition
        _current_recognition = None
    if r:
        try:
            r.stop()
        except Exception:
            pass


class ASRCallback:
    """Send recognition results to the interface and navigation service.

    Stop keywords reset the session immediately. Partial results update the interface;
    final results trigger a request only when speech playback is idle."""

    def __init__(
        self,
        on_sdk_error: Callable[[str], None],
        post: Callable[[asyncio.Future], None],
        ui_broadcast_partial,
        ui_broadcast_final,
        is_playing_now_fn: Callable[[], bool],
        start_response_fn,  # async (text)
        full_system_reset_fn,  # async (reason)
        interrupt_lock: asyncio.Lock,
    ):
        self._on_sdk_error = on_sdk_error
        self._post = post
        self._last_partial_for_ui: str = ""
        self._last_final_text: str = ""
        self._hot_interrupted: bool = False  # Debounce repeated stop keywords within one sentence.

        self._ui_partial = ui_broadcast_partial
        self._ui_final = ui_broadcast_final
        self._is_playing = is_playing_now_fn
        self._start_response = start_response_fn
        self._full_reset = full_system_reset_fn
        self._interrupt_lock = interrupt_lock

    def on_open(self):
        pass

    def on_close(self):
        pass

    def on_complete(self):
        pass

    def on_error(self, err):
        try:
            self._post(self._ui_partial(""))
            self._on_sdk_error(str(err))
        except Exception:
            pass

    def on_result(self, result):
        self._handle(result)

    def on_event(self, event):
        self._handle(event)

    def _has_hotword(self, text: str) -> bool:
        t = _normalize_cn(text)
        if not t:
            return False
        for w in INTERRUPT_KEYWORDS:
            if w and _normalize_cn(w) in t:
                return True
        return False

    def _handle(self, event: Any):
        if ASR_DEBUG_RAW:
            try:
                rawd = _safe_to_dict(event)
                print("[ASR EVENT RAW]", json.dumps(rawd, ensure_ascii=False), flush=True)
            except Exception:
                pass

        text, is_end = _extract_sentence(event)
        if text is None:
            return
        text = text.strip()
        if not text:
            return

        # Stop keywords reset playback without creating a navigation request.
        if not self._hot_interrupted and self._has_hotword(text):
            self._hot_interrupted = True

            async def _hot_reset():
                async with self._interrupt_lock:
                    print(f"[ASR HOTWORD] '{text}' -> FULL RESET, skip LLM", flush=True)
                    await self._full_reset("Hotword interrupt")

            try:
                self._post(_hot_reset())
            except Exception:
                pass
            return

        # Partial recognition results only update the interface.
        self._last_partial_for_ui = text
        try:
            print(f"[ASR PARTIAL] len={len(text)} text='{_shorten(text)}'", flush=True)
            self._post(self._ui_partial(self._last_partial_for_ui))
        except Exception:
            pass

        # Submit final recognition results only while playback is idle.
        if is_end is True:
            final_text = text
            try:
                print(f"[ASR FINAL]  len={len(final_text)} text='{final_text}'", flush=True)
                self._post(self._ui_final(final_text))
            except Exception:
                pass

            if (not self._is_playing()) and final_text:

                async def _run_final():
                    async with self._interrupt_lock:
                        print(f"[LLM INPUT TEXT] {final_text}", flush=True)
                        await self._start_response(final_text)

                try:
                    self._post(_run_final())
                except Exception:
                    pass

            self._last_partial_for_ui = ""
            self._last_final_text = ""
            self._hot_interrupted = False
