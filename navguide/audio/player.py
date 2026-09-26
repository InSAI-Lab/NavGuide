"""Play prerecorded navigation cues through the ESP32 speaker."""

from navguide.i18n import FULL_STOP, SENTENCE_ENDINGS, terms, text as localized_text

import os
from navguide.runtime.config import asset_path
import wave
import json
import asyncio
import threading
import queue
import time
from navguide.audio.stream import broadcast_pcm16_realtime
from navguide.audio.compression import compressed_audio_cache

AUDIO_BASE_DIR = asset_path("AUDIO_BASE_DIR", "music")

# Resolve assets from the project root unless overridden by the environment.
VOICE_DIR = asset_path("VOICE_DIR", "voice")
VOICE_MAP_FILE = os.path.join(VOICE_DIR, "map.zh-CN.json")

# Merge these base cues with the voice map at startup.
AUDIO_MAP = {
    localized_text("search.object_detected"): os.path.join(
        AUDIO_BASE_DIR, localized_text("audio.file_detected")
    ),
    localized_text("direction.up"): os.path.join(AUDIO_BASE_DIR, localized_text("audio.file_up")),
    localized_text("direction.down"): os.path.join(
        AUDIO_BASE_DIR, localized_text("audio.file_down")
    ),
    localized_text("direction.left"): os.path.join(
        AUDIO_BASE_DIR, localized_text("audio.file_left")
    ),
    localized_text("direction.right"): os.path.join(
        AUDIO_BASE_DIR, localized_text("audio.file_right")
    ),
    "OK": os.path.join(AUDIO_BASE_DIR, localized_text("audio.file_confirm")),
    localized_text("direction.front"): os.path.join(
        AUDIO_BASE_DIR, localized_text("audio.file_forward")
    ),
    localized_text("direction.back"): os.path.join(
        AUDIO_BASE_DIR, localized_text("audio.file_backward")
    ),
    localized_text("search.object_grasped"): os.path.join(
        AUDIO_BASE_DIR, localized_text("audio.file_grasped")
    ),
}

_audio_cache = {}

_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0
_worker_thread = None
_worker_loop = None
_is_playing = False
_playing_lock = threading.Lock()
_initialized = False
_last_play_ts = 0.0  # Last playback completion time determines leading silence.


def load_wav_file(filepath):
    """Load a WAV file as PCM audio resampled to 8 kHz."""
    if filepath in _audio_cache:
        return _audio_cache[filepath]

    if os.getenv("NAVGUIDE_COMPRESS_AUDIO", "1") == "1":
        compressed_data = compressed_audio_cache.load_and_compress(filepath)
        if compressed_data:
            _audio_cache[filepath] = compressed_data
            return compressed_data

    try:
        with wave.open(filepath, "rb") as wav:
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()

            if channels != 1:
                print(f"[AUDIO] Warning: {filepath} is not mono; using the first channel")
            if sampwidth != 2:
                print(f"[AUDIO] Warning: {filepath} is not 16-bit audio")

            frames = wav.readframes(wav.getnframes())

            # Use only the first channel of stereo input.
            if channels == 2:
                import audioop

                frames = audioop.tomono(frames, sampwidth, 1, 0)

            # Resample to 8 kHz while preserving pitch and duration.
            if framerate != 8000:
                import audioop

                frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                print(f"[AUDIO] Resampling: {filepath} {framerate}Hz -> 8000Hz")

            _audio_cache[filepath] = frames
            return frames

    except Exception as e:
        print(f"[AUDIO] Failed to load audio file {filepath}: {e}")
        return None


def _merge_voice_map():
    """Merge voice/map.zh-CN.json entries into AUDIO_MAP."""
    try:
        if not os.path.exists(VOICE_MAP_FILE):
            print(f"[AUDIO] Voice map not found: {VOICE_MAP_FILE}")
            return
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)
        added = 0
        for text, info in (m or {}).items():
            files = (info or {}).get("files") or []
            if not files:
                continue
            fname = files[0]
            fpath = os.path.join(VOICE_DIR, fname)
            if os.path.exists(fpath):
                AUDIO_MAP[text] = fpath
                added += 1
            else:
                print(f"[AUDIO] Mapped audio file missing: {fpath}")
        print(f"[AUDIO] Merged voice map entries: {added} entries")
    except Exception as e:
        print(f"[AUDIO] Failed to read voice map: {e}")


def preload_all_audio():
    """Load the configured audio files into memory."""
    print("[AUDIO] Preloading audio files...")
    loaded_count = 0

    for filepath in AUDIO_MAP.values():
        if os.path.exists(filepath) and load_wav_file(filepath):
            loaded_count += 1
    print(f"[AUDIO] Preload complete: {loaded_count} audio files")


def _audio_worker():
    """Run the audio playback worker."""
    global _worker_loop

    try:
        import ctypes
        import sys

        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 1  # THREAD_PRIORITY_ABOVE_NORMAL
            )
            print("[AUDIO] Audio thread priority increased")
    except Exception as e:
        print(f"[AUDIO] Failed to set thread priority: {e}")

    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)

    async def process_queue():
        while True:
            try:
                priority_data = await asyncio.get_event_loop().run_in_executor(
                    None, _audio_queue.get, True
                )
                if priority_data is None:
                    break
                if isinstance(priority_data, tuple) and len(priority_data) == 2:
                    _, audio_data = priority_data
                else:
                    audio_data = priority_data
                await _broadcast_audio_optimized(audio_data)
            except Exception as e:
                print(f"[AUDIO] Worker error: {e}")

    _worker_loop.run_until_complete(process_queue())


async def _broadcast_audio_optimized(pcm_data: bytes):
    """Add leading and trailing silence, then broadcast audio every 20 ms."""
    global _last_play_ts, _is_playing
    try:
        with _playing_lock:
            _is_playing = True
        # Input is mono PCM16 at 8 kHz.
        now = time.monotonic()
        idle_sec = now - (_last_play_ts or now)
        # Use longer leading silence after an idle interval.
        lead_ms = 160 if idle_sec > 3.0 else 60
        tail_ms = 40

        lead_silence = b"\x00" * (lead_ms * 8000 * 2 // 1000)  # 8k * 2B
        tail_silence = b"\x00" * (tail_ms * 8000 * 2 // 1000)

        full_audio = lead_silence + pcm_data + tail_silence

        # broadcast_pcm16_realtime owns recording to avoid duplicate audio.

        # The stream broadcaster applies the 20 ms pacing.
        await broadcast_pcm16_realtime(full_audio)

        _last_play_ts = time.monotonic()
    except Exception as e:
        print(f"[AUDIO] Audio broadcast failed: {e}")
    finally:
        with _playing_lock:
            _is_playing = False


def initialize_audio_system():
    """Initialize the audio cache and playback worker."""
    global _initialized, _worker_thread, _last_play_ts

    if _initialized:
        return

    _merge_voice_map()
    preload_all_audio()

    _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
    _worker_thread.start()
    _initialized = True
    _last_play_ts = 0.0

    if os.getenv("NAVGUIDE_COMPRESS_AUDIO", "1") == "1":
        stats = compressed_audio_cache.get_compression_stats()
        print(f"[AUDIO] Audio compression statistics:")
        print(f"  Files: {stats['files_cached']}")
        print(f"  Original size: {stats['total_original_size'] / 1024:.1f} KB")
        print(f"  Compressed size: {stats['total_compressed_size'] / 1024:.1f} KB")
        print(f"  Compression ratio: {stats['compression_ratio']:.1%}")
        print(f"  Saved: {stats['bytes_saved'] / 1024:.1f} KB")

    print("[AUDIO] Audio cache and playback worker initialized")


def play_audio_threadsafe(audio_key):
    """Queue a prerecorded cue safely from any thread."""
    global _audio_queue, _audio_priority

    if not _initialized:
        initialize_audio_system()

    if audio_key not in AUDIO_MAP:
        print(f"[AUDIO] Unknown audio key: {audio_key}")
        return

    filepath = AUDIO_MAP[audio_key]
    pcm_data = _audio_cache.get(filepath)
    if pcm_data is None:
        print(f"[AUDIO] Audio not cached: {audio_key}")
        return

    if pcm_data and len(pcm_data) > 5 and pcm_data[0] in [0x01, 0x02]:
        pcm_data = compressed_audio_cache.decompress(pcm_data)
        if not pcm_data:
            print(f"[AUDIO] Decompression failed: {audio_key}")
            return

    # Bound queued cues to prevent stale guidance.
    queue_size = _audio_queue.qsize()

    with _playing_lock:
        currently_playing = _is_playing

    if queue_size > 0 and not currently_playing:
        # When idle, replace queued cues with the latest guidance.
        print(f"[AUDIO] Clearing queue ({queue_size} items) to play the latest cue")
        while True:
            try:
                _audio_queue.get_nowait()
            except queue.Empty:
                break
    elif queue_size > 1 and currently_playing:
        # Discard a backlog longer than one cue during playback.
        print(f"[AUDIO] Clearing audio backlog ({queue_size} items)")
        while True:
            try:
                _audio_queue.get_nowait()
            except queue.Empty:
                break
    try:
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, pcm_data))
        if queue_size >= 1:
            print(f"[AUDIO] Playback queue size: {queue_size + 1}")
    except queue.Full:
        print(f"[AUDIO] Queue full, dropping: {audio_key}")
        pass


_last_voice_time = 0
_last_voice_text = ""
_voice_cooldown = 1.0  # Minimum interval between identical cues, in seconds.

VOICE_PRIORITY = {"obstacle": 100, "direction": 50, "straight": 10, "other": 30}


def play_voice_text(text: str):
    """Match a spoken cue against the voice map and play it.

    Try the original text and punctuation variants, then use a generic obstacle
    warning when the requested obstacle cue is unavailable."""
    global _last_voice_time, _last_voice_text

    if not text:
        return
    if not _initialized:
        initialize_audio_system()

    current_time = time.time()
    if text == _last_voice_text and current_time - _last_voice_time < _voice_cooldown:
        return

    candidates = []
    t = text.strip()
    candidates.append(t)
    if not t or t[-1] not in SENTENCE_ENDINGS:
        candidates.append(t + FULL_STOP)
    else:
        t2 = t.rstrip(SENTENCE_ENDINGS)
        if t2 and t2 != t:
            candidates.append(t2)

    for ck in candidates:
        if ck in AUDIO_MAP:
            play_audio_threadsafe(ck)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    # Fall back to a generic obstacle warning.
    if all(part in t for part in terms("warning.obstacle_parts")):
        fallback = localized_text("warning.obstacle_ahead")
        if fallback in AUDIO_MAP:
            play_audio_threadsafe(fallback)
            _last_voice_text = text
            _last_voice_time = current_time
            return

    base = t.rstrip(SENTENCE_ENDINGS)
    if base in AUDIO_MAP:
        play_audio_threadsafe(base)
        _last_voice_text = text
        _last_voice_time = current_time
        return
    if base + FULL_STOP in AUDIO_MAP:
        play_audio_threadsafe(base + FULL_STOP)
        _last_voice_text = text
        _last_voice_time = current_time
        return

    print(f"[AUDIO] No matching voice cue: {text}")
