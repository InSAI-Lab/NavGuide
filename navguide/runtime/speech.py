"""Local synthesis with cancellable pending guidance and stream-consumption triggers."""
from __future__ import annotations

import asyncio
import audioop
import contextlib
import io
import shutil
import time
import wave
from dataclasses import dataclass

SAMPLE_RATE = 8000


@dataclass
class SpeechJob:
    text: str
    created: float
    generation: int
    on_trigger: object = None
    is_valid: object = None
    triggered: bool = False


@dataclass
class SpeechChunk:
    pcm: bytes
    job: SpeechJob


class LocalSpeech:
    def __init__(self, command="espeak-ng", voice="cmn", enabled=False, max_pending_age=1.5):
        self.command, self.voice, self.enabled = command, voice, enabled
        self.max_pending_age = max_pending_age
        self.clients = set()
        self.pending = asyncio.Queue(maxsize=1)
        self.task = None
        self.closed = False
        self.last_error = None
        self.trigger_count = 0
        self.generation = 0
        self._synthesis_task = None

    def start(self):
        if self.enabled and self.task is None and not self.closed:
            if shutil.which(self.command) is None:
                raise RuntimeError(f"Install {self.command} or disable NAVGUIDE_SPEECH_ENABLED")
            self.task = asyncio.create_task(self._worker())

    @staticmethod
    def _clear(queue):
        while not queue.empty():
            queue.get_nowait()

    def _clear_audio(self):
        self._clear(self.pending)
        for queue in list(self.clients):
            self._clear(queue)

    def submit(self, text: str, on_trigger=None, is_valid=None):
        if self.closed or not self.enabled or not text or not self.clients:
            return False
        self.generation += 1
        self._clear_audio()
        if self._synthesis_task and not self._synthesis_task.done():
            self._synthesis_task.cancel()
        self.pending.put_nowait(SpeechJob(text[:500], time.monotonic(), self.generation,
                                         on_trigger, is_valid))
        return True

    def _valid(self, job):
        if self.closed or job.generation != self.generation:
            return False
        if not job.triggered and time.monotonic() - job.created > self.max_pending_age:
            return False
        try:
            return job.is_valid is None or bool(job.is_valid())
        except Exception:
            return False

    async def stop(self):
        """Invalidate queued PCM immediately and terminate any active synthesis."""
        self.generation += 1
        self._clear_audio()
        if self._synthesis_task and not self._synthesis_task.done():
            task = self._synthesis_task
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def next_chunk(self, queue, timeout=0.5):
        """Count a trigger only when a valid PCM chunk is consumed by the stream.

        This is the server stream trigger, not confirmation of audible playback.
        A job shared by several listeners contributes a single trigger sample.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            chunk = await asyncio.wait_for(queue.get(), remaining)
            if not self._valid(chunk.job) or not chunk.pcm:
                continue
            if not chunk.job.triggered:
                chunk.job.triggered = True
                self.trigger_count += 1
                if chunk.job.on_trigger:
                    chunk.job.on_trigger(time.monotonic())
            return chunk.pcm

    async def synthesize(self, text):
        process = await asyncio.create_subprocess_exec(
            self.command, "--stdout", "-v", self.voice, text,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError("Local speech synthesis failed")
        with wave.open(io.BytesIO(output), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                raise ValueError("TTS output must be mono PCM16")
            data = wav.readframes(wav.getnframes())
            return audioop.ratecv(data, 2, 1, wav.getframerate(), SAMPLE_RATE, None)[0]

    async def _worker(self):
        while not self.closed:
            job = await self.pending.get()
            if not self._valid(job):
                continue
            try:
                self._synthesis_task = asyncio.create_task(self.synthesize(job.text))
                pcm = await self._synthesis_task
                if not self._valid(job):
                    continue
                for offset in range(0, len(pcm), 320):
                    if not self._valid(job) or not self.clients:
                        break
                    for queue in list(self.clients):
                        if queue.full():
                            queue.get_nowait()
                        queue.put_nowait(SpeechChunk(pcm[offset:offset + 320], job))
                    await asyncio.sleep(0.020)
            except asyncio.CancelledError:
                if self.closed:
                    raise
                # A newer job or stop() cancelled synthesis, not the worker.
                continue
            except Exception as exc:
                self.last_error = type(exc).__name__
            finally:
                self._synthesis_task = None

    async def close(self):
        self.closed = True
        await self.stop()
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        self.clients.clear()


def wav_stream_header():
    import struct
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 0x7FFFFFF0 + 36,
                       b"WAVE", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2,
                       2, 16, b"data", 0x7FFFFFF0)
