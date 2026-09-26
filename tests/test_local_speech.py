"""PCM delivery, cancellation and freshness tests without a speech executable."""
import asyncio
import time

from navguide.runtime.speech import LocalSpeech


async def wait_for(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


def test_trigger_occurs_only_on_valid_pcm_consumption_once_per_job():
    async def scenario():
        speech = LocalSpeech(enabled=True)
        async def synthesize(text):
            return b"\x01\x00" * 400
        speech.synthesize = synthesize
        first, second = asyncio.Queue(maxsize=8), asyncio.Queue(maxsize=8)
        speech.clients.update((first, second))
        speech.task = asyncio.create_task(speech._worker())
        triggers = []
        try:
            assert speech.submit("example", triggers.append)
            await wait_for(lambda: not first.empty())
            assert speech.trigger_count == 0 and not triggers
            assert await speech.next_chunk(first, 0.1)
            assert speech.trigger_count == 1 and len(triggers) == 1
            assert await speech.next_chunk(second, 0.1)
            assert speech.trigger_count == 1
        finally:
            await speech.close()
    asyncio.run(scenario())


def test_stop_cancels_synthesis_and_erases_pending_pcm():
    async def scenario():
        speech = LocalSpeech(enabled=True)
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def synthesize(text):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        speech.synthesize = synthesize
        queue = asyncio.Queue(maxsize=8)
        speech.clients.add(queue)
        speech.task = asyncio.create_task(speech._worker())
        try:
            speech.submit("old")
            await asyncio.wait_for(started.wait(), 1)
            await speech.stop()
            assert cancelled.is_set()
            assert queue.empty() and speech.pending.empty()
            assert speech.trigger_count == 0
            async def ready(text):
                return b"\x01\x00" * 160
            speech.synthesize = ready
            speech.submit("new")
            assert await speech.next_chunk(queue, 1)
            await speech.stop()
            assert queue.empty()
        finally:
            await speech.close()
    asyncio.run(scenario())


def test_new_job_replaces_old_pcm_and_invalid_scene_never_triggers():
    async def scenario():
        speech = LocalSpeech(enabled=True)
        async def synthesize(text):
            return (b"\x01\x00" if text == "old" else b"\x02\x00") * 160
        speech.synthesize = synthesize
        queue = asyncio.Queue(maxsize=8)
        speech.clients.add(queue)
        speech.task = asyncio.create_task(speech._worker())
        valid = True
        try:
            speech.submit("old")
            await wait_for(lambda: not queue.empty())
            speech.submit("new", is_valid=lambda: valid)
            await wait_for(lambda: not queue.empty())
            assert await speech.next_chunk(queue, 0.1) == b"\x02\x00" * 160
            assert speech.trigger_count == 1
            speech.submit("new", is_valid=lambda: valid)
            await wait_for(lambda: not queue.empty())
            valid = False
            try:
                await speech.next_chunk(queue, 0.01)
                raise AssertionError("invalid content was delivered")
            except asyncio.TimeoutError:
                pass
            assert speech.trigger_count == 1
        finally:
            await speech.close()
    asyncio.run(scenario())


def test_expired_pending_job_and_closed_service_do_not_trigger():
    async def scenario():
        speech = LocalSpeech(enabled=True, max_pending_age=0.01)
        queue = asyncio.Queue(maxsize=8)
        speech.clients.add(queue)
        assert speech.submit("old")
        speech.pending._queue[0].created -= 1
        async def synthesize(text):
            raise AssertionError("expired job should not reach TTS")
        speech.synthesize = synthesize
        speech.task = asyncio.create_task(speech._worker())
        await wait_for(speech.pending.empty)
        assert queue.empty() and speech.trigger_count == 0
        await speech.close()
        assert not speech.submit("after close")
    asyncio.run(scenario())
