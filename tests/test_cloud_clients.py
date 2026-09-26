from tests.language_data import text as localized_text
import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from navguide.cloud.config import CloudSettings, CloudUnavailable, cloud_available
from navguide.cloud.provider import QwenProvider, UpstreamFailure
from navguide.cloud.labels import async_extract_english_label, extract_english_label


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in (
        "DASHSCOPE_API_KEY",
        "NAVGUIDE_CLOUD_URL",
        "NAVGUIDE_CLOUD_TOKEN",
        "CLOUD_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NAVGUIDE_CLOUD_ENABLED", "false")


def test_imports_work_without_optional_sdk_or_credentials(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    for name in ("navguide.cloud.descriptions", "navguide.cloud.labels"):
        importlib.reload(importlib.import_module(name))
    assert not cloud_available()
    assert extract_english_label(localized_text("object.red_bull")) == ("Red_Bull", "local")
    assert extract_english_label("road") == ("road", "local")
    assert extract_english_label(localized_text("target.unrecognized")) == ("", "fallback")


def test_disabled_cloud_makes_no_request():
    async def scenario():
        from navguide.cloud.descriptions import stream_chat

        with pytest.raises(CloudUnavailable):
            await stream_chat([{"type": "text", "text": "hello"}]).__anext__()

    asyncio.run(scenario())


def test_local_aliases_and_sync_call_do_not_block_event_loop(monkeypatch):
    async def scenario():
        assert await async_extract_english_label(localized_text("command.find_red_bull")) == (
            "Red_Bull",
            "local",
        )
        assert extract_english_label(localized_text("target.unknown_label")) == ("", "fallback")
        assert await async_extract_english_label(localized_text("target.unknown_label")) == (
            "",
            "fallback",
        )

    asyncio.run(scenario())


def test_cloud_configuration_rejects_plaintext_external_endpoint(monkeypatch):
    monkeypatch.setenv("NAVGUIDE_CLOUD_URL", "http://example.com")
    assert not cloud_available()
    with pytest.raises(ValueError):
        CloudSettings.from_env()


def test_sdk_stream_uses_async_io_and_releases_connections():
    async def scenario():
        observed = {}
        ticks = []

        class SSE(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.sleep(0.02)
                for delta in ({"content": "hello"}, {"audio": {"data": "AA=="}}):
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": "mock",
                                "created": 1,
                                "model": "test",
                                "object": "chat.completion.chunk",
                                "choices": [{"index": 0, "delta": delta}],
                            }
                        )
                        + "\n\n"
                    ).encode()
                yield b'data: {"choices":[],"usage":{"total_tokens":1}}\n\ndata: [DONE]\n\n'

            async def aclose(self):
                observed["closed"] = True

        async def handler(request):
            observed["payload"] = json.loads(request.content)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SSE())

        def factory(**kwargs):
            return AsyncOpenAI(
                **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            )

        async def ticker():
            await asyncio.sleep(0.005)
            ticks.append(True)

        task = asyncio.create_task(ticker())
        pieces = [
            p
            async for p in QwenProvider(CloudSettings(api_key="mock"), factory).stream(
                [{"type": "text", "text": "hello"}]
            )
        ]
        await task
        assert ticks and observed["closed"]
        assert pieces[0].text_delta == "hello" and pieces[1].audio_b64 == "AA=="
        assert observed["payload"]["stream"] is True
        assert observed["payload"]["audio"] == {"voice": "Cherry", "format": "wav"}

    asyncio.run(scenario())


def test_sdk_timeout_and_cancellation_close_stream():
    async def scenario(cancel):
        closed = []
        reading = asyncio.Event()
        release_read = asyncio.Event()

        class Slow(httpx.AsyncByteStream):
            async def __aiter__(self):
                reading.set()
                await release_read.wait()
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                closed.append(True)

        def factory(**kwargs):
            return AsyncOpenAI(
                **kwargs,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(
                            200, headers={"content-type": "text/event-stream"}, stream=Slow()
                        )
                    )
                )
            )

        timeout = 30.0 if cancel else 5.0
        stream = QwenProvider(
            CloudSettings(api_key="mock", timeout_seconds=timeout), factory
        ).stream([])
        task = asyncio.create_task(stream.__anext__())
        if cancel:
            await asyncio.wait_for(reading.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(UpstreamFailure) as result:
                await task
            assert result.value.code == "upstream_timeout"
        assert reading.is_set()
        assert closed

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_label_upstream_failure_never_invents_target(monkeypatch):
    async def failed(self, query):
        raise UpstreamFailure()

    monkeypatch.setenv("NAVGUIDE_CLOUD_ENABLED", "true")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "mock")
    monkeypatch.setattr(QwenProvider, "label", failed)
    assert asyncio.run(async_extract_english_label(localized_text("object.fire_extinguisher"))) == (
        "",
        "fallback",
    )


def test_gateway_transport_parses_completion_and_detects_truncation(monkeypatch):
    from navguide.cloud.descriptions import stream_chat

    original_client = httpx.AsyncClient
    monkeypatch.setenv("NAVGUIDE_CLOUD_ENABLED", "true")
    monkeypatch.setenv("NAVGUIDE_CLOUD_URL", "https://gateway.example")
    monkeypatch.setenv("NAVGUIDE_CLOUD_TOKEN", "mock")

    async def scenario(terminated):
        def handler(request):
            assert request.headers["Authorization"] == "Bearer mock"
            data = '{"type":"delta","text_delta":"hello"}\n'
            if terminated:
                data += '{"type":"done"}\n'
            return httpx.Response(200, text=data)

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(handler)),
        )
        if terminated:
            assert [p.text_delta async for p in stream_chat([])] == ["hello"]
        else:
            with pytest.raises(CloudUnavailable):
                [p async for p in stream_chat([])]

    asyncio.run(scenario(True))
    asyncio.run(scenario(False))
