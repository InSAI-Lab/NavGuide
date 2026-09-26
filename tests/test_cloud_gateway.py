from tests.language_data import text as localized_text
import asyncio
import json

import httpx

from navguide.cloud.config import CloudSettings
from navguide.cloud.gateway import GatewaySettings, create_app
from navguide.cloud.provider import StreamPiece

TOKEN = "test-token-" + "x" * 40
AUTH = {"Authorization": "Bearer " + TOKEN}
BODY = {"content": [{"type": "text", "text": "Describe surroundings"}]}


class Provider:
    def __init__(self, error=None, midstream=False, delay=0):
        self.error, self.midstream, self.delay = error, midstream, delay
        self.called = 0
        self.closed = False

    async def label(self, query):
        self.called += 1
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return "fire extinguisher"

    async def stream(self, content, voice, audio_format):
        self.called += 1
        try:
            await asyncio.sleep(self.delay)
            if self.error and not self.midstream:
                raise self.error
            yield StreamPiece(text_delta="A chair")
            if self.error:
                raise self.error
            yield StreamPiece(audio_b64="AA==")
        finally:
            self.closed = True


def settings(**kwargs):
    return GatewaySettings(
        provider=CloudSettings(api_key="mock", timeout_seconds=0.1), token=TOKEN, **kwargs
    )


def test_health_does_not_call_upstream_and_missing_config_fails_closed():
    async def scenario():
        provider = Provider()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(GatewaySettings(), provider)),
            base_url="http://test",
        ) as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/readyz")).status_code == 503
            assert (await client.post("/v1/describe", json=BODY, headers=AUTH)).status_code == 503
        assert provider.called == 0

    asyncio.run(scenario())


def test_authentication_and_input_validation_precede_upstream():
    async def scenario():
        provider = Provider()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings(), provider)),
            base_url="http://test",
        ) as client:
            assert (await client.post("/v1/describe", json=BODY)).status_code == 401
            assert (
                await client.post("/v1/label", json={"query": "  "}, headers=AUTH)
            ).status_code == 422
            invalid = {
                "content": [{"type": "image_url", "image_url": {"url": "http://localhost/private"}}]
            }
            result = await client.post("/v1/describe", json=invalid, headers=AUTH)
            assert result.status_code == 422
            assert "private" not in result.text
        assert provider.called == 0

    asyncio.run(scenario())


def test_size_limit_covers_declared_and_chunked_bodies():
    async def scenario():
        provider = Provider()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_app(settings(max_request_bytes=100), provider)
            ),
            base_url="http://test",
        ) as client:
            assert (
                await client.post("/v1/label", content=b"x" * 101, headers=AUTH)
            ).status_code == 413

            async def body():
                yield b"x" * 60
                yield b"x" * 60

            assert (await client.post("/v1/label", content=body(), headers=AUTH)).status_code == 413
        assert provider.called == 0

    asyncio.run(scenario())


def test_stream_and_label_contract():
    async def scenario():
        provider = Provider()
        app = create_app(settings(max_concurrent_requests=1), provider)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.post("/v1/describe", json=BODY, headers=AUTH)
            assert result.status_code == 200
            events = [json.loads(line) for line in result.text.splitlines()]
            assert events[0]["text_delta"] == "A chair"
            assert events[1]["audio_b64"] == "AA=="
            assert events[-1] == {"type": "done"}
            assert provider.closed
            assert not app.state.capacity.locked()
            assert (
                await client.post(
                    "/v1/label",
                    json={"query": localized_text("object.fire_extinguisher")},
                    headers=AUTH,
                )
            ).json() == {"label": "fire extinguisher"}

    asyncio.run(scenario())


def test_upstream_error_before_and_during_stream_redacts_exception():
    async def scenario(midstream):
        provider = Provider(error=RuntimeError("private upstream secret"), midstream=midstream)
        app = create_app(settings(max_concurrent_requests=1), provider)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            result = await client.post("/v1/describe", json=BODY, headers=AUTH)
            assert result.status_code == (200 if midstream else 502)
            assert "secret" not in result.text
            if midstream:
                assert json.loads(result.text.splitlines()[-1]) == {
                    "type": "error",
                    "code": "upstream_unavailable",
                }
        assert provider.closed and not app.state.capacity.locked()

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_upstream_timeout_and_concurrency_release():
    async def scenario():
        provider = Provider(delay=1)
        app = create_app(settings(max_concurrent_requests=1), provider)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/v1/describe", json=BODY, headers=AUTH))
            await asyncio.sleep(0.01)
            second = await client.post("/v1/label", json={"query": "chair"}, headers=AUTH)
            assert second.status_code == 429
            assert (await first).status_code == 504
            assert not app.state.capacity.locked()
            provider.delay = 0
            assert (
                await client.post("/v1/label", json={"query": "chair"}, headers=AUTH)
            ).status_code == 200

    asyncio.run(scenario())


def test_cancelled_request_releases_capacity_and_upstream():
    async def scenario():
        provider = Provider(delay=1)
        app = create_app(settings(max_concurrent_requests=1), provider)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(client.post("/v1/describe", json=BODY, headers=AUTH))
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert provider.closed and not app.state.capacity.locked()

    asyncio.run(scenario())
