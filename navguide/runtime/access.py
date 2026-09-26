"""Shared HTTP and WebSocket device access middleware."""
from fastapi.responses import JSONResponse
from urllib.parse import urlsplit
from navguide.runtime.imu import token_matches


class DeviceAccess:
    """ASGI authentication and bounded HTTP request bodies, including chunked uploads."""
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        public = scope["path"] in {
            "/", "/api/health", "/api/ready",
            "/static/localization.js", "/static/locales/zh-CN.json",
        }
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        if not token and scope["type"] == "websocket":
            token = next((p[5:] for p in scope.get("subprotocols", []) if p.startswith("auth.")), "")
        origin = headers.get(b"origin", b"").decode("latin-1")
        host = headers.get(b"host", b"").decode("latin-1")
        origin_ok = not origin or origin in {f"http://{host}", f"https://{host}"}
        host_ok = True
        if not self.settings.device_token:
            # A Host header alone is not a trusted origin for an unauthenticated
            # loopback service. This also closes the DNS-rebinding case.
            try:
                local_hosts = {"127.0.0.1", "::1", "localhost"}
                host_ok = urlsplit(f"http://{host}").hostname in local_hosts
                if origin:
                    origin_ok = origin_ok and urlsplit(origin).hostname in local_hosts
            except ValueError:
                host_ok = origin_ok = False
        if not public and (not token_matches(token, self.settings.device_token) or not origin_ok or not host_ok):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await JSONResponse({"detail": "Device authentication required"}, 401)(scope, receive, send)
            return
        if scope["type"] == "http" and scope.get("method") in {"POST", "PUT", "PATCH"}:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.settings.max_frame_bytes:
                    await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()
            return await self.app(scope, replay, send)
        await self.app(scope, receive, send)
