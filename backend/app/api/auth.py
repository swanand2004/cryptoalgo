from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from app.config import get_settings

# Password gate for running the dashboard on a public URL.
#
# OFF unless DASHBOARD_PASSWORD is set, so local development is unchanged.
# Once the site is reachable from the internet it MUST be on: the API can
# switch paper -> live, fire the kill switch and close positions, and none of
# those routes check who is asking.
#
# Pure ASGI rather than @app.middleware("http") on purpose: Starlette's HTTP
# middleware never sees websocket scopes, and /ws/dashboard streams live
# equity and positions -- gating HTTP alone would leave it wide open.
#
# The browser's own Basic-auth prompt does the login (no login page to build
# or break). A successful login also sets a signed cookie, because browsers do
# not reliably attach Basic credentials to a WebSocket handshake but always
# send same-origin cookies.

COOKIE_NAME = "dash_auth"
COOKIE_MAX_AGE = 30 * 24 * 3600
EXEMPT_PATHS = {"/api/health", "/api/health/trading"}  # uptime monitors must reach these without a password


def _token(user: str, password: str) -> str:
    # Derived from the credentials, so changing the password logs every
    # existing session out without a server-side session store.
    return hmac.new(password.encode(), f"dash:{user}".encode(), hashlib.sha256).hexdigest()


class DashboardAuthMiddleware:
    def __init__(self, app, user: str | None = None, password: str | None = None):
        self.app = app
        settings = get_settings()
        self.user = user if user is not None else settings.dashboard_user
        self.password = password if password is not None else settings.dashboard_password

    @property
    def enabled(self) -> bool:
        return bool(self.password)

    def _headers(self, scope) -> dict[str, str]:
        return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

    def _cookie_ok(self, headers: dict[str, str]) -> bool:
        expected = _token(self.user, self.password)
        for part in headers.get("cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE_NAME and hmac.compare_digest(value, expected):
                return True
        return False

    def _basic_ok(self, headers: dict[str, str]) -> bool:
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("basic "):
            return False
        try:
            user, _, password = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
        except Exception:
            return False
        # Both compared, both constant-time, so timing reveals neither.
        user_ok = secrets.compare_digest(user.encode(), self.user.encode())
        pass_ok = secrets.compare_digest(password.encode(), self.password.encode())
        return user_ok and pass_ok

    async def __call__(self, scope, receive, send):
        if not self.enabled or scope["type"] not in ("http", "websocket") or scope["path"] in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        headers = self._headers(scope)
        if self._cookie_ok(headers):
            await self.app(scope, receive, send)
            return

        if self._basic_ok(headers):
            if scope["type"] == "websocket":
                await self.app(scope, receive, send)
                return
            secure = "; Secure" if headers.get("x-forwarded-proto", scope.get("scheme")) == "https" else ""
            cookie = (
                f"{COOKIE_NAME}={_token(self.user, self.password)}; Path=/; Max-Age={COOKIE_MAX_AGE}; "
                f"HttpOnly; SameSite=Strict{secure}"
            ).encode()

            async def send_with_cookie(message):
                if message["type"] == "http.response.start":
                    message = {**message, "headers": [*message.get("headers", []), (b"set-cookie", cookie)]}
                await send(message)

            await self.app(scope, receive, send_with_cookie)
            return

        if scope["type"] == "websocket":
            # Close before accept -> the handshake fails with 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"www-authenticate", b'Basic realm="cryptoalgo", charset="UTF-8"'),
                (b"content-type", b"text/plain; charset=utf-8"),
            ],
        })
        await send({"type": "http.response.body", "body": b"Login required"})
