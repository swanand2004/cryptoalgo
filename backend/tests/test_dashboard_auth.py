import base64

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.auth import COOKIE_NAME, DashboardAuthMiddleware


def _app(password: str) -> FastAPI:
    app = FastAPI()

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/kill_switch")
    async def kill():
        return {"killed": True}

    @app.websocket("/ws/dashboard")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"equity": 1})
        await websocket.close()

    app.add_middleware(DashboardAuthMiddleware, user="admin", password=password)
    return app


def _basic(user: str, password: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_disabled_without_password_so_local_dev_is_unchanged():
    client = TestClient(_app(""))
    assert client.post("/api/kill_switch").status_code == 200


def test_write_route_refused_without_credentials():
    client = TestClient(_app("s3cret"))
    r = client.post("/api/kill_switch")
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Basic")


def test_wrong_password_refused():
    client = TestClient(_app("s3cret"))
    assert client.post("/api/kill_switch", headers=_basic("admin", "nope")).status_code == 401
    assert client.post("/api/kill_switch", headers=_basic("root", "s3cret")).status_code == 401


def test_health_stays_open_for_uptime_pingers():
    assert TestClient(_app("s3cret")).get("/api/health").status_code == 200


def test_login_sets_cookie_that_alone_admits_later_requests():
    client = TestClient(_app("s3cret"))
    r = client.post("/api/kill_switch", headers=_basic("admin", "s3cret"))
    assert r.status_code == 200
    assert COOKIE_NAME in r.cookies
    assert client.post("/api/kill_switch").status_code == 200  # cookie only


def test_forged_cookie_refused():
    client = TestClient(_app("s3cret"))
    client.cookies.set(COOKIE_NAME, "0" * 64)
    assert client.post("/api/kill_switch").status_code == 401


def test_websocket_refused_without_credentials():
    # The live equity/positions stream is the leak an HTTP-only gate would miss.
    client = TestClient(_app("s3cret"))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/dashboard") as ws:
            ws.receive_json()


def test_websocket_admitted_with_login_cookie():
    client = TestClient(_app("s3cret"))
    client.post("/api/kill_switch", headers=_basic("admin", "s3cret"))
    with client.websocket_connect("/ws/dashboard") as ws:
        assert ws.receive_json() == {"equity": 1}


def test_changing_password_invalidates_old_cookie():
    old = TestClient(_app("s3cret"))
    old.post("/api/kill_switch", headers=_basic("admin", "s3cret"))
    new = TestClient(_app("rotated"))
    new.cookies.set(COOKIE_NAME, old.cookies[COOKIE_NAME])
    assert new.post("/api/kill_switch").status_code == 401


def test_password_from_settings_env_file_is_enforced(monkeypatch):
    # Regression: the gate used to read os.environ only, so a password placed
    # in backend/.env (loaded into Settings, not os.environ) left the site open.
    from app.config import get_settings

    monkeypatch.setenv("DASHBOARD_PASSWORD", "from-settings")
    get_settings.cache_clear()
    try:
        app = FastAPI()

        @app.get("/x")
        async def x():
            return {}

        app.add_middleware(DashboardAuthMiddleware)  # no explicit password
        client = TestClient(app)
        assert client.get("/x").status_code == 401
        assert client.get("/x", headers=_basic("admin", "from-settings")).status_code == 200
    finally:
        monkeypatch.delenv("DASHBOARD_PASSWORD")
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_trading_health_is_503_until_equity_is_recorded_then_200():
    import datetime as dt

    import httpx

    from app.data.db import SessionLocal
    from app.data.models import EquitySnapshot
    from app.main import app as real_app

    transport = httpx.ASGITransport(app=real_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.get("/api/health/trading")).status_code == 503  # nothing recorded yet
        async with SessionLocal() as session:
            session.add(EquitySnapshot(mode="paper", balance=1.0, unrealized_pnl=0.0, equity=1.0,
                                       timestamp=dt.datetime.now(dt.timezone.utc)))
            await session.commit()
        r = await client.get("/api/health/trading")
        assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("family,expected", [("ipv6", "::"), ("ipv4", "0.0.0.0"), ("auto", None)])
def test_delta_client_pins_the_source_address_family(family, expected):
    # Delta whitelists the SOURCE IP; a dual-stack server alternates between
    # its addresses unless the family is pinned.
    from app.config import Settings
    from app.core.delta_client import DeltaRestClient

    client = DeltaRestClient(Settings(delta_api_key="k", delta_api_secret="s", delta_ip_family=family))
    pool = client._client._transport._pool
    assert getattr(pool, "_local_address", None) == expected
