"""Owner Telegram alerts (app/notify/telegram.py): off by default, rate-limited,
and a failure can never raise into (or block) trading. httpx is mocked."""
import asyncio
import logging
import types

import httpx
import pytest

from app.config import Settings
from app.notify import telegram

TOKEN = "123456:TEST-TOKEN-never-logged"
ON = Settings(telegram_bot_token=TOKEN, telegram_chat_id="42")
OFF = Settings()


@pytest.fixture(autouse=True)
def _reset_state():
    telegram._last_sent.clear()
    telegram._suppressed.clear()
    telegram._equity.update(down_since=None, alerted=False)
    yield
    telegram._last_sent.clear()
    telegram._suppressed.clear()
    telegram._equity.update(down_since=None, alerted=False)


@pytest.fixture
def calls(monkeypatch):
    """Every request telegram.py sends, captured by a mock transport (no network)."""
    class _Sent(list):
        behaviour: dict

    sent = _Sent()
    behaviour = {"handler": lambda req: httpx.Response(200, json={"ok": True})}

    async def handler(request):
        sent.append(request)
        out = behaviour["handler"](request)
        return await out if asyncio.iscoroutine(out) else out

    real = httpx.AsyncClient
    monkeypatch.setattr(telegram.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    sent.behaviour = behaviour
    return sent


async def _drain():
    if telegram._tasks:
        await asyncio.gather(*list(telegram._tasks))


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(telegram, "_now", lambda: now["t"])
    return now


# --- off by default ---------------------------------------------------------

def test_settings_default_to_off():
    assert OFF.telegram_bot_token == "" and OFF.telegram_chat_id == ""
    assert not telegram.is_enabled(OFF)


async def test_disabled_makes_no_calls(calls):
    assert telegram.notify("x", "hello", settings=OFF) is False
    assert telegram.notify("x", "hello", settings=Settings(telegram_bot_token=TOKEN)) is False  # chat id missing
    # The hooks read the real settings: conftest blanks the env, so they are off too.
    telegram.trade_opened("ETHUSD", "long", 1, 2500.0)
    telegram.trade_closed("ETHUSD", "long", 1, 2500.0, 2510.0, 0.01, 0.002, "signal")
    telegram.kill_switch_on("test")
    telegram.equity_read_failed("TimeoutError")
    await _drain()
    assert calls == [] and not telegram._tasks


async def test_enabled_sends_one_message(calls):
    assert telegram.notify("x", "hello", settings=ON) is True
    await _drain()
    assert len(calls) == 1
    assert calls[0].url.path == f"/bot{TOKEN}/sendMessage"
    body = calls[0].content.decode()
    assert '"chat_id":"42"' in body.replace(" ", "") and "hello" in body


# --- rate limit ---------------------------------------------------------------

async def test_rate_limit_one_per_event_per_minute(calls, clock):
    assert telegram.notify("kill_switch", "first", settings=ON)
    assert not telegram.notify("kill_switch", "second", settings=ON)
    assert not telegram.notify("kill_switch", "third", settings=ON)
    assert telegram.notify("trade_close:BTCUSD", "other key", settings=ON)  # its own budget
    clock["t"] += 59
    assert not telegram.notify("kill_switch", "still inside the minute", settings=ON)
    clock["t"] += 2
    assert telegram.notify("kill_switch", "after a minute", settings=ON)
    await _drain()
    texts = [c.content.decode() for c in calls]
    assert len(calls) == 3
    assert "+3 more 'kill_switch' alert(s)" in texts[-1]


async def test_trade_alerts_are_keyed_per_symbol(calls):
    # A kill switch flattening several coins must report every one of them.
    for sym in ("BTCUSD", "ETHUSD", "SOLUSD"):
        telegram.notify(f"trade_close:{sym}", sym, settings=ON)
    await _drain()
    assert len(calls) == 3


# --- failures never raise -------------------------------------------------------

@pytest.mark.parametrize("failure", ["connect", "http500", "timeout", "garbage"])
async def test_failure_never_raises_and_never_logs_the_token(calls, monkeypatch, caplog, failure):
    async def slow(_req):
        await asyncio.sleep(5)
        return httpx.Response(200)

    def boom(req):
        if failure == "connect":
            raise httpx.ConnectError(f"cannot reach {req.url}", request=req)
        if failure == "http500":
            return httpx.Response(500, text="nope")
        if failure == "timeout":
            return slow(req)
        raise ValueError(f"unexpected {req.url}")

    calls.behaviour["handler"] = boom
    monkeypatch.setattr(telegram, "SEND_TIMEOUT_S", 0.05)
    with caplog.at_level(logging.WARNING, logger="app.notify.telegram"):
        assert telegram.notify("x", "hello", settings=ON) is True
        await _drain()  # the background task itself must complete without an exception
    assert any("Telegram alert" in r.getMessage() for r in caplog.records)
    assert TOKEN not in caplog.text


def test_notify_outside_an_event_loop_is_a_no_op():
    assert telegram.notify("x", "hello", settings=ON) is False  # no running loop: swallowed


def test_httpx_request_log_never_shows_the_token(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").info('HTTP Request: POST https://api.telegram.org/bot%s/sendMessage "200"', TOKEN)
        logging.getLogger("httpx").info('HTTP Request: GET https://api.india.delta.exchange/v2/x "200"')
    assert TOKEN not in caplog.text
    assert "delta.exchange" in caplog.text  # other request lines are untouched


# --- hooks -----------------------------------------------------------------------

async def test_equity_outage_alerts_once_after_five_minutes_then_recovery(calls, clock, monkeypatch):
    monkeypatch.setattr(telegram, "get_settings", lambda: ON)  # the hooks read the (cached) app settings
    telegram.equity_read_failed("ConnectTimeout")
    clock["t"] += 200
    telegram.equity_read_failed("ConnectTimeout")
    await _drain()
    assert calls == []  # under 5 minutes: quiet
    clock["t"] += 101
    telegram.equity_read_failed("ConnectTimeout")
    clock["t"] += 120
    telegram.equity_read_failed("ConnectTimeout")  # same outage: no second alert
    await _drain()
    assert len(calls) == 1 and "EQUITY UNAVAILABLE" in calls[0].content.decode()
    telegram.equity_read_ok()
    telegram.equity_read_ok()
    await _drain()
    assert len(calls) == 2 and "recovered" in calls[1].content.decode()


async def test_short_outage_that_recovers_never_alerts(calls, clock, monkeypatch):
    monkeypatch.setattr(telegram, "get_settings", lambda: ON)  # the hooks read the (cached) app settings
    telegram.equity_read_failed("x")
    clock["t"] += 250
    telegram.equity_read_ok()
    clock["t"] += 250
    telegram.equity_read_failed("x")  # a NEW outage starts its own 5-minute clock
    await _drain()
    assert calls == []


async def test_kill_switch_alert_carries_the_reason(calls, monkeypatch):
    from app.data.app_state import is_kill_switch_on, set_kill_switch

    monkeypatch.setattr(telegram, "get_settings", lambda: ON)  # the hooks read the (cached) app settings
    await set_kill_switch(True, reason="daily loss limit -- test")
    await set_kill_switch(False)  # turning it off sends nothing
    await _drain()
    assert not await is_kill_switch_on()
    assert len(calls) == 1 and "daily loss limit -- test" in calls[0].content.decode()


async def test_kill_switch_still_engages_when_telegram_is_broken(calls, monkeypatch):
    from app.data.app_state import is_kill_switch_on, set_kill_switch

    monkeypatch.setattr(telegram, "get_settings", lambda: ON)  # the hooks read the (cached) app settings

    def broken(req):
        raise httpx.ConnectError("down", request=req)

    calls.behaviour["handler"] = broken
    await set_kill_switch(True)
    await _drain()
    assert await is_kill_switch_on()


def _runtime(settings, equity=14.0):
    async def get_equity():
        return equity

    async def get_positions():
        return {"ETHUSD:consolidated": types.SimpleNamespace(symbol="ETHUSD", size=1)}

    tracker = types.SimpleNamespace(day=__import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).date(), starting_equity=13.0)
    return types.SimpleNamespace(
        settings=settings,
        order_manager=types.SimpleNamespace(get_equity=get_equity, get_positions=get_positions),
        risk_manager=types.SimpleNamespace(daily_pnl=tracker),
    )


async def test_daily_summary_off_sends_nothing(calls):
    await telegram.send_daily_summary(_runtime(OFF))
    await _drain()
    assert calls == []


async def test_daily_summary_content(calls):
    await telegram.send_daily_summary(_runtime(ON))
    await _drain()
    text = calls[0].content.decode()
    assert len(calls) == 1
    for part in ("Yesterday ", "Yesterday's P&L", "Equity: 14.00", "+1.00 USD", "ETHUSD +1", "Trades closed yesterday: 0"):
        assert part in text


def test_summary_job_is_scheduled_at_09_05_ist():
    from app.scheduler import build_scheduler

    async def _noop(*_a, **_k):
        return None

    class _Stub:
        def __getattr__(self, _name):
            return _noop

    runtime = types.SimpleNamespace(
        settings=Settings(delta_api_key="k", delta_api_secret="s"), cache=None, paper_broker=_Stub(),
        live_broker=_Stub(), order_manager=_Stub(), risk_manager=_Stub(), registry=types.SimpleNamespace(symbols=[]),
    )
    job = build_scheduler(runtime).get_job("telegram_daily_summary")
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert (fields["hour"], fields["minute"]) == ("3", "35")  # UTC scheduler: 03:35 UTC = 09:05 IST


async def test_live_broker_reports_open_and_close_with_pnl(monkeypatch):
    """The real hooks in LiveBroker._sync_product / _record_close, on a fake Delta."""
    import datetime as dt

    from tests.test_live_mode import _broker, _FakeDelta

    seen = []
    monkeypatch.setattr(telegram, "notify", lambda event, text, settings=None: seen.append((event, text)) or True)
    now = dt.datetime.now(dt.timezone.utc)
    fake = _FakeDelta([{"size": 1, "entry_price": "2678.2", "margin": "2.5", "liquidation_price": "2440",
                        "realized_pnl": "0"}])
    broker = _broker(fake)
    await broker.sync_fills_and_positions()
    await broker.sync_fills_and_positions()  # unchanged position: no second "opened"
    fake.fills = [{"side": "sell", "size": "1", "price": "2683.4", "commission": "0.0158",
                   "created_at": now.isoformat().replace("+00:00", "Z")}]
    fake.remote_positions = []
    await broker.sync_fills_and_positions()
    assert [e for e, _ in seen] == ["trade_open:ETHUSD", "trade_close:ETHUSD"]
    assert "LONG ETHUSD" in seen[0][1] and "2,678.2" in seen[0][1]
    assert "+0.0362 USD net" in seen[1][1]  # (2683.4 - 2678.2) * 0.01 - 0.0158


async def test_live_balance_outage_feeds_the_equity_alert(monkeypatch):
    from app.execution.live_broker import LiveEquityUnavailable
    from tests.test_live_mode import _broker, _FakeDelta

    failed, ok = [], []
    monkeypatch.setattr(telegram, "equity_read_failed", lambda reason: failed.append(reason))
    monkeypatch.setattr(telegram, "equity_read_ok", lambda: ok.append(1))
    fake = _FakeDelta()
    broker = _broker(fake)
    fake.balance_error = True
    with pytest.raises(LiveEquityUnavailable):
        await broker.get_equity()
    assert failed == ["DeltaApiError"]
    fake.balance_error = False
    await broker.get_equity()
    assert ok == [1]
