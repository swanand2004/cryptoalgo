"""Regression tests for the bugs found in the 2026-10-06 VPS journal review.

Every test here is offline: fake transports and fake clients only.
"""
import asyncio
import logging

import httpx
import pytest

from app.config import Settings
from app.core.deadman_switch import DeadmanSwitch
from app.core.delta_client import DeltaRestClient
from app.execution.broker_base import OrderRequest, PositionView
from tests.test_live_mode import _broker, _FakeDelta


def _client_with(handler) -> DeltaRestClient:
    client = DeltaRestClient(Settings(delta_api_key="k", delta_api_secret="s"))
    client._client = httpx.AsyncClient(base_url="https://x", transport=httpx.MockTransport(handler))
    return client


# ------------------------------------------------ delta_client connect retries

@pytest.mark.asyncio
async def test_get_is_retried_after_a_connect_timeout(monkeypatch):
    # VPS: ~550 balance fallbacks, 30+ skipped position syncs and one failed
    # startup in 10 h, all httpx.ConnectTimeout on a GET that never connected.
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    attempts = []

    def handler(request):
        attempts.append(request.headers.get("timestamp"))
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("")
        return httpx.Response(200, json={"success": True, "result": [{"asset_symbol": "USD", "balance": "13.9"}]})

    result = await _client_with(handler).request("GET", "/v2/wallet/balances", auth=True)
    assert result[0]["balance"] == "13.9"
    assert len(attempts) == 2 and all(attempts)  # re-signed for the retry


@pytest.mark.asyncio
async def test_get_connect_failures_still_raise_once_retries_are_exhausted(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    attempts = []

    def handler(request):
        attempts.append(1)
        raise httpx.ConnectError("Temporary failure in name resolution")

    with pytest.raises(httpx.ConnectError):
        await _client_with(handler).request("GET", "/v2/products", max_retries=3)
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_order_posts_are_never_retried_on_connect_failure(monkeypatch):
    # The live order path must behave exactly as before: fail fast, no resend.
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    attempts = []

    def handler(request):
        attempts.append(1)
        raise httpx.ConnectTimeout("")

    with pytest.raises(httpx.ConnectTimeout):
        await _client_with(handler).request("POST", "/v2/orders", json_body={"size": 1}, auth=True)
    assert len(attempts) == 1


async def _no_sleep(_seconds):
    return None


# ------------------------------------------------ live balance fallback logging

class _TimeoutDelta(_FakeDelta):
    async def get_balances(self):
        if getattr(self, "balance_error", False):
            raise httpx.ConnectTimeout("")  # stringifies to "" -- the old "()" log
        return await super().get_balances()


@pytest.mark.asyncio
async def test_balance_fallback_names_the_error_and_never_reports_a_negative_age(caplog):
    fake = _TimeoutDelta()
    broker = _broker(fake)
    assert await broker.get_equity() == pytest.approx(3.67)
    fake.balance_error = True
    broker.invalidate_balance_cache()  # force a real (failing) read
    with caplog.at_level(logging.WARNING, logger="app.execution.live_broker"):
        assert await broker.get_equity() == pytest.approx(3.67)
    msg = next(r.getMessage() for r in caplog.records if "Live balance fetch failed" in r.getMessage())
    assert "ConnectTimeout" in msg
    assert "()" not in msg and "-" not in msg.split("from ")[-1]


# ------------------------------------------------ deadman switch survives network errors

class _NetFlakyHeartbeat:
    def __init__(self):
        self.register_attempts = 0
        self.heartbeats = 0

    async def create_heartbeat(self, body):
        self.register_attempts += 1
        if self.register_attempts == 1:
            raise httpx.ConnectTimeout("")

    async def send_heartbeat(self, heartbeat_id, ttl_ms):
        self.heartbeats += 1
        if self.heartbeats == 1:
            raise httpx.ConnectTimeout("")


@pytest.mark.asyncio
async def test_deadman_loop_survives_network_errors_and_keeps_its_registration(monkeypatch):
    # VPS: 21 "Task exception was never retrieved ... DeadmanSwitch._loop()
    # ... ConnectTimeout" -- the heartbeat task died on every network blip.
    import app.core.deadman_switch as dms

    monkeypatch.setattr(dms, "NETWORK_RETRY_SECONDS", 0.001)
    client = _NetFlakyHeartbeat()
    switch = DeadmanSwitch(client, send_interval_seconds=0.05)
    task = asyncio.create_task(switch._loop())
    await asyncio.sleep(0.2)
    assert not task.done(), "the loop must not die on a network error"
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert client.register_attempts == 2  # failed once, then registered; never re-registered
    assert client.heartbeats >= 2 and switch._registered is True


@pytest.mark.asyncio
async def test_deadman_retries_quickly_after_a_network_miss(monkeypatch):
    # 20 s interval + a missed heartbeat would overrun the 30 s TTL; the
    # resend after a network miss must come well inside it.
    import app.core.deadman_switch as dms

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(dms.asyncio, "sleep", fake_sleep)
    client = _NetFlakyHeartbeat()
    client.register_attempts = 1  # registration succeeds
    switch = DeadmanSwitch(client, send_interval_seconds=20.0)
    with pytest.raises(asyncio.CancelledError):
        await switch._loop()
    assert sleeps == [dms.NETWORK_RETRY_SECONDS, 20.0]


# ------------------------------------------------ external series for unmapped symbols

@pytest.mark.asyncio
async def test_binance_external_series_skip_unmapped_symbols_without_a_keyerror():
    # VPS: 18 KeyError tracebacks an hour for the six filler coins.
    from app.data import ingest_external as ie

    assert "XLMUSD" not in ie.BINANCE_SYMBOL
    assert await ie.fetch_binance_funding_history("XLMUSD", limit=3) == []
    assert await ie.fetch_binance_open_interest_hist("XLMUSD", period="1h", limit=6) == []
    assert await ie.fetch_binance_long_short_ratio("XLMUSD", period="1h", limit=6) == []


# ------------------------------------------------ incremental funding refresh

@pytest.mark.asyncio
async def test_funding_refresh_resumes_from_the_window_not_from_2019(monkeypatch):
    # VPS: every 15 minutes the job re-downloaded ~29k settlements since 2019.
    import app.data.ingest_binance_history as ibh
    from app.data.ingest_rest import upsert_candles

    starts = {}

    async def fake_fetch(perp, start_ms=ibh._FUNDING_START_MS, client=None):
        starts[perp] = start_ms
        return [{"time": 2_000_000_000, "open": 0.01, "high": 0.01, "low": 0.01, "close": 0.01, "volume": 0.0}]

    monkeypatch.setattr(ibh, "fetch_binance_funding_history_deep", fake_fetch)
    newest_s = 1_900_000_000
    await upsert_candles("BFUNDHIST:BTCUSD", "8h",
                         [{"time": newest_s, "open": 0.01, "high": 0.01, "low": 0.01, "close": 0.01, "volume": 0.0}])

    window_ms = 1_950_000_000_000  # newer than the stored row
    await ibh.backfill_deep_funding(["BTCUSD", "ETHUSD"], start_ms=window_ms)
    assert starts["BTCUSDT"] == newest_s * 1000  # reaches back to the newest stored print
    assert starts["ETHUSDT"] == ibh._FUNDING_START_MS  # nothing stored: full history

    await upsert_candles("BFUNDHIST:BTCUSD", "8h",
                         [{"time": 1_960_000_000, "open": 0.01, "high": 0.01, "low": 0.01, "close": 0.01, "volume": 0.0}])
    await ibh.backfill_deep_funding(["BTCUSD"], start_ms=window_ms)
    assert starts["BTCUSDT"] == window_ms  # up to date: just the recent window

    await ibh.backfill_deep_funding(["BTCUSD"])  # default: the one-time deep backfill
    assert starts["BTCUSDT"] == ibh._FUNDING_START_MS


# ------------------------------------------------ gold gap repair

@pytest.mark.asyncio
async def test_gold_gap_repair_loads_binance_once_for_all_pre_xaut_holes(monkeypatch):
    # VPS: one ccxt client (and full market reload) per hole; 10 permanent 1h
    # holes, and a slow fapi/dapi exchangeInfo failed the repair on 7/8 startups.
    import app.gold.ingest as gi
    from app.data.ingest_rest import upsert_candles

    created = []

    class _Ex:
        enableRateLimit = False
        closed = 0

        def __init__(self):
            created.append(self)

        async def fetch_ohlcv(self, symbol, timeframe, since, limit):
            return []

        async def close(self):
            self.closed += 1

    monkeypatch.setattr(gi.ccxt_async, "binance", _Ex)
    t0 = 1_700_000_000 - 1_700_000_000 % 3600  # before XAUT's Delta launch
    times = [t0, t0 + 2 * 3600, t0 + 4 * 3600, t0 + 6 * 3600]  # three one-bar holes
    await upsert_candles(gi.GOLD_SPOT_SYMBOL, "1h",
                         [{"time": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0} for t in times])

    await gi.gold_repair_gaps()
    assert len(created) == 1 and created[0].closed == 1


def test_gold_binance_client_loads_spot_markets_only():
    import app.gold.ingest as gi

    ex = gi._new_binance()
    try:
        assert ex.options["fetchMarkets"]["types"] == ["spot"]
        assert ex.enableRateLimit is True
    finally:
        asyncio.run(ex.close())


# ------------------------------------------------ logging volume

def test_apscheduler_per_run_info_lines_are_not_logged():
    import app.main  # noqa: F401  (configures logging on import)

    lg = logging.getLogger("apscheduler.executors.default")
    assert not lg.isEnabledFor(logging.INFO)
    assert lg.isEnabledFor(logging.WARNING)  # missed/skipped runs still show
    assert lg.isEnabledFor(logging.ERROR)  # job exceptions still show


# ------------------------------------------------ W4: shared live balance cache

class _CountingDelta(_FakeDelta):
    def __init__(self, delay=0.0, **kw):
        super().__init__(**kw)
        self.balance_calls = 0
        self.delay = delay

    async def get_balances(self):
        self.balance_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return await super().get_balances()


@pytest.mark.asyncio
async def test_balance_is_shared_for_a_few_seconds_then_refetched(monkeypatch):
    import app.execution.live_broker as lb

    clock = [1000.0]
    monkeypatch.setattr(lb.time, "monotonic", lambda: clock[0])
    fake = _CountingDelta()
    broker = _broker(fake)
    for _ in range(5):
        assert await broker.get_equity() == pytest.approx(3.67)
    assert fake.balance_calls == 1
    fake.balance = 5.0
    clock[0] += lb.BALANCE_CACHE_TTL_S - 0.1
    assert await broker.get_equity() == pytest.approx(3.67)  # still inside the TTL
    clock[0] += 0.2
    assert await broker.get_equity() == pytest.approx(5.0)  # never older than the TTL
    assert fake.balance_calls == 2
    assert lb.BALANCE_CACHE_TTL_S <= 5.0


@pytest.mark.asyncio
async def test_concurrent_equity_reads_share_one_balance_request():
    fake = _CountingDelta(delay=0.05)
    broker = _broker(fake)
    values = await asyncio.gather(*(broker.get_equity() for _ in range(6)))
    assert values == [pytest.approx(3.67)] * 6 and fake.balance_calls == 1


@pytest.mark.asyncio
async def test_an_order_invalidates_the_cached_balance():
    fake = _CountingDelta()
    broker = _broker(fake)
    await broker.get_equity()
    fake.balance = 3.50  # the fill paid fees
    await broker.place_order(OrderRequest(symbol="ETHUSD", side="buy", size=1, strategy="consolidated", leverage=5))
    assert await broker.get_equity() != pytest.approx(3.67)
    assert fake.balance_calls == 2


@pytest.mark.asyncio
async def test_a_read_in_flight_during_an_order_is_not_cached(monkeypatch):
    # A balance read that started BEFORE a fill must not be served as fresh after it.
    fake = _CountingDelta(delay=0.05)
    broker = _broker(fake)
    first = asyncio.ensure_future(broker.get_equity())
    await asyncio.sleep(0.01)
    broker.invalidate_balance_cache()  # what place_order / the sync do on a fill
    fake.balance = 3.0
    await first
    assert await broker.get_equity() == pytest.approx(3.0)
    assert fake.balance_calls == 2


@pytest.mark.asyncio
async def test_a_fill_seen_by_the_sync_invalidates_the_cached_balance():
    fake = _CountingDelta()
    broker = _broker(fake)
    await broker.get_equity()
    fake.remote_positions = [{"size": 1, "entry_price": "2750", "margin": "3.06",
                              "liquidation_price": "2480", "realized_pnl": "0"}]
    await broker.sync_fills_and_positions()  # e.g. a bracket or manual fill on Delta
    await broker.get_equity()
    assert fake.balance_calls == 2


# ------------------------------------------------ W5: analyst gate reasons

@pytest.mark.asyncio
async def test_open_best_trade_reports_the_gate_for_every_coin():
    from app.analyst import execution as ex
    from app.data.app_state import set_kill_switch, set_trading_mode
    from tests.test_analyst_execution import _OM, _call, _runtime

    await set_trading_mode("live")
    await set_kill_switch(False)
    held = {"BNBUSD:consolidated": PositionView("BNBUSD", 1, 600, 0, 0, strategy="consolidated")}
    results = {
        "ETHUSD": _call("ETHUSD", "BUY", 0.6),
        "SOLUSD": _call("SOLUSD", "BUY", 0.15),  # below the entry score floor (analyst_min_score)
        "BTCUSD": _call("BTCUSD", "SELL", -0.7, 85000, 88000, 82000),  # book is long BTC
        "BNBUSD": _call("BNBUSD", "BUY", 0.65),  # ranks above ETH, but the book already holds BNB
        "ADAUSD": {**_call("ADAUSD", "WAIT", 0.1)},
        "XLMUSD": _call("XLMUSD", "BUY", 0.45),  # eligible but ETH ranks first
    }
    rt = _runtime(_OM(positions=held), {"ETHUSD": 0.4, "BTCUSD": 0.5, "BNBUSD": 0.5, "XLMUSD": 0.5})
    gates = {}
    trade = await ex.open_best_trade(rt, results, gates=gates)
    assert trade.symbol == "ETHUSD"
    assert gates["ETHUSD"] == "OPENED"
    assert gates["SOLUSD"].startswith("score<")
    assert gates["BTCUSD"] == "book_disagrees"
    assert gates["BNBUSD"] == "book_holds_coin"
    assert gates["ADAUSD"] == "wait"
    assert gates["XLMUSD"] == "outranked"

    gates2 = {}
    assert await ex.open_best_trade(rt, results, gates=gates2) is None  # ETH trade is open now
    assert gates2["XLMUSD"] == "trade_open" and gates2["ADAUSD"] == "wait"

    await set_kill_switch(True)
    gates3 = {}
    await ex.open_best_trade(rt, {"XLMUSD": results["XLMUSD"]}, gates=gates3)
    assert gates3 == {"XLMUSD": "kill_switch"}
    await set_kill_switch(False)


@pytest.mark.asyncio
async def test_analyst_run_logs_one_info_summary_with_gates(monkeypatch, caplog):
    import types

    from app.analyst import execution as ex
    from app.analyst import service as svc

    async def fake_open(runtime, results, *, gates=None, **kw):
        gates.update({"ETHUSD": "book_disagrees"})
        return None

    async def fake_inputs(runtime, symbol):
        return object()

    call = {"symbol": "ETHUSD", "decision": "BUY", "conviction": 6, "factors": []}
    monkeypatch.setattr(ex, "open_best_trade", fake_open)
    monkeypatch.setattr(svc, "analyst_symbols", lambda rt: ["ETHUSD"])
    monkeypatch.setattr(svc, "gather_inputs", fake_inputs)
    monkeypatch.setattr(svc, "analyse", lambda symbol, inp: types.SimpleNamespace(to_dict=lambda: dict(call)))
    monkeypatch.setattr(svc, "_to_live_price", lambda r, price: None)
    monkeypatch.setattr(svc, "live_flow_readings", lambda rt, symbol: [])
    monkeypatch.setattr(svc, "_stable_refreshed", {"at": 1e18})
    with caplog.at_level(logging.INFO, logger="app.analyst.service"):
        await svc.run_analysis(types.SimpleNamespace())
    lines = [r for r in caplog.records if r.getMessage().startswith("Analyst: ")]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "ETHUSD BUY 6/10 book_disagrees" in lines[0].getMessage()


@pytest.mark.asyncio
async def test_risk_supervisor_reads_bypass_the_balance_cache():
    # Balance THEN deposit list: a cached pre-deposit balance paired with a
    # fresh list would read the deposit as a loss and trip the daily-loss halt.
    from app.execution.live_broker import fresh_balance

    fake = _CountingDelta()
    broker = _broker(fake)
    await broker.get_equity()
    fake.balance = 15.0  # a deposit landed
    with fresh_balance():
        assert await broker.get_equity() == pytest.approx(15.0)
    assert fake.balance_calls == 2


@pytest.mark.asyncio
async def test_external_flows_are_not_served_from_a_stale_cache():
    from app.execution.live_broker import LiveBroker

    rows = []

    class _Client:
        async def request(self, method, path, params=None, auth=False):
            return list(rows)

    broker = LiveBroker.__new__(LiveBroker)
    broker.client = _Client()
    broker._flows_cache = None
    assert await broker.external_flows() == []
    rows.append({"transaction_type": "deposit", "amount": "11.76", "asset_symbol": "USD",
                 "created_at": "2026-10-06T07:00:00Z", "uuid": "dep-1"})
    assert [f.uid for f in await broker.external_flows()] == ["dep-1"]


# ------------------------------------------------ #8: market order fill price

class _FillDelta(_FakeDelta):
    def __init__(self, response, **kw):
        super().__init__(**kw)
        self.response = response

    async def place_order(self, body):
        await super().place_order(body)
        return dict(self.response)


@pytest.mark.asyncio
async def test_market_order_reports_the_average_fill_not_the_protection_limit():
    # Delta returns a market order's protection bound in limit_price; the fill
    # is average_fill_price.
    fake = _FillDelta({"id": 7, "state": "closed", "unfilled_size": 0,
                       "limit_price": "2887.50", "average_fill_price": "2750.25"})
    result = await _broker(fake).place_order(
        OrderRequest(symbol="ETHUSD", side="buy", size=1, strategy="consolidated", leverage=5))
    assert result.avg_fill_price == pytest.approx(2750.25)


@pytest.mark.asyncio
async def test_fill_price_falls_back_to_limit_price_when_no_average_is_given():
    for missing in ({}, {"average_fill_price": None}, {"average_fill_price": ""}):
        fake = _FillDelta({"id": 7, "state": "closed", "unfilled_size": 0, "limit_price": "2751", **missing})
        result = await _broker(fake).place_order(
            OrderRequest(symbol="ETHUSD", side="buy", size=1, strategy="consolidated", leverage=5))
        assert result.avg_fill_price == pytest.approx(2751.0)
