import datetime as dt
import types

import pytest

import app.options.s7 as s7
from app.config import Settings
from app.options.s7 import (
    decide,
    entry_price,
    in_entry_window,
    intrinsic,
    next_entry_time,
    option_fee,
    select_contract,
    settle_pnl,
)

UTC = dt.timezone.utc
FRI = dt.datetime(2026, 10, 9, 12, 0, tzinfo=UTC)  # a Friday


def _row(kind, strike, delta, expiry, *, ask=100.0, mark=95.0, spot=85000.0, cv=0.001):
    return {
        "symbol": f"{'C' if kind == 'call' else 'P'}-BTC-{int(strike)}-{expiry:%d%m%y}", "type": kind,
        "strike": float(strike), "expiry": expiry, "delta": delta, "ask": ask, "mark": mark, "bid": None,
        "iv": 0.4, "spot": spot, "contract_value": cv,
    }


# ------------------------------------------------------------------ pure rules

def test_decide_uses_strict_point_two_threshold():
    assert decide(0.21) == "CALL"
    assert decide(-0.21) == "PUT"
    assert decide(0.2) == "NONE" and decide(-0.2) == "NONE" and decide(0.0) == "NONE"
    assert decide(None) == "NONE" and decide(float("nan")) == "NONE"


def test_entry_price_prefers_ask_then_mark():
    assert entry_price({"ask": 120.0, "mark": 100.0}) == (120.0, "ask")
    assert entry_price({"ask": None, "mark": 100.0}) == (100.0, "mark")
    assert entry_price({"ask": None, "mark": 0.0}) == (None, None)


def test_select_contract_picks_seven_day_expiry_and_nearest_040_delta():
    e_tue = FRI + dt.timedelta(days=1)  # too close (< 2 days)
    e_wk = FRI + dt.timedelta(days=7)
    e_2wk = FRI + dt.timedelta(days=14)
    rows = [
        _row("call", 85000, 0.40, e_tue),
        _row("call", 86000, 0.55, e_wk),
        _row("call", 88000, 0.38, e_wk),  # |0.02| -> winner
        _row("call", 87000, 0.44, e_wk),  # |0.04|
        _row("call", 88000, 0.40, e_2wk),  # exact delta but wrong expiry
        _row("put", 82000, -0.41, e_wk),
        _row("put", 81000, -0.30, e_wk),
        _row("put", 83000, -0.40, e_wk, ask=None, mark=0.0),  # no usable price
    ]
    call = select_contract(rows, "CALL", FRI)
    assert call["strike"] == 88000 and call["expiry"] == e_wk
    put = select_contract(rows, "PUT", FRI)
    assert put["strike"] == 82000
    assert select_contract(rows, "NONE", FRI) is None
    assert select_contract([r for r in rows if r["type"] == "put"], "CALL", FRI) is None


def test_select_contract_ignores_rows_without_delta():
    e = FRI + dt.timedelta(days=7)
    rows = [_row("call", 88000, None, e), _row("call", 90000, 0.2, e)]
    assert select_contract(rows, "CALL", FRI)["strike"] == 90000


def test_option_fee_is_min_of_notional_and_premium_cap_plus_gst():
    # notional cap binds: 0.01% of $85 = 0.0085 < 3.5% of $1.5 = 0.0525
    assert option_fee(85.0, 1.5) == pytest.approx(0.0085 * 1.18)
    # premium cap binds for a cheap option: 3.5% of $0.10 = 0.0035 < 0.0085
    assert option_fee(85.0, 0.10) == pytest.approx(0.0035 * 1.18)
    assert option_fee(85.0, 0.0) == 0.0


def test_intrinsic():
    assert intrinsic("call", 85000, 86000) == 1000
    assert intrinsic("CALL", 85000, 84000) == 0
    assert intrinsic("put", 85000, 84000) == 1000
    assert intrinsic("PUT", 85000, 86000) == 0


def test_settle_pnl_itm_charges_settlement_fee():
    cv, prem = 0.001, 1500.0
    entry_fee = option_fee(85000 * cv, prem * cv)
    out = settle_pnl(contract_value=cv, premium=prem, entry_fee=entry_fee, settle_value=3000.0, underlying=88000.0)
    settle_fee = option_fee(88000 * cv, 3000 * cv)
    assert out["settle_fee"] == pytest.approx(settle_fee)
    assert out["pnl_usd"] == pytest.approx(3.0 - settle_fee - 1.5 - entry_fee)
    assert out["pnl_pct"] == pytest.approx(out["pnl_usd"] / (1.5 + entry_fee))


def test_settle_pnl_otm_loses_premium_plus_entry_fee_only():
    out = settle_pnl(contract_value=0.01, premium=60.0, entry_fee=0.02, settle_value=0.0, underlying=2400.0)
    assert out["settle_fee"] == 0.0
    assert out["pnl_usd"] == pytest.approx(-0.62)
    assert out["pnl_pct"] == pytest.approx(-1.0)


def test_entry_window_and_next_entry():
    assert in_entry_window(FRI)
    assert in_entry_window(FRI + dt.timedelta(minutes=59))
    assert not in_entry_window(FRI + dt.timedelta(hours=1))
    assert not in_entry_window(FRI - dt.timedelta(minutes=1))
    assert not in_entry_window(FRI + dt.timedelta(days=1))
    assert next_entry_time(FRI - dt.timedelta(days=3)) == FRI
    assert next_entry_time(FRI + dt.timedelta(minutes=10)) == FRI
    assert next_entry_time(FRI + dt.timedelta(hours=2)) == FRI + dt.timedelta(days=7)


# ------------------------------------------------------------ job (DB) tests

def _chain_for(underlying):
    e = FRI + dt.timedelta(days=7)
    cv = 0.001 if underlying == "BTC" else 0.01
    return [
        _row("call", 88000, 0.39, e, ask=1500.0, cv=cv),
        _row("put", 82000, -0.40, e, ask=None, mark=1400.0, cv=cv),
    ]


@pytest.mark.asyncio
async def test_week_entry_settlement_and_snapshot(monkeypatch):
    calls = {"chain": 0}

    async def fake_chain(underlying, force=False):
        calls["chain"] += 1
        return _chain_for(underlying), 0.0

    monkeypatch.setattr(s7, "get_chain", fake_chain)
    runtime = types.SimpleNamespace(live_signals={"BTCUSD": 0.6, "ETHUSD": 0.05})

    await s7.run_s7_tick(runtime, now=FRI + dt.timedelta(minutes=1))
    await s7.run_s7_tick(runtime, now=FRI + dt.timedelta(minutes=6))  # idempotent within the week
    snap = await s7.s7_snapshot(now=FRI + dt.timedelta(minutes=7))
    rows = {r["underlying"]: r for r in snap["rows"]}
    assert len(snap["rows"]) == 2 and calls["chain"] == 1
    btc, eth = rows["BTC"], rows["ETH"]
    assert btc["decision"] == "CALL" and btc["is_open"] and btc["strike"] == 88000
    assert btc["premium"] == 1500.0 and btc["premium_source"] == "ask" and btc["signal_source"] == "live"
    assert btc["entry_fee"] == pytest.approx(option_fee(85000 * 0.001, 1.5))
    assert eth["decision"] == "NONE" and not eth["is_open"] and eth["symbol"] is None

    # Expiry + 5 min: Delta has published the settlement price -> settles ITM.
    async def fake_product(symbol):
        return {"state": "expired", "settlement_price": "2000.0"}

    monkeypatch.setattr(s7, "fetch_product", fake_product)
    expiry = FRI + dt.timedelta(days=7)
    await s7.run_s7_tick(runtime, now=expiry + dt.timedelta(minutes=2))  # too early: grace period
    assert (await s7.s7_snapshot())["stats"]["open"] == 1
    await s7.run_s7_tick(runtime, now=expiry + dt.timedelta(minutes=5))
    snap = await s7.s7_snapshot(now=expiry + dt.timedelta(minutes=6))
    btc = next(r for r in snap["rows"] if r["underlying"] == "BTC" and r["week"] == "2026-10-09")
    assert not btc["is_open"] and btc["settle_source"] == "delta"
    assert btc["underlying_at_expiry"] == 90000.0 and btc["settle_value"] == 2000.0
    want = settle_pnl(contract_value=0.001, premium=1500.0, entry_fee=btc["entry_fee"],
                      settle_value=2000.0, underlying=90000.0)
    assert btc["pnl_usd"] == pytest.approx(want["pnl_usd"])
    st = snap["stats"]
    assert st["settled"] == 1 and st["wins"] == 1 and st["total_pnl_usd"] == pytest.approx(want["pnl_usd"])
    # The settlement tick was itself a Friday 12:05 -> the next week was entered.
    assert any(r["week"] == "2026-10-16" for r in snap["rows"])


@pytest.mark.asyncio
async def test_settlement_falls_back_to_binance_close_after_two_hours(monkeypatch):
    from app.data.db import SessionLocal
    from app.data.models import Candle

    async def fake_chain(underlying, force=False):
        return _chain_for(underlying), 0.0

    async def no_product(symbol):
        return {"state": "live", "settlement_price": None}

    monkeypatch.setattr(s7, "get_chain", fake_chain)
    monkeypatch.setattr(s7, "fetch_product", no_product)
    runtime = types.SimpleNamespace(live_signals={"BTCUSD": -0.5, "ETHUSD": 0.0})
    await s7.enter_week(runtime, "BTC", FRI)
    expiry = FRI + dt.timedelta(days=7)
    async with SessionLocal() as session:
        session.add(Candle(symbol="BINANCE:BTCUSD", resolution="1h", time=int(expiry.timestamp()) - 3600,
                           open=80000, high=80500, low=79500, close=80000, volume=1))
        await session.commit()
    assert await s7.settle_due(expiry + dt.timedelta(hours=1)) == 0  # Delta not ready, too early for fallback
    assert await s7.settle_due(expiry + dt.timedelta(hours=2)) == 1
    row = (await s7.s7_snapshot())["rows"][0]
    assert row["decision"] == "PUT" and row["premium_source"] == "mark" and row["settle_source"] == "binance_1h"
    assert row["settle_value"] == 2000.0 and row["underlying_at_expiry"] == 80000.0


@pytest.mark.asyncio
async def test_entry_outside_window_does_nothing_and_missing_signal_retries(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("chain must not be read")

    monkeypatch.setattr(s7, "get_chain", boom)
    runtime = types.SimpleNamespace(live_signals={"BTCUSD": 0.9})
    await s7.run_s7_tick(runtime, now=FRI - dt.timedelta(hours=1))
    assert (await s7.s7_snapshot())["rows"] == []

    async def no_signal(rt, u):
        return None, None

    monkeypatch.setattr(s7, "read_signal", no_signal)
    assert await s7.enter_week(runtime, "BTC", FRI) is None
    assert (await s7.s7_snapshot())["rows"] == []


def test_scheduler_registers_the_options_job():
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
    ids = {job.id for job in build_scheduler(runtime).get_jobs()}
    assert "options_s7" in ids


def test_s7_package_never_touches_orders_or_private_endpoints():
    import inspect

    import app.options.chain as chain_mod

    src = inspect.getsource(s7) + inspect.getsource(chain_mod)
    for forbidden in ("place_order", "cancel_order", "DeltaRestClient", "order_manager", "api_key", "/v2/orders"):
        assert forbidden not in src
