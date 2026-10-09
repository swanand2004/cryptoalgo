"""Analyst shadow variants (cycle 8): C (no entry at a top-third ATR rank), B
(SELL only at score <= -0.40), C+B and the "base" control run as PAPER books
next to the live Analyst. They must never send an order, never block or alter
the live trade, never count in the live stats, and do nothing when the setting
is off."""
import types

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, delete, select, text

from app.analyst import execution as ex
from app.analyst import service as svc
from app.config import Settings
from app.data.app_state import set_kill_switch, set_trading_mode
from app.data.db import SessionLocal, _add_missing_columns
from app.data.models import AnalystTrade
from app.execution.broker_base import OrderResult, PositionView


def _s(**kw):
    kw.setdefault("analyst_risk_cap", 0)
    kw.setdefault("analyst_min_liq_to_stop", 0)
    kw.setdefault("analyst_target_leverage", 0)
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


def _call(sym, decision, score, entry=2700.0, stop=2550.0, tp=2850.0):
    return {"symbol": sym, "decision": decision, "conviction": int(round(abs(score) * 10)), "score": score,
            "size_factor": 1.0, "entry": entry, "stop_loss": stop, "take_profit": tp, "families": {}, "notes": [],
            "factors": [], "atr": 100.0, "time": "2026-10-08T00:00:00+00:00"}


class _OM:
    def __init__(self):
        self.positions, self.orders, self.closed = {}, [], []

    async def get_equity(self):
        return 14.0

    async def get_positions(self):
        return self.positions

    async def submit(self, req):
        self.orders.append(req)
        self.positions[f"{req.symbol}:consolidated"] = PositionView(req.symbol, req.size, 2701.0, 0, 0, strategy="consolidated")
        return OrderResult(1, "x", "closed", req.size, 2701.0, 0.0)

    async def close_position(self, symbol, strategy):
        self.closed.append(symbol)
        self.positions.pop(f"{symbol}:{strategy}", None)
        return OrderResult(2, "y", "closed", 1, 2600.0, 0.0)


class _NoOM:
    """Any use of the order manager fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"shadow variants touched order_manager.{name}")


def _runtime(om, signals, shadow=True, engine="combined", prices=None):
    reg = types.SimpleNamespace(get=lambda s: types.SimpleNamespace(contract_value=0.01 if s == "ETHUSD" else 0.001))
    prices = prices if prices is not None else {"ETHUSD": 2700.0, "BTCUSD": 85000.0, "SOLUSD": 150.0}
    return types.SimpleNamespace(settings=_s(trading_engine=engine, analyst_shadow_variants=shadow), order_manager=om,
                                 registry=reg, live_signals=signals,
                                 cache=types.SimpleNamespace(get_last_price=prices.get), _prices=prices)


async def _rows():
    async with SessionLocal() as s:
        return (await s.execute(select(AnalystTrade).order_by(AnalystTrade.id))).scalars().all()


# ------------------------------------------------------------- setting + migration
def test_setting_defaults_on():
    assert _s().analyst_shadow_variants is True
    assert _s(analyst_shadow_variants=False).analyst_shadow_variants is False


def test_migration_adds_the_variant_column(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as conn:  # an analyst_trades table from before the column existed
        conn.execute(text("CREATE TABLE analyst_trades (id INTEGER PRIMARY KEY, symbol VARCHAR(32), side VARCHAR(8))"))
        conn.execute(text("INSERT INTO analyst_trades (symbol, side) VALUES ('ETHUSD', 'BUY')"))
        _add_missing_columns(conn)
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(analyst_trades)"))}
        assert "variant" in cols
        assert conn.execute(text("SELECT variant FROM analyst_trades")).scalar() is None  # old rows = live records
        _add_missing_columns(conn)  # idempotent
    eng.dispose()


# ------------------------------------------------------------- gates
def test_variant_gates():
    buy, weak_sell, strong_sell = _call("ETHUSD", "BUY", 0.25), _call("ETHUSD", "SELL", -0.30), _call("ETHUSD", "SELL", -0.45)
    for r in (buy, weak_sell, strong_sell):
        assert ex.variant_gate("base", r, 0.99) is None  # the control has no extra rule
    # C: skip only when the rank is in the top third (> 2/3); unknown rank = no gate
    assert ex.variant_gate("C", buy, 0.70) == "vol_top_third"
    assert ex.variant_gate("C", buy, 0.66) is None and ex.variant_gate("C", buy, None) is None
    assert ex.variant_gate("C", weak_sell, 0.10) is None
    # B: SELL only at score <= -0.40; BUYs untouched
    assert ex.variant_gate("B", weak_sell, None) == "weak_short"
    assert ex.variant_gate("B", strong_sell, 0.99) is None and ex.variant_gate("B", buy, 0.99) is None
    assert ex.variant_gate("B", _call("ETHUSD", "SELL", -0.40), None) is None
    # C+B: both
    assert ex.variant_gate("CB", weak_sell, 0.1) == "weak_short"
    assert ex.variant_gate("CB", strong_sell, 0.9) == "vol_top_third"
    assert ex.variant_gate("CB", strong_sell, 0.5) is None


def _hourly(days, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=24 * days, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close}, index=idx)


def test_atr_rank_uses_past_data_only():
    h = _hourly(300)
    assert ex.atr_rank(h.iloc[: 24 * 100]) is None  # < 120 days: no gate
    cut = 24 * 200
    base = ex.atr_rank(h.iloc[:cut])
    assert base is not None and 0.0 <= base <= 1.0
    future = h.copy()
    future.iloc[cut:, future.columns.get_loc("high")] *= 1.5  # wild bars AFTER the decision time
    assert ex.atr_rank(future.iloc[:cut]) == base  # ... cannot change it
    spike = h.iloc[:cut].copy()
    spike.iloc[-48:, spike.columns.get_loc("high")] *= 1.08  # a volatility spike in the last two days
    assert ex.atr_rank(spike) > 2 / 3
    assert ex.atr_rank(None) is None  # bad input: no rank, never an error


# ------------------------------------------------------------- paper books, no orders
@pytest.mark.asyncio
async def test_variants_open_paper_books_and_never_call_the_order_manager():
    await set_trading_mode("live")
    await set_kill_switch(False)
    rt = _runtime(_NoOM(), {"ETHUSD": 0.4, "SOLUSD": -0.4, "BTCUSD": -0.4})
    results = {
        "ETHUSD": _call("ETHUSD", "BUY", 0.6),
        "SOLUSD": _call("SOLUSD", "SELL", -0.30, 150.0, 156.0, 144.0),
        "BTCUSD": _call("BTCUSD", "BUY", 0.6, 85000, 81850, 88150),  # the book is short BTC: no entry anywhere
    }
    new = await ex.open_variant_trades(rt, results, {"ETHUSD": 0.9, "SOLUSD": 0.2})
    got = sorted((t.symbol, t.variant) for t in new)
    assert got == [("ETHUSD", "B"), ("ETHUSD", "base"), ("SOLUSD", "C"), ("SOLUSD", "base")]
    assert all(t.contracts == 0 and not t.live and t.initial_stop == t.stop_loss for t in new)
    # one trade per coin per variant: the next run adds nothing while they are open
    assert await ex.open_variant_trades(rt, results, {"ETHUSD": 0.9, "SOLUSD": 0.2}) == []
    # exits: the live update_exit on a tick through the stop, on paper
    rt._prices["ETHUSD"] = 2540.0
    await ex.manage_open_trades(rt, {})
    eth = [t for t in await _rows() if t.symbol == "ETHUSD"]
    assert all(not t.is_open and t.result == "SL" and t.exit_price == 2540.0 for t in eth)
    assert all(t.r_multiple < -1.0 for t in eth)


@pytest.mark.asyncio
async def test_kill_switch_and_setting_off_mean_no_variant_trades():
    await set_trading_mode("paper")
    results = {"ETHUSD": _call("ETHUSD", "BUY", 0.6)}
    await set_kill_switch(True)
    assert await ex.open_variant_trades(_runtime(_NoOM(), {"ETHUSD": 0.4}), results, {}) == []
    await set_kill_switch(False)
    assert await ex.open_variant_trades(_runtime(_NoOM(), {"ETHUSD": 0.4}, shadow=False), results, {}) == []
    assert await _rows() == []


# ------------------------------------------------------------- the live trade is untouched
async def _live_scenario(shadow: bool):
    """Live ETH trade opened, variants run, price crosses the stop. A leftover
    OPEN variant row on BTC exists in both runs (must never block the live trade)."""
    async with SessionLocal() as s:
        await s.execute(delete(AnalystTrade))
        s.add(AnalystTrade(symbol="BTCUSD", side="BUY", conviction=6, score=0.6, entry=85000.0, stop_loss=81850.0,
                           take_profit=88150.0, initial_stop=81850.0, atr=2100.0, peak=85000.0, contracts=0,
                           variant="C"))
        await s.commit()
    await set_trading_mode("live")
    await set_kill_switch(False)
    om = _OM()
    rt = _runtime(om, {"ETHUSD": 0.4}, shadow=shadow)
    results = {"ETHUSD": _call("ETHUSD", "BUY", 0.6)}
    live = await ex.open_best_trade(rt, results)
    assert live is not None
    if ex.shadow_on(rt.settings):
        await ex.open_variant_trades(rt, results, {"ETHUSD": 0.1})
    rt._prices["ETHUSD"] = 2540.0
    await ex.manage_open_trades(rt, {})
    rows = await _rows()
    real = [t for t in rows if t.variant is None]
    assert len(real) == 1
    t = real[0]
    live_view = (t.symbol, t.side, t.entry, t.stop_loss, t.initial_stop, t.take_profit, t.contracts, t.live,
                 t.is_open, t.result, t.exit_price, round(t.r_multiple, 9))
    calls = ([(o.symbol, o.side, o.size, o.leverage) for o in om.orders], list(om.closed))
    return live_view, calls, [(v.symbol, v.variant, v.is_open, v.result) for v in rows if v.variant is not None]


@pytest.mark.asyncio
async def test_live_behaviour_is_identical_with_the_setting_on_or_off():
    on_view, on_calls, on_shadow = await _live_scenario(True)
    off_view, off_calls, off_shadow = await _live_scenario(False)
    assert on_view == off_view and on_calls == off_calls
    assert on_view[9] == "SL" and on_view[10] == 2600.0  # closed on the exchange at the fill, as before
    assert on_calls == ([("ETHUSD", "buy", 1, 3.0)], ["ETHUSD"])  # one order in, one close: the live trade only
    # on: the ETH variant books opened and closed on paper; off: nothing done, the leftover row untouched
    assert sorted(on_shadow) == sorted([("BTCUSD", "C", True, None)] + [
        ("ETHUSD", v, False, "SL") for v in ("base", "C", "B", "CB")])
    assert off_shadow == [("BTCUSD", "C", True, None)]


@pytest.mark.asyncio
async def test_a_failing_shadow_step_never_breaks_the_live_run(monkeypatch):
    monkeypatch.setattr(svc, "_latest", {})
    monkeypatch.setattr(svc, "_latest_gates", {})
    monkeypatch.setattr(svc, "_stable_refreshed", {"at": 1e18})
    call = _call("ETHUSD", "BUY", 0.6)

    async def fake_inputs(runtime, symbol):
        return object()

    opened = {}

    async def fake_open(runtime, results, *, gates=None, **kw):
        opened.update(results)

    async def boom(*a, **k):
        raise RuntimeError("shadow failure")

    monkeypatch.setattr(svc, "analyst_symbols", lambda rt: ["ETHUSD"])
    monkeypatch.setattr(svc, "gather_inputs", fake_inputs)
    monkeypatch.setattr(svc, "analyse", lambda symbol, inp: types.SimpleNamespace(to_dict=lambda: dict(call)))
    monkeypatch.setattr(svc, "_to_live_price", lambda r, price: None)
    monkeypatch.setattr(svc, "live_flow_readings", lambda rt, symbol: [])
    monkeypatch.setattr(ex, "open_best_trade", fake_open)
    monkeypatch.setattr(ex, "open_variant_trades", boom)
    res = await svc.run_analysis(_runtime(_NoOM(), {"ETHUSD": 0.4}))
    assert res["ETHUSD"] == call and opened["ETHUSD"] == call


# ------------------------------------------------------------- stats
@pytest.mark.asyncio
async def test_snapshot_keeps_variants_out_of_live_stats_and_reports_the_race(monkeypatch):
    monkeypatch.setattr(svc, "_latest", {})
    import datetime as dt

    t0 = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)

    def row(variant, r, minutes, is_open=False):
        return AnalystTrade(symbol="ETHUSD", side="BUY", conviction=6, score=0.6, entry=2700.0, stop_loss=2550.0,
                            take_profit=2850.0, initial_stop=2550.0, contracts=0, variant=variant, is_open=is_open,
                            r_multiple=None if is_open else r, result=None if is_open else "TP",
                            entry_time=t0 + dt.timedelta(minutes=minutes))

    async with SessionLocal() as s:
        s.add_all([row(None, -1.0, -600),  # live, before the race started
                   row(None, 0.9, 10), row("base", 0.9, 0), row("base", -1.0, 5), row("C", 0.9, 0),
                   row("B", 0.0, 0, is_open=True)])
        await s.commit()
    rt = _runtime(_NoOM(), {})
    snap = await svc.analyst_snapshot(rt)
    assert snap["stats"]["closed"] == 2 and snap["stats"]["total_r"] == pytest.approx(-0.1)  # live rows only
    assert all(r["live"] is False for r in snap["closed_trades"]) and len(snap["closed_trades"]) == 2
    assert snap["open_trades"] == []  # the open B row is not a live trade
    race = {r["key"]: r for r in snap["variants"]["rows"]}
    assert snap["variants"]["enabled"] and snap["variants"]["since"].startswith("2026-10-08T00:00:00")
    assert list(race) == ["live", "base", "C", "B", "CB"]
    assert race["live"]["closed"] == 1 and race["live"]["total_r"] == pytest.approx(0.9)  # same span only
    assert race["base"]["closed"] == 2 and race["base"]["win_rate"] == 0.5 and race["base"]["avg_r"] == pytest.approx(-0.05)
    assert race["C"]["closed"] == 1 and race["B"]["closed"] == 0 and race["B"]["open"] == 1
    assert race["CB"]["closed"] == 0 and race["CB"]["avg_r"] is None
    off = await svc.analyst_snapshot(_runtime(_NoOM(), {}, shadow=False))
    assert off["variants"] == {"enabled": False, "since": None, "rows": []} and off["stats"] == snap["stats"]
