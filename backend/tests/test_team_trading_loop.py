"""Team test agent, cycle 1: the trend book's interplay with the Analyst and the
size-aware universe (app/trading_loop.py). The book must never resize or close a
coin the Analyst holds for real; tests marked xfail(strict=True) pin real bugs."""
import datetime as dt
import types

import pandas as pd
import pytest

from app import trading_loop as tl
from app.config import Settings
from app.data.app_state import set_kill_switch, set_trading_mode
from app.data.db import SessionLocal
from app.data.models import AnalystTrade

UTC = dt.timezone.utc
SYMS = ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"]


def _s(**kw):
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


async def _analyst_trade(symbol, contracts=1, is_open=True):
    async with SessionLocal() as s:
        s.add(AnalystTrade(symbol=symbol, side="BUY", conviction=7, score=0.7, size_factor=1.0, entry=100.0,
                           stop_loss=97.0, initial_stop=97.0, take_profit=103.0, contracts=contracts,
                           live=contracts > 0, is_open=is_open, entry_time=dt.datetime.now(UTC), reasons={}))
        await s.commit()


# ================================================================ _analyst_held_symbols
@pytest.mark.asyncio
async def test_analyst_held_is_only_open_real_positions_and_only_when_the_analyst_executes():
    await _analyst_trade("ETHUSD", contracts=1)
    await _analyst_trade("BTCUSD", contracts=0)  # paper record: the book may trade BTC
    await _analyst_trade("SOLUSD", contracts=2, is_open=False)  # closed
    for engine in ("combined", "analyst"):
        rt = types.SimpleNamespace(settings=_s(trading_engine=engine))
        assert await tl._analyst_held_symbols(rt) == {"ETHUSD"}
    assert await tl._analyst_held_symbols(types.SimpleNamespace(settings=_s(trading_engine="book"))) == set()


# ================================================================ run_trading_tick wiring
def _frame():
    return pd.DataFrame({"time": [0], "close": [100.0]})


def _tick_runtime(engine="combined", live_signals=None):
    return types.SimpleNamespace(
        settings=_s(trading_engine=engine), registry=types.SimpleNamespace(symbols=list(SYMS)),
        live_signals=live_signals if live_signals is not None else {},
    )


@pytest.fixture
def patched_tick(monkeypatch):
    """Replaces the data/sizing halves of run_trading_tick so only its routing
    (who is ticked, who is closed, who is skipped) is under test."""
    calls = {"tick": [], "close": []}
    state = {"active": set(SYMS), "frames": {s: _frame() for s in SYMS},
             "signals": {s: pd.Series([0.5]) for s in SYMS}}

    async def build(runtime):
        return dict(state["frames"]), dict(state["signals"])

    async def active(runtime, frames):
        return state["active"]

    async def tick(runtime, symbol, **kw):
        calls["tick"].append(symbol)

    async def close(runtime, symbol):
        calls["close"].append(symbol)

    monkeypatch.setattr(tl, "_build_ladder_frames", build)
    monkeypatch.setattr(tl, "_active_universe", active)
    monkeypatch.setattr(tl, "_tick_symbol", tick)
    monkeypatch.setattr(tl, "_close_inactive", close)
    monkeypatch.setattr(tl, "compute_book_scalar", lambda frames, signals, **kw: (1.0, 0.5))
    monkeypatch.setattr(tl, "symbol_weights", lambda frames: {s: 1.0 / len(frames) for s in frames})
    return calls, state


@pytest.mark.asyncio
async def test_the_book_never_ticks_a_coin_the_analyst_holds(patched_tick):
    calls, _ = patched_tick
    await set_kill_switch(False)
    await _analyst_trade("ETHUSD", contracts=1)
    await tl.run_trading_tick(_tick_runtime("combined"))
    assert "ETHUSD" not in calls["tick"] and sorted(calls["tick"]) == ["BNBUSD", "BTCUSD", "SOLUSD"]
    assert calls["close"] == []


@pytest.mark.asyncio
async def test_the_book_never_closes_an_analyst_coin_outside_todays_universe(patched_tick):
    calls, state = patched_tick
    await set_kill_switch(False)
    state["active"] = {"BTCUSD"}
    await _analyst_trade("ETHUSD", contracts=1)
    await tl.run_trading_tick(_tick_runtime("combined"))
    assert calls["tick"] == ["BTCUSD"]
    assert sorted(calls["close"]) == ["BNBUSD", "SOLUSD"]  # ETH (the Analyst's) is left alone


@pytest.mark.asyncio
async def test_under_the_book_engine_an_analyst_paper_record_does_not_block_the_book(patched_tick):
    calls, _ = patched_tick
    await set_kill_switch(False)
    await _analyst_trade("ETHUSD", contracts=0)
    await tl.run_trading_tick(_tick_runtime("book"))
    assert "ETHUSD" in calls["tick"]


@pytest.mark.asyncio
async def test_signals_for_every_coin_feed_the_analyst_even_when_the_book_cannot_trade(patched_tick):
    calls, state = patched_tick
    await set_kill_switch(False)
    state["active"] = set()  # nothing affordable today
    sig = {}
    await tl.run_trading_tick(_tick_runtime("combined", sig))
    assert sig == {s: 0.5 for s in SYMS} and calls == {"tick": [], "close": []}


@pytest.mark.asyncio
async def test_kill_switch_stops_the_tick_before_any_routing(patched_tick):
    calls, _ = patched_tick
    await set_kill_switch(True)
    try:
        await tl.run_trading_tick(_tick_runtime("combined"))
    finally:
        await set_kill_switch(False)
    assert calls == {"tick": [], "close": []}


@pytest.mark.asyncio
async def test_analyst_engine_means_the_book_sends_no_orders(patched_tick):
    calls, state = patched_tick
    await set_kill_switch(False)
    state["active"] = {"BTCUSD", "ETHUSD"}
    sig = {}
    await tl.run_trading_tick(_tick_runtime("analyst", sig))
    assert sig  # the signals are still computed for the Analyst
    assert calls == {"tick": [], "close": []}


@pytest.mark.asyncio
async def test_a_coin_with_a_stale_frame_does_not_keep_its_old_signal(patched_tick):
    calls, state = patched_tick
    await set_kill_switch(False)
    sig = {"ETHUSD": 0.5}  # yesterday's value
    del state["frames"]["ETHUSD"], state["signals"]["ETHUSD"]  # today ETH's frame is stale -> dropped
    await tl.run_trading_tick(_tick_runtime("combined", sig))
    assert "ETHUSD" not in sig or sig["ETHUSD"] is None


# ================================================================ _active_universe
def _universe_runtime(equity, prices, core="BTCUSD,ETHUSD"):
    settings = _s(symbols="BTCUSD,ETHUSD,ADAUSD", core_symbols=core)
    cvs = {"BTCUSD": 0.001, "ETHUSD": 0.01, "ADAUSD": 1.0}
    reg = types.SimpleNamespace(symbols=["BTCUSD", "ETHUSD", "ADAUSD"],
                                get=lambda s: types.SimpleNamespace(contract_value=cvs[s]))
    box = {"equity": equity}

    async def get_equity():
        return box["equity"]

    return types.SimpleNamespace(settings=settings, registry=reg, cache=types.SimpleNamespace(get_last_price=prices.get),
                                 order_manager=types.SimpleNamespace(get_equity=get_equity)), box


@pytest.mark.asyncio
async def test_universe_unknown_core_price_is_none_and_is_not_cached():
    await set_trading_mode("live")
    rt, _ = _universe_runtime(100.0, {"ETHUSD": 2700.0})  # no BTC price anywhere
    assert await tl._active_universe(rt, {}) is None
    assert getattr(rt, "_universe_cache", None) is None
    # the ladder frame's close is the fallback price
    frames = {"BTCUSD": pd.DataFrame({"close": [85000.0]})}
    assert await tl._active_universe(rt, frames) == {"ETHUSD", "ADAUSD"}  # $85 BTC > $50; ETH $27 fits


@pytest.mark.asyncio
async def test_universe_zero_equity_is_none():
    await set_trading_mode("live")
    rt, _ = _universe_runtime(0.0, {"ETHUSD": 2700.0, "BTCUSD": 85000.0})
    assert await tl._active_universe(rt, {}) is None


@pytest.mark.asyncio
async def test_universe_is_re_decided_when_the_mode_changes():
    await set_trading_mode("live")
    rt, box = _universe_runtime(14.0, {"ETHUSD": 2700.0, "BTCUSD": 85000.0})
    assert await tl._active_universe(rt, {}) == {"ADAUSD"}
    box["equity"] = 5000.0
    assert await tl._active_universe(rt, {}) == {"ADAUSD"}  # same day, same mode: sticky
    await set_trading_mode("paper")
    assert await tl._active_universe(rt, {}) == {"BTCUSD", "ETHUSD"}  # paper account: decided afresh


# ================================================================ book_margin_leverage
def test_book_leverage_is_capped_at_max_leverage_and_ignores_empty_signals():
    s = _s(max_leverage=10.0)
    signals = {"A": pd.Series([1.0]), "B": pd.Series([], dtype=float)}
    assert tl.book_margin_leverage(s, signals, {"A": 1.0, "B": 1.0}, 50.0) == 10.0  # 50/0.85 = 59 -> capped
    assert tl.book_margin_leverage(s, {}, {}, 3.0) == 1.0  # flat book: 1x, never 0
    assert tl.book_margin_leverage(s, {"A": pd.Series([0.5])}, {}, 3.0) == 1.0  # unweighted leg counts 0


def test_book_leverage_counts_shorts_by_magnitude_and_rounds_up():
    s = _s()
    signals = {"A": pd.Series([0.2, -0.9]), "B": pd.Series([0.3])}
    # gross = 0.9*0.5 + 0.3*0.5 = 0.6; x k 2.0 = 1.2 / 0.85 = 1.41 -> 2x
    assert tl.book_margin_leverage(s, signals, {"A": 0.5, "B": 0.5}, 2.0) == 2.0
