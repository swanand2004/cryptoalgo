"""Regression (2026-10-07): the Analyst must never treat a partially stored
hour as closed. The Binance 1h refresh stores the forming hour every ~15 min,
so just after an hour ends the newest stored row can be that hour with only
its first minutes in it. The old cache (keyed on the newest row's time only)
also kept serving a frame read before later refreshes rewrote that row."""
from __future__ import annotations

import pandas as pd

from app.analyst import service as svc
from tests.test_analyst_live_bar import _DB, _hourly, _minutes, _runtime


def _setup(monkeypatch, *, forming_row: bool):
    h = _hourly(drift=0.0006)
    start = pd.Timestamp.now(tz="UTC").floor("h")
    newest = start if forming_row else start - pd.Timedelta(hours=1)
    h.index = h.index + (newest - h.index[-1])
    minutes = _minutes(start, max(1, int((pd.Timestamp.now(tz="UTC") - start).total_seconds() // 60) + 1),
                       float(h["close"].iloc[-2]), 0.0)
    db = _DB(h, minutes)
    monkeypatch.setattr(svc, "load_candles_df", db.load)
    monkeypatch.setattr(svc, "_hourly_cache", {})
    monkeypatch.setattr(svc, "_live_calls", {})
    return h, db


def test_final_bars_steady_state_drops_only_the_forming_hour():
    h = _hourly()
    now = h.index[-1] + pd.Timedelta(minutes=20)
    out, pending = svc.final_bars(h, now)
    assert not pending
    pd.testing.assert_frame_equal(out, h.iloc[:-1], check_freq=False)


def test_final_bars_holds_back_a_past_hour_no_refresh_has_finalised():
    h = _hourly()
    now = h.index[-1] + pd.Timedelta(hours=1, minutes=3)  # the newest row's hour ended 3 min ago
    out, pending = svc.final_bars(h, now)
    assert pending
    pd.testing.assert_frame_equal(out, h.iloc[:-1], check_freq=False)


async def test_closed_mode_skips_the_partial_previous_hour(monkeypatch):
    h, _ = _setup(monkeypatch, forming_row=False)  # no refresh since the hour ended
    inp = await svc.gather_inputs(_runtime(False), "ETHUSD")
    # the old code analysed h.iloc[-1] (stored mid-hour) as if it had closed
    assert inp.hourly.index[-1] == h.index[-2]
    pd.testing.assert_frame_equal(inp.hourly, h.iloc[:-1][["open", "high", "low", "close"]],
                                  check_freq=False, check_names=False)


async def test_live_mode_adds_no_forming_bar_after_an_unfinalised_hour(monkeypatch):
    h, db = _setup(monkeypatch, forming_row=False)
    inp = await svc.gather_inputs(_runtime(True), "ETHUSD")
    assert inp.hourly.index[-1] == h.index[-2]  # no gap, no partial bar
    assert ("ETHUSD", "1m") not in db.asked


async def test_cache_reloads_when_a_refresh_rewrites_the_newest_rows(monkeypatch):
    h, db = _setup(monkeypatch, forming_row=True)
    rt = _runtime(False)
    first = await svc.gather_inputs(rt, "ETHUSD")
    assert first.hourly.index[-1] == h.index[-2]
    # A later refresh rewrites the previous hour (same times, new close).
    db.hourly = h.copy()
    db.hourly.iloc[-2, db.hourly.columns.get_loc("close")] *= 1.01
    second = await svc.gather_inputs(rt, "ETHUSD")
    assert second.hourly["close"].iloc[-1] == db.hourly["close"].iloc[-2]
    assert second.hourly["close"].iloc[-1] != first.hourly["close"].iloc[-1]


async def test_unchanged_store_is_served_from_the_cache(monkeypatch):
    h, db = _setup(monkeypatch, forming_row=True)
    rt = _runtime(False)
    await svc.gather_inputs(rt, "ETHUSD")
    deep = sum(1 for s, r in db.asked if s == "BINANCE:ETHUSD" and r == "1h")
    await svc.gather_inputs(rt, "ETHUSD")
    # second run: only the 2-row key probe, not the deep read again
    assert sum(1 for s, r in db.asked if s == "BINANCE:ETHUSD" and r == "1h") == deep + 1


async def test_top_of_hour_refresh_fetches_recent_1h_only(monkeypatch):
    import types

    from app import scheduler
    from app.data import ingest_binance_history as ibh

    seen = {}

    async def fake(symbols=None, resolutions=("1h", "4h"), start_ms=0):
        seen.update(symbols=symbols, resolutions=resolutions, start_ms=start_ms)
        return {}

    monkeypatch.setattr(ibh, "backfill_deep_history", fake)
    rt = types.SimpleNamespace(registry=types.SimpleNamespace(all=lambda: [types.SimpleNamespace(symbol="BTCUSD")]))
    await scheduler._binance_hour_close_refresh(rt)
    assert seen["symbols"] == ["BTCUSD"] and seen["resolutions"] == ("1h",)
    age_h = (pd.Timestamp.now(tz="UTC").timestamp() * 1000 - seen["start_ms"]) / 3.6e6
    assert 2.9 < age_h < 3.1
