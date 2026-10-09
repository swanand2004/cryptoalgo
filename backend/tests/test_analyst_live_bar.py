"""Live-bar mode of the Analyst (Settings.analyst_live_bar): the forming hourly
bar from stored 1m candles, the hold rule against flip-flopping, and proof that
OFF is the closed-bar behaviour exactly."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from app.analyst import service as svc
from app.analyst.factors import Inputs, analyse
from app.config import Settings


def _hourly(n_days: int = 320, drift: float = 0.0006, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_days * 24
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close}, index=idx)


def _minutes(start: pd.Timestamp, n: int, p0: float, step: float) -> pd.DataFrame:
    t = int(start.timestamp()) + 60 * np.arange(n)
    c = p0 + step * np.arange(1, n + 1)
    return pd.DataFrame({"time": t, "open": c - step, "high": c + 0.05, "low": c - step - 0.05, "close": c,
                         "volume": 1.0})


# ------------------------------------------------------------ the forming bar
def test_forming_bar_is_built_from_this_hours_minutes_only():
    h = _hourly()
    now = h.index[-1] + pd.Timedelta(minutes=37, seconds=20)  # 37 min into the last stored hour
    m = pd.concat([_minutes(h.index[-1] - pd.Timedelta(minutes=10), 10, 50.0, 1.0),  # previous hour: ignored
                   _minutes(h.index[-1], 38, 200.0, 0.5)])
    out, n = svc.with_forming_bar(h, m, now)
    assert n == 38
    assert len(out) == len(h)  # stored partial row for this hour replaced, not duplicated
    assert out.index[-1] == h.index[-1] and out.index.is_monotonic_increasing
    bar = out.iloc[-1]
    assert bar["open"] == 200.0  # the hour's first 1m open
    assert bar["close"] == 200.0 + 0.5 * 38
    assert bar["high"] == 200.0 + 0.5 * 38 + 0.05 and bar["low"] == 200.0 - 0.05
    pd.testing.assert_frame_equal(out.iloc[:-1], h.iloc[:-1], check_freq=False)
    assert out.index.dtype == h.index.dtype and all(out[c].dtype == np.float64 for c in out)


def test_no_fresh_minutes_means_closed_bars_only():
    h = _hourly()
    now = h.index[-1] + pd.Timedelta(minutes=30)
    for m in (None, pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"]),
              _minutes(h.index[-1], 5, 100.0, 0.1)):  # newest minute 25 min old: feed down
        out, n = svc.with_forming_bar(h, m, now)
        assert n == 0
        pd.testing.assert_frame_equal(out, h.iloc[:-1], check_freq=False)


def test_score_moves_inside_the_hour():
    h = _hourly()
    closed = h.iloc[:-1]
    start = h.index[-1]
    last = float(closed["close"].iloc[-1])
    a0 = analyse("BTCUSD", Inputs(hourly=closed, strategy_signal=0.5))
    frame, _ = svc.with_forming_bar(h, _minutes(start, 30, last, -last * 0.004), start + pd.Timedelta(minutes=30))
    a1 = analyse("BTCUSD", Inputs(hourly=frame, strategy_signal=0.5))
    assert a1.score != a0.score and a1.price < a0.price  # a 12% slide this hour is seen now, not at the close


# ------------------------------------------------------------ the hold rule
def _call(decision="BUY", score=0.3):
    return {"decision": decision, "score": score, "notes": []}


def test_call_must_hold_before_it_counts(monkeypatch):
    monkeypatch.setattr(svc, "_live_calls", {})
    s = types.SimpleNamespace(analyst_live_bar_hold_minutes=15, analyst_min_score=0.20)
    t0 = 1_000_000.0
    for k in range(15):  # minutes 0..14: still forming
        r = _call()
        svc.hold_call(r, "ETHUSD", t0 + 60 * k, s)
        assert r["decision"] == "WAIT" and "forming" in r["notes"][-1]
    r = _call()
    svc.hold_call(r, "ETHUSD", t0 + 60 * 15, s)
    assert r["decision"] == "BUY" and r["live_bar"]["held_s"] == 900


def test_flip_wait_or_gap_starts_the_clock_again(monkeypatch):
    monkeypatch.setattr(svc, "_live_calls", {})
    s = types.SimpleNamespace(analyst_live_bar_hold_minutes=15, analyst_min_score=0.20)
    t = 0.0
    for k in range(20):
        svc.hold_call(_call(), "ETHUSD", t + 60 * k, s)
    r = _call("SELL", -0.3)  # flip: SELL starts from zero
    svc.hold_call(r, "ETHUSD", t + 60 * 20, s)
    assert r["decision"] == "WAIT"
    r = _call("WAIT", 0.1)
    svc.hold_call(r, "ETHUSD", t + 60 * 21, s)
    assert r["decision"] == "WAIT" and "ETHUSD" not in svc._live_calls
    for k in range(16):
        svc.hold_call(_call(), "ETHUSD", t + 60 * (22 + k), s)
    r = _call()
    svc.hold_call(r, "ETHUSD", t + 60 * 37 + svc.LIVE_CALL_GAP_S + 1, s)  # missed runs: start over
    assert r["decision"] == "WAIT"
    r = _call("BUY", 0.21)
    svc.hold_call(r, "BTCUSD", t, types.SimpleNamespace(analyst_live_bar_hold_minutes=0, analyst_min_score=0.20))
    assert r["decision"] == "BUY"  # hold 0 = no rule
    r = _call("BUY", 0.25)  # below a raised floor: not a tradeable call, left as it is
    svc.hold_call(r, "SOLUSD", t, types.SimpleNamespace(analyst_live_bar_hold_minutes=15, analyst_min_score=0.35))
    assert r["decision"] == "BUY" and "live_bar" not in r


# ------------------------------------------------------------ OFF = unchanged, ON = wired
class _DB:
    def __init__(self, hourly: pd.DataFrame, minutes: pd.DataFrame):
        self.hourly, self.minutes, self.asked = hourly, minutes, []

    async def load(self, symbol, resolution, limit=500):
        self.asked.append((symbol, resolution))
        if symbol.startswith("BINANCE:") and resolution == "1h":
            df = self.hourly.reset_index(drop=True).assign(time=(self.hourly.index.astype("int64") // 10**9).to_numpy(),
                                                          volume=0.0)
            return df.iloc[-limit:].reset_index(drop=True)
        if resolution == "1m" and not symbol.startswith(("BINANCE:", "MARK:")):
            return self.minutes.iloc[-limit:].reset_index(drop=True)
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


def _runtime(live_bar: bool, hold: int = 15):
    settings = Settings(delta_api_key="k", delta_api_secret="s", analyst_live_bar=live_bar, analyst_live_bar_hold_minutes=hold, analyst_symbols="ETHUSD")
    return types.SimpleNamespace(settings=settings, registry=types.SimpleNamespace(symbols=["ETHUSD"]),
                                 live_signals={"ETHUSD": 0.6}, cache=None)


def _setup(monkeypatch):
    h = _hourly(drift=0.0006)
    start = pd.Timestamp.now(tz="UTC").floor("h")
    h.index = h.index + (start - h.index[-1])  # the newest stored row = the current, unfinished hour
    last = float(h["close"].iloc[-2])
    minutes = _minutes(start, max(1, int((pd.Timestamp.now(tz="UTC") - start).total_seconds() // 60) + 1),
                       last, -last * 0.00001)
    db = _DB(h, minutes)
    monkeypatch.setattr(svc, "load_candles_df", db.load)
    monkeypatch.setattr(svc, "_hourly_cache", {})
    monkeypatch.setattr(svc, "_live_calls", {})
    monkeypatch.setattr(svc, "_stable_refreshed", {"at": 1e18})
    return h, db


async def test_off_is_the_closed_bar_behaviour_exactly(monkeypatch):
    h, db = _setup(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("live-bar code ran with the setting off")

    monkeypatch.setattr(svc, "with_forming_bar", boom)
    monkeypatch.setattr(svc, "hold_call", boom)
    rt = _runtime(False)
    inp = await svc.gather_inputs(rt, "ETHUSD")
    pd.testing.assert_frame_equal(inp.hourly, h.iloc[:-1][["open", "high", "low", "close"]], check_freq=False, check_names=False)
    assert ("ETHUSD", "1m") not in db.asked  # the 1m candles are never even read

    opened = {}

    async def fake_open(runtime, results, *, gates=None, **kw):
        opened.update(results)

    from app.analyst import execution as ex

    monkeypatch.setattr(ex, "open_best_trade", fake_open)
    res = await svc.run_analysis(rt)
    ref = analyse("ETHUSD", inp).to_dict()
    assert res["ETHUSD"] == ref and opened["ETHUSD"] == ref  # same dict as the plain closed-bar analysis
    assert "live_bar" not in res["ETHUSD"]


async def test_on_scores_the_forming_bar_and_holds_the_call(monkeypatch):
    h, db = _setup(monkeypatch)
    rt = _runtime(True, hold=15)
    inp = await svc.gather_inputs(rt, "ETHUSD")
    assert ("ETHUSD", "1m") in db.asked
    assert inp.hourly.index[-1] == h.index[-1] and inp.hourly.index.name == "time"  # the current hour is in the frame ...
    assert inp.hourly["close"].iloc[-1] == db.minutes["close"].iloc[-1]  # ... at the latest 1m close

    from app.analyst import execution as ex

    async def fake_open(runtime, results, *, gates=None, **kw):
        return None

    monkeypatch.setattr(ex, "open_best_trade", fake_open)
    raw = analyse("ETHUSD", inp)
    assert raw.decision == "BUY"  # the synthetic uptrend calls BUY
    res = await svc.run_analysis(rt)
    assert res["ETHUSD"]["decision"] == "WAIT" and res["ETHUSD"]["live_bar"]["call"] == "BUY"
    svc._live_calls["ETHUSD"] = ("BUY", svc._live_calls["ETHUSD"][1] - 16 * 60, svc._live_calls["ETHUSD"][2])
    res = await svc.run_analysis(rt)
    assert res["ETHUSD"]["decision"] == "BUY"


def test_settings_default_off():
    s = Settings(delta_api_key="k", delta_api_secret="s")
    assert s.analyst_live_bar is False and s.analyst_live_bar_hold_minutes == 15
