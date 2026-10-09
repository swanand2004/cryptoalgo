"""The Analyst's speed-ups change nothing it decides (efficiency, 2026-10-06).

factors.py now (1) builds daily/4h bars with numpy instead of pandas resample,
(2) computes the daily ATR with numpy, (3) slices sorted side series by
position, and (4) memoises everything derived from price alone per distinct
hourly frame, so the every-minute run only re-reads the live inputs.
structure.py remembers the swing pivots of a daily frame. These tests pin that
the outputs are identical to the original code:

- golden values produced by the ORIGINAL modules (frozen copy, before any of
  these changes) on synthetic data with gaps and NaN holes;
- 200 random historical timestamps per coin from the local database (skipped
  when it is absent): the optimised analyse(), on a memo miss AND a memo hit,
  equals the reference path that runs the original pandas helpers with the
  memo off;
- each fast helper equals the pandas expression it replaced.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import sqlite3
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from app.analyst import factors as F
from app.analyst import structure as st
from app.strategy.gann_angle import swing_pivots

DB = pathlib.Path(__file__).resolve().parent.parent / "cryptoalgo.db"
COINS = ("BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD")
PER_COIN = 200


# ------------------------------------------------------------ the original helpers
def _orig_daily_atr(daily: pd.DataFrame, n: int = 14) -> float:
    h, l, c = daily["high"], daily["low"], daily["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def _orig_last(s, now, max_age):
    if s is None or len(s) == 0:
        return None
    s = s[s.index <= now].dropna()
    if len(s) == 0 or now - s.index[-1] > max_age:
        return None
    return s


def _orig_pivots(daily):
    return swing_pivots(daily["high"].to_numpy(), daily["low"].to_numpy(), 2)


@contextlib.contextmanager
def _original_path():
    """analyse() exactly as before: pandas resample/ATR/slicing, no memo."""
    with mock.patch.object(F, "_resample", F._resample_pandas), \
            mock.patch.object(F, "daily_atr", _orig_daily_atr), \
            mock.patch.object(F, "_last", _orig_last), \
            mock.patch.object(st, "_pivots", _orig_pivots), \
            mock.patch.object(F, "PRICE_MEMO_ENABLED", False):
        yield


def _key(a) -> tuple:
    return (a.decision, a.score, a.conviction, a.stop_loss, a.take_profit) if a is not None else None


def _full(a) -> str:
    return json.dumps(a.to_dict() if a is not None else None, sort_keys=True, default=str)


# ------------------------------------------------------------ synthetic golden values
def _synthetic(seed: int, drift: float, n_days: int = 340) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_days * 24
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    openp = np.r_[close[0], close[:-1]]
    high = np.maximum(openp, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(openp, close) * (1 - rng.uniform(0, 0.004, n))
    idx = pd.date_range("2024-03-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close}, index=idx)
    df = df.drop(df.index[rng.choice(n, n // 50, replace=False)])  # missing hours
    df.iloc[rng.choice(len(df), 30, replace=False), 0] = np.nan  # a few holes in opens
    return df


# (seed, drift, bars dropped from the end, decision, score, conviction, stop, target)
# produced by the ORIGINAL factors.py/structure.py, before the speed-ups.
GOLDEN = [
    (1, 0.0004, 0, 'BUY', 0.8133333333333332, 8, 1555.1691988850494, 1770.0273502938608),
    (1, 0.0004, 97, 'BUY', 0.27989889002692575, 3, 1380.9904802008423, 1567.1507737071274),
    (1, 0.0004, 333, 'WAIT', -0.14115866902299373, 1, None, None),
    (2, -0.0004, 0, 'SELL', -0.6333819474253377, 6, 9.303318488555202, 8.122873006727556),
    (2, -0.0004, 97, 'WAIT', -0.1592172877665225, 2, None, None),
    (2, -0.0004, 333, 'WAIT', -0.1309078174961469, 1, None, None),
    (3, 0.0, 0, 'WAIT', 0.048657894329771835, 0, None, None),
    (3, 0.0, 97, 'WAIT', 0.06004234690712012, 1, None, None),
    (3, 0.0, 333, 'BUY', 0.3244444444444444, 3, 118.7137807566606, 134.1981451809446),
    (4, 0.0002, 0, 'WAIT', 0.24548163543425822, 2, None, None),
    (4, 0.0002, 97, 'BUY', 0.5190695035880348, 5, 1331.8232572450968, 1466.0143851575788),
    (4, 0.0002, 333, 'BUY', 0.48382744758448504, 5, 975.7114963564755, 1105.7638050007458),
]


@pytest.mark.parametrize("seed,drift,cut,decision,score,conviction,stop,target", GOLDEN)
def test_matches_values_from_the_original_code(seed, drift, cut, decision, score, conviction, stop, target, monkeypatch):
    # GOLDEN was recorded with DECISION_LINE 0.25; the line moved to 0.20 on
    # 2026-10-07 (config.analyst_min_score). This test pins the computation,
    # not the threshold, so it keeps the line the values were recorded at.
    monkeypatch.setattr(F, "DECISION_LINE", 0.25)
    F.clear_price_memo()
    h = _synthetic(seed, drift)
    win = h.iloc[: len(h) - cut]
    sig = 0.6 if seed % 2 else -0.6
    for frame in (win, win.copy(deep=True)):  # memo miss, then memo hit
        a = F.analyse("BTCUSD", F.Inputs(hourly=frame, strategy_signal=sig))
        assert _key(a) == (decision, score, conviction, stop, target)


# ------------------------------------------------------------ 200 historical timestamps per coin
def _load(con, sym, res):
    df = pd.read_sql_query("select time, open, high, low, close from candles where symbol=? and resolution=? "
                           "order by time", con, params=[sym, res])
    if df.empty:
        return None
    return df.set_index(pd.to_datetime(df["time"], unit="s", utc=True))[["open", "high", "low", "close"]]


def _series(df, shift_day=False):
    if df is None:
        return None
    s = df["close"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if shift_day:
        s.index = s.index + pd.Timedelta(days=1)
    return s


def _tail(s, ts, limit):
    if s is None:
        return None
    s = s[s.index <= ts]
    return s.iloc[-limit:] if len(s) else None


@pytest.mark.skipif(not DB.exists(), reason="needs the local research database backend/cryptoalgo.db")
@pytest.mark.parametrize("sym", COINS)
def test_identical_on_200_random_historical_timestamps(sym):
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        hourly = _load(con, f"BINANCE:{sym}", "1h")
        if hourly is None or len(hourly) < 24 * 400:
            pytest.skip(f"no hourly history for {sym}")
        hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
        side = {  # name: (series, the limit service.gather_inputs reads)
            "funding": (_series(_load(con, f"BFUND:{sym}", "8h")), 60),
            "open_interest": (_series(_load(con, f"BOI:{sym}", "1h")), 200),
            "long_short_ratio": (_series(_load(con, f"BLSR:{sym}", "1h")), 48),
            "fear_greed": (_series(_load(con, "FNG:GLOBAL", "1d"), shift_day=True), 30),
            "news": (_series(_load(con, f"NEWS:{sym}", "1h")), 72),
            "dollar": (_series(_load(con, "MACRO:DXY", "1d"), shift_day=True), 120),
            "implied_vol": (_series(_load(con, f"DVOL:{sym}", "1h")), 24 * 95),
        }
    finally:
        con.close()

    rng = np.random.default_rng(COINS.index(sym) + 2026)
    n = len(hourly)
    recent = int(np.searchsorted(hourly.index, hourly.index[-1] - pd.Timedelta(days=120)))
    # 3/4 anywhere in history, 1/4 in the last 120 days (where the side inputs exist)
    picks = np.r_[rng.integers(24 * 330, n, PER_COIN * 3 // 4), rng.integers(recent, n, PER_COIN - PER_COIN * 3 // 4)]
    F.clear_price_memo()
    decided = 0
    for i in picks:
        win = hourly.iloc[i - 24 * 330 + 1: i + 1]
        ts = win.index[-1]
        kw = {k: _tail(s, ts, lim) for k, (s, lim) in side.items()}
        sig = float(rng.uniform(-1, 1)) if rng.random() < 0.8 else None
        ev = [(ts.to_pydatetime() + pd.Timedelta(hours=5), "CPI")] if rng.random() < 0.2 else []
        with _original_path():
            ref = F.analyse(sym, F.Inputs(hourly=win, strategy_signal=sig, events=ev, **kw))
        miss = F.analyse(sym, F.Inputs(hourly=win, strategy_signal=sig, events=ev, **kw))
        hit = F.analyse(sym, F.Inputs(hourly=win.copy(deep=True), strategy_signal=sig, events=ev, **kw))
        for got in (miss, hit):
            assert _key(got) == _key(ref), (sym, ts)
            assert _full(got) == _full(ref), (sym, ts)  # every factor, reading and level too
        decided += ref is not None and ref.decision != "WAIT"
    assert decided > 0  # the sample exercised stop/target placement


# ------------------------------------------------------------ the memo itself
def test_memo_reuses_price_work_but_reads_live_inputs_fresh():
    F.clear_price_memo()
    h = _synthetic(7, 0.0003)
    calls = []
    real = F._price_block
    with mock.patch.object(F, "_price_block", lambda x: calls.append(1) or real(x)):
        a = F.analyse("BTCUSD", F.Inputs(hourly=h, strategy_signal=0.9))
        b = F.analyse("BTCUSD", F.Inputs(hourly=h.copy(deep=True), strategy_signal=-0.9,
                                         events=[(h.index[-1].to_pydatetime() + pd.Timedelta(hours=2), "FOMC")]))
        assert len(calls) == 1  # same closed bars: price work done once
        c = F.analyse("BTCUSD", F.Inputs(hourly=h.iloc[:-1], strategy_signal=0.9))
        assert len(calls) == 2  # one bar fewer is a different frame
        h2 = h.copy(deep=True)
        h2.iloc[-5, h2.columns.get_loc("high")] *= 1.0000001  # one value nudged
        F.analyse("BTCUSD", F.Inputs(hourly=h2, strategy_signal=0.9))
        assert len(calls) == 3
    with _original_path():
        a0 = F.analyse("BTCUSD", F.Inputs(hourly=h, strategy_signal=0.9))
        b0 = F.analyse("BTCUSD", F.Inputs(hourly=h, strategy_signal=-0.9,
                                          events=[(h.index[-1].to_pydatetime() + pd.Timedelta(hours=2), "FOMC")]))
        c0 = F.analyse("BTCUSD", F.Inputs(hourly=h.iloc[:-1], strategy_signal=0.9))
    assert _full(a) == _full(a0) and _full(b) == _full(b0) and _full(c) == _full(c0)
    assert b.size_factor == 0.5 and any(f.key == "strategy" and f.score == -0.9 for f in b.factors)


def test_callers_cannot_corrupt_the_memo():
    F.clear_price_memo()
    h = _synthetic(8, 0.0003)
    a = F.analyse("BTCUSD", F.Inputs(hourly=h, strategy_signal=0.5))
    before = _full(a)
    for f in a.factors:
        f.score = 99.0
    a.supports.clear()
    a.resistances.append(-1.0)
    b = F.analyse("BTCUSD", F.Inputs(hourly=h, strategy_signal=0.5))
    assert _full(b) == before


# ------------------------------------------------------------ the fast helpers
def _frames():
    rng = np.random.default_rng(11)
    for tz in ("UTC", None):
        for n in (1, 5, 30, 24 * 70 + 7):
            idx = pd.date_range("2025-01-01 03:00", periods=n, freq="h", tz=tz)
            v = 100 + rng.normal(0, 1, (n, 4)).cumsum(axis=0)
            df = pd.DataFrame(v, index=idx, columns=["open", "high", "low", "close"])
            if n > 30:
                df = df.drop(df.index[rng.choice(n, n // 4, replace=False)])  # gaps, incl. whole days
                for col in range(4):
                    df.iloc[rng.choice(len(df), 40, replace=False), col] = np.nan
                df.iloc[100:130] = np.nan  # a fully empty stretch
            yield df


@pytest.mark.parametrize("rule", ["1D", "4h"])
def test_resample_equals_pandas(rule):
    for df in _frames():
        pd.testing.assert_frame_equal(F._resample(df, rule), F._resample_pandas(df, rule), check_freq=False)


def test_resample_falls_back_to_pandas_for_other_timezones():
    idx = pd.date_range("2025-01-01", periods=24 * 5, freq="h", tz="Asia/Kolkata")
    df = pd.DataFrame(np.arange(24 * 5 * 4, dtype=float).reshape(-1, 4), index=idx,
                      columns=["open", "high", "low", "close"])
    pd.testing.assert_frame_equal(F._resample(df, "1D"), F._resample_pandas(df, "1D"))


def test_daily_atr_equals_the_pandas_formula():
    for df in _frames():
        d = F._resample_pandas(df, "1D")
        if len(d):
            assert F.daily_atr(d) == _orig_daily_atr(d) or (np.isnan(F.daily_atr(d)) and np.isnan(_orig_daily_atr(d)))


def test_pivot_memo_equals_swing_pivots():
    for df in _frames():
        d = F._resample_pandas(df, "1D")
        assert st._pivots(d) == _orig_pivots(d)
        assert st._pivots(d.copy()) == _orig_pivots(d)
