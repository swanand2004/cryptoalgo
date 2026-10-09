"""ML book filter v1 (app/ml/book_filter.py).

1. OFF and SHADOW leave sizing byte-identical: the trading-loop hook hands back the very same weights
   object and never even reads the stored multipliers.
2. LIVE applies fresh multipliers, clips the book to book_max_leverage, and falls back to 1.0 when the
   record is stale or missing.
3. Parity: on stored history the production feature code reproduces the research pipeline's frozen-model
   predictions (scratchpad team/ai_ml, cycle 5) -- full-grid, tail-window and end-to-end decide().
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.ml import book_filter as bf

DAY, HOUR = 86400, 3600
REAL_DB = pathlib.Path(__file__).resolve().parent.parent / "cryptoalgo.db"

# (symbol, decision instant = 00:00 UTC, research-pipeline P(book right), research S) -- from
# scratchpad team/ai_ml/parity_ref.json (frozen model applied by the validated pipeline).
REF = [
    ["BTCUSD", 1510790400, 0.4445096330005813, 0.0], ["BTCUSD", 1566691200, 0.420291311973047, -0.31476390355298156],
    ["BTCUSD", 1622678400, 0.4139366226907652, -0.21413538404393598],
    ["BTCUSD", 1678665600, 0.4128318182089897, -0.285272499295708],
    ["BTCUSD", 1734652800, 0.47025231496609743, 0.11477625925482245],
    ["BTCUSD", 1790640000, 0.4617449666043844, 0.2770377791496687],
    ["ETHUSD", 1510790400, 0.44325017908390274, 0.0], ["ETHUSD", 1566691200, 0.4292117823649988, -0.3997836232544935],
    ["ETHUSD", 1622678400, 0.39559863255768496, 0.0], ["ETHUSD", 1678665600, 0.42342036660863, -0.23078485577068575],
    ["ETHUSD", 1734652800, 0.4528147645528784, -0.5737721772648914],
    ["ETHUSD", 1790640000, 0.4508787653707347, 0.30755100932164875],
    ["SOLUSD", 1604966400, 0.4365640611302153, 0.0], ["SOLUSD", 1642032000, 0.4362620304436297, -0.2851268539051123],
    ["SOLUSD", 1679184000, 0.4531001963772855, 0.1487987577624747],
    ["SOLUSD", 1716336000, 0.5471212569520812, 0.4493506774602223],
    ["SOLUSD", 1753488000, 0.46703815487883676, -0.27356264177166656],
    ["SOLUSD", 1790640000, 0.4759819032636248, 0.3589639684229915],
    ["BNBUSD", 1517788800, 0.4023406922999343, 0.0], ["BNBUSD", 1572307200, 0.5070242736354654, 0.3015411388483586],
    ["BNBUSD", 1626912000, 0.43270443337246367, -0.3734927777320204],
    ["BNBUSD", 1681430400, 0.485164428777207, 0.4653718902044093],
    ["BNBUSD", 1736035200, 0.48124512023930277, 0.42927177937696437],
    ["BNBUSD", 1790640000, 0.44534794451934123, 0.201004020702424],
]


def _signals(vals: dict[str, float]) -> dict[str, pd.Series]:
    return {s: pd.Series([0.0, v]) for s, v in vals.items()}


def _rec(decision_ts: int, ms: dict[str, float], stale: bool = False) -> dict:
    return {"decision_ts": decision_ts, "applies_from_ts": decision_ts + HOUR,
            "applies_until_ts": decision_ts + HOUR + DAY, "stale": stale, "reasons": [],
            "coins": {c: {"p": 0.5, "m": m, "S": 0.3} for c, m in ms.items()}}


# ----------------------------------------------------------------------------- off / shadow
@pytest.mark.parametrize("mode", ["off", "shadow", "garbage"])
async def test_off_and_shadow_return_the_same_weights_object(mode, monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("off/shadow must not read multipliers")

    monkeypatch.setattr(bf, "current_multipliers", boom)
    w = {"BTCUSD": 0.4, "ETHUSD": 0.3, "SOLUSD": 0.2, "BNBUSD": 0.1}
    snapshot = dict(w)
    out = await bf.book_filter_weights(SimpleNamespace(ml_book_filter=mode, book_max_leverage=8.0), w,
                                       _signals({"BTCUSD": 0.5}), 2.0)
    assert out is w and w == snapshot


def test_settings_defaults():
    from app.config import Settings

    s = Settings()
    assert s.ml_book_filter == "shadow"
    assert bf.mode_of(s) == "shadow"
    assert s.book_vol_fast_up is False


async def test_shadow_job_stores_without_touching_sizing(monkeypatch):
    now = dt.datetime(2026, 10, 8, 0, 5, tzinfo=dt.timezone.utc)
    d0 = int(now.timestamp()) // DAY * DAY

    async def fake_inputs():
        return {}, {}, pd.Series(dtype=float)

    monkeypatch.setattr(bf, "_load_inputs", fake_inputs)
    monkeypatch.setattr(bf, "decide", lambda f, fu, fg, ts: _rec(ts, {"BTCUSD": 1.5, "ETHUSD": 0.5}))
    rec = await bf.run_daily_job(SimpleNamespace(ml_book_filter="shadow"), now=now)
    assert rec["mode"] == "shadow" and rec["decision_ts"] == d0
    hist = await bf.get_history()
    assert len(hist) == 1 and hist[0]["coins"]["BTCUSD"]["m"] == 1.5
    # sizing is still untouched in shadow, even with a fresh record in force
    w = {"BTCUSD": 0.5, "ETHUSD": 0.5}
    assert await bf.book_filter_weights(SimpleNamespace(ml_book_filter="shadow"), w, _signals({"BTCUSD": 1}), 1.0) is w
    # the API reports it
    from app.api.routes_ml import get_book_filter

    body = await get_book_filter(days=10)
    assert body["latest"]["decision_ts"] == d0 and len(body["history"]) == 1
    # "off" does nothing at all
    assert await bf.run_daily_job(SimpleNamespace(ml_book_filter="off"), now=now) is None


async def test_stale_attempt_is_retried_then_stored_on_final(monkeypatch):
    now = dt.datetime(2026, 10, 8, 0, 5, tzinfo=dt.timezone.utc)

    async def fake_inputs():
        return {}, {}, pd.Series(dtype=float)

    monkeypatch.setattr(bf, "_load_inputs", fake_inputs)
    rec = await bf.run_daily_job(SimpleNamespace(ml_book_filter="shadow"), now=now)
    assert rec["stale"] and all(v["m"] == 1.0 for v in rec["coins"].values())
    assert await bf.get_history() == []  # not stored: the scheduler retries
    await bf.run_daily_job(SimpleNamespace(ml_book_filter="shadow"), now=now, final_attempt=True)
    assert len(await bf.get_history()) == 1


# ----------------------------------------------------------------------------- live
async def test_live_applies_fresh_multipliers_and_clips(monkeypatch):
    d0 = 1_790_640_000
    await bf._store(_rec(d0, {"BTCUSD": 1.5, "ETHUSD": 0.5, "SOLUSD": 1.0, "BNBUSD": 1.0}))
    now = dt.datetime.fromtimestamp(d0 + 2 * HOUR, dt.timezone.utc)
    assert await bf.current_multipliers(now) == {"BTCUSD": 1.5, "ETHUSD": 0.5, "SOLUSD": 1.0, "BNBUSD": 1.0}
    # before 01:00 the new record is not yet in force
    assert await bf.current_multipliers(dt.datetime.fromtimestamp(d0 + 600, dt.timezone.utc)) == {}

    w = {"BTCUSD": 0.25, "ETHUSD": 0.25, "SOLUSD": 0.25, "BNBUSD": 0.25}
    sig = _signals({"BTCUSD": 1.0, "ETHUSD": 1.0, "SOLUSD": 1.0, "BNBUSD": 1.0})
    eff = bf.apply_to_weights(w, sig, 2.0, {"BTCUSD": 1.5, "ETHUSD": 0.5}, max_book_leverage=8.0)
    assert eff == {"BTCUSD": 0.375, "ETHUSD": 0.125, "SOLUSD": 0.25, "BNBUSD": 0.25}
    # clip: gross 1.0 * 1.5 * 8 = 12 > 8 -> scaled down to exactly the cap, never up
    eff = bf.apply_to_weights(w, sig, 8.0, {s: 1.5 for s in w}, max_book_leverage=8.0)
    gross = sum(eff.values()) * 8.0
    assert gross == pytest.approx(8.0)
    eff = bf.apply_to_weights(w, sig, 1.0, {s: 0.5 for s in w}, max_book_leverage=8.0)
    assert sum(eff.values()) == pytest.approx(0.5)


async def test_live_falls_back_to_one_when_stale_or_missing(monkeypatch):
    w = {"BTCUSD": 0.6, "ETHUSD": 0.4}
    sig = _signals({"BTCUSD": 0.5, "ETHUSD": -0.5})
    live = SimpleNamespace(ml_book_filter="live", book_max_leverage=8.0)
    assert await bf.book_filter_weights(live, w, sig, 2.0) == w  # nothing stored
    d0 = (int(dt.datetime.now(dt.timezone.utc).timestamp()) - HOUR) // DAY * DAY  # window covers now
    await bf._store(_rec(d0, {"BTCUSD": 1.5, "ETHUSD": 0.5}, stale=True))
    assert await bf.book_filter_weights(live, w, sig, 2.0) == w


def test_decide_flags_unfinalised_bar_as_stale():
    d = 1_790_640_000
    t = np.arange(d - 100 * HOUR, d, HOUR)  # stops at the 23:00 bar: no 00:00 bar yet
    df = pd.DataFrame({"time": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    fr = pd.Series([0.0001], index=[d])
    rec = bf.decide({c: df for c in bf.MAJORS}, {c: fr for c in bf.MAJORS}, pd.Series([50.0], index=[d - DAY]), d)
    assert rec["stale"] and any("not yet final" in r for r in rec["reasons"])
    assert all(v["m"] == 1.0 for v in rec["coins"].values())
    assert rec["applies_from_ts"] == d + HOUR and rec["applies_until_ts"] == d + HOUR + DAY


def test_multiplier_map():
    _, spec = bf.load_model()
    lo, hi = spec["lo"], spec["hi"]
    assert bf.multiplier(lo - 1e-9, lo, hi) == 0.5
    assert bf.multiplier((lo + hi) / 2, lo, hi) == 1.0
    assert bf.multiplier(hi + 1e-9, lo, hi) == 1.5
    assert len(spec["features"]) == 166


# ----------------------------------------------------------------------------- parity with research
def _real_inputs(until_ts: int | None = None):
    con = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    frames, funding = {}, {}
    for c in bf.MAJORS:
        df = pd.read_sql("select time, open, high, low, close, volume from candles where symbol=? and "
                         "resolution='1h' order by time", con, params=(f"BINANCE:{c}",)).drop_duplicates("time")
        frames[c] = df if until_ts is None else df[df.time <= until_ts]
        fd = pd.read_sql("select time, close from candles where symbol=? and resolution='8h' order by time",
                         con, params=(f"BFUNDHIST:{c}",)).drop_duplicates("time")
        funding[c] = pd.Series(fd["close"].to_numpy() / 100.0, index=fd["time"].astype("int64").to_numpy())
    fg = pd.read_sql("select time, close from candles where symbol='FNG:GLOBAL' and resolution='1d' order by time",
                     con).drop_duplicates("time")
    con.close()
    return frames, funding, pd.Series(fg["close"].astype(float).to_numpy(), index=fg["time"].astype("int64").to_numpy())


needs_db = pytest.mark.skipif(not REAL_DB.exists(), reason="local research DB not present")


@needs_db
def test_parity_full_grid_matches_research_pipeline():
    booster, spec = bf.load_model()
    frames, funding, fng = _real_inputs(until_ts=1_790_640_000)
    hr = bf.build_hourly(frames, funding)
    grid = hr["grid"]
    for c in bf.MAJORS:
        pts = [r for r in REF if r[0] == c]
        rows = np.array([int(np.where(grid == r[1] - HOUR)[0][0]) for r in pts])
        X = bf.feature_rows(hr, fng, rows)
        X = X[X.coin == c].reset_index(drop=True)
        p = booster.predict(X[spec["features"]])
        for k, r in enumerate(pts):
            assert p[k] == pytest.approx(r[2], abs=1e-9), (c, r[1])
            assert X["S"].iloc[k] == pytest.approx(r[3], abs=1e-12)


@needs_db
def test_parity_end_to_end_decide_and_tail_window():
    d = 1_790_640_000  # 2026-09-29 00:00 UTC, last reference decision
    frames, funding, fng = _real_inputs(until_ts=d)  # includes the 00:00 bar (finality proof), nothing later
    rec = bf.decide(frames, funding, fng, d)
    assert not rec["stale"], rec["reasons"]
    _, spec = bf.load_model()
    for c, ts, p, s in REF:
        if ts == d:
            assert rec["coins"][c]["p"] == pytest.approx(p, abs=1e-9), c
            assert rec["coins"][c]["m"] == bf.multiplier(p, spec["lo"], spec["hi"])
