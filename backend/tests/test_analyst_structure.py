"""Chart patterns, Gann and levels for the Analyst (app/analyst/structure.py)."""
import numpy as np
import pandas as pd
import pytest

from app.analyst import structure as st


def _daily(closes, spread=1.0):
    c = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(c), freq="D", tz="UTC")
    return pd.DataFrame({"open": c, "high": c + spread, "low": c - spread, "close": c}, index=idx)


def _zigzag(points, steps=5):
    out = []
    for a, b in zip(points, points[1:]):
        out += list(np.linspace(a, b, steps, endpoint=False))
    return out + [points[-1]]


def test_higher_highs_and_higher_lows_read_as_an_uptrend_and_mirror():
    up = _daily(_zigzag([100, 110, 104, 116, 109, 122, 115, 128]))
    assert st.market_structure(up)[0] == 1.0
    down = _daily(_zigzag([128, 115, 122, 109, 116, 104, 110, 98]))
    assert st.market_structure(down)[0] == -1.0


def test_double_top_is_confirmed_only_through_its_neckline():
    forming = _daily(_zigzag([100, 120, 110, 120.3, 112]))
    score, text = st.double_pattern(forming, atr=2.0)
    assert score == -0.3 and "possible double top" in text
    broken = _daily(_zigzag([100, 120, 110, 120.3, 104]))
    assert st.double_pattern(broken, atr=2.0)[0] == -1.0


def test_compression_then_breakout():
    wide = list(np.linspace(100, 160, 30)) + list(np.linspace(160, 100, 30))
    tight = [100 + (i % 2) for i in range(20)]
    assert st.compression_breakout(_daily(wide + tight + [100.5]))[0] == 0.0  # coiling
    assert st.compression_breakout(_daily(wide + tight + [110.0]))[0] == 1.0


def test_levels_and_the_bounce_and_rejection_zones():
    sup, res = [100.0, 95.0], [110.0]
    assert st.level_position(100.5, sup, res, atr=2.0)[0] == 0.5
    assert st.level_position(109.5, sup, res, atr=2.0)[0] == -0.5
    assert st.level_position(105.0, sup, res, atr=2.0)[0] == 0.0


def test_stop_goes_beyond_the_nearest_level_and_tp1_before_the_next():
    # long at 100, ATR 2: support 97 is 1.5 ATR away -> stop 96.5; risk 3.5; resistance 106 is 1.7R -> TP1 105.8
    stop, target, how = st.levels_plan("BUY", 100.0, 2.0, [97.0], [106.0], stop_atr=1.5)
    assert stop == pytest.approx(96.5) and target == pytest.approx(105.8) and "level" in how
    # no usable levels -> 1.5 ATR stop and a 1R target
    stop, target, _ = st.levels_plan("BUY", 100.0, 2.0, [99.5], [130.0], stop_atr=1.5)
    assert stop == pytest.approx(97.0) and target == pytest.approx(103.0)
    # short mirrors
    stop, target, _ = st.levels_plan("SELL", 100.0, 2.0, [94.0], [103.0], stop_atr=1.5)
    assert stop == pytest.approx(103.5) and target == pytest.approx(94.2)


def test_key_levels_split_around_price():
    d = _daily(_zigzag([100, 110, 104, 116, 109, 122, 115, 118]))
    sup, res = st.key_levels(d, 118.0)
    assert all(x < 118 for x in sup) and all(x > 118 for x in res) and sup and res


def test_gann_readings_cover_angle_swing_and_square_of_nine():
    d = _daily(list(np.linspace(100, 150, 80)))
    keys = [k for k, *_ in st.gann_readings(d)]
    assert keys == ["gann_angle", "gann_swing", "gann_sq9"]
