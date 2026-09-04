import numpy as np
import pandas as pd
import pytest

from app.gold.engine import _price_levels


def _frame(n=200, high=2100.0, low=1900.0, close=2000.0, atr_pct=0.01):
    """A frame whose extremes are pinned so the expected levels are exact."""
    rng = np.random.default_rng(0)
    mid = pd.Series(np.full(n, close) + rng.normal(0, 1.0, n))
    df = pd.DataFrame({
        "close": mid, "high": mid + 5.0, "low": mid - 5.0,
        "atr_pct_14": np.full(n, atr_pct),
    })
    # Plant the window extremes explicitly.
    df.loc[df.index[5], "high"] = high
    df.loc[df.index[6], "low"] = low
    df.loc[df.index[-1], "close"] = close
    return df


def test_buy_confirms_above_resistance_and_invalidates_below():
    lv = _price_levels(_frame(), horizon_bars=4, action="BUY")
    assert lv["breakout_above"] == pytest.approx(lv["resistance"])
    assert lv["breakdown_below"] == pytest.approx(lv["support"])
    # ATR-based invalidation sits below the entry price for a long.
    assert lv["invalidation"] < 2000.0


def test_sell_mirrors_the_buy_case():
    """For a short, breaking DOWN through support is the confirmation and
    rallying through resistance is what breaks the call -- the opposite of a
    long, so the fields must flip rather than always reporting the high."""
    lv = _price_levels(_frame(), horizon_bars=4, action="SELL")
    assert lv["breakout_above"] == pytest.approx(lv["support"])
    assert lv["breakdown_below"] == pytest.approx(lv["resistance"])
    assert lv["invalidation"] > 2000.0  # stop sits above a short


def test_hold_still_reports_the_range_but_has_nothing_to_invalidate():
    lv = _price_levels(_frame(), horizon_bars=4, action="HOLD")
    assert lv["resistance"] > lv["support"]
    assert lv["invalidation"] is None  # no position to be wrong about


def test_invalidation_scales_with_volatility_not_a_fixed_percentage():
    """A fixed percentage stop means completely different things in a calm
    market and a violent one. Doubling ATR must widen the level."""
    calm = _price_levels(_frame(atr_pct=0.005), horizon_bars=4, action="BUY")
    wild = _price_levels(_frame(atr_pct=0.02), horizon_bars=4, action="BUY")
    assert wild["invalidation"] < calm["invalidation"]
    assert wild["atr"] > calm["atr"]


def test_window_scales_with_the_prediction_horizon():
    """A 1-hour call should not be judged against a 3-month range, and a
    1-week call should not be judged against the last few bars."""
    short = _price_levels(_frame(n=600), horizon_bars=2, action="BUY")
    long = _price_levels(_frame(n=600), horizon_bars=40, action="BUY")
    assert long["range_window_bars"] > short["range_window_bars"]


def test_distances_are_signed_relative_to_current_price():
    lv = _price_levels(_frame(), horizon_bars=4, action="BUY")
    assert lv["distance_to_breakout_pct"] > 0   # resistance is above
    assert lv["distance_to_breakdown_pct"] < 0  # support is below


def test_missing_columns_return_nothing_rather_than_inventing_levels():
    assert _price_levels(pd.DataFrame({"close": [1, 2, 3]}), 4, "BUY") == {}
    assert _price_levels(None, 4, "BUY") == {}
    assert _price_levels(pd.DataFrame(), 4, "BUY") == {}


def test_too_little_history_returns_nothing():
    """Levels from a handful of bars would be noise dressed as structure."""
    assert _price_levels(_frame(n=10), horizon_bars=4, action="BUY") == {}
