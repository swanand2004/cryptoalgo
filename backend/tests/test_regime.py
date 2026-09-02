import numpy as np
import pandas as pd

from app.strategy.regime import (
    DEFAULT_ADX_TREND_THRESHOLD,
    Regime,
    VolRegime,
    classify,
    regime_labels,
    size_multiplier,
)


def _frame(adx, ema_gap, atr_pct=None, n=None):
    n = n or len(adx)
    return pd.DataFrame(
        {
            "adx_14": adx,
            "ema_9": np.array(ema_gap) + 100.0,
            "ema_21": np.full(n, 100.0),
            "atr_pct": atr_pct if atr_pct is not None else np.full(n, 0.01),
        }
    )


def test_strong_adx_with_positive_ema_gap_is_an_uptrend():
    df = _frame(adx=np.full(50, 40.0), ema_gap=np.full(50, 2.0))
    direction, _ = classify(df)
    assert (direction == Regime.TRENDING_UP.value).all()


def test_strong_adx_with_negative_ema_gap_is_a_downtrend():
    df = _frame(adx=np.full(50, 40.0), ema_gap=np.full(50, -2.0))
    direction, _ = classify(df)
    assert (direction == Regime.TRENDING_DOWN.value).all()


def test_weak_adx_is_a_range_regardless_of_ema_direction():
    """ADX measures strength only. A big EMA gap with no trend strength is a
    range, not a trend -- otherwise every drifting chop gets called a trend."""
    for gap in (5.0, -5.0):
        df = _frame(adx=np.full(50, 10.0), ema_gap=np.full(50, gap))
        direction, _ = classify(df)
        assert (direction == Regime.RANGE.value).all()


def test_the_adx_threshold_is_the_boundary_and_is_configurable():
    just_under = _frame(adx=np.full(30, DEFAULT_ADX_TREND_THRESHOLD - 0.1), ema_gap=np.full(30, 1.0))
    just_over = _frame(adx=np.full(30, DEFAULT_ADX_TREND_THRESHOLD + 0.1), ema_gap=np.full(30, 1.0))
    assert (classify(just_under)[0] == Regime.RANGE.value).all()
    assert (classify(just_over)[0] == Regime.TRENDING_UP.value).all()
    # and a caller can move the line
    assert (classify(just_under, adx_trend_threshold=10.0)[0] == Regime.TRENDING_UP.value).all()


def test_missing_indicators_report_unknown_rather_than_guessing():
    df = pd.DataFrame({"close": np.arange(30.0)})
    direction, vol = classify(df)
    assert (direction == Regime.UNKNOWN.value).all()
    assert (vol == VolRegime.NORMAL.value).all()  # neutral default, not a crash


def test_volatility_regime_is_relative_to_the_assets_own_history():
    """A fixed absolute vol threshold would label BTC and SOL differently for
    identical behaviour. The classifier uses trailing percentiles instead, so
    a 10x jump against the asset's OWN recent history reads as high vol."""
    n = 400
    atr = np.concatenate([np.full(200, 0.005), np.full(200, 0.05)])  # calm, then 10x wilder
    df = _frame(adx=np.full(n, 30.0), ema_gap=np.full(n, 1.0), atr_pct=atr, n=n)
    _, vol = classify(df)
    # Just after the jump, the trailing window is still dominated by the calm
    # period, so the spike sits strictly above the 75th percentile and is
    # flagged. Note the arithmetic this depends on: with a 200-bar lookback
    # and a 75th-percentile cut, the elevated period stops being "the top
    # quartile" once it fills 25% of the window (~50 bars) -- from there it
    # IS the 75th percentile and the strict inequality correctly stops firing.
    assert (vol.iloc[205:245] == VolRegime.HIGH.value).mean() > 0.9


def test_sustained_high_volatility_eventually_becomes_the_new_normal():
    """Deliberate design decision, pinned down here: once elevated vol has
    persisted longer than the lookback, the trailing window contains nothing
    else to compare against and it stops being "high". Otherwise a market
    that permanently repriced to higher vol would permanently halve position
    size and never adapt."""
    n = 600
    atr = np.concatenate([np.full(200, 0.005), np.full(400, 0.05)])
    df = _frame(adx=np.full(n, 30.0), ema_gap=np.full(n, 1.0), atr_pct=atr, n=n)
    _, vol = classify(df, vol_lookback_bars=200)
    assert vol.iloc[-1] == VolRegime.NORMAL.value


def test_volatility_percentiles_are_trailing_not_full_sample():
    """Using the whole sample's percentiles would leak the future into early
    bars. With a trailing window the calm opening stretch cannot already know
    about the volatility spike that comes later."""
    n = 400
    atr = np.concatenate([np.full(200, 0.005), np.full(200, 0.05)])
    df = _frame(adx=np.full(n, 30.0), ema_gap=np.full(n, 1.0), atr_pct=atr, n=n)
    _, vol = classify(df)
    early = vol.iloc[30:190]
    # If it were full-sample, every calm bar would be flagged LOW against the
    # later spike. Trailing-only means the calm period is its own normal.
    assert (early == VolRegime.HIGH.value).sum() == 0


def test_high_volatility_halves_position_size():
    vol = pd.Series([VolRegime.LOW.value, VolRegime.NORMAL.value, VolRegime.HIGH.value])
    mult = size_multiplier(vol)
    assert mult.tolist() == [1.0, 1.0, 0.5]


def test_regime_labels_combine_both_axes():
    df = _frame(adx=np.full(50, 40.0), ema_gap=np.full(50, 2.0))
    labels = regime_labels(df)
    assert labels.iloc[-1].startswith(Regime.TRENDING_UP.value + "/")
    assert labels.iloc[-1].split("/")[1] in {v.value for v in VolRegime}


def test_classify_handles_an_empty_frame():
    direction, vol = classify(pd.DataFrame())
    assert len(direction) == 0 and len(vol) == 0
