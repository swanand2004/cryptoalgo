import numpy as np
import pandas as pd
import pytest

from app.strategy.time_series_momentum import TimeSeriesMomentumStrategy


def _trending_df(n: int, drift_per_bar: float, atr: float = 1.0) -> pd.DataFrame:
    close = 100.0 + np.arange(n) * drift_per_bar
    return pd.DataFrame({"close": close, "atr_14": [atr] * n})


def test_uptrend_produces_positive_signal():
    strat = TimeSeriesMomentumStrategy(short_lookback=10, long_lookback=30)
    df = _trending_df(60, drift_per_bar=0.5)
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] > 0


def test_downtrend_produces_negative_signal():
    strat = TimeSeriesMomentumStrategy(short_lookback=10, long_lookback=30)
    df = _trending_df(60, drift_per_bar=-0.5)
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] < 0


def test_flat_price_produces_zero_signal():
    strat = TimeSeriesMomentumStrategy(short_lookback=10, long_lookback=30)
    df = _trending_df(60, drift_per_bar=0.0)
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] == pytest.approx(0.0)


def test_stronger_trend_saturates_to_full_conviction():
    strat = TimeSeriesMomentumStrategy(short_lookback=10, long_lookback=30, vol_normalizer=3.0)
    weak = _trending_df(60, drift_per_bar=0.05)
    strong = _trending_df(60, drift_per_bar=5.0)
    weak_signal = strat.generate_signal(weak).iloc[-1]
    strong_signal = strat.generate_signal(strong).iloc[-1]
    assert 0 < weak_signal < strong_signal
    assert strong_signal == pytest.approx(1.0)


def test_missing_required_columns_fails_closed():
    strat = TimeSeriesMomentumStrategy()
    df = pd.DataFrame({"time": [0, 1, 2]})
    signal = strat.generate_signal(df)
    assert (signal == 0.0).all()


def test_zero_atr_does_not_crash_or_produce_inf():
    strat = TimeSeriesMomentumStrategy(short_lookback=5, long_lookback=15)
    df = _trending_df(30, drift_per_bar=1.0, atr=0.0)
    signal = strat.generate_signal(df)
    assert np.isfinite(signal.to_numpy()).all()


def test_required_warmup_bars_matches_long_lookback():
    strat = TimeSeriesMomentumStrategy(short_lookback=20, long_lookback=100)
    assert strat.required_warmup_bars() == 105
