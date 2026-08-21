import numpy as np
import pandas as pd
import pytest

from app.features.feature_pipeline import FEATURE_COLUMNS, build_feature_frame


def _candles(n: int, base_price: float, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.004, n)
    close = base_price * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(10, 100, n)
    time = np.arange(n) * 900
    return pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_feature_columns_contain_no_raw_price_scale_indicators():
    """Regression test: ema_9/21/50, atr_14, and macd/macd_signal/macd_hist
    are stored in RAW price (or price-difference) units -- non-stationary
    across a training window where price can easily move double-digit
    percent. FEATURE_COLUMNS (what the ML model actually trains on) must
    use the normalized (price- or ATR-relative) versions instead.
    """
    raw_price_scale_columns = {"ema_9", "ema_21", "ema_50", "atr_14", "macd", "macd_signal", "macd_hist"}
    assert raw_price_scale_columns.isdisjoint(FEATURE_COLUMNS)


def test_normalized_ema_features_are_stationary_across_wildly_different_price_levels():
    """The whole point of the fix: the SAME relative EMA-spread pattern,
    reproduced at two completely different absolute price levels (as if the
    same symbol traded at $100 early in its history and $100,000 today),
    must produce comparable normalized feature values -- unlike the raw
    ema_9 column, whose absolute values would differ by 1000x and be
    meaningless to compare directly.
    """
    low_price = _candles(200, base_price=100.0, seed=11)
    high_price = _candles(200, base_price=100_000.0, seed=11)  # same RNG seed -> same relative pattern

    low_features = build_feature_frame(low_price)
    high_features = build_feature_frame(high_price)

    tail_low = low_features["ema_9_norm"].iloc[-50:].to_numpy()
    tail_high = high_features["ema_9_norm"].iloc[-50:].to_numpy()
    assert np.allclose(tail_low, tail_high, atol=1e-6)

    # The RAW column, by contrast, differs by ~1000x between the two --
    # confirming this test actually distinguishes normalized from raw.
    raw_low = low_features["ema_9"].iloc[-1]
    raw_high = high_features["ema_9"].iloc[-1]
    assert raw_high / raw_low > 500


def test_atr_pct_is_comparable_across_price_levels():
    low_price = _candles(200, base_price=50.0, seed=5)
    high_price = _candles(200, base_price=50_000.0, seed=5)

    low_features = build_feature_frame(low_price)
    high_features = build_feature_frame(high_price)

    assert low_features["atr_pct"].iloc[-1] == pytest.approx(high_features["atr_pct"].iloc[-1], rel=1e-3)


def test_normalized_features_have_no_infinite_values():
    features = build_feature_frame(_candles(300, base_price=2500.0))
    for col in FEATURE_COLUMNS:
        assert np.isfinite(features[col].dropna().to_numpy()).all(), f"{col} has non-finite values"
