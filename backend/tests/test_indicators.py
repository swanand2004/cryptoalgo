import numpy as np
import pandas as pd
import pytest

from app.features.indicators import efficiency_ratio, hurst_exponent


def test_straight_line_trend_has_efficiency_ratio_near_one():
    close = pd.Series(np.arange(1, 31, dtype=float))  # perfectly straight uptrend
    er = efficiency_ratio(close, period=10)
    assert er.iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_zigzag_has_efficiency_ratio_near_zero():
    # Alternates 101/99 every bar -- covers a lot of bar-to-bar ground but
    # ends up almost exactly where it started 10 bars ago.
    n = 30
    close = pd.Series([101.0 if i % 2 == 0 else 99.0 for i in range(n)])
    er = efficiency_ratio(close, period=10)
    assert er.iloc[-1] < 0.3


def test_efficiency_ratio_is_bounded_0_to_1():
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    er = efficiency_ratio(close, period=10)
    valid = er.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_efficiency_ratio_handles_zero_path_length_without_error():
    close = pd.Series([100.0] * 20)  # perfectly flat -- both numerator and denominator are 0
    er = efficiency_ratio(close, period=10)
    assert np.isfinite(er.dropna().to_numpy()).all()  # NaN (from 0/0), not inf or a crash


def test_efficiency_ratio_requires_period_bars_of_warmup():
    close = pd.Series(np.arange(1, 31, dtype=float))
    er = efficiency_ratio(close, period=10)
    assert er.iloc[:10].isna().all()
    assert er.iloc[10:].notna().all()


def test_hurst_exponent_requires_window_bars_of_warmup():
    # First valid value lands at index window-1 (99 for window=100): that's
    # the first row where `window` total closes (hence window-1 log-returns,
    # after the internal NaN filter drops close.pct_change's own leading
    # NaN) are available -- not an extra full window on top of that.
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 250))))
    h = hurst_exponent(close, window=100)
    assert h.iloc[:99].isna().all()
    assert h.iloc[99:].notna().all()


def test_hurst_exponent_near_half_for_a_pure_random_walk():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    h = hurst_exponent(close, window=200)
    # Finite-sample R/S estimates are noisy even for a textbook random walk --
    # a loose band around 0.5, not an exact match.
    assert 0.3 < h.iloc[-1] < 0.7


def test_hurst_exponent_is_higher_for_a_persistent_trending_series():
    # Positive autocorrelation injected directly into the RETURNS (not just
    # a constant drift, which alone would still read H~0.5) -- genuine
    # persistence, the textbook definition of what pushes Hurst above 0.5.
    rng = np.random.default_rng(3)
    n = 400
    returns = np.zeros(n)
    noise = rng.normal(0, 0.01, n)
    for i in range(1, n):
        returns[i] = 0.4 * returns[i - 1] + noise[i]
    close = pd.Series(100 * np.exp(np.cumsum(returns)))
    h_random = hurst_exponent(pd.Series(100 * np.exp(np.cumsum(noise))), window=200).iloc[-1]
    h_persistent = hurst_exponent(close, window=200).iloc[-1]
    assert h_persistent > h_random


def test_hurst_exponent_is_lower_for_an_anti_persistent_zigzag():
    n = 300
    close = pd.Series([100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(n)])
    h = hurst_exponent(close, window=200)
    assert h.iloc[-1] < 0.4
