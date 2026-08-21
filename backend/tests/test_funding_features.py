import numpy as np
import pandas as pd
import pytest

from app.features.funding import infer_bars_per_hour
from app.features.feature_pipeline import build_feature_frame


def test_infer_bars_per_hour_for_15m_candles():
    times = pd.Series(np.arange(0, 3600 * 10, 900))  # 15m spacing
    assert infer_bars_per_hour(times) == pytest.approx(4.0)


def test_infer_bars_per_hour_for_1h_candles():
    times = pd.Series(np.arange(0, 3600 * 200, 3600))  # 1h spacing
    assert infer_bars_per_hour(times) == pytest.approx(1.0)


def test_infer_bars_per_hour_for_1m_candles():
    times = pd.Series(np.arange(0, 3600 * 5, 60))  # 1m spacing
    assert infer_bars_per_hour(times) == pytest.approx(60.0)


def test_infer_bars_per_hour_handles_empty_or_single_row():
    assert infer_bars_per_hour(pd.Series([], dtype=float)) == 1.0
    assert infer_bars_per_hour(pd.Series([100.0])) == 1.0


def _candles(n: int, step_seconds: int, start_price: float = 100.0) -> pd.DataFrame:
    times = np.arange(0, n * step_seconds, step_seconds)
    prices = start_price + np.cumsum(np.random.default_rng(0).normal(0, 0.1, size=n))
    return pd.DataFrame(
        {
            "time": times,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1.0,
        }
    )


def test_funding_z_window_spans_roughly_one_week_regardless_of_resolution():
    """Regression test: funding_z_score's window used to be a bare 168-bar
    count reused verbatim across resolutions, so it meant ~1 week at 1h
    candles but only ~42 hours at 15m candles -- the "extreme funding"
    threshold was silently ~4x more reactive at the system's actual default
    resolution (15m) than the design intended. build_feature_frame must
    scale the window so it's genuinely ~168 HOURS regardless of resolution.
    """
    n = 2000
    candles_15m = _candles(n, step_seconds=900)
    candles_1h = _candles(n, step_seconds=3600)

    funding_times = np.arange(0, n * 3600, 3600)  # hourly funding realizations
    funding_values = np.sin(np.arange(len(funding_times)) / 20) * 0.001
    funding_df = pd.DataFrame({"time": funding_times, "value": funding_values})

    feats_15m = build_feature_frame(candles_15m, funding=funding_df)
    feats_1h = build_feature_frame(candles_1h, funding=funding_df)

    # Both should produce a z-score column with the same general shape of
    # rolling-window "burn-in" (NaN early on) roughly proportional to the
    # SAME time span (1 week), not the same BAR count.
    z_15m = feats_15m["funding_z"]
    z_1h = feats_1h["funding_z"]

    # At 15m resolution, 1 week = 672 bars; at 1h resolution, 1 week = 168
    # bars. Before the fix, both would have used a fixed 168-bar window.
    first_valid_15m = z_15m.replace(0.0, np.nan).first_valid_index() or 0
    first_valid_1h = z_1h.replace(0.0, np.nan).first_valid_index() or 0

    # The 15m series should need roughly 4x as many BARS to warm up as the
    # 1h series, since it needs the same number of HOURS (~1 week) to fill
    # its rolling window.
    assert first_valid_15m > first_valid_1h * 3
