from __future__ import annotations

import numpy as np
import pandas as pd


def infer_bars_per_hour(time_seconds: pd.Series) -> float:
    """Infers how many bars represent one hour from the median spacing
    between consecutive timestamps (in seconds).

    A window meant to span "1 week" only actually spans a week if it's sized
    in BARS-per-resolution, not a bare bar count reused across resolutions --
    168 bars is a week at 1h candles, but only ~42 hours at 15m candles. This
    lets callers convert an hours-based window (168 = 1 week) into the right
    bar count for whatever resolution the data is actually at.
    """
    diffs = time_seconds.diff().dropna()
    if diffs.empty:
        return 1.0
    median_seconds = diffs.median()
    if not median_seconds or median_seconds <= 0:
        return 1.0
    return 3600.0 / median_seconds


def funding_z_score(funding_rate: pd.Series, window: int = 168) -> pd.Series:
    """Z-score of the funding rate vs its trailing window (default 168 = 1 week
    of hourly funding realizations). Extreme positive z => crowded longs paying
    shorts richly (fade candidate); extreme negative => crowded shorts.
    """
    mean = funding_rate.rolling(window).mean()
    std = funding_rate.rolling(window).std()
    # np.nan (not pd.NA) keeps the result a plain float64 Series -- mixing in
    # pd.NA silently upcasts to object dtype, which then makes downstream
    # fillna/arithmetic emit FutureWarnings and lose vectorized performance.
    return (funding_rate - mean) / std.replace(0, np.nan)


def basis(mark_price: pd.Series, spot_price: pd.Series) -> pd.Series:
    """(mark - spot) / spot -- positive means the perpetual trades rich to spot."""
    return (mark_price - spot_price) / spot_price
