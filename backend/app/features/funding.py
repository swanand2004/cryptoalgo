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


# Delta's published funding parameters (verified 2026-09-02 against the live
# /v2/tickers funding_rate for all 3 symbols -- SOLUSD reproduced exactly,
# BTC/ETH to within the residual explained by using an instantaneous premium
# instead of Delta's 8-hour TWAP).
FUNDING_INTEREST_RATE_PCT = 0.01  # 0.01% per 8h
FUNDING_CLAMP_PCT = 0.05  # (-0.05%, +0.05%) band


def funding_rate_from_premium(
    premium_pct: pd.Series,
    interest_rate_pct: float = FUNDING_INTEREST_RATE_PCT,
    clamp_pct: float = FUNDING_CLAMP_PCT,
) -> pd.Series:
    """Delta's actual funding rate, derived from the premium index.

        funding = avg_premium + clamp(interest_rate - avg_premium, -0.05%, +0.05%)

    This exists because of a real bug found on 2026-09-02: Delta's
    `FUNDING:{symbol}` CANDLE series does not contain the funding rate at all
    -- it contains the PREMIUM INDEX, (mark - index) / index, in percent.
    Confirmed three ways: the series' live value (-0.0447) matched the live
    mark-vs-spot basis (-0.0431%) while Delta's own ticker reported funding of
    +0.01%; the sign was opposite; and the magnitude was ~4x off. Feeding that
    series in as "funding_rate" meant this project had two features measuring
    the premium and none measuring funding.

    Note the clamp binds whenever the premium sits outside roughly
    [-0.04%, +0.06%], and while it binds funding is just premium plus a
    CONSTANT -- which is why the derived `funding_z` z-score barely moves
    versus the buggy one (a z-score is invariant to adding a constant). The
    raw LEVEL, which is what any carry calculation depends on, was materially
    wrong.

    `premium_pct` must be in PERCENT (as Delta's FUNDING: candles are), not a
    fraction, and should already be the 8h average -- see
    `average_premium_pct`.
    """
    adjustment = (interest_rate_pct - premium_pct).clip(lower=-clamp_pct, upper=clamp_pct)
    return premium_pct + adjustment


def average_premium_pct(premium_pct: pd.Series, bars_per_hour: float) -> pd.Series:
    """Delta computes funding from the 8-HOUR TWAP of the premium, not the
    instantaneous reading -- so a series sampled more often than 8-hourly has
    to be averaged to that window before the funding formula applies.
    """
    window = max(1, round(8 * bars_per_hour))
    return premium_pct.rolling(window, min_periods=1).mean()
