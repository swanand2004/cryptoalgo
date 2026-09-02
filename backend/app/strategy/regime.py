from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    """What kind of market this bar is in.

    Deliberately two INDEPENDENT axes rather than one flat label, because
    "trending" and "volatile" are not alternatives -- a market can trend
    quietly or chop violently, and those need opposite responses (size up a
    quiet trend, size down a violent range). `classify` returns both.
    """

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE = "range"
    UNKNOWN = "unknown"


class VolRegime(str, Enum):
    LOW = "low_vol"
    NORMAL = "normal_vol"
    HIGH = "high_vol"


# Defaults chosen from established convention, NOT fitted to this project's
# data -- fitting regime boundaries to the same history the strategy is then
# evaluated on is how a sweep launders overfitting into "regime awareness".
#
# ADX 25 is the textbook trend/no-trend line (Wilder's own threshold, still
# the standard). The vol percentiles are symmetric tails of the asset's OWN
# trailing distribution rather than absolute vol numbers, so they mean the
# same thing across BTC/ETH/SOL and across calm and wild months.
DEFAULT_ADX_TREND_THRESHOLD = 25.0
DEFAULT_VOL_LOOKBACK_BARS = 200
DEFAULT_LOW_VOL_PERCENTILE = 0.25
DEFAULT_HIGH_VOL_PERCENTILE = 0.75


def classify(
    df: pd.DataFrame,
    adx_trend_threshold: float = DEFAULT_ADX_TREND_THRESHOLD,
    vol_lookback_bars: int = DEFAULT_VOL_LOOKBACK_BARS,
    low_vol_percentile: float = DEFAULT_LOW_VOL_PERCENTILE,
    high_vol_percentile: float = DEFAULT_HIGH_VOL_PERCENTILE,
) -> tuple[pd.Series, pd.Series]:
    """Returns (directional_regime, volatility_regime), one label per bar.

    Every input is already computed by build_feature_frame -- this classifies,
    it does not re-derive indicators, so there is exactly one definition of
    ADX/EMA/ATR in the codebase.

    Strictly causal: the volatility percentile is taken over a TRAILING window
    (shifted by one bar), never the full-sample distribution. Using the whole
    sample's percentiles would leak the future into every early bar and is a
    classic way to manufacture a regime filter that only works in hindsight.
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=object), pd.Series(dtype=object)

    adx = df["adx_14"] if "adx_14" in df.columns else pd.Series(np.nan, index=df.index)

    # Direction from EMA structure, strength from ADX. Both must agree before
    # a bar is called a trend; ADX is strength-only and says nothing about
    # which way, so on its own it cannot separate up from down.
    if "ema_9" in df.columns and "ema_21" in df.columns:
        ema_gap = df["ema_9"] - df["ema_21"]
    elif "ema_9_norm" in df.columns and "ema_21_norm" in df.columns:
        ema_gap = df["ema_9_norm"] - df["ema_21_norm"]
    else:
        ema_gap = pd.Series(np.nan, index=df.index)

    trending = adx >= adx_trend_threshold
    direction = pd.Series(Regime.RANGE.value, index=df.index, dtype=object)
    direction[trending & (ema_gap > 0)] = Regime.TRENDING_UP.value
    direction[trending & (ema_gap < 0)] = Regime.TRENDING_DOWN.value
    direction[adx.isna() | ema_gap.isna()] = Regime.UNKNOWN.value

    # Volatility relative to the asset's own recent history. atr_pct (ATR
    # divided by price) rather than raw ATR, so a $77,000 BTC and a $98 SOL
    # are on the same scale.
    if "atr_pct" in df.columns:
        vol = df["atr_pct"]
    elif "realized_vol_20" in df.columns:
        vol = df["realized_vol_20"]
    else:
        vol = pd.Series(np.nan, index=df.index)

    window = max(20, min(vol_lookback_bars, n))
    trailing = vol.shift(1).rolling(window, min_periods=max(20, window // 4))
    low_cut = trailing.quantile(low_vol_percentile)
    high_cut = trailing.quantile(high_vol_percentile)

    # STRICT inequalities, deliberately. With `<=` / `>=`, a flat stretch of
    # volatility (where the 25th and 75th percentiles collapse onto the same
    # value) satisfies BOTH conditions on every bar, which silently labelled
    # calm, perfectly steady markets as HIGH volatility. Requiring the current
    # reading to sit strictly OUTSIDE the trailing band means:
    #   - flat vol            -> vol equals both cuts -> neither fires -> NORMAL
    #   - a genuine spike     -> strictly above the band -> HIGH
    #   - sustained high vol  -> becomes its own band -> back to NORMAL
    # which is the intended behaviour in all three cases.
    vol_regime = pd.Series(VolRegime.NORMAL.value, index=df.index, dtype=object)
    vol_regime[vol < low_cut] = VolRegime.LOW.value
    vol_regime[vol > high_cut] = VolRegime.HIGH.value
    vol_regime[vol.isna() | low_cut.isna() | high_cut.isna()] = VolRegime.NORMAL.value

    return direction, vol_regime


def regime_labels(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Single combined label per bar, e.g. "trending_up/high_vol" -- the form
    the backtest report's per-regime breakdown consumes."""
    direction, vol = classify(df, **kwargs)
    return direction.astype(str) + "/" + vol.astype(str)


# How much of normal size to take in each volatility regime. Sizing DOWN in
# high vol is the whole point: a fixed contract count in a market whose ATR
# just doubled is silently a doubled risk position.
VOL_SIZE_MULTIPLIER = {
    VolRegime.LOW.value: 1.0,
    VolRegime.NORMAL.value: 1.0,
    VolRegime.HIGH.value: 0.5,
}


def size_multiplier(vol_regime: pd.Series) -> pd.Series:
    return vol_regime.map(VOL_SIZE_MULTIPLIER).fillna(1.0).astype(float)
