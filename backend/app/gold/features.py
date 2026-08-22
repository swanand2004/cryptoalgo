from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.astro.panchang import (
    all_pairs_aspect_score,
    benefic_malefic_aspect_score,
    combust_planet_count,
    graha_yuddha_intensity,
    hora_bias,
    malefic_retrograde_count,
    nakshatra_lord_bias,
    tithi_class_bias,
    vara_lord_bias,
    yoga_severity,
)
from app.features.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    efficiency_ratio,
    hurst_exponent,
    macd,
    realized_vol,
    rsi,
)
from app.gold.macro_calendar import macro_event_proximity

# A deliberately large GRID of the same underlying indicator families at MANY
# different parameterizations, rather than one instance of each -- this is
# gold's OWN research feature bank, distinct in composition from the crypto
# FEATURE_COLUMNS (app/features/feature_pipeline.py): no funding/OI/long-
# short-ratio/DVOL/astrology (none of that applies to a spot commodity index),
# instead a much wider technical parameter sweep plus autocorrelation and
# seasonality, which the crypto pipeline doesn't use at all. Every reading in
# this bank is also what the dedicated Gold page displays directly -- the
# "tons of calculations" are the visible research surface, not just hidden
# model inputs.
GOLD_FEATURE_COLUMNS: list[str] = [
    # Trend (EMA spread, normalized by the slower EMA so the reading is
    # scale-free across gold's own price history) at 3 timescale pairs.
    "ema_spread_9_21",
    "ema_spread_21_50",
    "ema_spread_50_200",
    # RSI at 4 lookbacks -- short (7) through long (28).
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "rsi_28",
    # MACD histogram (normalized by price) at a standard and a slower
    # parameter set -- two different "speeds" of the same construction.
    "macd_hist_12_26_9",
    "macd_hist_24_52_18",
    # Rate-of-change / momentum at 5 horizons spanning intraday to ~quarterly
    # (at whatever resolution the frame is built at).
    "roc_1",
    "roc_5",
    "roc_10",
    "roc_20",
    "roc_60",
    # Mean-reversion: Bollinger z-score at 3 lookbacks.
    "bb_z_10",
    "bb_z_20",
    "bb_z_50",
    # Mean-reversion: distance from SMA in rolling-std units, at 4 windows.
    "ma_z_20",
    "ma_z_50",
    "ma_z_100",
    "ma_z_200",
    # Volatility regime: realized vol at 3 windows.
    "realized_vol_10",
    "realized_vol_20",
    "realized_vol_50",
    # Volatility regime: ATR as a fraction of price, at 3 lookbacks.
    "atr_pct_7",
    "atr_pct_14",
    "atr_pct_28",
    # Trendiness (two independent measures, each at multiple scales).
    "efficiency_ratio_10",
    "efficiency_ratio_20",
    "efficiency_ratio_50",
    "adx_14",
    "adx_28",
    # Long-memory regime: rolling Hurst exponent at 3 windows (trending vs
    # mean-reverting vs random-walk, at 3 different lookback scales).
    "hurst_50",
    "hurst_100",
    "hurst_200",
    # Autocorrelation of returns at 3 lags -- a direct read on whether
    # consecutive moves reinforce (momentum) or reverse (mean-reversion).
    "autocorr_lag1",
    "autocorr_lag5",
    "autocorr_lag10",
    # Seasonality, cyclically encoded (sin/cos) so e.g. December and January
    # sit close together instead of maximally far apart as raw integers
    # would -- learned empirically by the model from this series' own
    # history, not asserted from outside claims about "gold seasonality".
    "day_of_week_sin",
    "day_of_week_cos",
    "month_of_year_sin",
    "month_of_year_cos",
    # Classic technical levels: where price sits in its recent trading
    # range (0 = at the recent swing low, 1 = at the recent swing high) --
    # the same 0/23.6/38.2/50/61.8/100% Fibonacci retracement framework
    # widely used in real gold technical analysis, at 2 lookback scales.
    "fib_position_50",
    "fib_position_100",
    # Linear-regression trend channel: OLS slope (normalized by price) and
    # R-squared (how orderly/channel-like the trend is, vs. noisy) at 2
    # windows -- a standard quant trend-quality pair, independent of the
    # EMA-spread/efficiency-ratio/ADX measures above.
    "linreg_slope_20",
    "linreg_r2_20",
    "linreg_slope_50",
    "linreg_r2_50",
    # Financial astrology: real, time-only planetary-transit calculations
    # (no natal/genesis chart -- gold, unlike a crypto asset, has no listing
    # "birth moment" to build one from; these read the CURRENT sky only,
    # exactly how actual financial astrology treats a commodity). Reuses
    # app/astro/panchang.py's existing, independently-tested calculations.
    # astro_all_pairs_aspect_score is the one mechanism already VALIDATED
    # for crypto (see project memory) -- the rest are included here as
    # genuinely untested-for-gold research inputs, not established.
    "astro_all_pairs_aspect_score",
    "astro_benefic_malefic_score",
    "astro_yoga_severity",
    "astro_malefic_retrograde_count",
    "astro_combust_count",
    "astro_tithi_class_bias",
    "astro_vara_lord_bias",
    "astro_hora_bias",
    "astro_nakshatra_lord_bias",
    "astro_graha_yuddha_intensity",
    # Real macro/forex calendar: proximity to the nearest of {FOMC rate
    # decision, CPI release, NFP jobs report} -- the specific US economic
    # events most documented to move gold (via real-yield/USD-strength
    # expectations) and forex broadly, verified against the Fed's/BLS's own
    # published 2026 schedules (see app/gold/macro_calendar.py). The 3
    # nearest-event-type flags let the model tell a rate-decision-driven
    # move apart from an inflation-print-driven or labor-market-driven one,
    # not just "an event is close."
    "macro_event_proximity",
    "macro_nearest_is_fomc",
    "macro_nearest_is_cpi",
    "macro_nearest_is_nfp",
    # Deep analysis: SECOND-ORDER composites built from the raw indicators
    # above (a real "ensemble of signals" technique, not a new raw
    # calculation) -- computed and re-evaluated every tick, same as
    # everything else. deep_trend_consensus/deep_meanrev_signal are two
    # competing directional readings; deep_regime_confidence (how strongly
    # trending vs. choppy the market is right now, from efficiency
    # ratio/ADX/Hurst) decides how much to trust the trend reading over the
    # reversion one; deep_composite_score is that regime-adaptive blend
    # (plus a small astrology tilt), clipped to [-1, 1]. Only uses base
    # columns present in BOTH the full and long-term-reduced feature sets
    # (see GOLD_FEATURE_COLUMNS_LONG_TERM below) so these composites are
    # valid for every horizon, not just short_term/intraday.
    "deep_trend_consensus",
    "deep_meanrev_signal",
    "deep_regime_confidence",
    "deep_astro_consensus",
    "deep_composite_score",
]

# Longest lookback actually used above (hurst_200) plus a safety margin, so
# every feature has genuinely warmed up before a row is considered valid.
# Fine for the short-term (1h) model, which has ~3000 rows of real history
# to spare -- see GOLD_FEATURE_COLUMNS_LONG_TERM below for why the long-term
# (1d) model can't afford this.
GOLD_WARMUP_BARS = 210

# The long-term model trains on `.DEXAUTUSD` DAILY bars, and (checked live
# against Delta's own history, 2026-08-21) only ~127 days of that history
# exist at all -- nowhere near the 200+ rows the 100/200-bar-lookback
# features above need to ever produce a single non-NaN reading. Rather than
# silently training on zero rows (every one of them NaN on those columns),
# the long-term model uses this deliberately SMALLER feature set (dropping
# ma_z_100/200, hurst_100/200, and fib_position_100 -- whose longest
# surviving lookback is 60 bars, not 200 or 100) and a correspondingly
# smaller warmup. The astro_* features need no warmup at all (point-in-time
# astronomical calculations, valid from row 0), so all of them stay in. This
# is an honest capability gap, not a design preference: as more daily
# history accumulates from here on, the full-size feature set becomes
# viable and should replace this one.
GOLD_FEATURE_COLUMNS_LONG_TERM: list[str] = [
    c for c in GOLD_FEATURE_COLUMNS
    if c not in {"ma_z_100", "ma_z_200", "hurst_100", "hurst_200", "fib_position_100"}
]
GOLD_WARMUP_BARS_LONG_TERM = 65


def _roc(close: pd.Series, periods: int) -> pd.Series:
    return close.pct_change(periods)


def _ma_zscore(close: pd.Series, window: int) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std.replace(0, np.nan)


def _fib_position(close: pd.Series, window: int) -> pd.Series:
    """Where price sits within its trailing `window`-bar range, on the
    standard 0-1 Fibonacci-retracement scale (0 = at the recent swing low,
    1 = at the recent swing high) -- the same range real gold technical
    analysis draws its 23.6/38.2/50/61.8% retracement lines across.
    """
    rolling_max = close.rolling(window).max()
    rolling_min = close.rolling(window).min()
    span = (rolling_max - rolling_min).replace(0, np.nan)
    return (close - rolling_min) / span


def _linreg_slope_r2(close: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS linear-regression trend channel: slope (normalized by
    price, so it's a scale-free %-per-bar rate) and R-squared (how well an
    actual straight line fits the last `window` closes -- high R2 means an
    orderly channel, low R2 means a noisy/choppy one). A standard quant
    trend-quality pair, independent of the EMA/efficiency-ratio/ADX trend
    measures elsewhere in this bank.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = float(np.sum(x_centered**2))

    def _fit(y: np.ndarray) -> tuple[float, float]:
        y_mean = y.mean()
        slope = float(np.sum(x_centered * (y - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return slope, r2

    slopes = np.full(len(close), np.nan)
    r2s = np.full(len(close), np.nan)
    values = close.to_numpy()
    for i in range(window - 1, len(values)):
        slope, r2 = _fit(values[i - window + 1 : i + 1])
        slopes[i] = slope / values[i]  # normalize by price -> scale-free
        r2s[i] = r2
    return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)


def _row_sign_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Mean of the SIGN of each listed column, per row -- e.g. 6 of 10
    trend indicators reading positive and 4 negative gives +0.2, not an
    average of their (very differently-scaled) raw magnitudes. This is the
    actual "consensus" computation: how many independent readings agree on
    direction, not how large any one of them is.

    Rows during warmup (before the underlying indicators have any real
    values yet) are legitimately all-NaN here -- expected, and dropped
    later the same way every other warmup row is, not a bug -- so the
    resulting "Mean of empty slice" RuntimeWarning is suppressed rather
    than left to spam logs on every single feature build.
    """
    signs = np.sign(df[cols].to_numpy(dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = np.nanmean(signs, axis=1)
    return pd.Series(values, index=df.index)


def _deep_analysis_layers(out: pd.DataFrame) -> pd.DataFrame:
    """The "deep analysis" pass: second-order composites built ON TOP of
    the already-computed indicators above, not a new raw calculation --
    real ensemble-of-signals engineering, re-evaluated every tick just like
    everything feeding it. Only reads base columns present in BOTH
    GOLD_FEATURE_COLUMNS and GOLD_FEATURE_COLUMNS_LONG_TERM (e.g. hurst_50/
    bb_z_*/ma_z_20/50, never hurst_100/200 or ma_z_100/200), so every one of
    these composites is valid for every horizon, not just short_term/
    intraday. Must run AFTER every base column it reads has already been
    computed onto `out`.
    """
    trend_cols = [
        "ema_spread_9_21", "ema_spread_21_50", "ema_spread_50_200",
        "macd_hist_12_26_9", "macd_hist_24_52_18",
        "roc_5", "roc_10", "roc_20", "linreg_slope_20", "linreg_slope_50",
    ]
    # Bollinger/MA z-scores read as MEAN-REVERSION pressure here (distinct
    # from the Trend layer above): a high positive z-score means price sits
    # well above its recent mean, which this layer reads as "due to revert
    # DOWN" -- hence the negation.
    meanrev_cols = ["bb_z_10", "bb_z_20", "bb_z_50", "ma_z_20", "ma_z_50"]
    # Only the genuinely SIGNED astro columns (positive=benefic tilt,
    # negative=malefic) -- yoga_severity/malefic_retrograde_count/
    # combust_count/graha_yuddha_intensity are non-negative MAGNITUDES
    # (severity/count/tightness, never actually negative), so including
    # their sign would silently skew this "consensus" toward always-
    # positive regardless of what they're reporting.
    astro_cols = [
        "astro_all_pairs_aspect_score", "astro_benefic_malefic_score", "astro_tithi_class_bias",
        "astro_vara_lord_bias", "astro_hora_bias", "astro_nakshatra_lord_bias",
    ]

    out["deep_trend_consensus"] = _row_sign_mean(out, trend_cols)
    out["deep_meanrev_signal"] = -_row_sign_mean(out, meanrev_cols)
    out["deep_astro_consensus"] = _row_sign_mean(out, astro_cols)

    # Trend-STRENGTH (magnitude-only, no direction) from efficiency ratio +
    # ADX -- how much conviction to have at all, regardless of which way.
    efficiency_mean = out[["efficiency_ratio_10", "efficiency_ratio_20", "efficiency_ratio_50"]].mean(axis=1)
    adx_mean_normalized = (out[["adx_14", "adx_28"]].mean(axis=1) / 100.0).clip(0, 1)
    out["deep_regime_confidence"] = ((efficiency_mean + adx_mean_normalized) / 2).clip(0, 1)

    # hurst_50 (not 100/200, so this stays valid for long_term too) as the
    # trend-vs-reversion MIX weight: >0.5 trending -> trust the trend
    # reading more; <0.5 mean-reverting -> trust the reversion reading more.
    trend_weight = out["hurst_50"].clip(0, 1)
    blended_direction = trend_weight * out["deep_trend_consensus"] + (1 - trend_weight) * out["deep_meanrev_signal"]
    out["deep_composite_score"] = (
        out["deep_regime_confidence"] * blended_direction + 0.15 * out["deep_astro_consensus"]
    ).clip(-1, 1)

    return out


def _autocorrelation(returns: pd.Series, lag: int, window: int = 50) -> pd.Series:
    """Rolling lag-N autocorrelation of returns: positive means a move tends
    to be followed by another move in the SAME direction (momentum at that
    lag); negative means it tends to reverse (mean-reversion at that lag).
    """
    return returns.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag + 1 else np.nan, raw=False
    )


def build_gold_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Builds gold's own research feature bank from raw OHLCV candles (only
    `time`/`open`/`high`/`low`/`close` are required -- the underlying
    `.DEXAUTUSD` spot index carries no real trade volume, so nothing here
    depends on it). Every column in GOLD_FEATURE_COLUMNS is computed here;
    `build_gold_feature_frame(...)`'s LAST row is also what the live
    /api/gold/snapshot endpoint reads for "current reading" display.
    """
    out = candles.copy().reset_index(drop=True)
    close, high, low = out["close"], out["high"], out["low"]

    ema_9, ema_21, ema_50, ema_200 = ema(close, 9), ema(close, 21), ema(close, 50), ema(close, 200)
    out["ema_spread_9_21"] = (ema_9 - ema_21) / ema_21
    out["ema_spread_21_50"] = (ema_21 - ema_50) / ema_50
    out["ema_spread_50_200"] = (ema_50 - ema_200) / ema_200

    out["rsi_7"] = rsi(close, 7)
    out["rsi_14"] = rsi(close, 14)
    out["rsi_21"] = rsi(close, 21)
    out["rsi_28"] = rsi(close, 28)

    macd_std = macd(close, 12, 26, 9)
    macd_slow = macd(close, 24, 52, 18)
    out["macd_hist_12_26_9"] = macd_std["macd_hist"] / close
    out["macd_hist_24_52_18"] = macd_slow["macd_hist"] / close

    out["roc_1"] = _roc(close, 1)
    out["roc_5"] = _roc(close, 5)
    out["roc_10"] = _roc(close, 10)
    out["roc_20"] = _roc(close, 20)
    out["roc_60"] = _roc(close, 60)

    out["bb_z_10"] = bollinger(close, 10)["bb_z"]
    out["bb_z_20"] = bollinger(close, 20)["bb_z"]
    out["bb_z_50"] = bollinger(close, 50)["bb_z"]

    out["ma_z_20"] = _ma_zscore(close, 20)
    out["ma_z_50"] = _ma_zscore(close, 50)
    out["ma_z_100"] = _ma_zscore(close, 100)
    out["ma_z_200"] = _ma_zscore(close, 200)

    out["realized_vol_10"] = realized_vol(close, 10)
    out["realized_vol_20"] = realized_vol(close, 20)
    out["realized_vol_50"] = realized_vol(close, 50)

    out["atr_pct_7"] = atr(high, low, close, 7) / close
    out["atr_pct_14"] = atr(high, low, close, 14) / close
    out["atr_pct_28"] = atr(high, low, close, 28) / close

    out["efficiency_ratio_10"] = efficiency_ratio(close, 10)
    out["efficiency_ratio_20"] = efficiency_ratio(close, 20)
    out["efficiency_ratio_50"] = efficiency_ratio(close, 50)
    out["adx_14"] = adx(high, low, close, 14)
    out["adx_28"] = adx(high, low, close, 28)

    out["hurst_50"] = hurst_exponent(close, window=50)
    out["hurst_100"] = hurst_exponent(close, window=100)
    out["hurst_200"] = hurst_exponent(close, window=200)

    returns = close.pct_change()
    out["autocorr_lag1"] = _autocorrelation(returns, lag=1)
    out["autocorr_lag5"] = _autocorrelation(returns, lag=5)
    out["autocorr_lag10"] = _autocorrelation(returns, lag=10)

    times = pd.to_datetime(out["time"], unit="s", utc=True)
    day_of_week = times.dt.dayofweek
    month = times.dt.month
    out["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    out["month_of_year_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_of_year_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    out["fib_position_50"] = _fib_position(close, 50)
    out["fib_position_100"] = _fib_position(close, 100)

    out["linreg_slope_20"], out["linreg_r2_20"] = _linreg_slope_r2(close, 20)
    out["linreg_slope_50"], out["linreg_r2_50"] = _linreg_slope_r2(close, 50)

    times_utc = out["time"].to_numpy(dtype=float)
    out["astro_all_pairs_aspect_score"] = all_pairs_aspect_score(times_utc)
    out["astro_benefic_malefic_score"] = benefic_malefic_aspect_score(times_utc)
    out["astro_yoga_severity"] = yoga_severity(times_utc)
    out["astro_malefic_retrograde_count"] = malefic_retrograde_count(times_utc)
    out["astro_combust_count"] = combust_planet_count(times_utc)
    out["astro_tithi_class_bias"] = tithi_class_bias(times_utc)
    out["astro_vara_lord_bias"] = vara_lord_bias(times_utc)
    out["astro_hora_bias"] = hora_bias(times_utc)
    out["astro_nakshatra_lord_bias"] = nakshatra_lord_bias(times_utc)
    out["astro_graha_yuddha_intensity"] = graha_yuddha_intensity(times_utc)

    (
        out["macro_event_proximity"],
        out["macro_nearest_is_fomc"],
        out["macro_nearest_is_cpi"],
        out["macro_nearest_is_nfp"],
    ) = macro_event_proximity(times_utc)

    out = _deep_analysis_layers(out)

    return out


def gold_regime(hurst_100: float) -> str:
    """Classifies the current Hurst(100) reading into a plain-language
    regime label for display -- not a model feature by itself (hurst_100
    already is one), just a human-readable summary of it.
    """
    if pd.isna(hurst_100):
        return "unknown"
    if hurst_100 > 0.55:
        return "trending"
    if hurst_100 < 0.45:
        return "mean_reverting"
    return "random_walk"
