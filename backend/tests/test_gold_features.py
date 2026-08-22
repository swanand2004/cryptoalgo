import numpy as np
import pandas as pd
import pytest

from app.gold.features import (
    GOLD_FEATURE_COLUMNS,
    GOLD_FEATURE_COLUMNS_LONG_TERM,
    GOLD_WARMUP_BARS,
    GOLD_WARMUP_BARS_LONG_TERM,
    _row_sign_mean,
    build_gold_feature_frame,
    gold_regime,
)

ASTRO_COLUMNS = [
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
]


def _synthetic_gold_candles(n: int = 500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.003, n)
    close = 4_500 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    time = np.arange(n) * 3600  # hourly bars, no volume (spot index)

    return pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close})


def test_build_gold_feature_frame_produces_every_declared_column():
    candles = _synthetic_gold_candles()
    features = build_gold_feature_frame(candles)

    for col in GOLD_FEATURE_COLUMNS:
        assert col in features.columns, f"missing feature column: {col}"


def test_features_are_nan_during_warmup_and_valid_after():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    # EMA-based features (ema_spread_*) are defined from row 0 by
    # construction (an EWM has no strict warmup the way a rolling window
    # does) -- only genuinely rolling-window features are NaN this early.
    first_row = features.iloc[0]
    long_lookback_cols = ["ma_z_200", "hurst_200"]
    for col in long_lookback_cols:
        assert pd.isna(first_row[col]), f"{col} should still be NaN on the very first row"

    last_row = features.iloc[-1]
    for col in GOLD_FEATURE_COLUMNS:
        assert pd.notna(last_row[col]), f"{col} should be valid by the last row"


def test_rsi_columns_stay_within_the_0_100_bound():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    for col in ["rsi_7", "rsi_14", "rsi_21", "rsi_28"]:
        valid = features[col].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


def test_hurst_columns_stay_within_a_reasonable_bound():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    for col in ["hurst_50", "hurst_100", "hurst_200"]:
        valid = features[col].dropna()
        if len(valid) == 0:
            continue
        assert (valid >= 0).all() and (valid <= 1.5).all()


def test_cyclical_seasonality_encodings_stay_within_unit_circle_bounds():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    for col in ["day_of_week_sin", "day_of_week_cos", "month_of_year_sin", "month_of_year_cos"]:
        assert (features[col] >= -1.0001).all() and (features[col] <= 1.0001).all()


def test_no_volume_column_required():
    """The `.DEXAUTUSD` spot index carries no real trade volume -- the
    feature bank must not depend on it at all (unlike crypto's vwap)."""
    candles = _synthetic_gold_candles(n=300)
    assert "volume" not in candles.columns
    features = build_gold_feature_frame(candles)  # must not raise KeyError
    assert len(features) == len(candles)


def test_gold_regime_labels():
    assert gold_regime(0.7) == "trending"
    assert gold_regime(0.3) == "mean_reverting"
    assert gold_regime(0.5) == "random_walk"
    assert gold_regime(float("nan")) == "unknown"


def test_warmup_constant_covers_every_lookback_used():
    """GOLD_WARMUP_BARS must be at least as large as the longest single
    lookback actually used (hurst_200) -- otherwise training would silently
    treat still-warming-up rows as valid.
    """
    assert GOLD_WARMUP_BARS >= 200


def test_astrology_columns_are_present_and_valid_from_row_zero():
    """Astro features are point-in-time astronomical calculations (no
    rolling window) -- unlike the technical indicators, they need NO
    warmup at all and must be valid even on the very first row.
    """
    candles = _synthetic_gold_candles(n=200)
    features = build_gold_feature_frame(candles)

    for col in ASTRO_COLUMNS:
        assert col in features.columns
        assert pd.notna(features[col].iloc[0]), f"{col} should need no warmup"
        assert pd.notna(features[col].iloc[-1])


def test_astro_bias_columns_stay_within_their_declared_range():
    candles = _synthetic_gold_candles(n=200)
    features = build_gold_feature_frame(candles)

    for col in ["astro_tithi_class_bias", "astro_vara_lord_bias", "astro_hora_bias", "astro_nakshatra_lord_bias"]:
        assert (features[col] >= -1.0001).all() and (features[col] <= 1.0001).all()
    assert (features["astro_graha_yuddha_intensity"] >= 0).all() and (features["astro_graha_yuddha_intensity"] <= 1).all()
    assert (features["astro_yoga_severity"] >= 0).all() and (features["astro_yoga_severity"] <= 2).all()
    assert (features["astro_malefic_retrograde_count"] >= 0).all() and (features["astro_malefic_retrograde_count"] <= 2).all()
    assert (features["astro_combust_count"] >= 0).all() and (features["astro_combust_count"] <= 5).all()


def test_fibonacci_position_stays_within_0_1():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    for col in ["fib_position_50", "fib_position_100"]:
        valid = features[col].dropna()
        assert (valid >= -0.0001).all() and (valid <= 1.0001).all()


def test_linreg_r2_stays_within_0_1_and_slope_is_scale_free():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    for r2_col in ["linreg_r2_20", "linreg_r2_50"]:
        valid = features[r2_col].dropna()
        assert (valid >= -0.0001).all() and (valid <= 1.0001).all()
    # Slope normalized by price -> should be a small fraction, not a raw
    # price-unit number (gold trades around several thousand dollars).
    for slope_col in ["linreg_slope_20", "linreg_slope_50"]:
        valid = features[slope_col].dropna()
        assert (valid.abs() < 1.0).all()


def test_long_term_feature_columns_all_fit_within_their_smaller_warmup():
    """Every column in the reduced long-term list must actually be non-NaN
    once GOLD_WARMUP_BARS_LONG_TERM rows have passed -- the exact bug this
    reduced list exists to avoid (see GOLD_HORIZONS' docstring).
    """
    candles = _synthetic_gold_candles(n=GOLD_WARMUP_BARS_LONG_TERM + 20)
    features = build_gold_feature_frame(candles)

    last_row = features.iloc[-1]
    for col in GOLD_FEATURE_COLUMNS_LONG_TERM:
        assert pd.notna(last_row[col]), f"{col} should be valid within the long-term warmup budget"


MACRO_COLUMNS = ["macro_event_proximity", "macro_nearest_is_fomc", "macro_nearest_is_cpi", "macro_nearest_is_nfp"]


def test_macro_calendar_columns_are_present_and_valid_from_row_zero():
    """Real FOMC/CPI/NFP calendar proximity -- a pure calendar calculation,
    needs no warmup at all, unlike the technical indicators."""
    candles = _synthetic_gold_candles(n=200)
    features = build_gold_feature_frame(candles)

    for col in MACRO_COLUMNS:
        assert col in features.columns
        assert pd.notna(features[col].iloc[0])
        assert pd.notna(features[col].iloc[-1])


def test_macro_event_proximity_stays_within_0_1_and_exactly_one_type_flag_set():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    assert (features["macro_event_proximity"] >= -0.0001).all()
    assert (features["macro_event_proximity"] <= 1.0001).all()
    flag_sum = features["macro_nearest_is_fomc"] + features["macro_nearest_is_cpi"] + features["macro_nearest_is_nfp"]
    assert (flag_sum == 1.0).all()


def test_macro_columns_included_in_the_long_term_reduced_feature_set_too():
    """No rolling-window cost -- these belong in long_term's reduced list
    just as much as short_term's full one."""
    for col in MACRO_COLUMNS:
        assert col in GOLD_FEATURE_COLUMNS_LONG_TERM


DEEP_COLUMNS = [
    "deep_trend_consensus", "deep_meanrev_signal", "deep_regime_confidence",
    "deep_astro_consensus", "deep_composite_score",
]


def test_row_sign_mean_matches_a_hand_verified_case():
    """3 of 4 columns positive, 1 negative -> mean of signs = (1+1+1-1)/4 = 0.5."""
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [-0.5], "d": [3.0]})
    result = _row_sign_mean(df, ["a", "b", "c", "d"])
    assert result.iloc[0] == pytest.approx(0.5)


def test_deep_analysis_columns_are_present_and_valid_after_warmup():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    last_row = features.iloc[-1]
    for col in DEEP_COLUMNS:
        assert col in features.columns
        assert pd.notna(last_row[col])


def test_deep_regime_confidence_and_composite_score_stay_within_declared_bounds():
    candles = _synthetic_gold_candles(n=500)
    features = build_gold_feature_frame(candles)

    confidence = features["deep_regime_confidence"].dropna()
    assert (confidence >= -0.0001).all() and (confidence <= 1.0001).all()

    composite = features["deep_composite_score"].dropna()
    assert (composite >= -1.0001).all() and (composite <= 1.0001).all()


def test_deep_columns_included_in_the_long_term_reduced_feature_set_too():
    """Built only from base columns present in both feature sets (hurst_50,
    not hurst_100/200; ma_z_20/50, not ma_z_100/200) -- so these composites
    are valid, and included, for long_term too."""
    for col in DEEP_COLUMNS:
        assert col in GOLD_FEATURE_COLUMNS_LONG_TERM
