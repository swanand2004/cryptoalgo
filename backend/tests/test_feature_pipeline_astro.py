import numpy as np
import pandas as pd
import pytest

from app.features.feature_pipeline import FEATURE_COLUMNS, build_feature_frame

skyfield = pytest.importorskip("skyfield")

N = 100
_TIMES = pd.Series(1_700_000_000 + np.arange(N) * 900)

ASTRO_COLUMNS = [
    "astro_paksha_bias",
    "astro_nakshatra_lord_bias",
    "astro_tithi_class_bias",
    "astro_vara_lord_bias",
    "astro_hora_bias",
    "astro_aspect_score",
    "astro_yoga_severity",
    "astro_malefic_retrograde_count",
    "astro_combust_count",
    "astro_graha_yuddha_intensity",
    "astro_rahu_kaal",
    "astro_vishti_karana",
    "astro_dasha_bias",
    "astro_agreement_score",
    "astro_all_pairs_aspect_score",
    "astro_helio_aspect_score",
    "astro_declination_aspect_score",
    "astro_mars_velocity",
    "astro_jupiter_velocity",
    "astro_saturn_velocity",
]


def _candles() -> pd.DataFrame:
    close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, N)))
    return pd.DataFrame(
        {"time": _TIMES, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0}
    )


def test_every_astro_column_is_present_and_finite():
    df = build_feature_frame(_candles())
    for col in ASTRO_COLUMNS:
        assert col in df.columns
        assert np.isfinite(df[col].to_numpy()).all()


def test_dasha_bias_is_neutral_without_a_symbol_but_active_with_one():
    without_symbol = build_feature_frame(_candles())
    with_symbol = build_feature_frame(_candles(), symbol="BTCUSD")

    assert (without_symbol["astro_dasha_bias"] == 0.0).all()
    # BTCUSD has a real genesis chart -- at least sometimes non-neutral.
    assert (with_symbol["astro_dasha_bias"] != 0.0).any()


def test_dasha_bias_neutral_for_unrecognized_symbol():
    df = build_feature_frame(_candles(), symbol="DOGEUSD")
    assert (df["astro_dasha_bias"] == 0.0).all()


def test_agreement_score_is_the_signed_sum_of_seven_systems():
    df = build_feature_frame(_candles(), symbol="BTCUSD")
    expected = (
        np.sign(df["astro_paksha_bias"])
        + np.sign(df["astro_nakshatra_lord_bias"])
        + np.sign(df["astro_tithi_class_bias"])
        + np.sign(df["astro_vara_lord_bias"])
        + np.sign(df["astro_hora_bias"])
        + np.sign(df["astro_aspect_score"])
        + np.sign(df["astro_dasha_bias"])
    )
    assert np.allclose(df["astro_agreement_score"], expected)
    assert (df["astro_agreement_score"].abs() <= 7).all()


def test_deployed_vs_excluded_astro_columns():
    """astro_all_pairs_aspect_score (a Bradley-Barometer-style aggregate
    across all 21 pairs of the 7 classical bodies) passed honest
    walk-forward + a genuine 70/30 holdout -- accuracy up on ALL 3 symbols
    with no exceptions -- and is deployed. Every other astro_* column is
    computed and tested but did not clear that bar (see
    feature_pipeline.py's FEATURE_COLUMNS comment for the full story per
    column) and stays excluded.
    """
    assert "astro_all_pairs_aspect_score" in FEATURE_COLUMNS
    for col in ASTRO_COLUMNS:
        if col == "astro_all_pairs_aspect_score":
            continue
        assert col not in FEATURE_COLUMNS
