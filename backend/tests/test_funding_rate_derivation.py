import numpy as np
import pandas as pd
import pytest

from app.features.feature_pipeline import build_feature_frame
from app.features.funding import (
    FUNDING_CLAMP_PCT,
    FUNDING_INTEREST_RATE_PCT,
    average_premium_pct,
    funding_rate_from_premium,
)

N = 100
_TIMES = pd.Series(1_700_000_000 + np.arange(N) * 3600)  # hourly bars


def _candles() -> pd.DataFrame:
    close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, N)))
    return pd.DataFrame(
        {"time": _TIMES, "open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 1.0}
    )


def _series(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"time": _TIMES, "value": values})


def test_funding_formula_reproduces_deltas_own_published_numbers():
    """Verified live against Delta's /v2/tickers on 2026-09-02: SOLUSD's
    premium of -0.0059% produced exactly the 0.01% funding rate Delta itself
    reported. This is the check that the formula is Delta's, not invented."""
    premium = pd.Series([-0.0059])
    assert funding_rate_from_premium(premium).iloc[0] == pytest.approx(0.01, abs=1e-9)


def test_clamp_binds_when_premium_is_far_from_the_interest_rate():
    """funding = premium + clamp(0.01 - premium, -0.05, +0.05). A deeply
    negative premium pushes (0.01 - premium) past the +0.05 cap, so funding
    becomes premium + 0.05 exactly."""
    deep_negative = pd.Series([-0.20])
    assert funding_rate_from_premium(deep_negative).iloc[0] == pytest.approx(-0.20 + FUNDING_CLAMP_PCT)

    deep_positive = pd.Series([0.20])
    assert funding_rate_from_premium(deep_positive).iloc[0] == pytest.approx(0.20 - FUNDING_CLAMP_PCT)


def test_funding_equals_interest_rate_when_premium_is_inside_the_band():
    """Inside the band the clamp does nothing, so funding collapses to the
    interest rate regardless of the exact premium -- the flat region that
    makes most observed funding readings sit at 0.01%."""
    for premium in (-0.03, 0.0, 0.02, 0.04):
        result = funding_rate_from_premium(pd.Series([premium])).iloc[0]
        assert result == pytest.approx(FUNDING_INTEREST_RATE_PCT)


def test_funding_rate_is_not_the_premium_it_was_derived_from():
    """The regression this whole module exists for: the FUNDING: candle series
    carries the PREMIUM, and using it directly as the funding rate gave the
    wrong sign and ~4x the magnitude."""
    premium = pd.Series([-0.044])
    funding = funding_rate_from_premium(premium).iloc[0]
    assert funding > 0  # real funding was POSITIVE while the premium was negative
    assert abs(funding - premium.iloc[0]) > 0.04  # materially different, not a rounding nit


def test_average_premium_uses_an_eight_hour_window_scaled_to_resolution():
    premium = pd.Series([0.0] * 8 + [8.0] * 8)
    hourly = average_premium_pct(premium, bars_per_hour=1.0)
    # After 8 bars of 8.0 following 8 bars of 0.0, an 8-bar mean is exactly 8.0
    assert hourly.iloc[-1] == pytest.approx(8.0)
    # Halfway through the step, the 8-bar window still holds four 0.0 readings
    assert hourly.iloc[11] == pytest.approx(4.0)


def test_feature_frame_exposes_premium_and_derived_funding_separately():
    """Both quantities must be present and distinct -- before the fix the
    pipeline had two features measuring premium and none measuring funding."""
    df = build_feature_frame(_candles(), funding=_series([-0.044] * N))
    assert "premium_index" in df.columns
    assert "funding_rate" in df.columns
    assert df["premium_index"].iloc[-1] == pytest.approx(-0.044)
    assert df["funding_rate"].iloc[-1] == pytest.approx(-0.044 + FUNDING_CLAMP_PCT)
    assert df["funding_rate"].iloc[-1] > 0 > df["premium_index"].iloc[-1]


def test_funding_spread_compares_like_for_like_units():
    """Delta's funding is in percent, ccxt's Binance funding is a fraction.
    Subtracting them raw was a 100x mismatch that made Binance's leg ~500x
    too small to affect the result."""
    # Binance 0.0001 fraction == 0.01% == Delta's in-band funding rate,
    # so a like-for-like spread here must be ~0, not ~0.01.
    df = build_feature_frame(
        _candles(),
        funding=_series([0.0] * N),  # premium 0 -> funding lands at 0.01%
        binance_funding=_series([0.0001] * N),  # fraction: also 0.01%
    )
    assert df["funding_spread"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_missing_funding_series_still_degrades_to_neutral():
    df = build_feature_frame(_candles())
    assert (df["premium_index"] == 0.0).all()
    assert (df["funding_rate"] == 0.0).all()
    assert (df["funding_z"] == 0.0).all()


def test_premium_z_preserves_what_the_mislabeled_funding_z_used_to_carry():
    """The rename must not silently strip information. Post-fix `funding_z` is
    a z-score of true funding, which the clamp pins to a constant most of the
    time; `premium_z` carries the varying quantity the model was actually
    trained on."""
    from app.features.feature_pipeline import FEATURE_COLUMNS
    from app.features.funding import funding_z_score

    rng = np.random.default_rng(3)
    premium = list(rng.normal(-0.04, 0.01, N))  # realistic: mostly inside the clamp band
    df = build_feature_frame(_candles(), funding=_series(premium))

    assert "premium_z" in FEATURE_COLUMNS
    expected = funding_z_score(pd.Series(premium), window=168).fillna(0.0)
    assert df["premium_z"].iloc[-1] == pytest.approx(expected.iloc[-1], abs=1e-9)


def test_true_funding_is_mostly_flat_which_is_why_premium_z_is_needed():
    """Documents the measured reason premium_z exists: with the premium
    sitting inside the clamp band, true funding is literally constant, so its
    z-score carries no information at all."""
    inside_band = [-0.02] * N  # 0.01 - (-0.02) = 0.03, well inside +-0.05
    df = build_feature_frame(_candles(), funding=_series(inside_band))
    assert df["funding_rate"].nunique() == 1  # genuinely constant
    assert (df["funding_z"] == 0.0).all()  # a constant has no z-score
    assert (df["premium_z"] == 0.0).all()  # constant premium here too, by construction
