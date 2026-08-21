import numpy as np
import pandas as pd
import pytest

from app.features.feature_pipeline import FEATURE_COLUMNS, build_feature_frame

skyfield = pytest.importorskip("skyfield")

N = 150
_TIMES = pd.Series(1_700_000_000 + np.arange(N) * 900)


def _candles() -> pd.DataFrame:
    close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, N)))
    return pd.DataFrame(
        {"time": _TIMES, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0}
    )


def test_hurst_and_expiry_columns_present():
    df = build_feature_frame(_candles())
    assert "hurst_100" in df.columns
    assert "weekly_expiry_proximity" in df.columns
    assert np.isfinite(df["weekly_expiry_proximity"].to_numpy()).all()
    assert (df["weekly_expiry_proximity"] >= 0).all() and (df["weekly_expiry_proximity"] <= 1.0).all()


def test_hurst_and_expiry_not_yet_deployed_pending_validation():
    assert "hurst_100" not in FEATURE_COLUMNS
    assert "weekly_expiry_proximity" not in FEATURE_COLUMNS
