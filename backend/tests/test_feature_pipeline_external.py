import numpy as np
import pandas as pd
import pytest

from app.features.feature_pipeline import FEATURE_COLUMNS, build_feature_frame

N = 100
_TIMES = pd.Series(1_700_000_000 + np.arange(N) * 3600)  # hourly bars -> bars_per_hour == 1.0


def _candles() -> pd.DataFrame:
    close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, N)))
    return pd.DataFrame(
        {
            "time": _TIMES,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )


def _series(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"time": _TIMES, "value": values})


def test_missing_external_series_default_to_neutral():
    df = build_feature_frame(_candles())
    assert (df["fear_greed_z"] == 0.0).all()
    assert (df["funding_spread"] == 0.0).all()  # both delta and binance funding default to 0
    assert (df["binance_oi_change"] == 0.0).all()
    assert (df["binance_lsr_z"] == 0.0).all()
    assert (df["dvol_change"] == 0.0).all()
    assert (df["news_sentiment"] == 0.0).all()
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert "news_sentiment" not in FEATURE_COLUMNS  # zero real history on day one -- see the exclusion note


def test_fear_greed_scaled_to_minus_one_to_one():
    extreme_fear = build_feature_frame(_candles(), fear_greed=_series([0.0] * N))
    neutral = build_feature_frame(_candles(), fear_greed=_series([50.0] * N))
    extreme_greed = build_feature_frame(_candles(), fear_greed=_series([100.0] * N))

    assert extreme_fear["fear_greed_z"].iloc[-1] == -1.0
    assert neutral["fear_greed_z"].iloc[-1] == 0.0
    assert extreme_greed["fear_greed_z"].iloc[-1] == 1.0


def test_funding_spread_is_delta_minus_binance():
    """Updated 2026-09-02. This test previously asserted 0.0010 - 0.0004 =
    0.0006, which encoded TWO bugs it was silently protecting:

      1. The `funding` argument is Delta's FUNDING: series, which carries the
         PREMIUM INDEX, not the funding rate. Delta's real funding is derived
         from it: funding = premium + clamp(0.01 - premium, -0.05, +0.05).
         A premium of 0.0010% sits inside the clamp band, so funding = 0.01%.
      2. ccxt reports Binance funding as a FRACTION while Delta's is a
         PERCENT, so subtracting them raw was a 100x mismatch. 0.0004 as a
         fraction is 0.04%.

    Correct spread: 0.01% (Delta) - 0.04% (Binance) = -0.03%.
    """
    df = build_feature_frame(
        _candles(),
        funding=_series([0.0010] * N),
        binance_funding=_series([0.0004] * N),
    )
    assert df["funding_spread"].iloc[-1] == pytest.approx(-0.03)


def test_binance_oi_change_is_relative_not_absolute():
    """OI drifts over years just like price -- the feature must be a
    proportional change, not a raw-level difference, or it inherits the
    same non-stationarity problem already fixed for ema/atr/macd."""
    doubling_oi = [1_000_000.0 * (1.03 ** i) for i in range(N)]  # steady ~3%/bar compounding growth
    df = build_feature_frame(_candles(), binance_oi=_series(doubling_oi))
    # 24-bar-back pct_change of a steady 3%/bar compounding series:
    # (1.03**24 - 1) ~= 1.03, i.e. roughly +103%, regardless of the
    # absolute level reached by bar 100 -- proving it's relative, not raw.
    assert df["binance_oi_change"].iloc[-1] == pytest.approx(1.03**24 - 1, rel=0.05)


def test_binance_lsr_z_flags_a_crowding_extreme():
    lsr_values = [1.0] * (N - 1) + [3.5]  # flat positioning, then a sudden extreme spike
    df = build_feature_frame(_candles(), binance_lsr=_series(lsr_values))
    assert abs(df["binance_lsr_z"].iloc[-1]) > 3.0  # a genuine outlier vs its own recent history
    assert df["binance_lsr_z"].iloc[-2] == pytest.approx(0.0, abs=1e-6)  # flat history -> no signal


def test_dvol_change_defaults_neutral_when_absent_eg_solusd():
    """Deribit has no SOL options market at all -- dvol is simply never
    passed for SOLUSD, and must degrade to neutral, not error or NaN."""
    df = build_feature_frame(_candles())
    assert (df["dvol_change"] == 0.0).all()
    assert np.isfinite(df["dvol_change"].to_numpy()).all()


def test_news_sentiment_is_forward_filled_onto_the_bar_grid():
    """News lands sparsely (a handful of hourly buckets, not one per bar) --
    merge_asof direction='backward' must carry the last known reading
    forward rather than leaving gaps as NaN."""
    sparse_times = pd.Series([_TIMES.iloc[0], _TIMES.iloc[40]])
    sparse_news = pd.DataFrame({"time": sparse_times, "value": [0.7, -0.3]})
    df = build_feature_frame(_candles(), news=sparse_news)
    assert df["news_sentiment"].iloc[0] == pytest.approx(0.7)
    assert df["news_sentiment"].iloc[39] == pytest.approx(0.7)  # still carrying the first reading
    assert df["news_sentiment"].iloc[40] == pytest.approx(-0.3)
    assert df["news_sentiment"].iloc[-1] == pytest.approx(-0.3)
    assert np.isfinite(df["news_sentiment"].to_numpy()).all()
