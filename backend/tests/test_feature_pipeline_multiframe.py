import numpy as np
import pandas as pd
import pytest

from app.features.feature_pipeline import build_feature_frame

N = 100
_TIMES_15M = pd.Series(1_700_000_000 + np.arange(N) * 900)  # 15m bars


def _candles(n=N, times=None, seed=0, trend=0.0) -> pd.DataFrame:
    times = times if times is not None else _TIMES_15M
    close = pd.Series(100.0 + trend * np.arange(len(times)) + np.cumsum(np.random.default_rng(seed).normal(0, 0.3, len(times))))
    return pd.DataFrame(
        {"time": times, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0}
    )


def test_htf_features_default_neutral_when_absent():
    df = build_feature_frame(_candles())
    assert (df["htf_trend_norm"] == 0.0).all()
    assert (df["htf_rsi_14"] == 50.0).all()
    assert (df["htf_bb_z"] == 0.0).all()


def test_htf_trend_norm_reflects_the_higher_timeframe_direction_not_the_base_one():
    # Base (15m) grid is flat/noisy; the "4h" candles fed as
    # higher_timeframe_candles have a clear, strong uptrend -- htf_trend_norm
    # must reflect THAT, proving it's reading the higher timeframe, not
    # silently re-deriving something from the base resolution.
    base = _candles(trend=0.0)
    htf_times = pd.Series(1_700_000_000 + np.arange(40) * 14400)  # 4h bars
    htf = _candles(n=40, times=htf_times, trend=2.0)  # strong uptrend

    df = build_feature_frame(base, higher_timeframe_candles=htf)
    assert df["htf_trend_norm"].iloc[-1] > 0.3  # clearly positive, not just noise near 0


def test_htf_features_never_leak_a_still_forming_higher_timeframe_bar():
    """Regression test for a real, caught-before-shipping leak: a 15m row
    EARLY inside a still-forming 4h bar must NOT see that bar's own
    indicators (computed from its eventual close/high/low, i.e. genuine
    future information relative to that row) -- it must still see the
    PREVIOUS, already-closed 4h bar's reading. Only once the base-
    resolution row reaches or passes that 4h bar's actual close time
    should the newer reading appear. Caught empirically: this exact bug
    inflated SOLUSD's walk-forward accuracy from a real 49.9% to a
    fabricated 61.9% before being fixed.
    """
    htf_bar_seconds = 14400  # 4h
    htf_times = pd.Series([0, htf_bar_seconds])  # two 4h bars: [0,14400), [14400,28800)
    flat_close = pd.Series([100.0] * 30)
    trending_close = pd.Series(100.0 + 5.0 * np.arange(30))  # bar1: strong uptrend within itself
    htf = pd.DataFrame(
        {
            "time": htf_times,
            "open": [flat_close.iloc[-1], trending_close.iloc[0]],
            "high": [flat_close.iloc[-1] + 1, trending_close.max()],
            "low": [flat_close.iloc[-1] - 1, trending_close.min()],
            "close": [flat_close.iloc[-1], trending_close.iloc[-1]],  # bar1's close: only known at t=28800
            "volume": 1.0,
        }
    )

    # Base (15m) rows: one early inside bar1's still-forming span (15 min
    # after it opened), one right at bar1's actual close.
    base_times = pd.Series([htf_bar_seconds + 900, 2 * htf_bar_seconds])
    base = _candles(n=2, times=base_times)

    df = build_feature_frame(base, higher_timeframe_candles=htf)

    # The early row (15 min into bar1) must NOT reflect bar1's eventual
    # strong uptrend -- bar1 hasn't closed yet as of that row's timestamp.
    early_reading = df["htf_trend_norm"].iloc[0]
    assert not (early_reading > 0.3), (
        f"leaked bar1's still-forming close into an early row (htf_trend_norm={early_reading})"
    )

    # The row AT bar1's actual close time is fair game -- may or may not
    # yet reflect it depending on merge_asof's inclusive/exclusive boundary,
    # but must not error and must stay finite.
    assert np.isfinite(df["htf_trend_norm"].iloc[1])


def test_rsi_and_adx_slope_capture_trajectory_not_just_level():
    df = build_feature_frame(_candles(trend=0.5))  # steady uptrend -> RSI/ADX should trend too
    # Both slopes must actually vary across the series (not constant/always
    # zero), proving they carry information beyond the plain level columns.
    assert df["rsi_14_slope_5"].dropna().std() > 0
    assert df["adx_14_slope_5"].dropna().std() > 0


def test_ret_1_skew_requires_warmup_like_every_other_indicator():
    df = build_feature_frame(_candles())
    assert df["ret_1_skew_20"].iloc[:20].isna().all()
    assert df["ret_1_skew_20"].iloc[25:].notna().all()


def test_cross_asset_btc_feature_defaults_neutral_when_absent():
    df = build_feature_frame(_candles())
    assert (df["btc_ret_5"] == 0.0).all()


def test_cross_asset_btc_feature_reflects_btc_own_return():
    base = _candles(seed=1, trend=0.0)  # ETH/SOL's own (unrelated) price action
    btc = _candles(seed=2, trend=3.0)  # BTC rallying hard
    df = build_feature_frame(base, cross_asset_candles=btc)

    expected_btc_ret5 = btc["close"].pct_change(5)
    # merge_asof(direction="backward") on identical timestamps degenerates
    # to a direct alignment -- compare the tail where both are well past
    # warmup.
    assert df["btc_ret_5"].iloc[-1] == pytest.approx(expected_btc_ret5.iloc[-1], rel=1e-6)
    assert df["btc_ret_5"].iloc[-1] > 0  # BTC's rally should show up as a positive return


def test_deployed_vs_excluded_new_columns():
    """rsi_14_slope_5/adx_14_slope_5/ret_1_skew_20 passed honest walk-forward
    + holdout validation (accuracy up on all 3 symbols) and are deployed.
    htf_trend_norm/htf_rsi_14/htf_bb_z (helped only SOLUSD) and btc_ret_5
    (too small an effect to trust) did not, and stay computed-but-unused --
    see feature_pipeline.py's FEATURE_COLUMNS comment for the full story.
    """
    from app.features.feature_pipeline import FEATURE_COLUMNS

    for deployed_col in ("rsi_14_slope_5", "adx_14_slope_5", "ret_1_skew_20"):
        assert deployed_col in FEATURE_COLUMNS
    for excluded_col in ("htf_trend_norm", "htf_rsi_14", "htf_bb_z", "btc_ret_5"):
        assert excluded_col not in FEATURE_COLUMNS
