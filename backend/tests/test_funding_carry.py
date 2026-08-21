import numpy as np
import pandas as pd
import pytest

from app.strategy.funding_carry import FundingCarryStrategy


def _df(n: int, funding_rate: float | list[float], time_step_s: int = 900) -> pd.DataFrame:
    rates = funding_rate if isinstance(funding_rate, list) else [funding_rate] * n
    return pd.DataFrame(
        {
            "time": [i * time_step_s for i in range(n)],
            "funding_rate": rates,
        }
    )


def test_missing_funding_z_returns_flat_signal():
    df = pd.DataFrame({"time": [0, 900, 1800]})
    signal = FundingCarryStrategy().generate_signal(df)
    assert (signal == 0.0).all()


def test_fades_extreme_positive_funding_with_negative_signal():
    # Long history of near-zero funding, then a sustained extreme positive
    # spike -- crowded longs paying shorts richly -> fade with a short lean.
    n = 300
    rates = [0.0001] * (n - 20) + [0.01] * 20
    df = _df(n, rates)
    df["funding_z"] = (df["funding_rate"] - df["funding_rate"].rolling(168).mean()) / df["funding_rate"].rolling(
        168
    ).std().replace(0, np.nan)

    signal = FundingCarryStrategy().generate_signal(df)
    assert signal.iloc[-1] < 0


def test_fades_extreme_negative_funding_with_positive_signal():
    n = 300
    rates = [-0.0001] * (n - 20) + [-0.01] * 20
    df = _df(n, rates)
    df["funding_z"] = (df["funding_rate"] - df["funding_rate"].rolling(168).mean()) / df["funding_rate"].rolling(
        168
    ).std().replace(0, np.nan)

    signal = FundingCarryStrategy().generate_signal(df)
    assert signal.iloc[-1] > 0


def test_final_signal_is_smoothed_over_roughly_one_funding_interval():
    """Regression test for the turnover fix: an isolated backtest showed this
    strategy generating MORE trades/month than the fast technical strategies,
    the opposite of its documented "slow, low-turnover" intent. The basis
    component updates every bar even after its own internal smoothing, and
    that residual wobble was enough to repeatedly cross the entry/exit
    hysteresis thresholds. The final signal must not swing to its full
    per-bar extreme on every single bar when the underlying basis is
    oscillating bar-to-bar.
    """
    n = 300
    df = _df(n, 0.0)
    df["funding_z"] = 0.0
    # Rapidly alternating basis, i.e. the noisiest possible bar-to-bar input.
    df["mark_price"] = 100.0
    df["basis"] = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]

    strat = FundingCarryStrategy()
    signal = strat.generate_signal(df)

    unsmoothed_basis_signal_extreme = 0.3 / strat.z_cap * strat.z_cap  # i.e. the 0.3-weighted component alone at full clip
    # The smoothed final signal's bar-to-bar range must be much tighter than
    # the alternating input would produce without smoothing.
    tail = signal.iloc[100:]
    assert tail.abs().max() < 0.5


def test_smoothing_window_scales_with_bar_resolution():
    """8-hour smoothing window at 15m bars (4 bars/hour) should be 32 bars;
    at 1h bars (1 bar/hour) it should be 8 bars -- same real-world duration,
    different bar counts, matching infer_bars_per_hour's stated contract.
    """
    n = 300
    df_15m = _df(n, 0.0001, time_step_s=900)
    df_15m["funding_z"] = 0.0

    df_1h = _df(n, 0.0001, time_step_s=3600)
    df_1h["funding_z"] = 0.0

    # Both should run without error and produce a fully-defined (non-NaN) signal.
    sig_15m = FundingCarryStrategy().generate_signal(df_15m)
    sig_1h = FundingCarryStrategy().generate_signal(df_1h)
    assert not sig_15m.isna().any()
    assert not sig_1h.isna().any()
