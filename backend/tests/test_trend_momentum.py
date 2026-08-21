import pandas as pd
import pytest

from app.strategy.trend_momentum import TrendMomentumStrategy


def _df(adx_values: list[float]) -> pd.DataFrame:
    n = len(adx_values)
    return pd.DataFrame(
        {
            "ema_9": [104.0] * n,
            "ema_21": [100.0] * n,  # spread=4 -> clipped to the |3| cap -> full magnitude
            "atr_14": [1.0] * n,
            "macd_hist": [0.5] * n,  # agrees with the long bias
            "adx_14": adx_values,
        }
    )


def test_flat_high_adx_passes_full_signal():
    """An ADX that's high and NOT declining (a genuine, active trend) should
    be trusted at the level gate's full value.
    """
    strat = TrendMomentumStrategy()
    df = _df([40.0] * 10)
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] > 0.9  # near-full magnitude, gate fully open


def test_declining_adx_damps_signal_even_at_a_nominally_high_level():
    """Regression test: ADX is a lagging measure -- after a real trend ends
    it can stay elevated for a while purely on residual momentum from a move
    that's already over. Before this fix, a still-high-but-decaying ADX (the
    exact pattern observed live: BTC read 41.6, declining from 53 ten bars
    earlier, hours into a market that had gone flat) fully unlocked the
    signal regardless of trajectory. It must now be damped.
    """
    strat = TrendMomentumStrategy()
    # Declining from 53 -> 41.6 over the last 5 bars, mirroring the live
    # BTCUSD reading that motivated this fix.
    adx_values = [53.7, 52.8, 51.9, 51.3, 50.0, 49.0, 47.8, 44.6, 41.6]
    df = _df(adx_values)

    flat_df = _df([41.6] * len(adx_values))  # same ending level, no decline

    declining_signal = strat.generate_signal(df).iloc[-1]
    flat_signal = strat.generate_signal(flat_df).iloc[-1]

    assert declining_signal < flat_signal
    assert declining_signal < 0.5 * flat_signal  # meaningfully damped, not just nudged


def test_low_adx_still_fully_gates_regardless_of_slope():
    strat = TrendMomentumStrategy()
    df = _df([10.0] * 10)  # below adx_gate_low=15 throughout, flat
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] == 0.0


def test_missing_adx_column_leaves_signal_ungated():
    strat = TrendMomentumStrategy()
    df = _df([40.0] * 10).drop(columns=["adx_14"])
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] > 0.9


def _df_with_er(adx_values: list[float], er_values: list[float]) -> pd.DataFrame:
    df = _df(adx_values)
    df["efficiency_ratio_10"] = er_values
    return df


def test_high_efficiency_ratio_passes_full_signal_alongside_high_adx():
    strat = TrendMomentumStrategy()  # ER blend is on by default
    df = _df_with_er([40.0] * 10, [0.8] * 10)  # both measures agree: real trend
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] > 0.9


def test_low_efficiency_ratio_damps_but_does_not_zero_a_high_adx_signal():
    """Backtested across BTC/ETH/SOL (full-window, train/holdout, and 4
    threshold bands each): requiring ADX and ER to unanimously AGREE
    (multiplicative gate) consistently hurt ETHUSD and was flat-to-worse on
    BTCUSD, helping only SOLUSD -- a 1-of-3 result. A SOFT BLEND instead
    (each gate contributes half a vote) improved SOLUSD in every band
    tested, improved ETHUSD's out-of-sample holdout Sharpe with drawdown
    nearly halved, and left BTCUSD within noise of unchanged -- so ADX and
    ER are AVERAGED, not multiplied. That means a full disagreement (ADX
    maximally confident, ER reading pure chop) damps the signal by half
    rather than zeroing it outright, unlike the old AND gate this replaced.
    """
    strat = TrendMomentumStrategy()
    high_adx_high_er = _df_with_er([40.0] * 10, [0.8] * 10)
    high_adx_low_er = _df_with_er([40.0] * 10, [0.05] * 10)

    agree_signal = strat.generate_signal(high_adx_high_er).iloc[-1]
    disagree_signal = strat.generate_signal(high_adx_low_er).iloc[-1]

    assert disagree_signal < agree_signal
    assert disagree_signal > 0.0  # damped, but NOT fully gated -- the key difference from AND
    assert disagree_signal == pytest.approx(0.5 * agree_signal, abs=1e-9)


def test_efficiency_ratio_gate_can_be_disabled():
    strat = TrendMomentumStrategy(use_efficiency_ratio_gate=False)
    df = _df_with_er([40.0] * 10, [0.0] * 10)  # ER would damp the signal if it were active
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] > 0.9  # ignored -- ADX-only gates still pass it


def test_missing_efficiency_ratio_column_leaves_signal_gated_by_adx_only():
    strat = TrendMomentumStrategy()
    df = _df([40.0] * 10)  # no efficiency_ratio_10 column at all
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] > 0.9
