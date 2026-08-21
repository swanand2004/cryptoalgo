from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.base import Strategy


class TrendMomentumStrategy(Strategy):
    """EMA(9/21) cross direction, confirmed by MACD histogram sign, scaled by
    how extended price is relative to ATR (institutional trend-following style:
    ride the trend, but size down when already extended).

    Gated by ADX (trend strength): a naive EMA cross whipsaws constantly in a
    choppy/ranging market, since price keeps drifting back and forth across
    the two averages with no real direction. ADX distinguishes "actually
    trending" from "chopping sideways" -- the gate fades in smoothly between
    ADX 15-30 rather than a hard cutoff, so it doesn't itself become a source
    of on/off whipsaw at a single threshold.

    The level gate alone isn't enough, though: ADX is a smoothed, lagging
    measure over `adx_period` bars, so after a real trend ends it stays
    elevated for a while purely on residual momentum from a move that's
    already over -- e.g. reading 40+ for hours into a market that's gone
    flat, because the big move is still inside its lookback window. A second
    gate on ADX's own trajectory catches exactly that: a DECLINING ADX (the
    trend fading) damps the signal even while the absolute level still looks
    "trending" on paper, whereas a flat-or-rising ADX (an active, strengthening
    trend) is trusted at the level gate's full value.

    A THIRD gate on Kaufman's Efficiency Ratio (Perry Kaufman, "Trading
    Systems and Methods") adds a second opinion that isn't just another
    lagging derivative of the same ADX calculation: ER is a direct ratio
    (net displacement / total path length over the window), with no
    smoothing of its own, so it doesn't share ADX's specific lag failure
    mode.

    How the two gates are COMBINED matters more than whether ER is included
    at all -- backtested across all 3 symbols at their favorable timeframes
    (full-window, a 70/30 train/holdout split, and 3-4 different threshold
    bands each, to separate a real effect from a lucky cutoff):
      - Requiring BOTH to agree (multiplicative AND, trend_gate * er_gate)
        consistently hurt ETHUSD (Sharpe 2.89->2.22, return 81%->49%
        full-window, training-fold Sharpe going outright negative to -2.87)
        and was flat-to-worse on BTCUSD, helping only SOLUSD -- a 1-of-3
        result. The AND logic gives either gate unilateral veto power, and
        ADX and ER disagree often enough (they measure genuinely different
        things -- smoothed directional strength vs. raw path efficiency)
        that requiring unanimity mostly just throws away real trend trades
        that one of the two measures got right.
      - A SOFT BLEND (0.5*trend_gate + 0.5*er_gate) instead: SOLUSD improved
        in every single band tested (full-window Sharpe 2.95->3.62, holdout
        4.14->4.25), ETHUSD improved on the metric that matters most --
        out-of-sample holdout Sharpe 5.60->6.28 with max drawdown nearly
        halved (-19.4%->-12.6%) -- while BTCUSD stayed within noise of
        unchanged (full-window 4.84->4.80, holdout 5.91->5.89). No band
        produced anything like the AND version's ETHUSD damage. Averaging
        lets either measure's vote count instead of requiring unanimous
        agreement, which is why it doesn't share the AND version's failure
        mode of vetoing good trades over a disagreement between two
        imperfect, differently-flawed measures.
    This is why the combination is a fixed average, not a configurable mode:
    the choice was decided by the data, not left as an open question.
    """

    name = "trend_momentum"

    def __init__(
        self,
        ema_fast: str = "ema_9",
        ema_slow: str = "ema_21",
        adx_gate_low: float = 15.0,
        adx_gate_high: float = 30.0,
        adx_slope_bars: int = 5,
        adx_decay_points: float = 10.0,
        er_gate_low: float = 0.2,
        er_gate_high: float = 0.5,
        use_efficiency_ratio_gate: bool = True,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_gate_low = adx_gate_low
        self.adx_gate_high = adx_gate_high
        # How many bars back to measure ADX's trajectory, and how many ADX
        # points of decline over that span fully zeroes the decay gate.
        self.adx_slope_bars = adx_slope_bars
        self.adx_decay_points = adx_decay_points
        # Efficiency Ratio [0, 1] fades in over this range, same shape as the
        # ADX level gate. Toggleable for A/B backtesting against the
        # ADX-only baseline.
        self.er_gate_low = er_gate_low
        self.er_gate_high = er_gate_high
        self.use_efficiency_ratio_gate = use_efficiency_ratio_gate

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        ema_spread = (df[self.ema_fast] - df[self.ema_slow]) / df["atr_14"].replace(0, np.nan)
        direction = np.sign(ema_spread)
        macd_confirm = np.sign(df["macd_hist"])
        agreement = (direction == macd_confirm).astype(float)

        magnitude = ema_spread.clip(-3, 3).abs() / 3.0
        raw = direction * magnitude * (0.5 + 0.5 * agreement)

        if "adx_14" in df.columns:
            adx = df["adx_14"]
            gate_range = max(self.adx_gate_high - self.adx_gate_low, 1e-9)
            level_gate = ((adx - self.adx_gate_low) / gate_range).clip(0, 1)

            adx_slope = adx.diff(self.adx_slope_bars)
            decay_gate = (1.0 + adx_slope / self.adx_decay_points).clip(0, 1)

            trend_gate = (level_gate * decay_gate).fillna(0.0)

            if self.use_efficiency_ratio_gate and "efficiency_ratio_10" in df.columns:
                er_range = max(self.er_gate_high - self.er_gate_low, 1e-9)
                er_gate = ((df["efficiency_ratio_10"] - self.er_gate_low) / er_range).clip(0, 1).fillna(0.0)
                # Soft blend, not a multiplicative AND -- see class docstring
                # for why requiring unanimous agreement backtested worse.
                trend_gate = 0.5 * trend_gate + 0.5 * er_gate

            raw = raw * trend_gate

        return raw.clip(-1, 1).fillna(0.0)
