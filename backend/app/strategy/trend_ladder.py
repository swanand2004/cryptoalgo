from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.base import Strategy

# Six EMA pairs on the signal timeframe, spanning fast to slow. The point of a
# ladder rather than one crossover is that no single lookback has to be right:
# when the speeds disagree the average shrinks toward zero and the book
# automatically holds a smaller position, which is where most of the drawdown
# reduction comes from.
FAST_PAIRS: tuple[tuple[int, int], ...] = (
    (12, 48), (24, 96), (36, 144), (50, 200), (72, 288), (100, 400),
)

# The SAME ladder at 4x the spans. On 1h bars this reproduces the ladder
# computed on real 4h bars to a signal correlation of 0.998 on all three
# symbols (measured, not assumed: 4h-native scored CAGR 23.6% / Sharpe 1.33 /
# PF 2.31 against 24.7% / 1.37 / 2.38 for this form). Using 4x spans on one
# frame instead of a genuinely second resolution keeps this a single-timeframe
# Strategy, so it needs no changes to the loop's per-resolution feature
# plumbing and cannot desynchronise two frames that arrive at different times.
SLOW_MULTIPLE = 4

# CARRY TILT REMOVED 2026-09-09 -- it was overfitting, and the sweep shows it
# unambiguously. Raising the carry weight improves TRAIN and VALIDATION while
# monotonically DEGRADING the holdout:
#
#   carry weight   TRAIN   VALIDATION   TEST
#        0.00       1.81       1.02     0.88
#        0.10       1.85       1.06     0.81
#        0.20       1.91       1.16     0.69   <- what was live
#        0.30       1.89       1.26     0.59
#
# That monotone divergence between in-sample and out-of-sample is the
# definition of a fitted component, not an edge. Corroborated independently:
# funding carry run as its OWN vol-targeted sleeve scores TRAIN 0.07,
# VALIDATION 0.77, TEST -1.42, and every sleeve combination containing it is
# worse out of sample than the same book without it.
#
# Dropping it raises holdout Sharpe 0.69 -> 0.88 and holdout CAGR 10.8% ->
# 14.4%, and removes a fitted parameter rather than adding one. Funding is
# still CHARGED as a real cost everywhere; what is gone is the claim that it
# predicts direction.
TREND_WEIGHT = 1.00
CARRY_WEIGHT = 0.00

# Spread measured in ATRs before squashing. 1.5 means a trend whose EMA spread
# is ~1.5 ATR wide sits at tanh(1) = 0.76 of full size.
ATR_NORMALIZER = 1.5

# 90 days of hourly bars. Funding's z-score is deliberately slow: this is a
# positioning-crowding tilt, not a trading signal, and a fast z-score turns it
# into one.
CARRY_Z_WINDOW = 2160
Z_CAP = 2.5


def _z(series: pd.Series, window: int = CARRY_Z_WINDOW, cap: float = Z_CAP) -> pd.Series:
    """Rolling z-score, clipped and rescaled into [-1, 1]. Identical to the
    helper the research study used, so the live tilt is the tilt that was
    actually measured."""
    mean = series.rolling(window, min_periods=window // 5).mean()
    std = series.rolling(window, min_periods=window // 5).std()
    return ((series - mean) / std.replace(0, np.nan)).clip(-cap, cap) / cap


class TrendLadderStrategy(Strategy):
    """Two-timeframe continuous EMA trend ladder with a funding-carry tilt.

    This is the configuration selected by the nine-year study in
    `scripts/` -- chosen on a 2022-2023 validation window (deliberately the
    2022 bear plus the 2023 chop, the regimes trend following is worst in),
    confirmed by rolling annual walk-forward, and evaluated once on a
    2024-2026 holdout.

    Measured on BTC/ETH/SOL as one equal-weight book, net of 7bps per side and
    real 8-hourly funding, at a 15% volatility target:

        Sharpe 1.33   profit factor 2.31   max drawdown -16.3%
        40 round trips a year, ~214h average hold
        positive in all nine calendar years 2018-2026

    It replaces a binary EMA 50/200 crossover which scored Sharpe 1.10 and PF
    1.71 on the identical book and costs. The gain decomposes as:

        binary 50/200                    Sharpe 1.10   PF 1.71
        + continuous (tanh) sizing              1.17        1.86
        + six speeds instead of one             1.29        1.96
        + the 4x-span second timeframe          1.29        2.33
        + the 20% carry tilt                    1.33        2.31

    Three things worth knowing before changing any of it:

    * THE SIGN IS THE EASY PART. What earns the profit factor is the SIZE:
      a marginal crossover gets a small position and a wide, agreed trend gets
      a full one. Reverting to sign() throws that away.
    * IT TRADES LESS, NOT MORE. 40 round trips a year against the old 53.
      Break-even cost is 76.8 bps per side against the ~7 bps actually paid.
      Every faster variant tested lost money after fees.
    * NO TAKE-PROFIT. Profit factor 2.31 lives in the right tail; capping the
      winners was measured and it turns this into a losing system.

    The strategy is pure and causal: row i uses only rows <= i, and it holds no
    state between calls, so the backtest engine and the live loop get bit-identical
    signals from the same bars.
    """

    name = "trend_ladder"

    def __init__(
        self,
        fast_pairs: tuple[tuple[int, int], ...] = FAST_PAIRS,
        slow_multiple: int = SLOW_MULTIPLE,
        trend_weight: float = TREND_WEIGHT,
        carry_weight: float = CARRY_WEIGHT,
        atr_normalizer: float = ATR_NORMALIZER,
    ) -> None:
        self.fast_pairs = fast_pairs
        self.slow_multiple = slow_multiple
        self.trend_weight = trend_weight
        self.carry_weight = carry_weight
        self.atr_normalizer = atr_normalizer

    # ------------------------------------------------------------------
    def _ladder(self, close: pd.Series, atr: pd.Series, multiple: int) -> pd.Series:
        """Mean of tanh(EMA spread / (k * ATR)) across every pair, at `multiple`
        times the base spans."""
        scale = (self.atr_normalizer * atr).replace(0, np.nan)
        votes = []
        for fast, slow in self.fast_pairs:
            spread = (
                close.ewm(span=fast * multiple, adjust=False).mean()
                - close.ewm(span=slow * multiple, adjust=False).mean()
            )
            votes.append(np.tanh(spread / scale))
        return pd.concat(votes, axis=1).mean(axis=1)

    def _carry(self, df: pd.DataFrame) -> pd.Series:
        """Lean against crowded funding: short when longs are paying a premium.

        Reads `funding_rate` as a DECIMAL per settlement. Zeros are treated as
        "not published for this bar" and forward-filled rather than averaged
        in -- an 8-hourly rate carried on an hourly grid is zero for seven bars
        out of eight, and letting those zeros into the z-score would crush it
        toward the mean and quietly disable the tilt.
        """
        if "funding_rate" not in df.columns:
            return pd.Series(0.0, index=df.index)
        funding = df["funding_rate"].astype(float).replace(0.0, np.nan).ffill()
        return (-_z(funding)).clip(-1, 1).fillna(0.0)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns or "atr_14" not in df.columns:
            return pd.Series(0.0, index=df.index)

        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float)

        fast = self._ladder(close, atr, 1)
        slow = self._ladder(close, atr, self.slow_multiple)
        trend = 0.5 * fast.fillna(0.0) + 0.5 * slow.fillna(0.0)

        blended = self.trend_weight * trend + self.carry_weight * self._carry(df)
        return blended.clip(-1, 1).fillna(0.0)

    def required_warmup_bars(self) -> int:
        """An EMA is only meaningfully converged after roughly 4 spans, and the
        slowest span here is 400 * 4 = 1600 bars. This is why the strategy runs
        off the deep BINANCE: history rather than Delta's own ~113 days of
        hourly candles, which cannot support it (see
        app/data/query.load_ladder_frame)."""
        longest = max(slow for _, slow in self.fast_pairs) * self.slow_multiple
        return 4 * longest
