from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.base import Strategy

# W.D. Gann's 1x1 angle, read off his 2-day swing chart, on DAILY bars.
#
# Of Gann's methods this is the one that survived testing (study 2026-10-02,
# BTC/ETH/SOL/BNB, 2017-2026, 7 bps/side + real funding, book vol-targeted):
#
#                               TRAIN <=2021   VAL 2022-23   TEST 2024-26
#   trend_ladder alone             1.44           0.71          0.82
#   50/50 ladder + gann_angle      1.47           0.98          1.10   (maxDD -30% -> -25%)
#
# Better in all three windows, better on TEST for every one of the four coins,
# block-bootstrap P(blend better) = 96.5%. It helps because it is a genuinely
# different bet: book-return correlation with the ladder is only 0.26.
#
# What FAILED the same test and is deliberately not here: the Square of 9
# (its levels did no better than the same grid shifted half a step -- a
# control with no Gann meaning), Gann time cycles (30/45/60/90/120/144/180/360
# days beat only 80% of RANDOM cycle sets), the 2x1 angle, and the 3-day swing
# chart (matched the ladder on TRAIN, collapsed on VALIDATION).
#
# The one judgement call Gann leaves open is the price scale of an angle --
# "one unit of price per unit of time" means nothing until you say what a unit
# of price is. Here it is the daily ATR at the pivot, which makes the 1x1 the
# same steepness in volatility terms on every coin and at every price level.
# It was fixed before testing, not tuned.

SWING_DAYS = 2  # Gann's 2-day swing chart
ANGLE_RATIO = 1.0  # the 1x1
ATR_DAYS = 14
DAY_SECONDS = 86_400


def swing_pivots(high: np.ndarray, low: np.ndarray, n: int = SWING_DAYS) -> list[tuple[int, float, str, int]]:
    """Gann's n-day swing chart. The swing turns up after n consecutive higher
    highs and down after n consecutive lower lows; the extreme of the swing it
    leaves behind is a pivot. Each pivot is (index, price, 'top'|'bottom',
    confirmed_at) -- `confirmed_at` is the bar on which the turn became known,
    which is the earliest the pivot may be used."""
    pivots: list[tuple[int, float, str, int]] = []
    swing_dir, ext_i = 0, 0
    up_run = dn_run = 0
    for i in range(1, len(high)):
        up_run = up_run + 1 if high[i] > high[i - 1] else 0
        dn_run = dn_run + 1 if low[i] < low[i - 1] else 0
        if swing_dir >= 0 and dn_run >= n:
            if swing_dir == 1:
                pivots.append((ext_i, float(high[ext_i]), "top", i))
            swing_dir, ext_i = -1, i
        elif swing_dir <= 0 and up_run >= n:
            if swing_dir == -1:
                pivots.append((ext_i, float(low[ext_i]), "bottom", i))
            swing_dir, ext_i = 1, i
        else:
            if swing_dir == 1 and high[i] >= high[ext_i]:
                ext_i = i
            if swing_dir == -1 and low[i] <= low[ext_i]:
                ext_i = i
    return pivots


def daily_angle_signal(daily: pd.DataFrame, n: int = SWING_DAYS, ratio: float = ANGLE_RATIO) -> pd.Series:
    """+1 while the close holds above the 1x1 rising from the last confirmed
    swing bottom, -1 while it holds below the 1x1 falling from the last
    confirmed swing top, 0 once price crosses back through the angle (Gann:
    breaking the 1x1 means the move has lost its strength)."""
    high, low, close = (daily[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    true_range = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(true_range).ewm(alpha=1 / ATR_DAYS, adjust=False).mean().to_numpy()

    pivots = swing_pivots(high, low, n)
    signal = np.zeros(len(daily))
    k, current = 0, None
    for i in range(len(daily)):
        while k < len(pivots) and pivots[k][3] <= i:
            current = pivots[k]
            k += 1
        if current is None:
            continue
        pivot_i, pivot_price, kind, _ = current
        slope = ratio * atr[pivot_i]
        if kind == "bottom":
            signal[i] = 1.0 if close[i] > pivot_price + slope * (i - pivot_i) else 0.0
        else:
            signal[i] = -1.0 if close[i] < pivot_price - slope * (i - pivot_i) else 0.0
    return pd.Series(signal, index=daily.index)


class GannAngleStrategy(Strategy):
    """Gann 1x1 angle on daily bars, served on the hourly ladder frame.

    Reads the same deep BINANCE: hourly frame as trend_ladder, builds UTC daily
    bars from it, and gives every hourly row the signal decided at the close of
    the PREVIOUS UTC day. Today's partial day is never used, so the value
    cannot repaint intraday and backtest and live see identical numbers.
    """

    name = "gann_angle"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if not {"time", "high", "low", "close"}.issubset(df.columns) or df.empty:
            return pd.Series(0.0, index=df.index)

        day = (df["time"].astype("int64") // DAY_SECONDS).to_numpy()
        daily = (
            df.assign(_day=day)
            .groupby("_day", sort=True)
            .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        )
        decided = daily_angle_signal(daily)
        # Day D's decision applies from the first bar of day D+1 onward.
        prior_day = pd.Series(day - 1)
        values = prior_day.map(decided).fillna(0.0).to_numpy()
        return pd.Series(values, index=df.index)

    def required_warmup_bars(self) -> int:
        # ATR convergence plus a few swings, in hourly rows. Far inside the
        # ladder's own 6,400, which is what the ensemble actually waits for.
        return 60 * 24
