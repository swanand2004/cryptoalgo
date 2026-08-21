from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.base import Strategy


class MeanReversionStrategy(Strategy):
    """Fades Bollinger-band z-score extremes, gated by RSI agreement, sized down
    when realized vol is elevated (a stretched band means less in a vol spike),
    and dampened by ADX -- fading a move during a genuine strong trend (a real
    breakout, not noise) is exactly what kills naive mean-reversion, so this
    strategy backs off precisely where trend_momentum turns on.

    Entry requires the FULL z_entry threshold (previously gated at half of
    it), which was firing on fairly mundane 0.75-std wobbles rather than
    genuinely stretched conditions.
    """

    name = "mean_reversion"

    def __init__(
        self,
        z_entry: float = 1.5,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        # Was 30.0 -- but trend_momentum's own ADX gate (adx_gate_low=15,
        # adx_gate_high=30) ramps UP from 15 to full conviction by 30. With
        # this dampener not starting until 30, both strategies could be at
        # full/near-full weight simultaneously through ADX 15-30 -- exactly
        # the window where a trend is just forming, the single worst time to
        # fade a move. Starting the dampener at 15 backs mean_reversion off
        # in step with trend_momentum ramping on, instead of only after
        # trend_momentum is already at full conviction.
        adx_trend_cutoff: float = 15.0,
    ) -> None:
        self.z_entry = z_entry
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.adx_trend_cutoff = adx_trend_cutoff

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        z = df["bb_z"]
        rsi = df["rsi_14"]

        fade = -z.clip(-3, 3) / 3.0  # price above upper band (z>0) -> bearish fade

        rsi_confirm_short = (rsi > self.rsi_overbought).astype(float)
        rsi_confirm_long = (rsi < self.rsi_oversold).astype(float)
        confirm = np.where(fade < 0, rsi_confirm_short, rsi_confirm_long)

        vol_median = df["realized_vol_20"].rolling(100, min_periods=20).median()
        vol_ratio = (df["realized_vol_20"] / vol_median.replace(0, np.nan)).clip(0.2, 3.0)
        vol_dampener = (1.0 / vol_ratio).clip(0.2, 1.0)

        raw = fade * (0.4 + 0.6 * confirm) * vol_dampener

        if "adx_14" in df.columns:
            # Full weight below the cutoff, fading out to 0 by 1.5x the cutoff
            # -- avoids fading genuinely strong trends.
            range_ = self.adx_trend_cutoff * 0.5
            trend_dampener = (1.0 - (df["adx_14"] - self.adx_trend_cutoff) / range_).clip(0, 1)
            raw = raw * trend_dampener.fillna(1.0)

        raw = raw.where(z.abs() >= self.z_entry, 0.0)
        return raw.clip(-1, 1).fillna(0.0)
