from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.funding import infer_bars_per_hour
from app.strategy.base import Strategy


class FundingCarryStrategy(Strategy):
    """Institutional-style carry: fade crowded positioning implied by extreme
    funding rates / rich basis. Very positive funding (longs paying shorts
    heavily) => lean short; very negative => lean long. This is a slow,
    low-turnover signal meant to blend with the faster technical strategies,
    not to trade alone.
    """

    name = "funding_carry"

    def __init__(self, z_cap: float = 2.5) -> None:
        self.z_cap = z_cap

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "funding_z" not in df.columns:
            return pd.Series(0.0, index=df.index)
        z = df["funding_z"].astype(float)
        signal = -(z.clip(-self.z_cap, self.z_cap) / self.z_cap)

        if "basis" in df.columns and df["basis"].notna().any():
            # Basis is derived from mark/spot prices that update every bar --
            # far noisier than funding (which barely moves within a funding
            # interval). Smoothing the raw series first, before z-scoring,
            # keeps this component genuinely slow like the rest of the
            # strategy instead of injecting bar-to-bar price noise into what's
            # supposed to be a low-turnover carry signal.
            smoothed_basis = df["basis"].rolling(8, min_periods=1).mean()
            # Window sized in hours (168 = 1 week), scaled to actual bar
            # resolution -- see infer_bars_per_hour's docstring. A bare 168
            # would mean ~1 week at 1h candles but only ~42 hours at 15m.
            basis_window = max(2, round(168 * infer_bars_per_hour(df["time"]))) if "time" in df.columns else 168
            basis_z = (
                smoothed_basis - smoothed_basis.rolling(basis_window).mean()
            ) / smoothed_basis.rolling(basis_window).std()
            basis_signal = -(basis_z.clip(-self.z_cap, self.z_cap) / self.z_cap)
            # Funding is the primary carry driver; basis is a secondary confirmer.
            signal = 0.7 * signal.fillna(0.0) + 0.3 * basis_signal.fillna(0.0)

        # Final smoothing over one funding interval (~8h, the real-world cadence
        # this signal should move at): an isolated backtest exposed this
        # strategy generating MORE trades per month than the fast technical
        # strategies -- the opposite of "slow, low-turnover" as documented.
        # The basis component still updates every bar even after its own
        # internal smoothing, and that residual bar-to-bar wobble was enough
        # to repeatedly cross the entry/exit hysteresis thresholds. Smoothing
        # the final output directly enforces the intended cadence instead of
        # hoping the upstream smoothing was enough.
        if "time" in df.columns:
            smoothing_window = max(2, round(8 * infer_bars_per_hour(df["time"])))
            signal = signal.rolling(smoothing_window, min_periods=1).mean()

        return signal.clip(-1, 1).fillna(0.0)
