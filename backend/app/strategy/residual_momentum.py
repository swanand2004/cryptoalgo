from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.base import Strategy


class ResidualMomentumStrategy(Strategy):
    """Momentum on the part of a symbol's move that BTC does NOT explain.

    Why this exists, when the book already holds three momentum strategies:
    those three measured 0.733 mean pairwise return correlation with each
    other -- about 1.3 effective independent bets out of 9 (see
    [[project-risk-controls]]) -- because they are all directional
    trend-followers reading the same market-wide move. Adding a fourth
    trend-follower would buy nothing.

    Residual momentum is the one signal measured in this project that is
    materially LESS correlated with plain momentum than the existing
    strategies are with each other: 0.339 on ETH and 0.558 on SOL, against
    the book's own 0.733. The mechanism is structural rather than
    statistical -- 68.9% of ETH's variance and 55.1% of SOL's is the common
    BTC factor, so 31% and 45% respectively is idiosyncratic movement that a
    directional book cannot see by construction.

    Signal: rolling-beta-hedged return, accumulated and normalised by its own
    residual volatility (so it is comparable across assets and regimes), then
    clipped to [-1, 1].

    NOT tradeable on the base asset itself: BTC's beta to BTC is exactly 1,
    so its residual is identically zero. `build_feature_frame` emits NaN for
    that case rather than floating-point noise, and this strategy returns a
    flat 0.0 signal when the columns are absent or unusable -- it simply
    abstains on BTC instead of inventing a signal.
    """

    name = "residual_momentum"

    def __init__(self, fast: int = 12, slow: int = 48, cap: float = 2.0) -> None:
        self.fast = fast
        self.slow = slow
        self.cap = cap

    def warmup_bars(self) -> int:
        # The beta window inside build_feature_frame is the binding constraint,
        # not `slow` -- it is sized in hours and scaled to the bar resolution,
        # so leave generous headroom rather than assuming a bar count.
        return 220

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        fast_col, slow_col = f"resid_mom_{self.fast}", f"resid_mom_{self.slow}"
        if fast_col not in df.columns or slow_col not in df.columns:
            return pd.Series(0.0, index=df.index)

        fast = df[fast_col].astype(float)
        slow = df[slow_col].astype(float)
        # All-NaN means either no cross-asset data was supplied or this IS the
        # base asset -- abstain rather than fabricate a signal.
        if not fast.notna().any() or not slow.notna().any():
            return pd.Series(0.0, index=df.index)

        # Two horizons voting, same barbell logic as tsmom: a short lookback
        # for responsiveness and a longer one for persistence, so a single
        # bar's residual spike cannot carry the signal on its own.
        combined = 0.5 * fast + 0.5 * slow
        signal = (combined / self.cap).clip(-1.0, 1.0)
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
