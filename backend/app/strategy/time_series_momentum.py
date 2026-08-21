from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.base import Strategy


class TimeSeriesMomentumStrategy(Strategy):
    """Time-series ("absolute") momentum: a barbell of a SHORT and a LONG
    lookback horizon, each independently voting long/short/flat on the SIGN
    of its own trailing cumulative price return, volatility-normalized (ATR-
    relative, scaled by sqrt(lookback) the way realized vol scales with the
    square root of time) before being combined -- a materially different
    construction from `trend_momentum`'s single EMA(9/21) crossover: this
    looks at raw price displacement over two genuinely different horizons
    rather than the spread between one pair of moving averages, and applies
    no ADX/trend-strength gate at all.

    Grounded in real, heavily-cited research rather than invented from
    scratch:
      - Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum" (Journal of
        Financial Economics) -- the original result, replicated across 58
        futures markets: sign of trailing excess return predicts the sign of
        the next period's return, and volatility-scaled position sizing
        contributes MORE to the strategy's edge than the raw momentum signal
        alone (monthly alpha 1.27% with vol scaling vs. 0.41% without, in
        their sample).
      - Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-
        Following Investing" (AQR) -- combines 1/3/12-month horizons; more
        recent replications (e.g. arXiv:2510.23150) find a "barbell" of just
        a short and a long horizon captures nearly all of the benefit, since
        a medium horizon adds little once both ends are present. That's the
        structure used here.
      - Crypto-specific replications (e.g. Bitcoin intraday time-series
        momentum studies, multi-horizon EMA-momentum studies reporting ~32%
        annualized returns) find the effect persists intraday in BTC/ETH, not
        just at the traditional monthly horizon the original paper used --
        which is why this is implemented with a BAR-COUNT lookback (tunable
        per the resolution it actually runs at) rather than hardcoded
        calendar months.

    Volatility-scaled POSITION SIZING (the part the research says matters
    most) is deliberately NOT duplicated in here -- this codebase already
    applies ATR-vol-targeted sizing downstream for every strategy (see
    app/risk/position_sizing.py), so this class's job is only the direction/
    conviction signal, exactly like every other Strategy in this codebase.
    """

    name = "tsmom"

    def __init__(self, short_lookback: int = 20, long_lookback: int = 100, vol_normalizer: float = 3.0) -> None:
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
        # ATR-normalized cumulative moves at/beyond this many "horizon-scaled
        # ATRs" saturate that horizon's vote at full conviction (+-1).
        self.vol_normalizer = vol_normalizer

    def _horizon_vote(self, close: pd.Series, atr: pd.Series, lookback: int) -> pd.Series:
        cum_return = close - close.shift(lookback)
        # Volatility scales with sqrt(time) under a random-walk assumption --
        # the same reasoning behind annualizing daily vol by sqrt(252) -- so
        # this approximates "how big a move is typical over `lookback` bars"
        # from the single-bar ATR already computed for every other strategy,
        # rather than needing a second, differently-windowed vol estimate.
        horizon_vol = atr * np.sqrt(lookback)
        normalized = cum_return / horizon_vol.replace(0, np.nan)
        return (normalized / self.vol_normalizer).clip(-1, 1)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns or "atr_14" not in df.columns:
            return pd.Series(0.0, index=df.index)

        close = df["close"].astype(float)
        atr = df["atr_14"].astype(float).replace(0, np.nan)

        short_vote = self._horizon_vote(close, atr, self.short_lookback)
        long_vote = self._horizon_vote(close, atr, self.long_lookback)

        # Equal-weight barbell -- per Hurst/Ooi/Pedersen (2017) and its
        # replications, short+long captures most of trend-following's
        # benefit; a medium horizon adds little once both ends are present.
        signal = 0.5 * short_vote.fillna(0.0) + 0.5 * long_vote.fillna(0.0)
        return signal.clip(-1, 1).fillna(0.0)

    def required_warmup_bars(self) -> int:
        return self.long_lookback + 5
