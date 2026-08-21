from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """A Strategy turns a feature dataframe into a signal series in [-1, 1] per bar.

    Signals are computed vectorized over the *entire* history so the exact same
    `generate_signal` implementation powers both the backtest engine (full replay)
    and live trading (called on each new bar, only the last value is used) --
    this is what keeps backtest and live from silently diverging.

    Implementations MUST be causal: row i may only depend on rows <= i.
    """

    name: str = "strategy"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """Returns a float Series aligned to df.index, values in [-1, 1].
        Positive = bullish/long bias, negative = bearish/short bias.
        """
        raise NotImplementedError

    def required_warmup_bars(self) -> int:
        """Number of leading bars whose signal should be ignored (indicator warmup)."""
        return 50
