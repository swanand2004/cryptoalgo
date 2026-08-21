from __future__ import annotations

import pandas as pd

from app.strategy.base import Strategy


class OrderbookImbalanceStrategy(Strategy):
    """Short-horizon microstructure signal: order-book depth imbalance blended
    with recent trade-flow imbalance. This is the closest thing to an HFT-style
    signal achievable over a REST/WS API (no colocation) -- useful mainly for
    fine-tuning entry/exit timing of the slower strategies, not as a standalone
    directional bet, so its ensemble weight should stay modest.

    Only meaningful once `book_imbalance`/`trade_flow_imbalance` columns are
    present (populated live from the market cache, or from historical
    orderbook_snapshots once enough has been collected for backtesting).
    """

    name = "orderbook_imbalance"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "book_imbalance" not in df.columns and "trade_flow_imbalance" not in df.columns:
            return pd.Series(0.0, index=df.index)

        book = df.get("book_imbalance", pd.Series(0.0, index=df.index)).fillna(0.0)
        flow = df.get("trade_flow_imbalance", pd.Series(0.0, index=df.index)).fillna(0.0)
        signal = 0.6 * book + 0.4 * flow
        return signal.clip(-1, 1)

    def required_warmup_bars(self) -> int:
        return 1
