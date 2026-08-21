import numpy as np
import pandas as pd
import pytest

from app.strategy.orderbook_imbalance import OrderbookImbalanceStrategy


def test_returns_zero_when_columns_entirely_absent():
    df = pd.DataFrame({"close": [1, 2, 3]})
    strategy = OrderbookImbalanceStrategy()
    signal = strategy.generate_signal(df)
    assert (signal == 0.0).all()


def test_only_last_row_populated_still_produces_a_live_signal():
    """This is exactly the shape trading_loop.py produces: historical bars have
    no stored order-book state (NaN), only the current/last bar gets live data
    from the market cache.
    """
    n = 10
    df = pd.DataFrame({"close": np.arange(n, dtype=float)})
    df["book_imbalance"] = np.nan
    df["trade_flow_imbalance"] = np.nan
    df.loc[df.index[-1], "book_imbalance"] = 0.8
    df.loc[df.index[-1], "trade_flow_imbalance"] = 0.4

    strategy = OrderbookImbalanceStrategy()
    signal = strategy.generate_signal(df)

    assert (signal.iloc[:-1] == 0.0).all()
    assert signal.iloc[-1] == pytest.approx(0.6 * 0.8 + 0.4 * 0.4)
