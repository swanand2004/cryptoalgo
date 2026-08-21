from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrderbookState:
    symbol: str
    bids: list[list[float]] = field(default_factory=list)  # [[price, size], ...] desc by price
    asks: list[list[float]] = field(default_factory=list)  # asc by price
    timestamp: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def imbalance(self) -> Optional[float]:
        """(bid_depth - ask_depth) / (bid_depth + ask_depth) over visible levels, in [-1, 1]."""
        bid_depth = sum(size for _, size in self.bids)
        ask_depth = sum(size for _, size in self.asks)
        total = bid_depth + ask_depth
        if total == 0:
            return None
        return (bid_depth - ask_depth) / total


class MarketDataCache:
    """Thread-safe in-memory latest-state cache fed by the WS ingestion loop.

    Read by the paper broker (for simulated fills), risk manager (for mark
    prices / circuit breakers), strategies (for live features) and the
    dashboard API (for a fast "current state" snapshot) without hitting the DB.
    """

    def __init__(self, trade_buffer_size: int = 5000) -> None:
        self._lock = threading.Lock()
        self.ticker: dict[str, dict] = {}
        self.orderbook: dict[str, OrderbookState] = {}
        self.mark_price: dict[str, float] = {}
        self.spot_price: dict[str, float] = {}
        self.funding_rate: dict[str, dict] = {}
        self.latest_candle: dict[tuple[str, str], dict] = {}
        self.system_status: dict = {"status": "unknown"}
        self.trade_buffer: deque[dict] = deque(maxlen=trade_buffer_size)
        self.pending_trades: list[dict] = []
        # EMA state for microstructure readings (book/trade-flow imbalance),
        # keyed by (symbol, "book"|"flow") -- see smooth_imbalance.
        self._imbalance_ema: dict[tuple[str, str], float] = {}

    def update_ticker(self, symbol: str, data: dict) -> None:
        with self._lock:
            self.ticker[symbol] = data

    def update_orderbook(self, symbol: str, bids: list, asks: list, timestamp: int) -> None:
        with self._lock:
            self.orderbook[symbol] = OrderbookState(symbol=symbol, bids=bids, asks=asks, timestamp=timestamp)

    def get_orderbook(self, symbol: str) -> Optional[OrderbookState]:
        with self._lock:
            return self.orderbook.get(symbol)

    def update_mark_price(self, symbol: str, price: float) -> None:
        with self._lock:
            self.mark_price[symbol] = price

    def update_spot_price(self, symbol: str, price: float) -> None:
        with self._lock:
            self.spot_price[symbol] = price

    def update_funding_rate(self, symbol: str, data: dict) -> None:
        with self._lock:
            self.funding_rate[symbol] = data

    def add_trade(self, trade: dict) -> None:
        with self._lock:
            self.trade_buffer.append(trade)
            self.pending_trades.append(trade)

    def drain_pending_trades(self) -> list[dict]:
        with self._lock:
            drained, self.pending_trades = self.pending_trades, []
            return drained

    def update_candle(self, symbol: str, resolution: str, candle: dict) -> None:
        with self._lock:
            self.latest_candle[(symbol, resolution)] = candle

    def drain_latest_candles(self) -> dict[tuple[str, str], dict]:
        with self._lock:
            drained, self.latest_candle = self.latest_candle, {}
            return drained

    def snapshot_book_state(
        self,
    ) -> tuple[dict[str, "OrderbookState"], dict[str, dict], dict[str, dict]]:
        """Lock-protected shallow copies of orderbook/ticker/funding_rate for
        callers (e.g. the periodic DB flush) that need to iterate all symbols
        at once -- iterating the live dicts directly from outside this class
        races with update_orderbook/update_ticker/update_funding_rate (all
        of which mutate under the lock) and can raise "dictionary changed
        size during iteration" if a WS message lands mid-flush.
        """
        with self._lock:
            return dict(self.orderbook), dict(self.ticker), dict(self.funding_rate)

    def smooth_imbalance(self, symbol: str, key: str, raw: Optional[float], span: int = 5) -> Optional[float]:
        """EMA over successive live readings of a microstructure signal
        (order-book depth imbalance, trade-flow imbalance), persisted here
        because it's the one object that actually lives across trading
        ticks.

        Without this, book_imbalance/trade_flow_imbalance only ever exist
        for the CURRENT row of a feature frame that's rebuilt from scratch
        every tick (see trading_loop._tick_symbol) -- every other row is
        NaN-turned-0, so any smoothing applied afterward (the ensemble's own
        EMA) has no real history to smooth over and reduces to ~half the
        instantaneous reading, with zero memory of the previous tick. That
        let OrderbookImbalanceStrategy's signal flip sign essentially every
        tick, since order-book snapshots genuinely do reverse minute to
        minute -- producing rapid open/close/flip churn that mostly just
        paid fees. This keeps a real running EMA across ticks instead.

        `raw=None` (no orderbook/trade data this tick) holds the last known
        smoothed value rather than resetting to 0.
        """
        state_key = (symbol, key)
        with self._lock:
            if raw is None:
                return self._imbalance_ema.get(state_key)
            prev = self._imbalance_ema.get(state_key)
            alpha = 2.0 / (span + 1)
            smoothed = raw if prev is None else prev + alpha * (raw - prev)
            self._imbalance_ema[state_key] = smoothed
            return smoothed

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Best-effort current price: orderbook mid, else mark price, else last trade."""
        with self._lock:
            ob = self.orderbook.get(symbol)
            if ob and ob.mid:
                return ob.mid
            if symbol in self.mark_price:
                return self.mark_price[symbol]
            for t in reversed(self.trade_buffer):
                if t.get("symbol") == symbol:
                    return t.get("price")
            return None


_cache: MarketDataCache | None = None


def get_market_cache() -> MarketDataCache:
    global _cache
    if _cache is None:
        _cache = MarketDataCache()
    return _cache
