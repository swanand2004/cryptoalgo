from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderRequest:
    symbol: str
    side: str  # "buy" | "sell"
    size: int  # contracts, always positive
    order_type: str = "market"  # "market" | "limit"
    limit_price: Optional[float] = None
    reduce_only: bool = False
    client_order_id: Optional[str] = None
    strategy: Optional[str] = None
    # ATR at signal time, used to set stop-loss/take-profit levels when this
    # order opens a new position. None on closing orders (not needed).
    atr: Optional[float] = None
    # Trade conviction (the ensemble signal's magnitude, 0..1) at signal
    # time, used to select this trade's leverage tier when it opens a new
    # position (see position_sizing.select_leverage). Irrelevant on closing
    # orders -- reduce-only orders never open a position, so they never
    # select a leverage tier. Defaults to full conviction so any caller that
    # doesn't have a real signal (e.g. a manual order) gets the ceiling.
    quality: float = 1.0
    # Explicit exchange leverage for this order (live: set on Delta before it
    # is placed). None -> the conviction tier, select_leverage(quality).
    leverage: Optional[float] = None
    # Small-account minimum lot (trading_loop._min_lot_target): ONE contract
    # the normal sizing rounded to zero, already checked against the
    # account-level leverage cap -- so the per-trade margin caps, which can
    # never fit a whole contract on a tiny account, are not re-applied.
    min_lot: bool = False


@dataclass
class OrderResult:
    order_id: int  # our internal DB id
    broker_order_id: Optional[str]
    state: str  # "open" | "closed" | "cancelled" | "rejected"
    filled_size: int
    avg_fill_price: Optional[float]
    fee: float
    reason: Optional[str] = None
    # Realized PnL this specific fill locked in (0.0 for an order that only
    # opened/extended a position). Fed into RiskManager.record_realized_pnl
    # by OrderManager so the daily-loss kill switch actually sees losses.
    realized_pnl: float = 0.0


@dataclass
class PositionView:
    symbol: str
    size: int
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    strategy: Optional[str] = None
    entry_time: Optional[dt.datetime] = None
    leverage: Optional[float] = None
    fees_paid: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    # Delta's own UI shows both: size in lots (contracts) AND what that
    # implies in USD terms, plus the margin that leverage choice locks up.
    notional_usd: float = 0.0  # abs(size) * contract_value * current price
    margin_usd: float = 0.0  # notional_usd / leverage
    liquidation_price: Optional[float] = None  # Delta's, live only


class Broker(ABC):
    mode: str

    @abstractmethod
    async def place_order(self, req: OrderRequest, exit_reason: str = "signal") -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, order_id: int) -> None: ...

    @abstractmethod
    async def get_positions(self) -> dict[str, PositionView]:
        """Keyed by f"{symbol}:{strategy}" -- every enabled strategy can hold
        its own independent position per symbol, so symbol alone isn't a
        unique key. Use PositionView.symbol / .strategy, not the dict key.
        """
        ...

    @abstractmethod
    async def get_equity(self) -> float: ...
