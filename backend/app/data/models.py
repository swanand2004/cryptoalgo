from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Candle(Base):
    """OHLCV bars. `symbol` may be a plain product symbol (BTCUSD) or a prefixed
    pseudo-symbol Delta uses for candle history: MARK:BTCUSD, FUNDING:BTCUSD, OI:ETHUSD.
    """

    __tablename__ = "candles"
    __table_args__ = (Index("ix_candles_symbol_res_time", "symbol", "resolution", "time", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64))
    resolution: Mapped[str] = mapped_column(String(8))
    time: Mapped[int] = mapped_column(Integer)  # unix seconds, candle open time
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)


class OrderbookSnapshot(Base):
    __tablename__ = "orderbook_snapshots"
    __table_args__ = (Index("ix_ob_symbol_ts", "symbol", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[int] = mapped_column(Integer)  # microseconds since epoch
    best_bid: Mapped[float] = mapped_column(Float)
    best_ask: Mapped[float] = mapped_column(Float)
    best_bid_size: Mapped[float] = mapped_column(Float)
    best_ask_size: Mapped[float] = mapped_column(Float)
    bids: Mapped[list] = mapped_column(JSON)  # top-15 [[price, size], ...]
    asks: Mapped[list] = mapped_column(JSON)


class TradeTick(Base):
    __tablename__ = "trade_ticks"
    __table_args__ = (Index("ix_trades_symbol_ts", "symbol", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[int] = mapped_column(Integer)  # microseconds since epoch
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)
    buyer_role: Mapped[str] = mapped_column(String(8))  # "m" maker / "t" taker


class FundingRateTick(Base):
    __tablename__ = "funding_rate_ticks"
    __table_args__ = (Index("ix_funding_symbol_ts", "symbol", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[int] = mapped_column(Integer)
    funding_rate: Mapped[float] = mapped_column(Float)
    next_funding_realization: Mapped[int] = mapped_column(Integer)


class TickerSnapshot(Base):
    """Periodic snapshot of the ticker channel -- powers the dashboard's live
    market view and gives us mark/spot/OI/turnover history without re-deriving
    it from raw candles.
    """

    __tablename__ = "ticker_snapshots"
    __table_args__ = (Index("ix_ticker_symbol_ts", "symbol", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[int] = mapped_column(Integer)
    mark_price: Mapped[float] = mapped_column(Float)
    spot_price: Mapped[float] = mapped_column(Float, nullable=True)
    mark_change_24h: Mapped[float] = mapped_column(Float, nullable=True)
    best_bid: Mapped[float] = mapped_column(Float, nullable=True)
    best_ask: Mapped[float] = mapped_column(Float, nullable=True)
    oi_contracts: Mapped[float] = mapped_column(Float, nullable=True)
    turnover_usd: Mapped[float] = mapped_column(Float, nullable=True)


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (Index("ix_orders_symbol_mode_state", "symbol", "mode", "state"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    broker_order_id: Mapped[str] = mapped_column(String(64), nullable=True)
    client_order_id: Mapped[str] = mapped_column(String(32), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(8))  # "paper" | "live"
    side: Mapped[str] = mapped_column(String(8))
    order_type: Mapped[str] = mapped_column(String(16))
    size: Mapped[int] = mapped_column(Integer)
    unfilled_size: Mapped[int] = mapped_column(Integer)
    limit_price: Mapped[float] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float] = mapped_column(Float, nullable=True)
    reduce_only: Mapped[bool] = mapped_column(Boolean, default=False)
    state: Mapped[str] = mapped_column(String(16))  # open/pending/closed/cancelled
    strategy: Mapped[str] = mapped_column(String(32), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class Fill(Base):
    __tablename__ = "fills"
    __table_args__ = (Index("ix_fills_symbol_mode_ts", "symbol", "mode", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(8))
    side: Mapped[str] = mapped_column(String(8))
    size: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    role: Mapped[str] = mapped_column(String(8), default="taker")
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)


class Position(Base):
    """One row per (symbol, mode, strategy). Every enabled strategy trades its
    own independent paper position per symbol, in addition to the blended
    "ensemble" position -- so `strategy` is part of identity, not just
    metadata, and multiple rows can exist for the same symbol at once.
    """

    __tablename__ = "positions"
    __table_args__ = (Index("ix_positions_symbol_mode_strategy", "symbol", "mode", "strategy", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(8))
    strategy: Mapped[str] = mapped_column(String(32))
    size: Mapped[int] = mapped_column(Integer, default=0)  # + long, - short, 0 flat
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    liquidation_price: Mapped[float] = mapped_column(Float, nullable=True)
    margin: Mapped[float] = mapped_column(Float, default=0.0)
    # Metadata captured when the position was opened from flat (or flipped
    # direction) -- carried forward until it closes, at which point it's used
    # to write the ClosedTrade record below.
    entry_time: Mapped[dt.datetime] = mapped_column(nullable=True)
    leverage: Mapped[float] = mapped_column(Float, nullable=True)
    fees_paid: Mapped[float] = mapped_column(Float, default=0.0)  # cumulative since this position opened
    stop_loss_price: Mapped[float] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float] = mapped_column(Float, nullable=True)
    # Trailing-stop bookkeeping: peak_price is the best price seen since entry
    # (highest for a long, lowest for a short); stop_distance is the fixed
    # $ gap (atr_stop_multiple * ATR-at-entry) the stop trails behind it by.
    # stop_loss_price only ever moves favorably as peak_price improves -- it
    # never loosens.
    peak_price: Mapped[float] = mapped_column(Float, nullable=True)
    stop_distance: Mapped[float] = mapped_column(Float, nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class ClosedTrade(Base):
    """A completed round trip: opened from flat (or via a direction flip) and
    later closed back to flat (or flipped again). This is what the dashboard's
    trade history shows -- paired entry/exit, not raw individual fills.
    """

    __tablename__ = "closed_trades"
    __table_args__ = (Index("ix_closed_trades_symbol_mode_exit", "symbol", "mode", "exit_time"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(8))
    side: Mapped[str] = mapped_column(String(8))  # "long" | "short"
    strategy: Mapped[str] = mapped_column(String(32), nullable=True)
    size: Mapped[int] = mapped_column(Integer)  # contracts closed (always positive)
    leverage: Mapped[float] = mapped_column(Float, nullable=True)  # notional / equity at entry
    entry_price: Mapped[float] = mapped_column(Float)
    entry_time: Mapped[dt.datetime] = mapped_column()
    exit_price: Mapped[float] = mapped_column(Float)
    exit_time: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    exit_reason: Mapped[str] = mapped_column(String(24), default="signal")  # "signal" | "stop_loss" | "take_profit"
    realized_pnl: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    __table_args__ = (Index("ix_equity_mode_ts", "mode", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    balance: Mapped[float] = mapped_column(Float)
    unrealized_pnl: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    __table_args__ = (Index("ix_predictions_symbol_ts", "symbol", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    horizon_bars: Mapped[int] = mapped_column(Integer)
    predicted_return: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)


class StrategyConfig(Base):
    __tablename__ = "strategy_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(32), unique=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class AppState(Base):
    """Singleton key/value rows for small pieces of runtime state -- trading mode,
    kill-switch flag, etc. Kept generic instead of one column per flag.
    """

    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    value: Mapped[str] = mapped_column(String(256))
    updated_at: Mapped[dt.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class AnalystTrade(Base):
    """A PAPER trade taken by the rules-based Analyst (app/analyst): opened when
    a scheduled analysis says BUY/SELL and no trade is open on that symbol,
    closed ONLY at its stop or target. Never sent to an exchange -- it is the
    forward record that decides whether the Analyst deserves real money."""

    __tablename__ = "analyst_trades"
    __table_args__ = (Index("ix_analyst_trades_symbol_open", "symbol", "is_open"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))  # "BUY" | "SELL"
    conviction: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    size_factor: Mapped[float] = mapped_column(Float, default=1.0)
    entry_time: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    entry: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    exit_time: Mapped[dt.datetime] = mapped_column(nullable=True)
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    result: Mapped[str] = mapped_column(String(8), nullable=True)  # "TP" | "SL"
    r_multiple: Mapped[float] = mapped_column(Float, nullable=True)  # net of 7 bps/side
    reasons: Mapped[dict] = mapped_column(JSON, default=dict)  # the analysis it was taken on
    # Live execution (Settings.trading_engine = "analyst") and the dynamic exit.
    live: Mapped[bool] = mapped_column(Boolean, default=False)  # True = a real Delta order
    contracts: Mapped[int] = mapped_column(Integer, default=0)
    initial_stop: Mapped[float] = mapped_column(Float, nullable=True)  # the stop at entry (1R)
    atr: Mapped[float] = mapped_column(Float, nullable=True)  # daily ATR at entry
    peak: Mapped[float] = mapped_column(Float, nullable=True)  # best price since entry
    extended: Mapped[bool] = mapped_column(Boolean, default=False)  # ran past TP1 on a still-strong call
    # Shadow variants (Settings.analyst_shadow_variants): NULL = the real Analyst record;
    # "base" / "C" / "B" / "CB" = a paper-only forward test of a candidate rule. Never ordered,
    # never counted in the live stats (app/analyst/execution.py, shadow section).
    variant: Mapped[str] = mapped_column(String(8), nullable=True, default=None)


class OptionsPaperTrade(Base):
    """One week of the PAPER "S7" options tracker (app/options/s7.py). Every
    Friday 12:00 UTC per underlying (BTC, ETH) it records the trend-book signal
    and the decision -- buy the ~7-day 0.40-delta call, the -0.40-delta put, or
    nothing -- then holds the option to expiry and settles it at intrinsic
    value. Never sent to an exchange: it is the forward record that decides
    whether S7 ever deserves real money. Weeks with no trade are stored too
    (decision "NONE") so the record shows every signal, not only the trades."""

    __tablename__ = "options_paper_trades"
    __table_args__ = (Index("ux_options_paper_week_underlying", "week", "underlying", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    week: Mapped[str] = mapped_column(String(10))  # entry date YYYY-MM-DD (the Friday)
    underlying: Mapped[str] = mapped_column(String(8))  # "BTC" | "ETH"
    signal: Mapped[float] = mapped_column(Float, nullable=True)
    signal_source: Mapped[str] = mapped_column(String(16), nullable=True)  # "live" | "trend_ladder"
    decision: Mapped[str] = mapped_column(String(8))  # "CALL" | "PUT" | "NONE"
    note: Mapped[str] = mapped_column(String(256), nullable=True)
    entry_time: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    symbol: Mapped[str] = mapped_column(String(40), nullable=True)  # e.g. C-BTC-90000-161026
    strike: Mapped[float] = mapped_column(Float, nullable=True)
    expiry: Mapped[dt.datetime] = mapped_column(nullable=True)
    contract_value: Mapped[float] = mapped_column(Float, nullable=True)  # underlying units per contract
    spot_at_entry: Mapped[float] = mapped_column(Float, nullable=True)
    delta_at_entry: Mapped[float] = mapped_column(Float, nullable=True)
    iv_at_entry: Mapped[float] = mapped_column(Float, nullable=True)
    premium: Mapped[float] = mapped_column(Float, nullable=True)  # price per 1 underlying unit
    premium_source: Mapped[str] = mapped_column(String(8), nullable=True)  # "ask" | "mark"
    entry_fee: Mapped[float] = mapped_column(Float, nullable=True)  # USD, incl. 18% GST
    is_open: Mapped[bool] = mapped_column(Boolean, default=False)
    settled_at: Mapped[dt.datetime] = mapped_column(nullable=True)
    underlying_at_expiry: Mapped[float] = mapped_column(Float, nullable=True)
    settle_value: Mapped[float] = mapped_column(Float, nullable=True)  # intrinsic per 1 underlying unit
    settle_source: Mapped[str] = mapped_column(String(16), nullable=True)  # "delta" | "binance_1h"
    settle_fee: Mapped[float] = mapped_column(Float, nullable=True)  # USD, 0 when it expires worthless
    pnl_usd: Mapped[float] = mapped_column(Float, nullable=True)  # net of both fees, per 1 contract
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)  # pnl / (premium paid + entry fee)
