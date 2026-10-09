from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select

from app.data.db import SessionLocal
from app.data.models import Candle, ClosedTrade, EquitySnapshot, Fill, Order


async def load_candles_df(symbol: str, resolution: str, limit: int = 500) -> pd.DataFrame:
    async with SessionLocal() as session:
        stmt = (
            select(Candle)
            .where(Candle.symbol == symbol, Candle.resolution == resolution)
            .order_by(Candle.time.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = list(reversed(result.scalars().all()))
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(
        [{"time": r.time, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume} for r in rows]
    )


async def load_pseudo_series_df(symbol: str, resolution: str, limit: int = 2000) -> pd.DataFrame:
    """Loads a FUNDING:/MARK:/OI: candle series as a flat [time, value] frame
    (using the close of each bar as the point value) for merge_asof joins.
    """
    df = await load_candles_df(symbol, resolution, limit)
    if df.empty:
        return pd.DataFrame(columns=["time", "value"])
    return df[["time", "close"]].rename(columns={"close": "value"})


# Enough hourly bars to converge the ladder's slowest 1600-span EMA (4 spans
# = 6400) with room to spare. BINANCE:BTCUSD holds ~79,000, SOL ~53,000.
LADDER_BARS = 12_000


async def load_ladder_frame(symbol: str, limit: int = LADDER_BARS) -> pd.DataFrame:
    """The signal frame for TrendLadderStrategy: deep hourly prices plus real
    perp funding, assembled exactly the way the nine-year study assembled it.

    WHY THIS DOES NOT USE DELTA'S OWN CANDLES. The ladder's slowest EMA spans
    1600 hourly bars and needs roughly four spans to converge -- about 267
    days. Delta's own 1h history is ~113 days and the live loop loads 400 bars
    of it, so the slow half of the ladder would be pure warm-up garbage on
    every tick. BINANCE:{symbol} carries 2017-onward hourly history, is the
    series the strategy was actually validated on, and is the global reference
    price Delta's own contracts track. Execution still happens on Delta at
    Delta's live price -- only the SIGNAL is computed from this series.

    Funding comes from BFUNDHIST:{symbol} (real 8-hourly perp settlements back
    to 2019) rather than Delta's FUNDING: series, which carries the premium
    index -- sign-inverted and ~4x off, see app/features/funding.py. Feeding
    the premium index to a carry tilt would invert it.

    Returns a frame indexed 0..n-1 with time/open/high/low/close/volume,
    `atr_14`, and `funding_rate` as a DECIMAL per settlement (0.0 on bars with
    no settlement -- TrendLadderStrategy forward-fills those itself).
    """
    df = await load_candles_df(f"BINANCE:{symbol}", "1h", limit=limit)
    if df.empty:
        return df

    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    # Wilder smoothing, matching app/features/indicators.py and the study.
    df["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    funding = await load_candles_df(f"BFUNDHIST:{symbol}", "8h", limit=limit)
    if funding.empty:
        df["funding_rate"] = 0.0
        return df

    # BFUNDHIST stores the rate in PERCENT; the strategy wants a decimal.
    settlements = (
        funding[["time", "close"]]
        .assign(funding_rate=lambda f: f["close"].astype(float) / 100.0)
        .drop(columns="close")
        .drop_duplicates(subset="time")
        .sort_values("time")
    )
    merged = df.merge(settlements, on="time", how="left")
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
    return merged


async def load_orders(mode: str, symbol: str | None = None, limit: int = 100) -> list[Order]:
    async with SessionLocal() as session:
        stmt = select(Order).where(Order.mode == mode)
        if symbol:
            stmt = stmt.where(Order.symbol == symbol)
        stmt = stmt.order_by(Order.created_at.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def load_fills(mode: str, symbol: str | None = None, limit: int = 100) -> list[Fill]:
    async with SessionLocal() as session:
        stmt = select(Fill).where(Fill.mode == mode)
        if symbol:
            stmt = stmt.where(Fill.symbol == symbol)
        stmt = stmt.order_by(Fill.created_at.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def load_closed_trades(
    mode: str,
    symbol: str | None = None,
    date_from: dt.datetime | None = None,
    date_to: dt.datetime | None = None,
    limit: int = 100,
) -> list[ClosedTrade]:
    """date_from/date_to filter on exit_time (when the trade actually
    closed) -- the same column the results are already sorted by, and the
    one the composite index (symbol, mode, exit_time) covers, so the filter
    stays efficient even as trade history grows. Both bounds are inclusive.
    """
    async with SessionLocal() as session:
        stmt = select(ClosedTrade).where(ClosedTrade.mode == mode)
        if symbol:
            stmt = stmt.where(ClosedTrade.symbol == symbol)
        if date_from is not None:
            stmt = stmt.where(ClosedTrade.exit_time >= date_from)
        if date_to is not None:
            stmt = stmt.where(ClosedTrade.exit_time <= date_to)
        stmt = stmt.order_by(ClosedTrade.exit_time.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def load_equity_curve(mode: str, limit: int = 500) -> list[EquitySnapshot]:
    async with SessionLocal() as session:
        stmt = (
            select(EquitySnapshot)
            .where(EquitySnapshot.mode == mode)
            .order_by(EquitySnapshot.timestamp.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(reversed(result.scalars().all()))
