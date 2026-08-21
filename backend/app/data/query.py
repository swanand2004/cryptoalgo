from __future__ import annotations

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


async def load_closed_trades(mode: str, symbol: str | None = None, limit: int = 100) -> list[ClosedTrade]:
    async with SessionLocal() as session:
        stmt = select(ClosedTrade).where(ClosedTrade.mode == mode)
        if symbol:
            stmt = stmt.where(ClosedTrade.symbol == symbol)
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
