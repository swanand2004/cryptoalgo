"""DB growth fixes (2026-10-07): orderbook snapshots on a shorter window, and
old 1m/5m candles pruned while every other resolution is left alone."""
import types

import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.data.db import SessionLocal
from app.data.models import Candle, OrderbookSnapshot, TradeTick
from app.data.retention import prune_fine_candles, prune_raw_ticks, run_retention

NOW = 1_790_000_000.0
DAY = 86_400
DAY_US = DAY * 1_000_000


def _candle(symbol, res, age_days):
    return Candle(symbol=symbol, resolution=res, time=int(NOW - age_days * DAY), open=1, high=1, low=1, close=1, volume=0)


async def _count(model, **where):
    async with SessionLocal() as session:
        stmt = select(func.count()).select_from(model)
        for k, v in where.items():
            stmt = stmt.where(getattr(model, k) == v)
        return (await session.execute(stmt)).scalar()


@pytest.mark.asyncio
async def test_orderbook_window_is_separate_from_the_other_raw_tables():
    now_us = int(NOW * 1_000_000)
    async with SessionLocal() as session:
        for age in (0.5, 1.5, 3):
            session.add(TradeTick(symbol="BTCUSD", timestamp=now_us - int(age * DAY_US), price=1.0, size=1.0, buyer_role="t"))
            session.add(OrderbookSnapshot(symbol="BTCUSD", timestamp=now_us - int(age * DAY_US), best_bid=1.0, best_ask=1.0,
                                          best_bid_size=1.0, best_ask_size=1.0, bids=[], asks=[]))
        await session.commit()
    deleted = await prune_raw_ticks(2, now_s=NOW, per_table_days={"orderbook_snapshots": 1})
    assert deleted["trade_ticks"] == 1 and deleted["orderbook_snapshots"] == 2
    assert await _count(TradeTick) == 2 and await _count(OrderbookSnapshot) == 1


@pytest.mark.asyncio
async def test_only_old_fine_candles_are_pruned():
    async with SessionLocal() as session:
        for sym in ("BTCUSD", "MARK:BTCUSD", "BINANCE:BTCUSD"):
            for res in ("1m", "5m", "15m", "1h", "4h", "1d"):
                for age in (0.1, 2, 6, 20, 400):
                    session.add(_candle(sym, res, age))
        await session.commit()
    deleted = await prune_fine_candles({"1m": 4, "5m": 14}, now_s=NOW)
    assert deleted == {"1m": 3 * 3, "5m": 3 * 2}
    for sym in ("BTCUSD", "MARK:BTCUSD", "BINANCE:BTCUSD"):
        assert await _count(Candle, symbol=sym, resolution="1m") == 2
        assert await _count(Candle, symbol=sym, resolution="5m") == 3
        for res in ("15m", "1h", "4h", "1d"):
            assert await _count(Candle, symbol=sym, resolution=res) == 5  # never touched


@pytest.mark.asyncio
async def test_zero_days_keeps_everything():
    async with SessionLocal() as session:
        session.add(_candle("BTCUSD", "1m", 100))
        await session.commit()
    assert await prune_fine_candles({"1m": 0, "5m": 0}, now_s=NOW) == {}
    assert await _count(Candle) == 1


@pytest.mark.asyncio
async def test_run_retention_uses_the_settings():
    s = Settings()
    assert (s.orderbook_retention_days, s.candle_1m_retention_days, s.candle_5m_retention_days) == (1.0, 4.0, 14.0)
    async with SessionLocal() as session:
        session.add(_candle("BTCUSD", "1m", 5))
        session.add(_candle("BTCUSD", "1m", 1))
        await session.commit()
    out = await run_retention(types.SimpleNamespace(raw_tick_retention_days=2.0, orderbook_retention_days=1.0,
                                                    candle_1m_retention_days=4.0, candle_5m_retention_days=14.0),
                              now_s=NOW)
    assert out["1m"] == 1 and "orderbook_snapshots" in out
    assert await _count(Candle) == 1
