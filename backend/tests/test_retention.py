import pytest
from sqlalchemy import func, select

from app.data.db import SessionLocal
from app.data.models import Candle, TradeTick
from app.data.retention import prune_raw_ticks

NOW = 1_790_000_000.0
DAY_US = 86_400 * 1_000_000


@pytest.mark.asyncio
async def test_old_raw_ticks_are_pruned_recent_ones_and_candles_are_kept():
    now_us = int(NOW * 1_000_000)
    async with SessionLocal() as session:
        for age_days in (0.5, 1.5, 3, 10):
            session.add(TradeTick(symbol="BTCUSD", timestamp=now_us - int(age_days * DAY_US), price=1.0, size=1.0, buyer_role="t"))
        session.add(Candle(symbol="BTCUSD", resolution="1m", time=int(NOW) - 30 * 86_400,
                           open=1, high=1, low=1, close=1, volume=0))
        await session.commit()

    deleted = await prune_raw_ticks(retention_days=2, now_s=NOW)

    assert deleted["trade_ticks"] == 2  # the 3- and 10-day-old ticks
    async with SessionLocal() as session:
        assert (await session.execute(select(func.count()).select_from(TradeTick))).scalar() == 2
        assert (await session.execute(select(func.count()).select_from(Candle))).scalar() == 1  # never pruned
