import datetime as dt

import pytest

from app.data.db import SessionLocal
from app.data.models import ClosedTrade
from app.data.query import load_closed_trades


async def _seed_trade(exit_time: dt.datetime, symbol: str = "BTCUSD") -> None:
    async with SessionLocal() as session:
        session.add(
            ClosedTrade(
                symbol=symbol, mode="paper", side="long", size=1,
                entry_price=100.0, entry_time=exit_time - dt.timedelta(hours=1),
                exit_price=101.0, exit_time=exit_time, realized_pnl=1.0, fees=0.0,
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_no_date_filter_returns_everything():
    await _seed_trade(dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc))
    await _seed_trade(dt.datetime(2026, 8, 15, tzinfo=dt.timezone.utc))
    await _seed_trade(dt.datetime(2026, 8, 30, tzinfo=dt.timezone.utc))

    rows = await load_closed_trades("paper")
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_date_from_excludes_earlier_trades():
    await _seed_trade(dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc))
    await _seed_trade(dt.datetime(2026, 8, 15, tzinfo=dt.timezone.utc))
    await _seed_trade(dt.datetime(2026, 8, 30, tzinfo=dt.timezone.utc))

    rows = await load_closed_trades("paper", date_from=dt.datetime(2026, 8, 10, tzinfo=dt.timezone.utc))
    assert len(rows) == 2
    assert all(r.exit_time >= dt.datetime(2026, 8, 10) for r in rows)


@pytest.mark.asyncio
async def test_date_to_excludes_later_trades():
    await _seed_trade(dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc))
    await _seed_trade(dt.datetime(2026, 8, 15, tzinfo=dt.timezone.utc))
    await _seed_trade(dt.datetime(2026, 8, 30, tzinfo=dt.timezone.utc))

    rows = await load_closed_trades("paper", date_to=dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc))
    assert len(rows) == 2
    assert all(r.exit_time <= dt.datetime(2026, 8, 20) for r in rows)


@pytest.mark.asyncio
async def test_date_range_is_inclusive_on_both_ends():
    boundary = dt.datetime(2026, 8, 15, 12, 0, tzinfo=dt.timezone.utc)
    await _seed_trade(boundary)

    rows = await load_closed_trades("paper", date_from=boundary, date_to=boundary)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_date_filter_combines_with_symbol_and_mode_filters():
    await _seed_trade(dt.datetime(2026, 8, 15, tzinfo=dt.timezone.utc), symbol="BTCUSD")
    await _seed_trade(dt.datetime(2026, 8, 15, tzinfo=dt.timezone.utc), symbol="ETHUSD")

    rows = await load_closed_trades(
        "paper", symbol="ETHUSD",
        date_from=dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc),
        date_to=dt.datetime(2026, 8, 31, tzinfo=dt.timezone.utc),
    )
    assert len(rows) == 1
    assert rows[0].symbol == "ETHUSD"


@pytest.mark.asyncio
async def test_no_trades_in_range_returns_empty_not_an_error():
    await _seed_trade(dt.datetime(2026, 8, 15, tzinfo=dt.timezone.utc))

    rows = await load_closed_trades("paper", date_from=dt.datetime(2027, 1, 1, tzinfo=dt.timezone.utc))
    assert rows == []
