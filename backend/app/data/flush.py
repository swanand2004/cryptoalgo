from __future__ import annotations

import logging
import time

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import Candle, FundingRateTick, OrderbookSnapshot, TickerSnapshot, TradeTick

logger = logging.getLogger(__name__)


async def flush_market_cache_to_db(cache: MarketDataCache) -> None:
    """Periodic job: drains buffered trades/candles and snapshots current
    orderbook/ticker/funding state into SQLite. Keeps the hot WS ingestion path
    free of per-message DB writes.
    """
    trades = cache.drain_pending_trades()
    if trades:
        rows = [
            {
                "symbol": t["symbol"],
                "timestamp": t["timestamp"],
                "price": t["price"],
                "size": t["size"],
                "buyer_role": t["buyer_role"],
            }
            for t in trades
        ]
        async with SessionLocal() as session:
            await session.execute(sqlite_insert(TradeTick), rows)
            await session.commit()

    candles = cache.drain_latest_candles()
    if candles:
        async with SessionLocal() as session:
            for (symbol, resolution), c in candles.items():
                if c.get("open") is None:
                    continue
                stmt = sqlite_insert(Candle).values(
                    symbol=symbol,
                    resolution=resolution,
                    time=c["time"],
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c.get("volume", 0.0) or 0.0,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "resolution", "time"],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                    },
                )
                await session.execute(stmt)
            await session.commit()

    now_us = int(time.time() * 1_000_000)
    orderbook, ticker, funding_rate = cache.snapshot_book_state()
    async with SessionLocal() as session:
        for symbol, ob in orderbook.items():
            session.add(
                OrderbookSnapshot(
                    symbol=symbol,
                    timestamp=ob.timestamp or now_us,
                    best_bid=ob.best_bid or 0.0,
                    best_ask=ob.best_ask or 0.0,
                    best_bid_size=ob.bids[0][1] if ob.bids else 0.0,
                    best_ask_size=ob.asks[0][1] if ob.asks else 0.0,
                    bids=ob.bids[:15],
                    asks=ob.asks[:15],
                )
            )
        for symbol, t in ticker.items():
            session.add(
                TickerSnapshot(
                    symbol=symbol,
                    timestamp=now_us,
                    mark_price=float(t.get("m", 0) or 0),
                    spot_price=cache.spot_price.get(symbol),
                    mark_change_24h=float(t.get("m24hc", 0) or 0),
                    best_bid=float((t.get("q") or [None])[0]) if t.get("q") and t["q"][0] else None,
                    best_ask=float((t.get("q") or [None] * 3)[2]) if t.get("q") and len(t["q"]) > 2 and t["q"][2] else None,
                    oi_contracts=float((t.get("oi") or [None])[0]) if t.get("oi") and t["oi"][0] else None,
                    turnover_usd=float((t.get("to") or [None, None])[1]) if t.get("to") and len(t["to"]) > 1 else None,
                )
            )
        for symbol, f in funding_rate.items():
            session.add(
                FundingRateTick(
                    symbol=symbol,
                    timestamp=int(f.get("timestamp") or now_us),
                    funding_rate=float(f.get("funding_rate") or 0),
                    next_funding_realization=int(f.get("next_funding_realization") or 0),
                )
            )
        await session.commit()
