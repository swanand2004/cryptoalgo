from __future__ import annotations

import logging
import time

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.core.delta_client import DeltaRestClient
from app.core.symbols import SymbolRegistry
from app.data.db import SessionLocal
from app.data.models import Candle

logger = logging.getLogger(__name__)

RESOLUTIONS = ["1m", "5m", "15m", "1h", "4h", "1d"]
MAX_CANDLES_PER_REQUEST = 2000

_RESOLUTION_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "1d": 86400,
    "1w": 604800,
}


async def upsert_candles(symbol: str, resolution: str, candles: list[dict]) -> int:
    if not candles:
        return 0
    rows = [
        {
            "symbol": symbol,
            "resolution": resolution,
            "time": c["time"],
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c.get("volume", 0.0) or 0.0,
        }
        for c in candles
    ]
    async with SessionLocal() as session:
        stmt = sqlite_insert(Candle).values(rows)
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
    return len(rows)


async def backfill_symbol_history(
    client: DeltaRestClient,
    symbol: str,
    resolution: str,
    lookback_seconds: int,
) -> int:
    """Pulls candle history in <=2000-candle windows until `lookback_seconds` in
    the past is covered. Works for plain symbols and the FUNDING:/MARK:/OI:
    prefixed pseudo-symbols Delta exposes on the same endpoint.
    """
    step = _RESOLUTION_SECONDS[resolution] * MAX_CANDLES_PER_REQUEST
    end = int(time.time())
    start_target = end - lookback_seconds
    total = 0
    cursor_end = end
    while cursor_end > start_target:
        cursor_start = max(cursor_end - step, start_target)
        candles = await client.get_candles(symbol, resolution, cursor_start, cursor_end)
        total += await upsert_candles(symbol, resolution, candles)
        if not candles:
            break
        cursor_end = cursor_start
    logger.info("Backfilled %d %s candles for %s", total, resolution, symbol)
    return total


async def backfill_all(client: DeltaRestClient, registry: SymbolRegistry, lookback_days: int = 90) -> None:
    lookback_seconds = lookback_days * 86400
    for product in registry.all():
        for resolution in RESOLUTIONS:
            # Cap lookback for very fine resolutions so we don't pull years of 1m bars.
            days = min(lookback_days, {"1m": 14, "5m": 30, "15m": 60}.get(resolution, lookback_days))
            await backfill_symbol_history(client, product.symbol, resolution, days * 86400)
            await backfill_symbol_history(client, f"MARK:{product.symbol}", resolution, days * 86400)
            # Spot index (e.g. .DEXBTUSD) feeds the "basis" feature
            # (mark_price - spot_price) -- same treatment as MARK:, since
            # build_feature_frame reads it the same way.
            if product.spot_index_symbol:
                await backfill_symbol_history(client, product.spot_index_symbol, resolution, days * 86400)
        await backfill_symbol_history(client, f"FUNDING:{product.symbol}", "1h", lookback_seconds)
        await backfill_symbol_history(client, f"OI:{product.symbol}", "1h", lookback_seconds)
