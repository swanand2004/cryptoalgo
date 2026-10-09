from __future__ import annotations

import logging
import time

from sqlalchemy import text

from app.data.db import SessionLocal

logger = logging.getLogger(__name__)

# Raw websocket captures that are written every 5 seconds and READ BY NOTHING
# (checked 2026-10-02: the only references outside models.py are the writer,
# app/data/flush.py, and deploy/slim_db.py, which drops them). Their consumers
# -- orderbook_imbalance and friends -- were deleted on 2026-09-04.
#
# Left alone they were the bulk of an 858 MB database, and at 24/7 uptime
# trade_ticks alone adds roughly 0.7-1.5 GB a month: on a 10-20 GB VPS that is
# a full disk, and a full disk stops SQLite writing equity snapshots and fills
# -- i.e. it stops the trading loop. Candles are NOT touched; the strategy and
# the models read those.
RAW_TICK_TABLES = ("trade_ticks", "orderbook_snapshots", "ticker_snapshots", "funding_rate_ticks")
_BATCH = 20_000  # small deletes, so the trading loop never waits long on the write lock


async def prune_raw_ticks(retention_days: float, now_s: float | None = None,
                          per_table_days: dict[str, float] | None = None) -> dict[str, int]:
    """Deletes raw tick rows older than `retention_days` (or the table's entry
    in `per_table_days`). Timestamps in these tables are microseconds since the
    epoch."""
    now = now_s if now_s is not None else time.time()
    deleted: dict[str, int] = {}
    for table in RAW_TICK_TABLES:
        days = (per_table_days or {}).get(table, retention_days)
        cutoff_us = int((now - days * 86_400) * 1_000_000)
        total = 0
        while True:
            async with SessionLocal() as session:
                result = await session.execute(
                    text(
                        f"DELETE FROM {table} WHERE rowid IN "
                        f"(SELECT rowid FROM {table} WHERE timestamp < :cutoff LIMIT :batch)"
                    ),
                    {"cutoff": cutoff_us, "batch": _BATCH},
                )
                await session.commit()
            total += result.rowcount or 0
            if (result.rowcount or 0) < _BATCH:
                break
        deleted[table] = total
    if any(deleted.values()):
        logger.info("Pruned raw ticks older than %.1f days: %s", retention_days, deleted)
    return deleted


# Fine candles (2026-10-07). 1m and 5m bars are written by the websocket feed
# and the startup backfill (14 / 30 days) for every product, MARK: and spot
# index symbol; nothing reads more than a few hours of them (the Analyst live
# bar reads 120 1m rows, the Markets chart 300 bars). Every other resolution --
# the 15m/1h/4h/1d series the strategies, ML and the deep BINANCE: history use
# -- is never touched here.
FINE_RESOLUTIONS = ("1m", "5m")


async def prune_fine_candles(days_by_resolution: dict[str, float], now_s: float | None = None) -> dict[str, int]:
    """Deletes 1m/5m candles older than the given days (0 or missing = keep).
    Works symbol by symbol so every delete is an index range scan on
    ix_candles_symbol_res_time (short write locks), never a table scan."""
    now = now_s if now_s is not None else time.time()
    async with SessionLocal() as session:
        symbols = [r[0] for r in (await session.execute(
            text("SELECT DISTINCT symbol FROM candles WHERE resolution IN ('1m', '5m')"))).all()]
    deleted: dict[str, int] = {}
    for resolution in FINE_RESOLUTIONS:
        days = float(days_by_resolution.get(resolution) or 0.0)
        if days <= 0:
            continue
        cutoff = int(now - days * 86_400)
        total = 0
        for symbol in symbols:
            while True:
                async with SessionLocal() as session:
                    result = await session.execute(
                        text(
                            "DELETE FROM candles WHERE rowid IN (SELECT rowid FROM candles "
                            "WHERE symbol = :s AND resolution = :r AND time < :cutoff LIMIT :batch)"
                        ),
                        {"s": symbol, "r": resolution, "cutoff": cutoff, "batch": _BATCH},
                    )
                    await session.commit()
                total += result.rowcount or 0
                if (result.rowcount or 0) < _BATCH:
                    break
        deleted[resolution] = total
    if any(deleted.values()):
        logger.info("Pruned fine candles: %s", deleted)
    return deleted


async def run_retention(settings, now_s: float | None = None) -> dict[str, int]:
    """The scheduled retention job: raw ticks (orderbook snapshots on their own,
    shorter window) and old fine candles. Each part fails on its own."""
    out: dict[str, int] = {}
    try:
        out.update(await prune_raw_ticks(
            settings.raw_tick_retention_days, now_s=now_s,
            per_table_days={"orderbook_snapshots": settings.orderbook_retention_days},
        ))
    except Exception:
        logger.exception("Raw tick prune failed")
    try:
        out.update(await prune_fine_candles(
            {"1m": settings.candle_1m_retention_days, "5m": settings.candle_5m_retention_days}, now_s=now_s,
        ))
    except Exception:
        logger.exception("Fine candle prune failed")
    return out
