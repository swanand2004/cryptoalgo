from __future__ import annotations

import datetime as dt
import logging
import time

import ccxt.async_support as ccxt_async

from app.data.ingest_rest import upsert_candles
from app.data.query import load_candles_df

logger = logging.getLogger(__name__)

# PAXG (Paxos Gold), a token backed 1:1 by real physical LBMA-vaulted gold,
# traded on Binance -- switched from Delta's own `.DEXAUTUSD` (Tether Gold)
# spot index on 2026-08-25 because it is a genuinely DEEPER data source, not
# just a different one: Binance's real PAXG/USDT history goes back to
# 2020-08-28 (~2200 daily bars), vs `.DEXAUTUSD`'s real history of only
# ~127-150 days. This is exactly the "genuinely new data source" the
# exhaustive 2026-08-22 research pass concluded was the actual lever left --
# more features/models/horizons on the SAME ~127 days had already been
# proven not to help (see [[project-trading-system-state]]).
#
# Honestly validated before switching (purged walk-forward + a genuine
# 70/30 holdout + effective-non-overlapping-sample-size correction, matching
# this project's established discipline): retraining on PAXG's deeper
# history produced a real, statistically significant improvement for the
# INTRADAY horizon only (holdout dir_acc 0.5245 vs the live baseline's
# 0.517, z=4.80 against an effective n of ~9566 -- comfortably past the
# |z|>2 bar this project's other validated findings use). short_term and
# long_term did NOT clear that bar even with 10-30x more rows (z=0.68 and
# z=-0.22 respectively) -- switched anyway for ONE consistent price source
# across the whole Gold page, but their accuracy must be read as "no proven
# edge, essentially a coin flip," not as improved.
GOLD_SPOT_SYMBOL = "PAXGUSDT"
GOLD_BINANCE_SYMBOL = "PAXG/USDT"

GOLD_RESOLUTIONS = ["15m", "1h", "1d"]
_CCXT_TIMEFRAME = {"15m": "15m", "1h": "1h", "1d": "1d"}

# How far back each resolution's ONE-TIME deep backfill reaches. 15m is
# deliberately capped at 2023-01-01 (~3.5 years) rather than PAXG's full
# 2020-08-28 inception -- that's the exact window the validation above used,
# and pulling the full ~6 years at 15m resolution (~211k candles) would
# roughly double the one-time backfill time for no validated benefit. 1h/1d
# use PAXG's full available history since both are cheap to fetch in full
# (~52k and ~2.2k candles respectively).
_DEEP_BACKFILL_SINCE = {
    "15m": dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc),
    "1h": dt.datetime(2020, 8, 28, tzinfo=dt.timezone.utc),
    "1d": dt.datetime(2020, 8, 28, tzinfo=dt.timezone.utc),
}


async def _fetch_ohlcv_since(timeframe: str, since_ms: int, max_batches: int = 400) -> list[dict]:
    """Pages through Binance's real OHLCV history for PAXG/USDT starting at
    `since_ms` (inclusive), 1000 candles per request (ccxt/Binance's own
    page size), until either the exchange stops returning full pages (i.e.
    we've reached "now") or `max_batches` is hit as a hard safety cap.
    """
    exchange = ccxt_async.binance()
    exchange.enableRateLimit = True
    rows: list[dict] = []
    since = since_ms
    try:
        for _ in range(max_batches):
            batch = await exchange.fetch_ohlcv(GOLD_BINANCE_SYMBOL, timeframe=timeframe, since=since, limit=1000)
            if not batch:
                break
            for ts_ms, o, h, l, c, v in batch:
                rows.append({"time": int(ts_ms / 1000), "open": o, "high": h, "low": l, "close": c, "volume": v or 0.0})
            since = batch[-1][0] + 1
            if len(batch) < 1000:
                break
    finally:
        await exchange.close()
    return rows


async def gold_deep_backfill() -> None:
    """One-time (per resolution) deep pull of PAXG/USDT's real Binance
    history. Only actually pages through years of history for a resolution
    that doesn't already have it (fewer than 1000 stored candles) -- so a
    server restart never re-fetches history it already has; incremental
    backfill (below) keeps it fresh from then on. Safe to call on every
    startup for exactly that reason.
    """
    for resolution in GOLD_RESOLUTIONS:
        try:
            existing = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=1000)
            if len(existing) >= 1000:
                continue
            since_ms = int(_DEEP_BACKFILL_SINCE[resolution].timestamp() * 1000)
            candles = await _fetch_ohlcv_since(_CCXT_TIMEFRAME[resolution], since_ms)
            await upsert_candles(GOLD_SPOT_SYMBOL, resolution, candles)
            logger.info("Gold deep backfill: stored %d %s PAXG/USDT candles", len(candles), resolution)
        except Exception:
            logger.exception("Gold deep backfill failed for resolution %s", resolution)


async def gold_incremental_backfill() -> None:
    """Keeps PAXG/USDT fresh between scheduled research ticks -- small,
    cheap recent windows per resolution, same pattern as ingest_rest's own
    incremental backfill for the crypto symbols.
    """
    now_ms = int(time.time() * 1000)
    windows_seconds = {"15m": 3 * 3600, "1h": 6 * 3600, "1d": 2 * 86400}
    for resolution, lookback_seconds in windows_seconds.items():
        try:
            candles = await _fetch_ohlcv_since(_CCXT_TIMEFRAME[resolution], now_ms - lookback_seconds * 1000)
            await upsert_candles(GOLD_SPOT_SYMBOL, resolution, candles)
        except Exception:
            logger.exception("Gold incremental %s backfill failed", resolution)
