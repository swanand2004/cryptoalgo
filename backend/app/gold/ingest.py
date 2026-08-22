from __future__ import annotations

import logging

from app.core.delta_client import DeltaRestClient
from app.data.ingest_rest import backfill_symbol_history

logger = logging.getLogger(__name__)

# Tether Gold (XAUT), a token backed 1:1 by real physical gold -- Delta only
# lists XAUT as options underlyings (no perpetual/spot product), but its
# `.DEXAUTUSD` spot index (the same feed those options mark against) is a
# genuine continuous OHLC series, confirmed live against Delta's own candles
# endpoint (2026-08-21): ~4000 15m bars (~42 days -- a narrower retention
# window at this fine granularity), ~3000 hourly bars (~126 days), and ~127
# daily bars of real history, growing every day going forward. No separate
# exchange integration needed -- this reuses the same DeltaRestClient/
# candles endpoint already wired into the rest of this project.
GOLD_SPOT_SYMBOL = ".DEXAUTUSD"

GOLD_RESOLUTIONS = ["15m", "1h", "1d"]


async def gold_initial_backfill(client: DeltaRestClient, lookback_days: int = 400) -> None:
    """One-time deep backfill at startup, mirroring ingest_rest.backfill_all
    for the crypto symbols -- pulls as much `.DEXAUTUSD` history as Delta
    actually has (its own retention naturally caps this; requesting more
    than exists is harmless, see backfill_symbol_history) at every
    resolution gold's own models need.
    """
    for resolution in GOLD_RESOLUTIONS:
        try:
            await backfill_symbol_history(client, GOLD_SPOT_SYMBOL, resolution, lookback_seconds=lookback_days * 86400)
        except Exception:
            logger.exception("Initial gold backfill failed for resolution %s", resolution)


async def gold_incremental_backfill(client: DeltaRestClient) -> None:
    """Keeps `.DEXAUTUSD` fresh between scheduled research ticks -- small,
    cheap windows per resolution, same pattern as scheduler.
    _incremental_backfill for the crypto symbols.
    """
    try:
        await backfill_symbol_history(client, GOLD_SPOT_SYMBOL, "15m", lookback_seconds=3 * 3600)
    except Exception:
        logger.exception("Incremental gold 15m backfill failed")
    try:
        await backfill_symbol_history(client, GOLD_SPOT_SYMBOL, "1h", lookback_seconds=6 * 3600)
    except Exception:
        logger.exception("Incremental gold 1h backfill failed")
    try:
        await backfill_symbol_history(client, GOLD_SPOT_SYMBOL, "1d", lookback_seconds=2 * 86400)
    except Exception:
        logger.exception("Incremental gold 1d backfill failed")
