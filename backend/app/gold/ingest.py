from __future__ import annotations

import datetime as dt
import logging
import time

import ccxt.async_support as ccxt_async
import httpx

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
#
# 2026-10-05: the live series is now Delta's own XAUTUSD perpetual (Tether
# Gold, 24/7, ~$200M/day, 0.01% fees) -- the contract the user would actually
# trade, so the page's price is Delta's price. XAUT only listed on Delta on
# 2026-04-17, too short for the daily board (needs ~250 bars) or the models,
# so the stored series is SPLICED: PAXG/USDT history before XAUT's first Delta
# bar, Delta XAUTUSD from then on. Both are 1:1 gold tokens trading within a
# few basis points of each other (both ~0.2% under spot), so the join is
# continuous at the scale of any indicator here.
GOLD_SPOT_SYMBOL = "XAUTUSD"  # the stored, spliced series every gold module reads
GOLD_DELTA_SYMBOL = "XAUTUSD"
PAXG_HISTORY_SYMBOL = "PAXGUSDT"  # pre-2026-10-05 rows, kept as the splice's history
GOLD_BINANCE_SYMBOL = "PAXG/USDT"
XAUT_LAUNCH = dt.datetime(2026, 4, 17, 10, 45, tzinfo=dt.timezone.utc)
_DELTA_REST = "https://api.india.delta.exchange"
_DELTA_PAGE_BARS = 1500

GOLD_RESOLUTIONS = ["15m", "1h", "1d"]
_CCXT_TIMEFRAME = {"15m": "15m", "1h": "1h", "1d": "1d"}
_BAR_SECONDS = {"15m": 900, "1h": 3600, "1d": 86400}

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


def _new_binance():
    """A ccxt Binance client that loads SPOT markets only. PAXG/USDT is a spot
    pair, but ccxt's default also pulls the USD-M and COIN-M futures
    exchangeInfo on first use -- the fapi/dapi calls that timed out and failed
    gold gap repair (1h) on 7 of 8 VPS startups on 2026-10-06."""
    exchange = ccxt_async.binance()
    options = getattr(exchange, "options", None)
    if isinstance(options, dict) and isinstance(options.get("fetchMarkets"), dict):
        options["fetchMarkets"]["types"] = ["spot"]
    exchange.enableRateLimit = True
    return exchange


async def _fetch_ohlcv_since(
    timeframe: str, since_ms: int, max_batches: int = 400, until_ms: int | None = None, exchange=None
) -> list[dict]:
    """Pages through Binance's real OHLCV history for PAXG/USDT starting at
    `since_ms` (inclusive), 1000 candles per request (ccxt/Binance's own
    page size), until either the exchange stops returning full pages (i.e.
    we've reached "now") or `max_batches` is hit as a hard safety cap.
    Pass `exchange` to reuse one client (and its loaded markets) across calls;
    the caller then owns closing it.
    """
    owns = exchange is None
    exchange = exchange or _new_binance()
    rows: list[dict] = []
    since = since_ms
    try:
        for _ in range(max_batches):
            batch = await exchange.fetch_ohlcv(GOLD_BINANCE_SYMBOL, timeframe=timeframe, since=since, limit=1000)
            if not batch:
                break
            for ts_ms, o, h, l, c, v in batch:
                if until_ms is not None and ts_ms >= until_ms:
                    break
                rows.append({"time": int(ts_ms / 1000), "open": o, "high": h, "low": l, "close": c, "volume": v or 0.0})
            since = batch[-1][0] + 1
            if len(batch) < 1000 or (until_ms is not None and since >= until_ms):
                break
    finally:
        if owns:
            await exchange.close()
    return rows


async def _fetch_delta_since(resolution: str, since_s: int, until_s: int | None = None) -> list[dict]:
    """Delta XAUTUSD candles from `since_s` (bar open, inclusive) up to
    `until_s` (exclusive) or now, oldest first. Delta returns newest-first and
    caps each response, so this pages forward in fixed windows."""
    bar = _BAR_SECONDS[resolution]
    end_s = until_s if until_s is not None else int(time.time())
    rows: dict[int, dict] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        start = since_s
        while start < end_s:
            stop = min(end_s, start + _DELTA_PAGE_BARS * bar)
            resp = await client.get(
                f"{_DELTA_REST}/v2/history/candles",
                params={"symbol": GOLD_DELTA_SYMBOL, "resolution": resolution, "start": start, "end": stop},
            )
            resp.raise_for_status()
            for c in resp.json().get("result") or []:
                t = int(c["time"])
                if since_s <= t < end_s:
                    rows[t] = {"time": t, "open": float(c["open"]), "high": float(c["high"]),
                               "low": float(c["low"]), "close": float(c["close"]),
                               "volume": float(c.get("volume") or 0.0)}
            start = stop
    return [rows[t] for t in sorted(rows)]


async def _paxg_history_before(resolution: str, before_s: int) -> list[dict]:
    """PAXG/USDT bars strictly before `before_s`: the rows already stored from
    when PAXG was the gold source, or fetched fresh from Binance if absent."""
    stored = await load_candles_df(PAXG_HISTORY_SYMBOL, resolution, limit=1_000_000)
    if len(stored) >= 1000:
        stored = stored[stored["time"] < before_s]
        rows = []
        for r in stored.to_dict("records"):
            rows.append({"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
                         "low": float(r["low"]), "close": float(r["close"]), "volume": float(r.get("volume") or 0.0)})
        return rows
    since_ms = int(_DEEP_BACKFILL_SINCE[resolution].timestamp() * 1000)
    return await _fetch_ohlcv_since(_CCXT_TIMEFRAME[resolution], since_ms, until_ms=before_s * 1000)


async def gold_deep_backfill() -> None:
    """One-time (per resolution) build of the spliced gold series: PAXG/USDT
    history up to XAUT's first Delta bar, Delta XAUTUSD from there. Only runs
    for a resolution that doesn't already have it (fewer than 1000 stored
    candles) -- so a restart never re-fetches history it already has;
    incremental backfill (below) keeps it fresh from then on. Safe to call on
    every startup for exactly that reason.
    """
    for resolution in GOLD_RESOLUTIONS:
        try:
            existing = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=1000)
            if len(existing) >= 1000:
                continue
            xaut = await _fetch_delta_since(resolution, int(XAUT_LAUNCH.timestamp()))
            splice_s = xaut[0]["time"] if xaut else int(time.time())
            paxg = await _paxg_history_before(resolution, splice_s)
            await upsert_candles(GOLD_SPOT_SYMBOL, resolution, paxg + xaut)
            logger.info(
                "Gold deep backfill %s: %d PAXG bars + %d Delta XAUTUSD bars (from %s)",
                resolution, len(paxg), len(xaut), dt.datetime.fromtimestamp(splice_s, dt.timezone.utc).isoformat(),
            )
        except Exception:
            logger.exception("Gold deep backfill failed for resolution %s", resolution)
    await gold_repair_gaps()


async def gold_incremental_backfill() -> None:
    """Keeps the gold series fresh from Delta XAUTUSD between research ticks -- small,
    cheap recent windows per resolution, same pattern as ingest_rest's own
    incremental backfill for the crypto symbols.
    """
    now_ms = int(time.time() * 1000)
    windows_seconds = {"15m": 3 * 3600, "1h": 6 * 3600, "1d": 2 * 86400}
    for resolution, lookback_seconds in windows_seconds.items():
        try:
            newest = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=1)
            newest_s = int(newest["time"].iloc[-1]) if len(newest) else None
            since_ms = incremental_start_ms(newest_s, now_ms, lookback_seconds)
            candles = await _fetch_delta_since(resolution, since_ms // 1000)
            await upsert_candles(GOLD_SPOT_SYMBOL, resolution, candles)
        except Exception:
            logger.exception("Gold incremental %s backfill failed", resolution)


def incremental_start_ms(newest_stored_s: int | None, now_ms: int, window_s: int) -> int:
    """Where an incremental refresh starts: the usual recent window, or the
    newest stored bar if that is OLDER.

    A fixed window alone (the old behaviour) only works while the app never
    stops for longer than it. It does stop -- the laptop sleeps -- and every
    longer outage became a permanent hole: 640 hourly and 2,779 15m PAXG bars
    were missing over 90 days when this was found (2026-10-02), silently
    corrupting every indicator and model built on them.
    """
    window_start = now_ms - window_s * 1000
    if newest_stored_s is None:
        return window_start
    return min(window_start, newest_stored_s * 1000)


def find_gaps(times: list[int], bar_seconds: int) -> list[tuple[int, int]]:
    """(first_missing_bar_open, next_present_bar_open) for every hole, in seconds.
    XAUT and PAXG both trade 24/7, so any step larger than one bar is missing data."""
    return [(a + bar_seconds, b) for a, b in zip(times, times[1:]) if b - a > bar_seconds]


async def gold_repair_gaps() -> None:
    """Refetches every hole in the stored history, range by range: from Delta
    for holes after XAUT's launch, from Binance PAXG before it. Run at startup;
    a hole the source itself has no data for simply stays a hole, costing one
    small request per startup.

    One Binance client serves every pre-XAUT hole. A fresh client per hole
    re-loaded Binance's market list each time (the 10 permanent 1h holes
    meant 10 market loads per startup), and one slow load failed the whole
    resolution."""
    exchange = None
    try:
        for resolution in GOLD_RESOLUTIONS:
            try:
                stored = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=1_000_000)
                gaps = find_gaps([int(t) for t in stored["time"]], _BAR_SECONDS[resolution])
                filled = 0
                for start_s, end_s in gaps:
                    if start_s >= XAUT_LAUNCH.timestamp():
                        candles = await _fetch_delta_since(resolution, start_s, until_s=end_s)
                    else:
                        exchange = exchange or _new_binance()
                        candles = await _fetch_ohlcv_since(
                            _CCXT_TIMEFRAME[resolution], start_s * 1000, until_ms=end_s * 1000, exchange=exchange
                        )
                    if candles:
                        await upsert_candles(GOLD_SPOT_SYMBOL, resolution, candles)
                        filled += len(candles)
                if gaps:
                    logger.info("Gold gap repair %s: %d holes, %d candles refilled", resolution, len(gaps), filled)
            except Exception:
                logger.exception("Gold gap repair failed for resolution %s", resolution)
    finally:
        if exchange is not None:
            await exchange.close()
