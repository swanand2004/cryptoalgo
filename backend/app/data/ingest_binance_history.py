from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.data.ingest_rest import upsert_candles

logger = logging.getLogger(__name__)

# Deep historical OHLCV from Binance -- free, no API key, ~9 years back.
#
# Why this exists: every negative result in this project traces to sample
# size, not to a shortage of feature sources. Delta gives 111 days of 1h
# candles (2,666 bars). Binance gives 9.1 years (79,296 bars) of the same
# hour, for free. The 1080-config sweep on 2026-09-02 produced a deflated
# Sharpe of 0.000 largely because trial-Sharpe dispersion was 6.09 -- which
# is what happens when every configuration is measured on a few dozen
# trades. More history is the only thing that fixes that.
#
# STORED UNDER A SEPARATE SYMBOL PREFIX ON PURPOSE. Delta's own candles are
# what live trading executes against and must not be silently blended with
# another venue's prints. BINANCE:BTCUSD is for BACKTESTING and ML TRAINING
# only; the live path never reads it.
BINANCE_SPOT_SYMBOL = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT"}
HISTORY_PREFIX = "BINANCE:"

# Binance kline interval codes for the resolutions this project uses.
_INTERVAL = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
_MAX_LIMIT = 1000
# Binance allows 1200 request-weight/minute and a limit=1000 klines call
# costs 2. This is far under that, but stays polite on a free endpoint.
_SLEEP_BETWEEN_CALLS = 0.12


async def fetch_binance_klines(
    binance_symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Pages through Binance spot klines, returning candle dicts in the same
    shape upsert_candles expects.

    Binance returns at most 1000 bars per call, so this walks forward from
    `start_ms` using the last returned open-time + 1ms as the next cursor --
    NOT a fixed time step, which would silently skip gaps where a market was
    halted or a symbol had not listed yet.
    """
    if end_ms is None:
        end_ms = int(time.time() * 1000)

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    out: list[dict] = []
    cursor = start_ms
    try:
        while cursor < end_ms:
            resp = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": binance_symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": _MAX_LIMIT,
                },
            )
            if resp.status_code == 429:
                logger.warning("Binance rate limited on %s %s -- backing off", binance_symbol, interval)
                await asyncio.sleep(5.0)
                continue
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            for k in rows:
                out.append({
                    "time": int(k[0]) // 1000,  # open time, unix seconds
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            last_open = int(rows[-1][0])
            if last_open + 1 <= cursor:  # no forward progress; stop rather than spin
                break
            cursor = last_open + 1
            if len(rows) < _MAX_LIMIT:
                break
            await asyncio.sleep(_SLEEP_BETWEEN_CALLS)
    finally:
        if owns_client:
            await client.aclose()
    return out


async def backfill_deep_history(
    symbols: list[str] | None = None,
    resolutions: tuple[str, ...] = ("1h", "4h"),
    start_ms: int = 0,
) -> dict[str, int]:
    """Backfills BINANCE:{symbol} candles for every requested resolution.

    `start_ms=0` means "from whenever Binance's own history begins" -- it
    clamps to the symbol's listing date on its own rather than needing one
    hardcoded per symbol (SOL listed in 2020, BTC in 2017).

    Each symbol/resolution is independently fault tolerant: one failing must
    not lose the others, matching every other ingestion path here.
    """
    symbols = symbols or list(BINANCE_SPOT_SYMBOL)
    counts: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for symbol in symbols:
            binance_symbol = BINANCE_SPOT_SYMBOL.get(symbol)
            if binance_symbol is None:
                logger.warning("No Binance mapping for %s -- skipped", symbol)
                continue
            for resolution in resolutions:
                interval = _INTERVAL.get(resolution)
                if interval is None:
                    logger.warning("No Binance interval for resolution %s -- skipped", resolution)
                    continue
                key = f"{HISTORY_PREFIX}{symbol}:{resolution}"
                try:
                    candles = await fetch_binance_klines(binance_symbol, interval, start_ms, client=client)
                    if candles:
                        await upsert_candles(f"{HISTORY_PREFIX}{symbol}", resolution, candles)
                    counts[key] = len(candles)
                    span_days = (candles[-1]["time"] - candles[0]["time"]) / 86400 if candles else 0
                    logger.info("Backfilled %d %s %s bars (%.0f days)", len(candles), symbol, resolution, span_days)
                except Exception:
                    logger.exception("Deep-history backfill failed for %s %s", symbol, resolution)
                    counts[key] = 0
    return counts
