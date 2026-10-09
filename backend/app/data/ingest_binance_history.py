from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.data.ingest_rest import upsert_candles
from app.data.query import load_candles_df

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
BINANCE_SPOT_SYMBOL = {
    "BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT",
    # Added 2026-09-07. A 30-asset universe study found no width that
    # generalises -- train, validation and test disagreed about the optimum,
    # and the rule that picked 29 symbols scored 0.54 on test against 0.86 for
    # six. BNB is the single addition that improved ALL THREE windows
    # (TRAIN 1.30 -> 1.50, VALIDATION 1.09 -> 1.16, TEST 0.67 -> 0.70), so it
    # is the only one taken. Everything wider is noise, and the weak names
    # measurably drag: LTC gross Sharpe -0.01, BCH 0.24, TRX 0.40.
    "BNBUSD": "BNBUSDT",
    # Small-account fillers (Settings.core_symbols, 2026-10-04): NOT part of
    # the core book -- they trade only while the balance cannot hold every
    # core symbol, chosen by contract size alone ($0.10-0.39 a contract).
    "HBARUSD": "HBARUSDT", "MANAUSD": "MANAUSDT", "XLMUSD": "XLMUSDT",
    "ADAUSD": "ADAUSDT", "TRXUSD": "TRXUSDT", "STXUSD": "STXUSDT",
}
HISTORY_PREFIX = "BINANCE:"
FUNDING_PREFIX = "BFUNDHIST:"

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


# Binance USDM perpetual funding-rate HISTORY, back to 2019-09-10 for BTCUSDT.
#
# Needed because funding_carry was cut from the live lineup on evidence that
# turned out to be invalid: it was evaluated against Delta's FUNDING: series,
# which carries the PREMIUM INDEX rather than the funding rate (sign-inverted
# and ~4x off -- see app/features/funding.py). The strategy has never had a
# fair test against real funding, and testing it needs deep FUNDING history,
# not just deep price history.
#
# Stored as BFUNDHIST:{symbol} to sit alongside BINANCE:{symbol} prices and
# stay clear of both Delta's live series and the shallow BFUND: series the
# hourly scheduler maintains.
_FUNDING_PERP = {
    "BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT", "BNBUSD": "BNBUSDT",
}
_FUNDING_START_MS = 1_567_000_000_000  # ~2019-08-28, before BTCUSDT perp listed


async def fetch_binance_funding_history_deep(
    binance_symbol: str,
    start_ms: int = _FUNDING_START_MS,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Every funding settlement (8-hourly) from `start_ms` to now.

    Returned in candle shape with the FRACTION converted to PERCENT, matching
    how Delta quotes funding and how app/features/funding.py expects it --
    the exact unit mismatch that made `funding_spread` meaningless until it
    was fixed on 2026-09-03.
    """
    owns = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    out: list[dict] = []
    cursor = start_ms
    try:
        while True:
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": binance_symbol, "startTime": cursor, "limit": 1000},
            )
            if resp.status_code == 429:
                await asyncio.sleep(5.0)
                continue
            resp.raise_for_status()
            rows = resp.json()
            if not isinstance(rows, list) or not rows:
                break
            for r in rows:
                pct = float(r["fundingRate"]) * 100.0  # fraction -> percent
                t = int(r["fundingTime"]) // 1000
                out.append({"time": t, "open": pct, "high": pct, "low": pct,
                            "close": pct, "volume": 0.0})
            last = int(rows[-1]["fundingTime"])
            if last + 1 <= cursor:
                break
            cursor = last + 1
            if len(rows) < 1000:
                break
            await asyncio.sleep(_SLEEP_BETWEEN_CALLS)
    finally:
        if owns:
            await client.aclose()
    return out


async def backfill_deep_funding(
    symbols: list[str] | None = None, start_ms: int | None = None
) -> dict[str, int]:
    """Funding settlements into BFUNDHIST:{symbol}.

    start_ms=None (the default) fetches the full history -- the one-time deep
    backfill. The 15-minute scheduler refresh passes a recent window instead:
    it used to re-download and re-upsert all ~29,000 settlements since 2019
    on every run (~30 Binance pages and ~29k row writes each time, every 15
    minutes) to pick up at most one new 8-hourly print per symbol. The window
    still reaches back to the newest STORED settlement, so an outage longer
    than the window cannot leave a permanent hole, and a symbol with nothing
    stored gets its full history.
    """
    symbols = symbols or list(_FUNDING_PERP)
    counts: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for symbol in symbols:
            perp = _FUNDING_PERP.get(symbol)
            if perp is None:
                continue
            symbol_start = _FUNDING_START_MS
            if start_ms is not None:
                newest = await load_candles_df(f"{FUNDING_PREFIX}{symbol}", "8h", limit=1)
                if len(newest):
                    symbol_start = min(start_ms, int(newest["time"].iloc[-1]) * 1000)
            try:
                rows = await fetch_binance_funding_history_deep(perp, start_ms=symbol_start, client=client)
                if rows:
                    await upsert_candles(f"{FUNDING_PREFIX}{symbol}", "8h", rows)
                counts[symbol] = len(rows)
                span = (rows[-1]["time"] - rows[0]["time"]) / 86400 if rows else 0
                logger.info("Backfilled %d %s funding settlements (%.0f days)", len(rows), symbol, span)
            except Exception:
                logger.exception("Deep funding backfill failed for %s", symbol)
                counts[symbol] = 0
    return counts
