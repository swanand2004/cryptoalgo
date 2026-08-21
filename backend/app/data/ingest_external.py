from __future__ import annotations

import asyncio
import logging
import time

import ccxt.async_support as ccxt_async
import httpx

from app.data.ingest_rest import upsert_candles

logger = logging.getLogger(__name__)

# Delta's BTCUSD/ETHUSD/SOLUSD -> the corresponding Binance USDM perpetual and
# Deribit options-currency symbol. SOL has no entry in DERIBIT_CURRENCY:
# Deribit runs no SOL options market at all (confirmed against the live API --
# get_volatility_index_data?currency=SOL returns an empty series, not an
# error), so there is no DVOL feature to compute for it. That's a real
# coverage gap, not a bug to work around.
BINANCE_SYMBOL = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT"}
DERIBIT_CURRENCY = {"BTCUSD": "BTC", "ETHUSD": "ETH"}

FNG_SYMBOL = "FNG:GLOBAL"


async def fetch_fear_greed_index(limit: int = 0) -> list[dict]:
    """Crypto Fear & Greed Index (alternative.me) -- free, no API key, one
    global daily reading (not per-symbol). `limit=0` returns full history.
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get("https://api.alternative.me/fng/", params={"limit": limit})
        resp.raise_for_status()
        payload = resp.json()
    candles = []
    for point in payload.get("data", []):
        v = float(point["value"])
        t = int(point["timestamp"])
        candles.append({"time": t, "open": v, "high": v, "low": v, "close": v, "volume": 0.0})
    return candles


async def fetch_binance_funding_history(delta_symbol: str, limit: int = 1000) -> list[dict]:
    """Binance USDM perpetual funding rate history -- free, no API key.
    Used to compute a cross-exchange funding SPREAD against Delta's own
    FUNDING:{symbol} series already ingested in ingest_rest.py: extreme
    divergence between venues signals relative positioning imbalance, a
    genuinely documented (if noisy) crowding/arbitrage signal.
    """
    binance_symbol = BINANCE_SYMBOL[delta_symbol]
    exchange = ccxt_async.binanceusdm()
    try:
        raw = await exchange.fetch_funding_rate_history(binance_symbol, limit=min(limit, 1000))
    finally:
        await exchange.close()
    candles = []
    for point in raw:
        v = float(point["fundingRate"])
        t = int(point["timestamp"] / 1000)
        candles.append({"time": t, "open": v, "high": v, "low": v, "close": v, "volume": 0.0})
    return candles


async def fetch_binance_open_interest_hist(delta_symbol: str, period: str = "1h", limit: int = 500) -> list[dict]:
    """Binance USDM aggregate open interest history -- free, no API key.
    Not part of ccxt's unified API, so hit the REST endpoint directly.
    Rising OI alongside rising price = fresh money confirming the trend;
    rising OI alongside falling price = fresh shorts piling in -- the
    standard institutional read on OI-vs-price divergence.
    """
    binance_symbol = BINANCE_SYMBOL[delta_symbol]
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": binance_symbol, "period": period, "limit": min(limit, 500)},
        )
        resp.raise_for_status()
        payload = resp.json()
    candles = []
    for point in payload:
        v = float(point["sumOpenInterest"])
        t = int(int(point["timestamp"]) / 1000)
        candles.append({"time": t, "open": v, "high": v, "low": v, "close": v, "volume": 0.0})
    return candles


async def fetch_binance_long_short_ratio(delta_symbol: str, period: str = "1h", limit: int = 500) -> list[dict]:
    """Binance USDM global long/short ACCOUNT ratio -- free, no API key.
    Extreme readings (heavily one-sided positioning) are the same "crowded
    trade" fragility signal that motivates funding-rate-based risk reads --
    a second, independent read on the same underlying phenomenon.
    """
    binance_symbol = BINANCE_SYMBOL[delta_symbol]
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": binance_symbol, "period": period, "limit": min(limit, 500)},
        )
        resp.raise_for_status()
        payload = resp.json()
    candles = []
    for point in payload:
        v = float(point["longShortRatio"])
        t = int(int(point["timestamp"]) / 1000)
        candles.append({"time": t, "open": v, "high": v, "low": v, "close": v, "volume": 0.0})
    return candles


async def fetch_deribit_dvol(delta_symbol: str, lookback_seconds: int, resolution_seconds: int = 3600) -> list[dict]:
    """Deribit's DVOL index -- the crypto analogue of VIX, a 30-day
    forward-looking implied volatility gauge derived from the full options
    surface. Free, no API key. BTC and ETH only (see DERIBIT_CURRENCY).

    Documented limitation acted on honestly rather than oversold: implied
    vol has been observed to react SLOWLY to sharp realized-vol changes
    (Deribit's own research), so this is included as a regime/positioning
    feature, not expected to be a precise short-term directional signal on
    its own.
    """
    currency = DERIBIT_CURRENCY.get(delta_symbol)
    if currency is None:
        return []
    end = int(time.time() * 1000)
    start = end - lookback_seconds * 1000
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            "https://www.deribit.com/api/v2/public/get_volatility_index_data",
            params={
                "currency": currency,
                "resolution": resolution_seconds,
                "start_timestamp": start,
                "end_timestamp": end,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    rows = payload.get("result", {}).get("data", [])
    candles = []
    for point in rows:
        t_ms, o, h, l, c = point
        candles.append({"time": int(t_ms / 1000), "open": o, "high": h, "low": l, "close": c, "volume": 0.0})
    return candles


async def fetch_google_trends(keywords: list[str], timeframe: str = "today 12-m") -> list[dict]:
    """Google search interest (0-100, relative) for the given keywords,
    averaged across them into one series. Free, but unofficial (scrapes
    Google Trends via pytrends) and can be rate-limited or blocked without
    warning -- treated as a best-effort, optional source: callers must
    handle an empty return, not assume it always succeeds.

    Documented limitation acted on honestly: academic studies find search
    interest AMPLIFIES whatever's already happening (attention rises with
    rallies and with selloffs alike) more than it cleanly leads price in one
    direction, so this is a regime/attention feature, not a directional one.
    """
    from pytrends.request import TrendReq

    def _fetch() -> list[dict]:
        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload(keywords, timeframe=timeframe)
        df = pytrends.interest_over_time()
        if df.empty:
            return []
        avg = df[keywords].mean(axis=1)
        out = []
        for ts, v in avg.items():
            t = int(ts.timestamp())
            out.append({"time": t, "open": float(v), "high": float(v), "low": float(v), "close": float(v), "volume": 0.0})
        return out

    try:
        return await asyncio.to_thread(_fetch)
    except Exception:
        logger.exception("Google Trends fetch failed for %s -- skipping, treated as unavailable", keywords)
        return []


async def backfill_external_all(symbols: list[str], lookback_days: int = 90) -> None:
    """Backfills every external pseudo-series used by build_feature_frame's
    optional external-data columns, for every symbol Delta trades. Each
    source is best-effort and independently fault-tolerant -- one source
    failing (e.g. Google Trends getting rate-limited) must not block the
    others or crash ingestion.
    """
    try:
        fng = await fetch_fear_greed_index(limit=0)
        await upsert_candles(FNG_SYMBOL, "1d", fng)
        logger.info("Backfilled %d Fear & Greed points", len(fng))
    except Exception:
        logger.exception("Fear & Greed backfill failed")

    for symbol in symbols:
        try:
            funding = await fetch_binance_funding_history(symbol, limit=1000)
            await upsert_candles(f"BFUND:{symbol}", "8h", funding)
            logger.info("Backfilled %d Binance funding points for %s", len(funding), symbol)
        except Exception:
            logger.exception("Binance funding backfill failed for %s", symbol)

        try:
            oi = await fetch_binance_open_interest_hist(symbol, period="1h", limit=500)
            await upsert_candles(f"BOI:{symbol}", "1h", oi)
            logger.info("Backfilled %d Binance OI points for %s", len(oi), symbol)
        except Exception:
            logger.exception("Binance OI backfill failed for %s", symbol)

        try:
            lsr = await fetch_binance_long_short_ratio(symbol, period="1h", limit=500)
            await upsert_candles(f"BLSR:{symbol}", "1h", lsr)
            logger.info("Backfilled %d Binance long/short points for %s", len(lsr), symbol)
        except Exception:
            logger.exception("Binance long/short ratio backfill failed for %s", symbol)

        try:
            dvol = await fetch_deribit_dvol(symbol, lookback_days * 86400)
            if dvol:
                await upsert_candles(f"DVOL:{symbol}", "1h", dvol)
                logger.info("Backfilled %d Deribit DVOL points for %s", len(dvol), symbol)
        except Exception:
            logger.exception("Deribit DVOL backfill failed for %s", symbol)
