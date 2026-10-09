from __future__ import annotations

import asyncio
import calendar
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
# BNBUSD joined the live universe on 2026-09-07 but was never added here, so
# every funding/OI/long-short refresh for it raised KeyError each cycle.
# The size-aware filler coins (ADA/HBAR/MANA/STX/TRX/XLM, 2026-10-05) hit the
# same KeyError -- 18 tracebacks an hour. Unmapped symbols now get an empty
# series instead. They are deliberately NOT added here: these series feed the
# Analyst's factor scores, so adding them would change its calls on those
# coins, which is a trading decision, not a bug fix.
BINANCE_SYMBOL = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "SOLUSD": "SOLUSDT", "BNBUSD": "BNBUSDT"}
DERIBIT_CURRENCY = {"BTCUSD": "BTC", "ETHUSD": "ETH"}

FNG_SYMBOL = "FNG:GLOBAL"

# Free, no-API-key crypto trade-press RSS feeds. Deliberately global outlets,
# not an India-specific source: Delta India's product is USD-quoted perps
# (see funding-carry research elsewhere in this project), so what moves
# BTC/ETH/SOL is global crypto news -- India-specific regulatory stories
# (RBI/SEBI, crypto tax) that are actually market-relevant already surface
# in this same global trade press, so one feed set covers both rather than
# needing a separate "local news" source.
NEWS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
}
NEWS_SYMBOL_KEYWORDS = {
    "BTCUSD": ("bitcoin", "btc"),
    "ETHUSD": ("ethereum", "eth", "ether"),
    "SOLUSD": ("solana", "sol"),
}
NEWS_GLOBAL_SYMBOL = "NEWS:GLOBAL"

# VADER's general-purpose lexicon misses crypto/finance jargon entirely --
# confirmed live against real headlines (2026-09-02): "The bearish 'Bart
# Simpson' pattern is back as bitcoin and XRP prices..." scored a flat 0.0,
# despite the word "bearish" appearing outright, because it simply isn't in
# VADER's base word list. This is VADER's OWN documented extension
# mechanism (updating `.lexicon`), not a different model or a hand-tuned
# override of its scoring logic.
#
# Deliberately a short, UNAMBIGUOUS list only. Words like "dip", "support",
# "resistance", or "correction" read as bullish or bearish depending on the
# surrounding sentence (buying a dip vs. price dipping; a correction down
# vs. a bug fix) -- adding those would misfire more often than the flat
# zero they're replacing, so they're left out on purpose rather than
# forgotten. Single tokens only: VADER looks up one word at a time, so a
# multi-word phrase as a dict key would silently never match anything.
CRYPTO_LEXICON_EXTENSION: dict[str, float] = {
    # bullish
    "bullish": 2.5, "rally": 2.0, "rallies": 2.0, "rallying": 2.0,
    "surge": 2.2, "surges": 2.2, "surging": 2.2,
    "soar": 2.3, "soars": 2.3, "soaring": 2.3,
    "rebound": 1.8, "rebounds": 1.8, "rebounding": 1.8,
    "breakout": 1.8, "moon": 1.5, "mooning": 1.8,
    "adoption": 1.5, "accumulate": 1.3, "accumulation": 1.3,
    "inflow": 1.5, "inflows": 1.5,
    "approval": 1.8, "approved": 1.8, "approves": 1.8,
    "ath": 2.0,
    # bearish
    "bearish": -2.5, "plunge": -2.5, "plunges": -2.5, "plunging": -2.5,
    "crash": -2.8, "crashes": -2.8, "crashing": -2.8,
    "dump": -1.8, "dumps": -1.8, "dumping": -1.8, "dumped": -1.8,
    "hack": -2.5, "hacked": -2.5, "hacking": -2.3,
    "exploit": -2.0, "exploited": -2.2, "breach": -2.0, "breached": -2.0,
    "stolen": -2.3, "theft": -2.0,
    "lawsuit": -1.5, "sues": -1.5, "sued": -1.5, "suing": -1.5,
    "ban": -2.0, "banned": -2.2, "banning": -2.0, "crackdown": -2.2,
    "liquidation": -2.0, "liquidated": -2.2, "liquidations": -2.0,
    "outflow": -1.5, "outflows": -1.5,
    "delist": -1.8, "delisted": -1.8, "delisting": -1.8,
    "insolvent": -2.5, "insolvency": -2.5, "bankruptcy": -2.8, "bankrupt": -2.8,
    "capitulation": -2.0, "selloff": -2.0, "sell-off": -2.0,
    "fraud": -2.5, "scam": -2.5, "rug": -2.3, "rugpull": -2.5,
}


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
    binance_symbol = BINANCE_SYMBOL.get(delta_symbol)
    if binance_symbol is None:
        return []  # unmapped symbol: no series, not a KeyError every hour (see BINANCE_SYMBOL)
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
    binance_symbol = BINANCE_SYMBOL.get(delta_symbol)
    if binance_symbol is None:
        return []  # unmapped symbol: no series, not a KeyError every hour (see BINANCE_SYMBOL)
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
    binance_symbol = BINANCE_SYMBOL.get(delta_symbol)
    if binance_symbol is None:
        return []  # unmapped symbol: no series, not a KeyError every hour (see BINANCE_SYMBOL)
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


async def fetch_crypto_news_headlines() -> list[dict]:
    """Latest headlines from NEWS_FEEDS -- free, no API key. Each is a plain
    RSS feed, so this only ever sees the CURRENT ~20-30 items per source,
    never historical archives: there is no free source of historical crypto
    news to backfill from, unlike Fear & Greed/Binance/Deribit above, which
    all expose a real history endpoint. This can only start accumulating
    real history from whenever it's first run, same limitation already
    documented for Binance OI/long-short/DVOL (see the FEATURE_COLUMNS
    exclusion note in feature_pipeline.py) -- and that limitation already
    measurably HURT the model there (short history reads as a spurious
    time-index proxy), so this is wired to compute and store the feature,
    not to feed it to the ML model, until there's enough real history to
    honestly walk-forward test it the same way.

    Independently fault-tolerant per source, same as every other fetch_*
    here -- one feed being down/blocked must not lose the other.
    """
    import feedparser

    headlines: list[dict] = []
    async with httpx.AsyncClient(
        timeout=15.0, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True
    ) as client:
        for source, url in NEWS_FEEDS.items():
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                parsed = feedparser.parse(resp.content)
                for entry in parsed.entries:
                    title = getattr(entry, "title", None)
                    published = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
                    if not title or published is None:
                        continue
                    # feedparser normalises published_parsed/updated_parsed to a
                    # UTC struct_time -- time.mktime() would be wrong here, since
                    # it interprets its input as LOCAL time and converts using
                    # the process's own timezone. On this IST (UTC+5:30) machine
                    # that silently shifted every headline's timestamp back by
                    # 5.5 hours, making "just published" news read as ~6 hours
                    # stale. calendar.timegm() is the UTC-correct inverse of
                    # gmtime()/feedparser's own normalisation.
                    t = calendar.timegm(published)
                    headlines.append({"time": t, "title": title, "source": source})
            except Exception:
                logger.exception("News feed fetch failed for %s -- skipping, treated as unavailable", source)
    return headlines


def score_headline_sentiment(headlines: list[dict]) -> list[dict]:
    """Attaches a VADER compound sentiment score in [-1, 1] to each headline.
    VADER (rule-based, lexicon + heuristics) rather than a transformer model
    like FinBERT: pure Python, no model download, no GPU/torch dependency --
    this runs on the same local Windows machine as the trading loop itself,
    and VADER was built and validated specifically for short, informal text,
    which a headline is close enough to. Extended with CRYPTO_LEXICON_EXTENSION
    for the crypto/finance jargon VADER's base lexicon has no opinion on at
    all (see that dict's own comment) -- still nowhere near a dedicated
    finance model like FinBERT, which is a real limitation acted on honestly,
    not hidden: see the walk-forward validation this feature is subject to
    before it's ever trusted with a trading decision.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CRYPTO_LEXICON_EXTENSION)
    return [
        {**h, "sentiment": analyzer.polarity_scores(h["title"])["compound"]}
        for h in headlines
    ]


def aggregate_news_sentiment(scored_headlines: list[dict], keywords: tuple[str, ...] | None = None) -> list[dict]:
    """Buckets scored headlines into hourly [time, open, high, low, close,
    volume] rows (candle-shaped, like every other pseudo-series here) --
    `close` is the mean sentiment for that hour, `volume` the headline count
    (so a downstream reader can tell one-headline noise from a genuine
    multi-story hour). `keywords`, if given, filters to headlines whose
    title mentions any of them (case-insensitive) -- e.g. NEWS_SYMBOL_KEYWORDS
    for a per-symbol series; omit for the NEWS_GLOBAL_SYMBOL market-wide mood.
    """
    if keywords is not None:
        scored_headlines = [
            h for h in scored_headlines if any(k in h["title"].lower() for k in keywords)
        ]
    if not scored_headlines:
        return []

    buckets: dict[int, list[float]] = {}
    for h in scored_headlines:
        hour = (h["time"] // 3600) * 3600
        buckets.setdefault(hour, []).append(h["sentiment"])

    candles = []
    for hour, scores in sorted(buckets.items()):
        mean_score = sum(scores) / len(scores)
        candles.append(
            {"time": hour, "open": mean_score, "high": mean_score, "low": mean_score,
             "close": mean_score, "volume": float(len(scores))}
        )
    return candles


async def fetch_and_store_news_sentiment(symbols: list[str]) -> None:
    """Fetches the current RSS window once, scores it once, and buckets it
    per symbol plus one global series -- rather than one fetch per symbol,
    which would just hit the same two feeds redundantly and let one keyword
    set's failure have nothing to do with another's.
    """
    try:
        headlines = await fetch_crypto_news_headlines()
        if not headlines:
            return
        scored = score_headline_sentiment(headlines)
    except Exception:
        logger.exception("News sentiment fetch/scoring failed")
        return

    try:
        glob = aggregate_news_sentiment(scored, keywords=None)
        if glob:
            await upsert_candles(NEWS_GLOBAL_SYMBOL, "1h", glob)
    except Exception:
        logger.exception("News sentiment global aggregation/store failed")

    for symbol in symbols:
        keywords = NEWS_SYMBOL_KEYWORDS.get(symbol)
        if keywords is None:
            continue
        try:
            per_symbol = aggregate_news_sentiment(scored, keywords=keywords)
            if per_symbol:
                await upsert_candles(f"NEWS:{symbol}", "1h", per_symbol)
        except Exception:
            logger.exception("News sentiment aggregation/store failed for %s", symbol)


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

    # No historical backfill possible for news (see fetch_crypto_news_headlines) --
    # this just captures whatever's currently in the RSS window as a starting point.
    try:
        await fetch_and_store_news_sentiment(symbols)
        logger.info("Captured initial news sentiment window for %s", symbols)
    except Exception:
        logger.exception("News sentiment initial capture failed")
