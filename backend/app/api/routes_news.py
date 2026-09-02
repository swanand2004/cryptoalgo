from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time

from fastapi import APIRouter

from app.data.ingest_external import (
    NEWS_GLOBAL_SYMBOL,
    NEWS_SYMBOL_KEYWORDS,
    fetch_crypto_news_headlines,
    score_headline_sentiment,
)
from app.data.query import load_candles_df

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news", tags=["news"])

# VADER's own documented convention (compound >= 0.05 positive, <= -0.05
# negative), not an arbitrary cutoff invented for this page -- keeps the
# label consistent with the score a reader can see right next to it.
_POSITIVE_THRESHOLD = 0.05
_NEGATIVE_THRESHOLD = -0.05

# Headlines are cheap to re-fetch but CoinDesk/CoinTelegraph are somebody
# else's servers. 3 minutes keeps the page feeling live (RSS items don't
# actually change much faster than that anyway) without hitting both feeds
# on every ~60s dashboard poll.
_CACHE_TTL_SECONDS = 180
_cache: dict = {"fetched_at": 0.0, "headlines": []}
# Serializes refreshes so N concurrent requests hitting an expired cache (a
# few browser tabs polling at once, say) fetch the feeds once, not N times.
_refresh_lock = asyncio.Lock()

# |sentiment| bands for the "impact" badge -- distinguishes a headline
# actually worth a second look from routine coverage, independent of
# direction. Not a claim about real-world price impact, just about how
# strongly-worded the coverage itself is.
_HIGH_IMPACT_THRESHOLD = 0.5
_NOTABLE_IMPACT_THRESHOLD = 0.2


def _label(score: float) -> str:
    if score >= _POSITIVE_THRESHOLD:
        return "bullish"
    if score <= _NEGATIVE_THRESHOLD:
        return "bearish"
    return "neutral"


def _impact(score: float) -> str | None:
    magnitude = abs(score)
    if magnitude >= _HIGH_IMPACT_THRESHOLD:
        return "high"
    if magnitude >= _NOTABLE_IMPACT_THRESHOLD:
        return "notable"
    return None


def _matched_symbols(title: str) -> list[str]:
    lowered = title.lower()
    return [sym for sym, keywords in NEWS_SYMBOL_KEYWORDS.items() if any(k in lowered for k in keywords)]


async def _get_headlines(force: bool = False) -> list[dict]:
    now = time.time()
    if not force and _cache["headlines"] and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return _cache["headlines"]

    async with _refresh_lock:
        # Re-check: another request may have refreshed it while we waited.
        now = time.time()
        if not force and _cache["headlines"] and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
            return _cache["headlines"]

        try:
            raw = await fetch_crypto_news_headlines()
            scored = score_headline_sentiment(raw)
        except Exception:
            logger.exception("Live news fetch failed for the news page -- serving last known cache")
            return _cache["headlines"]

        enriched = sorted(
            (
                {
                    "title": h["title"],
                    "source": h["source"],
                    "time": h["time"],
                    "sentiment": round(h["sentiment"], 3),
                    "label": _label(h["sentiment"]),
                    "impact": _impact(h["sentiment"]),
                    "symbols": _matched_symbols(h["title"]),
                }
                for h in scored
            ),
            key=lambda h: h["time"],
            reverse=True,
        )
        _cache["headlines"] = enriched
        _cache["fetched_at"] = now
        return enriched


@router.get("/headlines")
async def get_headlines(limit: int = 40):
    """Latest scored headlines (global + no-API-key crypto trade press --
    see app/data/ingest_external.py). Cached for a few minutes server-side,
    not fetched fresh on every dashboard poll.
    """
    headlines = await _get_headlines()
    fetched_at = (
        dt.datetime.fromtimestamp(_cache["fetched_at"], dt.timezone.utc).isoformat()
        if _cache["fetched_at"]
        else None
    )
    return {"fetched_at": fetched_at, "headlines": headlines[:limit]}


@router.get("/sentiment_history")
async def get_sentiment_history(symbol: str = "GLOBAL", hours: int = 72):
    """Hourly-bucketed mean sentiment + headline count for one symbol (or
    GLOBAL for the market-wide reading), from the stored NEWS:* series --
    see aggregate_news_sentiment in ingest_external.py. Real history only
    goes back to whenever this was first deployed (2026-09-02); there is no
    backfilled archive, same limitation documented in feature_pipeline.py.
    """
    series_symbol = NEWS_GLOBAL_SYMBOL if symbol.upper() == "GLOBAL" else f"NEWS:{symbol}"
    df = await load_candles_df(series_symbol, "1h", limit=max(1, hours))
    return [
        {"time": int(row["time"]), "sentiment": row["close"], "headline_count": int(row["volume"])}
        for _, row in df.iterrows()
    ]
