from __future__ import annotations

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
# else's servers -- this page is expected to be polled every ~60s by the
# dashboard, and hitting both feeds that often would be rude for no benefit
# (RSS items don't change that fast). One shared cache, refreshed lazily.
_CACHE_TTL_SECONDS = 300
_cache: dict = {"fetched_at": 0.0, "headlines": []}


def _label(score: float) -> str:
    if score >= _POSITIVE_THRESHOLD:
        return "bullish"
    if score <= _NEGATIVE_THRESHOLD:
        return "bearish"
    return "neutral"


def _matched_symbols(title: str) -> list[str]:
    lowered = title.lower()
    return [sym for sym, keywords in NEWS_SYMBOL_KEYWORDS.items() if any(k in lowered for k in keywords)]


async def _get_headlines(force: bool = False) -> list[dict]:
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
