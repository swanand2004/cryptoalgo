from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time

import pandas as pd
from fastapi import APIRouter

from app.config import get_settings
from app.data.query import load_candles_df
from app.features.feature_pipeline import build_feature_frame
from app.ml.predict import MLPredictor
from app.strategy.crypto_blend import MODEL_WEIGHT, snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

# Building a feature frame per symbol is not free (indicators, astrology,
# microstructure), and this endpoint is polled by a dashboard. Same shape as
# routes_news' cache: one shared lock, double-checked, with the stale copy
# served if a refresh fails rather than surfacing an error to a page that was
# rendering fine a minute ago.
_CACHE_TTL_SECONDS = 60
_cache: dict[str, object] = {"at": 0.0, "payload": None}
_lock = asyncio.Lock()

RESOLUTION = "1h"
CANDLE_LIMIT = 1500      # comfortably past the 720-bar z-score window
MACRO_LIMIT = 3000


async def _macro_series(symbol: str, grid: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """External series aligned onto the candle grid.

    Forward-filled, never interpolated: only the last ALREADY-PUBLISHED value
    is carried onto a later bar. Each series missing degrades that one factor
    to neutral rather than failing the request.
    """

    async def onto(sym: str, resolution: str) -> pd.Series | None:
        try:
            df = await load_candles_df(sym, resolution, limit=MACRO_LIMIT)
        except Exception:
            return None
        if df.empty:
            return None
        idx = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s", utc=True))
        s = pd.Series(df["close"].astype(float).to_numpy(), index=idx).groupby(level=0).last()
        return s.reindex(s.index.union(grid)).ffill().reindex(grid)

    dxy, real, be, fng, funding = await asyncio.gather(
        onto("MACRO:DXY", "1d"),
        onto("MACRO:REALRATE10Y", "1d"),
        onto("MACRO:BREAKEVEN10Y", "1d"),
        onto("FNG:GLOBAL", "1d"),
        onto(f"BFUNDHIST:{symbol}", "8h"),
    )
    return {"dxy": dxy, "real_rate": real, "breakeven": be, "fear_greed": fng, "funding": funding}


async def _news_series(symbol: str, grid: pd.DatetimeIndex) -> pd.Series | None:
    for sym in (f"NEWS:{symbol}", "NEWS:GLOBAL"):
        try:
            df = await load_candles_df(sym, "1h", limit=500)
        except Exception:
            continue
        if df.empty:
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s", utc=True))
        s = pd.Series(df["close"].astype(float).to_numpy(), index=idx).groupby(level=0).last()
        return s.reindex(s.index.union(grid)).ffill().reindex(grid)
    return None


async def _build_one(symbol: str) -> dict:
    candles = await load_candles_df(symbol, RESOLUTION, limit=CANDLE_LIMIT)
    if candles.empty or len(candles) < 300:
        return {
            "symbol": symbol, "ready": False,
            "reason": f"only {len(candles)} {RESOLUTION} bars ingested so far",
        }

    frame = await asyncio.to_thread(build_feature_frame, candles, symbol=symbol)
    grid = pd.DatetimeIndex(pd.to_datetime(frame["time"], unit="s", utc=True))

    macro, news = await asyncio.gather(_macro_series(symbol, grid), _news_series(symbol, grid))

    # The model's own score, on the tail row only -- predicting the whole
    # history to read the last value is pure waste.
    model_score: float | None = None
    model_ready = False
    predictor = MLPredictor(symbol, RESOLUTION)
    if predictor.is_ready:
        try:
            model_score = float(predictor.predict_score_batch(frame.tail(1)).iloc[-1])
            model_ready = True
        except Exception:
            logger.exception("crypto blend: model prediction failed for %s", symbol)

    frame_indexed = frame.set_index(grid)
    out = snapshot(
        symbol=symbol,
        frame=frame_indexed,
        model_score=model_score,
        macro={k: (v if v is None else v.reset_index(drop=True)) for k, v in macro.items()},
        news_sentiment=None if news is None else news.reset_index(drop=True),
        model_weight=get_settings().crypto_blend_model_weight,
    )
    out["ready"] = True
    out["model_ready"] = model_ready
    return out


async def _build_all() -> dict:
    settings = get_settings()
    symbols = list(settings.symbol_list)
    results = await asyncio.gather(*(_build_one(s) for s in symbols), return_exceptions=True)
    predictions = []
    for symbol, res in zip(symbols, results):
        if isinstance(res, Exception):
            logger.exception("crypto blend failed for %s", symbol, exc_info=res)
            predictions.append({"symbol": symbol, "ready": False, "reason": "computation failed"})
        else:
            predictions.append(res)
    return {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_weight": settings.crypto_blend_model_weight,
        "default_model_weight": MODEL_WEIGHT,
        "resolution": RESOLUTION,
        "predictions": predictions,
    }


@router.get("/crypto")
async def crypto_predictions():
    now = time.time()
    cached = _cache.get("payload")
    if cached is not None and now - float(_cache["at"]) < _CACHE_TTL_SECONDS:
        return cached

    async with _lock:
        now = time.time()
        cached = _cache.get("payload")
        if cached is not None and now - float(_cache["at"]) < _CACHE_TTL_SECONDS:
            return cached
        try:
            payload = await _build_all()
        except Exception:
            logger.exception("crypto predictions failed to build")
            if cached is not None:
                # A stale answer beats an error on a page that was fine a
                # minute ago; the timestamp shows the reader it is stale.
                return cached
            raise
        _cache["payload"] = payload
        _cache["at"] = time.time()
        return payload
