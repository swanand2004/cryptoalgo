from __future__ import annotations

from fastapi import APIRouter

from app.data.query import load_candles_df
from app.gold.engine import GOLD_SPOT_SYMBOL, load_gold_snapshot

router = APIRouter(prefix="/api/gold", tags=["gold"])


@router.get("/snapshot")
async def get_gold_snapshot():
    """Serves the last scheduler-computed research snapshot (price, the full
    parameter-bank readings, regime, and both horizon predictions) -- never
    recomputes on request, see gold.engine.build_gold_snapshot's docstring.
    """
    return load_gold_snapshot()


@router.get("/candles")
async def get_gold_candles(resolution: str = "1h", limit: int = 300):
    df = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=limit)
    return df.to_dict(orient="records")
