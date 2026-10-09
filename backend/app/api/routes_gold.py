from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter

from app.data.query import load_candles_df
from app.gold.engine import GOLD_SPOT_SYMBOL, load_gold_snapshot
from app.gold.signal_board import build_board

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


# (resolution, bar seconds, horizon bars, label, history to load). Hourly loads
# ~7 months: enough to warm the 200-bar indicators and still leave thousands of
# bars of track record, without rebuilding 52k rows on every refresh.
_BOARDS = (
    ("1d", 86_400, 5, "daily", 3_000),
    ("1h", 3_600, 4, "hourly", 5_000),
)
_board_cache: dict = {"at": 0.0, "value": None}


@router.get("/signal_board")
async def get_signal_board():
    """Technicals + Gann direction board for both timeframes. Cached 60s and
    built off the event loop -- the dashboard polls it, and the trading loop
    shares this process."""
    if _board_cache["value"] is not None and time.monotonic() - _board_cache["at"] < 60:
        return _board_cache["value"]
    frames = {label: await load_candles_df(GOLD_SPOT_SYMBOL, res, limit=limit) for res, _, _, label, limit in _BOARDS}

    def build():
        return {
            label: build_board(frames[label], bar_seconds=secs, horizon_bars=h, label=label)
            for _, secs, h, label, _ in _BOARDS
        }

    value = await asyncio.to_thread(build)
    _board_cache.update(at=time.monotonic(), value=value)
    return value
