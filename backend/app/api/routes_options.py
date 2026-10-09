from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException, Query

from app.options.chain import UNDERLYINGS, build_chain_view, get_chain
from app.options.s7 import NOTE, s7_snapshot

router = APIRouter(prefix="/api/options", tags=["options"])


@router.get("/chain")
async def get_option_chain(
    underlying: str = Query("BTC"),
    expiry: str | None = Query(None, description="DDMMYY, e.g. 161026; default the nearest with a day left"),
    width: int = Query(10, ge=1, le=40),
):
    """Read-only Delta option chain (public API, cached ~60 s). Paper only."""
    underlying = underlying.upper()
    if underlying not in UNDERLYINGS:
        raise HTTPException(status_code=400, detail=f"underlying must be one of {', '.join(UNDERLYINGS)}")
    try:
        rows, fetched = await get_chain(underlying)
    except Exception as exc:  # network / Delta down with nothing cached
        raise HTTPException(status_code=502, detail=f"Delta option chain unavailable: {exc}") from exc
    view = build_chain_view(rows, expiry_code=expiry, width=width)
    return {
        "underlying": underlying,
        "note": NOTE,
        "fetched_at": dt.datetime.fromtimestamp(fetched, dt.timezone.utc).isoformat(),
        **view,
    }


@router.get("/s7")
async def get_s7():
    """The PAPER S7 tracker: pre-registered rules, every weekly decision, P&L."""
    return await s7_snapshot()
