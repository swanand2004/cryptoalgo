from __future__ import annotations

from fastapi import APIRouter, Depends

from app.analyst.service import analyst_snapshot, gold_snapshot
from app.api.deps import get_runtime
from app.runtime import Runtime

router = APIRouter(prefix="/api/analyst", tags=["analyst"])


@router.get("")
async def get_analyst(runtime: Runtime = Depends(get_runtime)):
    """Latest rules-based analysis per symbol plus the Analyst's PAPER trades."""
    return await analyst_snapshot(runtime)


@router.get("/training")
async def get_analyst_training():
    """Nightly weight training: on/off, weights in use, the latest runs."""
    from app.analyst.training import training_status

    return training_status()


@router.get("/gold")
async def get_analyst_gold():
    """The Analyst on gold (Delta XAUTUSD): latest analysis + its paper trades."""
    return await gold_snapshot()
