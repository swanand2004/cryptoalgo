from __future__ import annotations

from fastapi import APIRouter, Query

from app.config import get_settings
from app.ml.book_filter import current_multipliers, get_history, mode_of

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.get("/book-filter")
async def get_book_filter(days: int = Query(60, ge=1, le=400)):
    """ML book filter v1: today's per-major multipliers plus history. Read-only.
    In "shadow" mode these numbers are logged only and never change sizing."""
    hist = await get_history()
    return {
        "mode": mode_of(get_settings()),
        "latest": hist[-1] if hist else None,
        "in_force": await current_multipliers(),
        "history": hist[-days:],
        "note": "multiplier 0.5 / 1.0 / 1.5 on the trend book's weight per major; "
                "applies 01:00-01:00 UTC; shadow = logged only, no effect on sizing",
    }
