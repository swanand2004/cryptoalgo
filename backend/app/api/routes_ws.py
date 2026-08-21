from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import ALLOWED_ORIGINS
from app.core.time_utils import to_iso_utc
from app.data.app_state import get_trading_mode
from app.runtime import get_runtime_sync

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/dashboard")
async def dashboard_feed(websocket: WebSocket) -> None:
    """Pushes a lightweight snapshot (mode, equity, positions, latest prices)
    to the dashboard every second, so unrealized PnL tracks live price ticks
    instead of lagging behind on a slow REST poll. One loop per connection --
    fine for the small number of simultaneous browser tabs this app expects.
    """
    # CORSMiddleware only inspects the HTTP scope -- it does NOT protect
    # WebSocket connections at all, so without this check any page open in
    # the same browser (or any local process) could connect here and read
    # live equity/positions/strategy names every second, regardless of the
    # CORS allowlist configured for the rest of the API. This is read-only
    # data exposure (the socket never calls receive(), so it can't place
    # orders), but it's still a real leak worth closing off.
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        runtime = get_runtime_sync()
    except RuntimeError:
        await websocket.close(code=1013)
        return

    try:
        while True:
            mode = await get_trading_mode()
            positions = await runtime.order_manager.get_positions()
            equity = await runtime.order_manager.get_equity()
            prices = {s: runtime.cache.get_last_price(s) for s in runtime.registry.symbols}

            await websocket.send_json(
                {
                    "type": "snapshot",
                    "mode": mode,
                    "equity": equity,
                    "prices": prices,
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "size": p.size,
                            "entry_price": p.entry_price,
                            "unrealized_pnl": p.unrealized_pnl,
                            "realized_pnl": p.realized_pnl,
                            "strategy": p.strategy,
                            "entry_time": to_iso_utc(p.entry_time),
                            "leverage": p.leverage,
                            "fees_paid": p.fees_paid,
                            "stop_loss_price": p.stop_loss_price,
                            "take_profit_price": p.take_profit_price,
                            "notional_usd": p.notional_usd,
                            "margin_usd": p.margin_usd,
                        }
                        for p in positions.values()
                    ],
                    "system_status": runtime.cache.system_status.get("status", "unknown"),
                }
            )
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.debug("Dashboard websocket disconnected")
    except Exception:
        logger.exception("Dashboard websocket error")
