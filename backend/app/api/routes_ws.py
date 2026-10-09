from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import is_allowed_origin
from app.core.time_utils import to_iso_utc
from app.data.app_state import get_trading_mode
from app.execution.live_broker import LiveEquityUnavailable
from app.risk.exit_levels import position_exit_levels
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
    if not is_allowed_origin(origin, websocket.headers.get("host")):
        # Logged, not silent: a bare 403 during the handshake gives no clue
        # WHICH origin was refused, which made this genuinely hard to diagnose
        # when the dev server started advertising a LAN URL.
        logger.warning("Dashboard websocket refused: origin %r is not permitted", origin)
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
            try:
                equity = await runtime.order_manager.get_equity()
            except LiveEquityUnavailable:
                equity = None  # Delta unreachable right now: show "--", keep the feed alive
            prices = {s: runtime.cache.get_last_price(s) for s in runtime.registry.symbols}
            # Where each trade will be closed (no exchange stop): the account
            # limits as prices -- see app/risk/exit_levels.py.
            levels = {key: await position_exit_levels(runtime, p, equity) for key, p in positions.items()}

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
                            # What actually closes the trade: the blended
                            # signal fading through the exit line.
                            "signal": runtime.live_signals.get(p.symbol),
                            "exit_threshold": runtime.settings.signal_exit_deadband,
                            "notional_usd": p.notional_usd,
                            "margin_usd": p.margin_usd,
                            "liquidation_price": p.liquidation_price,
                            **levels[key],
                        }
                        for key, p in positions.items()
                    ],
                    "system_status": runtime.cache.system_status.get("status", "unknown"),
                }
            )
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.debug("Dashboard websocket disconnected")
    except Exception:
        logger.exception("Dashboard websocket error")
