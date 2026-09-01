from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_runtime
from app.core.time_utils import to_iso_utc
from app.data.app_state import (
    get_state,
    get_trading_mode,
    is_kill_switch_on,
    set_kill_switch,
    set_state,
    set_trading_mode,
)
from app.data.query import load_closed_trades, load_equity_curve, load_fills, load_orders
from app.risk.position_sizing import required_margin
from app.runtime import Runtime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading", tags=["trading"])


class ModeUpdate(BaseModel):
    mode: str  # "paper" | "live"


class KillSwitchUpdate(BaseModel):
    on: bool


@router.get("/mode")
async def get_mode(runtime: Runtime = Depends(get_runtime)):
    kill_switch = await is_kill_switch_on()
    reason = None
    if kill_switch:
        equity = await runtime.order_manager.get_equity()
        reason = await runtime.risk_manager.kill_switch_status(equity)
    return {"mode": await get_trading_mode(), "kill_switch": kill_switch, "kill_switch_reason": reason}


@router.get("/signals")
async def get_live_signals(runtime: Runtime = Depends(get_runtime)):
    """Every live-trading strategy's CURRENT computed signal (each
    individual member, enabled or not -- the blended ensemble itself
    doesn't trade its own position anymore, so it isn't included), per
    symbol -- a read-only peek using the exact same feature-building path
    the live trading loop uses, without submitting any order. Powers the
    dashboard's live-signal panel.
    """
    from app.trading_loop import get_live_signals as _get_live_signals

    return {symbol: await _get_live_signals(runtime, symbol) for symbol in runtime.registry.symbols}


@router.get("/ml_predictions")
async def get_live_ml_predictions_endpoint(runtime: Runtime = Depends(get_runtime)):
    """The standalone ML prediction model's current confidence-gated score,
    one entry per SYMBOL -- not one of the 3 live-trading strategies (see
    /signals above), so deliberately kept as its own endpoint rather than
    folded into the per-strategy one. Still not live-trading; see the
    Walk-Forward Ledger for why.
    """
    from app.trading_loop import get_live_ml_predictions

    return await get_live_ml_predictions(runtime)


@router.post("/mode")
async def set_mode(body: ModeUpdate, runtime: Runtime = Depends(get_runtime)):
    if body.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
    if body.mode == "live" and not runtime.settings.has_credentials:
        raise HTTPException(
            status_code=400,
            detail="Cannot enable live trading: DELTA_API_KEY/DELTA_API_SECRET are not configured in backend/.env",
        )
    if await is_kill_switch_on() and body.mode == "live":
        raise HTTPException(status_code=400, detail="Kill switch is active -- reset it before enabling live trading")
    await set_trading_mode(body.mode)
    return {"mode": body.mode}


@router.get("/kill_switch")
async def get_kill_switch():
    return {"on": await is_kill_switch_on()}


@router.post("/kill_switch")
async def update_kill_switch(body: KillSwitchUpdate, runtime: Runtime = Depends(get_runtime)):
    await set_kill_switch(body.on)
    if not body.on:
        # Turning the switch off is the deliberate ACKNOWLEDGMENT that
        # clears a max-drawdown halt (nothing else does -- see
        # scheduler._drawdown_supervisor / _daily_kill_switch_reset). The
        # high-water mark has to be re-based to current equity at the same
        # moment: leaving the old peak in place would have the drawdown
        # supervisor re-halt on its very next 15s tick, since equity is of
        # course still just as far below that peak as it was a second ago.
        if (await get_state("drawdown_halt_active")) == "on":
            equity = await runtime.order_manager.get_equity()
            await set_state("drawdown_halt_active", "off")
            await set_state("equity_high_water_mark", str(equity))
            logger.info("Max-drawdown halt acknowledged -- high-water mark re-based to %.2f", equity)
    if body.on:
        # Flatten whichever broker was ACTUALLY live when the switch was
        # hit, before flipping the mode to paper -- flatten_all() with no
        # override reads the *current* mode, so switching to paper first
        # would make it flatten paper positions while real live positions
        # stay open and unmanaged.
        mode_to_flatten = await get_trading_mode()
        await runtime.order_manager.flatten_all(mode_override=mode_to_flatten)
        await set_trading_mode("paper")
    return {"on": body.on}


@router.post("/positions/{symbol}/{strategy}/close")
async def close_position(symbol: str, strategy: str, runtime: Runtime = Depends(get_runtime)):
    result = await runtime.order_manager.close_position(symbol, strategy)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no open position for {symbol}/{strategy}")
    return {"symbol": symbol, "strategy": strategy, "realized_pnl": result.realized_pnl, "state": result.state}


@router.get("/positions")
async def get_positions(runtime: Runtime = Depends(get_runtime)):
    positions = await runtime.order_manager.get_positions()
    return [
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
    ]


@router.get("/equity")
async def get_equity(runtime: Runtime = Depends(get_runtime)):
    return {"equity": await runtime.order_manager.get_equity(), "mode": await get_trading_mode()}


@router.get("/equity_curve")
async def get_equity_curve(mode: str = "paper"):
    rows = await load_equity_curve(mode)
    return [{"timestamp": to_iso_utc(r.timestamp), "equity": r.equity} for r in rows]


@router.get("/orders")
async def get_orders(mode: str = "paper", symbol: str | None = None):
    rows = await load_orders(mode, symbol)
    return [
        {
            "id": o.id,
            "symbol": o.symbol,
            "side": o.side,
            "order_type": o.order_type,
            "size": o.size,
            "unfilled_size": o.unfilled_size,
            "limit_price": o.limit_price,
            "state": o.state,
            "strategy": o.strategy,
            "created_at": to_iso_utc(o.created_at),
        }
        for o in rows
    ]


@router.get("/closed_trades")
async def get_closed_trades(
    mode: str = "paper",
    symbol: str | None = None,
    date_from: dt.datetime | None = None,
    date_to: dt.datetime | None = None,
    runtime: Runtime = Depends(get_runtime),
):
    rows = await load_closed_trades(mode, symbol, date_from=date_from, date_to=date_to)
    # t.leverage is a genuinely selected tier (fixed at entry, 10-50x by
    # conviction -- see select_leverage), so notional/leverage is meaningful
    # margin, not a rounded display artifact. required_margin is just a
    # defensive clamp so this can never show more than funds actually on
    # hand, however margin was computed upstream.
    funds = (
        await runtime.paper_broker.get_cash_balance()
        if mode == "paper"
        else await runtime.live_broker.get_equity()
    )
    result = []
    for t in rows:
        try:
            contract_value = runtime.registry.get(t.symbol).contract_value
        except KeyError:
            contract_value = None
        notional_usd = t.size * contract_value * t.entry_price if contract_value else None
        margin_usd = (
            required_margin(notional_usd / t.leverage, funds)
            if notional_usd is not None and t.leverage
            else notional_usd
        )
        result.append(
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "strategy": t.strategy,
                "size": t.size,
                "leverage": t.leverage,
                "entry_price": t.entry_price,
                "entry_time": to_iso_utc(t.entry_time),
                "exit_price": t.exit_price,
                "exit_time": to_iso_utc(t.exit_time),
                "exit_reason": t.exit_reason,
                "realized_pnl": t.realized_pnl,
                "fees": t.fees,
                "notional_usd": notional_usd,
                "margin_usd": margin_usd,
            }
        )
    return result


@router.get("/fills")
async def get_fills(mode: str = "paper", symbol: str | None = None):
    rows = await load_fills(mode, symbol)
    return [
        {
            "id": f.id,
            "symbol": f.symbol,
            "side": f.side,
            "size": f.size,
            "price": f.price,
            "fee": f.fee,
            "role": f.role,
            "created_at": to_iso_utc(f.created_at),
        }
        for f in rows
    ]
