from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import and_, or_, select

from app.data.db import SessionLocal
from app.data.models import Order
from app.execution.broker_base import OrderRequest, OrderResult

logger = logging.getLogger(__name__)

# Maker (passive limit) execution for the trading loop's rebalances.
#
# Delta charges 0.05% to take liquidity and 0.02% to provide it; the loop used
# to send every rebalance as a market order. trend_ladder holds positions for
# days, so waiting minutes for a passive fill costs it nearly nothing, while
# the measured fee drag is 0.027%/day of a 0.181%/day book (2026-10-02 gap
# analysis) -- most of it recoverable.
#
# Per rebalance: rest ONE post-only order at the touch (best bid to buy, best
# ask to sell), re-peg it each tick if the touch moves, and if it has not
# filled within `maker_timeout_seconds` of the FIRST attempt, cancel and take
# the rest at market. The clock deliberately survives re-pegs -- otherwise a
# market drifting away re-arms the timer every tick and the trade never
# happens, which for a trend follower is the expensive failure, not the fee.
#
# Exits forced by risk (kill switch, manual close, flatten) never come here;
# they stay market orders, and OrderManager cancels resting orders first.

# key "SYMBOL:strategy" -> (side, first attempt time). In memory: a restart
# simply restarts the clock, which at worst delays one fill by one timeout.
_clock: dict[str, tuple[str, dt.datetime]] = {}


def maker_enabled(settings, mode: str) -> bool:
    setting = getattr(settings, "maker_execution", "off")
    return setting == "all" or (setting == "paper" and mode == "paper")


def resting_clause():
    """Orders that may be working on the book: "open", or Delta's "pending"
    (accepted, not yet on the book) once Delta has acknowledged it."""
    return or_(Order.state == "open", and_(Order.state == "pending", Order.broker_order_id.is_not(None)))


async def open_orders(mode: str, symbol: str, strategy: str) -> list[Order]:
    async with SessionLocal() as session:
        result = await session.execute(
            select(Order).where(
                Order.mode == mode, Order.symbol == symbol, Order.strategy == strategy, resting_clause()
            )
        )
        return list(result.scalars().all())


async def _cancel(runtime, orders: list[Order]) -> bool:
    """Cancels `orders`. Returns True when one of them traded (or may have)
    since this tick read its position -- the cancel found it filled further
    than our row knew, or it had vanished with an unknown ending. The tick
    must then NOT submit on that stale position (it would double the trade);
    the next tick works from the refreshed one. An order that vanished
    unfilled (Delta's deadman switch cancels resting orders when heartbeats
    lapse) is simply unfilled: the caller re-pegs or falls back to market."""
    unsure = False
    for order in orders:
        if await runtime.order_manager.cancel(order.id) is False:
            unsure = True
    if orders:
        async with SessionLocal() as session:
            for order in orders:
                now_row = await session.get(Order, order.id)
                if now_row is not None and (now_row.unfilled_size or 0) < (order.unfilled_size or 0):
                    unsure = True
    return unsure


async def cancel_resting(runtime, mode: str, symbol: str, strategy: str) -> int:
    """Cancels every resting order for this position and stops its clock."""
    orders = await open_orders(mode, symbol, strategy)
    for order in orders:
        await runtime.order_manager.cancel(order.id)
    _clock.pop(f"{symbol}:{strategy}", None)
    return len(orders)


async def execute_as_maker(
    runtime, mode: str, req: OrderRequest, now: dt.datetime | None = None
) -> OrderResult | None:
    """Works `req` (a would-be market order) passively. Returns the result of
    whatever was submitted this tick, or None if a resting order was left
    working untouched."""
    now = now or dt.datetime.now(dt.timezone.utc)
    key = f"{req.symbol}:{req.strategy}"
    timeout = dt.timedelta(seconds=runtime.settings.maker_timeout_seconds)

    side_started = _clock.get(key)
    if side_started is None or side_started[0] != req.side:
        _clock[key] = side_started = (req.side, now)
    started = side_started[1]

    book = runtime.cache.get_orderbook(req.symbol)
    touch = None
    if book is not None:
        touch = book.best_bid if req.side == "buy" else book.best_ask

    resting = await open_orders(mode, req.symbol, req.strategy)
    if now - started >= timeout or not touch:
        reason = "timed out" if touch else "no order book"
        if await _cancel(runtime, resting):
            # Clock kept: the next tick takes whatever is still missing.
            logger.info("Maker %s for %s -- resting order traded meanwhile; re-reading position first", reason, key)
            return None
        _clock.pop(key, None)
        logger.info("Maker %s for %s -- taking %d at market", reason, key, req.size)
        return await runtime.order_manager.submit(req)

    for order in resting:
        # Smaller than asked is fine: risk limits may have capped it at
        # placement, and re-placing it every tick would only churn. Whatever
        # is still missing after it fills is worked on the next tick.
        # What is still working counts (unfilled_size, kept current from
        # Delta), not the original size: a part-filled order at the touch
        # whose remainder is all that is missing is left alone.
        remaining = order.unfilled_size if order.unfilled_size is not None else order.size
        keep = order.side == req.side and remaining <= req.size and order.limit_price is not None and (
            abs(order.limit_price / touch - 1) < 1e-9
        )
        if keep:
            return None  # still at the touch: let it work
    if await _cancel(runtime, resting):
        logger.info("Maker re-peg for %s -- resting order traded meanwhile; re-reading position first", key)
        return None

    result = await runtime.order_manager.submit(
        OrderRequest(
            symbol=req.symbol, side=req.side, size=req.size, order_type="limit", limit_price=touch,
            reduce_only=req.reduce_only, strategy=req.strategy, atr=req.atr, quality=req.quality,
        ),
        post_only=True,
    )
    logger.info(
        "Maker %s %d %s @ %.4f (%s, working %ds of %ds)",
        req.side, req.size, key, touch, result.state, int((now - started).total_seconds()), int(timeout.total_seconds()),
    )
    # A post-only order Delta refuses because it would have crossed (the
    # touch moved under us) comes back "rejected"/"cancelled" with nothing
    # filled. The clock keeps running: the next tick re-pegs at the new
    # touch, or takes at market once the timeout is reached.
    if result.state == "closed":
        _clock.pop(key, None)
    return result
