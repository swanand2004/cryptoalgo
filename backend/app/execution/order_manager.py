from __future__ import annotations

import logging

from app.core.symbols import SymbolRegistry
from app.data.app_state import get_trading_mode
from app.data.market_cache import MarketDataCache
from app.execution.broker_base import Broker, OrderRequest, OrderResult, PositionView
from app.execution.live_broker import LiveBroker
from app.execution.paper_broker import PaperBroker
from app.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderManager:
    """Single entry point for order flow. Reads the *current* trading mode on
    every call (not cached at construction time) and routes to the matching
    broker, so flipping the dashboard toggle takes effect on the very next
    order -- this is what makes paper and live behave identically except for
    where money actually moves.

    All risk checks happen here, before either broker ever sees the order, so
    neither broker implementation needs to duplicate limit logic.
    """

    def __init__(
        self,
        paper_broker: PaperBroker,
        live_broker: LiveBroker,
        risk_manager: RiskManager,
        registry: SymbolRegistry,
        cache: MarketDataCache,
    ) -> None:
        self.paper_broker = paper_broker
        self.live_broker = live_broker
        self.risk_manager = risk_manager
        self.registry = registry
        self.cache = cache

    async def current_broker(self) -> Broker:
        mode = await get_trading_mode()
        return self.live_broker if mode == "live" else self.paper_broker

    async def get_positions(self) -> dict[str, PositionView]:
        return await (await self.current_broker()).get_positions()

    async def get_equity(self) -> float:
        return await (await self.current_broker()).get_equity()

    async def submit(self, req: OrderRequest, *, post_only: bool = False) -> OrderResult:
        broker = await self.current_broker()
        equity = await broker.get_equity()

        allowed, reason = await self.risk_manager.trading_allowed(self.cache, equity)
        if not allowed and not req.reduce_only:
            logger.warning("Order blocked by risk manager: %s (%s)", reason, req)
            return OrderResult(0, None, "rejected", 0, None, 0.0, reason=reason)

        if not req.reduce_only and not req.min_lot:
            positions = await broker.get_positions()

            # Hard cap on concurrent open positions, account-wide. Only a
            # genuinely NEW entry from flat counts against it -- extending an
            # already-open position, or flipping its direction, keeps the
            # same key in `positions` (nonzero size already), so neither one
            # is blocked here even once the cap is reached.
            key = f"{req.symbol}:{req.strategy or 'manual'}"
            opening_from_flat = key not in positions or positions[key].size == 0
            if opening_from_flat and len(positions) >= self.risk_manager.settings.max_concurrent_positions:
                reason = f"max concurrent positions ({self.risk_manager.settings.max_concurrent_positions}) reached"
                logger.info("Order blocked: %s (rejecting new %s/%s entry)", reason, req.symbol, req.strategy)
                return OrderResult(0, None, "rejected", 0, None, 0.0, reason=reason)

            price = self.cache.get_last_price(req.symbol)
            product = self.registry.get(req.symbol)
            if price:
                # Aggregate notional across EVERY position regardless of which
                # strategy holds it -- each strategy trades its own independent
                # position now, but portfolio risk limits still apply to the
                # account as a whole, not per-strategy. Each position already
                # carries its own correctly-priced notional_usd (computed
                # against ITS OWN symbol's price) -- do not re-derive it here
                # using this order's symbol's price, which would silently
                # misprice every other symbol's position (e.g. multiplying a
                # SOL position's size by BTC's price when sizing a BTC order).
                current_notional = sum(p.notional_usd for p in positions.values())
                current_margin = sum(p.margin_usd for p in positions.values())
                signed = req.size if req.side == "buy" else -req.size
                capped = self.risk_manager.cap_position_size(
                    signed,
                    req.symbol,
                    price,
                    product.contract_value,
                    equity,
                    current_notional,
                    current_margin,
                    quality=req.quality,
                    leverage=req.leverage,
                )
                if abs(capped) < abs(signed):
                    logger.info("Order size capped by risk limits: %d -> %d for %s", signed, capped, req.symbol)
                req.size = abs(capped)

        if req.size == 0:
            return OrderResult(0, None, "rejected", 0, None, 0.0, reason="size reduced to 0 by risk limits")

        if post_only and broker.mode == "live":
            # Maker limit orders only (maker.py). Paper never crosses at the
            # touch it is given, so it needs no flag.
            result = await broker.place_order(req, post_only=True)
        else:
            result = await broker.place_order(req)
        if result.realized_pnl != 0.0:
            # Feeds the daily-loss kill switch -- without this it never sees
            # a single real loss and can never trip, regardless of how much
            # the account actually draws down.
            await self.risk_manager.record_realized_pnl(result.realized_pnl)
        return result

    async def cancel(self, order_id: int):
        """Live returns False when the order vanished with an unknown final
        state (it may have filled); paper returns None."""
        return await (await self.current_broker()).cancel_order(order_id)

    async def cancel_open_orders(self, broker, symbol: str | None = None, strategy: str | None = None) -> list[int]:
        """Cancels resting orders (all, or one symbol/strategy's). Forced exits
        call this FIRST: a maker order left resting would otherwise fill after
        the flatten and quietly reopen the position the kill switch just closed.

        A cancel that fails is logged and SKIPPED, never raised: the forced
        exit that follows must happen regardless. Returns the ids that failed."""
        from sqlalchemy import select

        from app.data.db import SessionLocal
        from app.data.models import Order
        from app.execution.maker import resting_clause

        async with SessionLocal() as session:
            stmt = select(Order.id).where(Order.mode == broker.mode, resting_clause())
            if symbol is not None:
                stmt = stmt.where(Order.symbol == symbol)
            if strategy is not None:
                stmt = stmt.where(Order.strategy == strategy)
            ids = list((await session.execute(stmt)).scalars().all())
        failed = []
        for order_id in ids:
            try:
                await broker.cancel_order(order_id)
            except Exception:
                logger.exception("Cancel of resting order %s failed -- continuing to close positions", order_id)
                failed.append(order_id)
        return failed

    async def process_resting_fills(self) -> None:
        """Fills paper limit orders the market has reached, and feeds their
        realized PnL to the daily-loss kill switch -- which only ever heard
        about market orders before, so a loss realized by a resting fill was
        invisible to it."""
        realized = await self.paper_broker.check_resting_orders()
        if realized:
            await self.risk_manager.record_realized_pnl(realized)

    async def flatten_all(self, mode_override: str | None = None, reason: str = "kill_switch") -> None:
        """Kill-switch action: reduce every open position to zero at market.
        `reason` is recorded as the closed trades' exit reason."""
        broker = self.live_broker if mode_override == "live" else (
            self.paper_broker if mode_override == "paper" else await self.current_broker()
        )
        # Never raises: a failed cancel must not stop the flatten below.
        failed_cancels = await self.cancel_open_orders(broker)
        positions = await broker.get_positions()
        failed_closes = []
        for pos in positions.values():
            if pos.size == 0:
                continue
            side = "sell" if pos.size > 0 else "buy"
            try:
                result = await broker.place_order(
                    OrderRequest(
                        symbol=pos.symbol,
                        side=side,
                        size=abs(pos.size),
                        order_type="market",
                        reduce_only=True,
                        strategy=pos.strategy,
                    ),
                    exit_reason=reason,
                )
            except Exception:
                # One symbol's failure must not leave the others open.
                logger.exception("Flatten of %s failed -- continuing with the rest", pos.symbol)
                failed_closes.append(pos.symbol)
                continue
            if result.state == "rejected":  # the live broker returns (not raises) an exchange rejection
                logger.error("Flatten of %s rejected by the exchange: %s", pos.symbol, result.reason)
                failed_closes.append(pos.symbol)
                continue
            if result.realized_pnl != 0.0:
                await self.risk_manager.record_realized_pnl(result.realized_pnl)
        for order_id in failed_cancels:
            # Positions are closed; one more try at whatever is still resting.
            try:
                await broker.cancel_order(order_id)
            except Exception:
                logger.exception("Resting order %s could NOT be cancelled after the flatten", order_id)
        if failed_closes:
            raise RuntimeError(f"flatten_all could not close: {', '.join(failed_closes)}")

    async def close_position(self, symbol: str, strategy: str) -> OrderResult | None:
        """Closes exactly ONE (symbol, strategy) position at market,
        reduce-only. Needed because disabling a strategy orphans whatever
        position it was holding -- the trading loop skips a disabled member
        entirely, so that position would otherwise only ever exit by
        happening to hit its stop-loss or take-profit, with no way to
        deliberately close it out. Returns None if there's nothing open to close.
        """
        broker = await self.current_broker()
        await self.cancel_open_orders(broker, symbol, strategy)
        positions = await broker.get_positions()
        pos = positions.get(f"{symbol}:{strategy}")
        if pos is None or pos.size == 0:
            return None
        side = "sell" if pos.size > 0 else "buy"
        result = await broker.place_order(
            OrderRequest(
                symbol=symbol, side=side, size=abs(pos.size), order_type="market", reduce_only=True, strategy=strategy
            ),
            exit_reason="manual",
        )
        if result.realized_pnl != 0.0:
            await self.risk_manager.record_realized_pnl(result.realized_pnl)
        return result
