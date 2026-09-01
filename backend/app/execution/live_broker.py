from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings, get_settings
from app.core.delta_client import DeltaApiError, DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import Fill, Order, Position
from app.execution.broker_base import Broker, OrderRequest, OrderResult, PositionView
from app.risk.position_sizing import stop_and_target
from app.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class LiveBroker(Broker):
    """Places real orders on Delta. Only ever reached when the trading mode is
    explicitly set to LIVE via the dashboard (the app always boots in PAPER
    mode). Deadman-switch heartbeat wiring lives in core/deadman_switch.py and
    is started alongside this broker by the trading loop, not here, so it can
    be unit tested independently.
    """

    mode = "live"

    def __init__(
        self,
        client: DeltaRestClient,
        registry: SymbolRegistry,
        cache: MarketDataCache,
        risk_manager: RiskManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.cache = cache
        self.risk_manager = risk_manager
        self.settings = settings or get_settings()
        # Delta's per-symbol realized_pnl accumulates for the life of a
        # position and resets when it fully closes -- tracked here so
        # sync_fills_and_positions can feed the daily-loss kill switch a
        # delta each poll instead of the raw (non-monotonic) total.
        self._last_synced_realized_pnl: dict[str, float] = {}

    async def _current_position_size(self, product_id: int) -> int:
        try:
            positions = await self.client.get_positions_margined(product_ids=str(product_id))
        except DeltaApiError:
            logger.exception("Failed to read current position before sizing a bracket order")
            return 0
        if not positions:
            return 0
        return int(positions[0].get("size", 0) or 0)

    def _round_to_tick(self, price: float, tick_size: float) -> str:
        if tick_size <= 0:
            return str(price)
        rounded = round(price / tick_size) * tick_size
        return f"{rounded:.10f}".rstrip("0").rstrip(".")

    async def _attach_bracket(self, product: ProductInfo, req: OrderRequest) -> None:
        """Attaches a native Delta bracket order (stop-loss + take-profit) to
        a position that was just opened fresh from flat. Delta allows only
        ONE bracket per open position (per its API docs), which is exactly
        why this is only ever called on a genuine flat->open transition, the
        same rule PaperBroker.open_or_extend uses for setting stop_loss_price/
        take_profit_price only once, at entry -- never on a later resize.

        The stop-loss leg carries `trail_amount`, Delta's own native trailing-
        stop mechanism (the stop price only ever ratchets favorably as the
        market moves in the position's favor, exactly like
        PaperBroker.check_stop_loss_take_profit's ratchet) -- so this trails
        on the exchange side, protecting the position even if this app or the
        laptop it runs on goes offline, which a locally-polled trail like
        paper's cannot do.
        """
        try:
            positions = await self.client.get_positions_margined(product_ids=str(product.product_id))
        except DeltaApiError:
            logger.exception("Bracket order NOT attached for %s -- could not read fill price", req.symbol)
            return
        if not positions or int(positions[0].get("size", 0) or 0) == 0:
            logger.error("Bracket order NOT attached for %s -- no open position found right after fill", req.symbol)
            return

        entry_price = float(positions[0].get("entry_price", 0) or 0)
        if entry_price <= 0:
            logger.error("Bracket order NOT attached for %s -- invalid entry_price from Delta", req.symbol)
            return

        direction = 1 if req.side == "buy" else -1
        stop_price, target_price = stop_and_target(
            direction, entry_price, req.atr, self.settings.atr_stop_multiple, self.settings.atr_take_profit_multiple
        )
        if stop_price is None or target_price is None:
            return
        trail_amount = self.settings.atr_stop_multiple * req.atr

        bracket_body = {
            "product_id": product.product_id,
            "product_symbol": product.symbol,
            "stop_loss_order": {
                "order_type": "market_order",
                "stop_price": self._round_to_tick(stop_price, product.tick_size),
                "trail_amount": self._round_to_tick(trail_amount, product.tick_size),
            },
            "take_profit_order": {
                "order_type": "market_order",
                "stop_price": self._round_to_tick(target_price, product.tick_size),
            },
            # Mark price, not last-traded, so a thin-book wick can't trigger
            # the stop on a price nobody could actually have been liquidated
            # at -- the same basis Delta's own liquidation engine uses.
            "bracket_stop_trigger_method": "mark_price",
        }
        try:
            await self.client.place_bracket_order(bracket_body)
            logger.info(
                "Bracket order attached for %s: stop=%s trail=%s target=%s",
                req.symbol, bracket_body["stop_loss_order"]["stop_price"], bracket_body["stop_loss_order"]["trail_amount"],
                bracket_body["take_profit_order"]["stop_price"],
            )
        except DeltaApiError:
            logger.exception(
                "Bracket order attach FAILED for %s -- position is open with NO stop-loss/take-profit protection",
                req.symbol,
            )

    async def _cleanup_bracket_if_flat(self, product: ProductInfo) -> None:
        """After a reduce-only order (kill switch, manual close, strategy
        disable), cancel any resting stop/bracket legs still attached to this
        product once its position is fully back to flat -- a bracket's SL/TP
        pair only auto-cancels each other when ONE of THEM triggers (per
        Delta's docs); a position closed some other way (this) would
        otherwise leave a stale stop order resting on the exchange, ready to
        misfire against whatever position opens on this symbol next.
        """
        size = await self._current_position_size(product.product_id)
        if size != 0:
            return
        try:
            await self.client.cancel_all_orders(
                {"product_id": product.product_id, "cancel_stop_orders": True}
            )
        except DeltaApiError:
            logger.exception("Failed to clean up resting stop orders for %s after flattening", product.symbol)

    async def place_order(self, req: OrderRequest) -> OrderResult:
        product = self.registry.get(req.symbol)
        body = {
            "product_id": product.product_id,
            "size": req.size,
            "side": req.side,
            "order_type": "market_order" if req.order_type == "market" else "limit_order",
            "reduce_only": req.reduce_only,
        }
        if req.order_type == "limit" and req.limit_price is not None:
            body["limit_price"] = str(req.limit_price)
        if req.client_order_id:
            body["client_order_id"] = req.client_order_id

        # Read BEFORE placing, so we know whether this fill is a genuine
        # flat->open transition (the only time a bracket gets attached) --
        # reading Delta's own authoritative position rather than our locally
        # polled copy (see scheduler._live_sync, up to 15s stale) to avoid a
        # race with a fill that just happened moments ago.
        pre_size = 0
        if not req.reduce_only and req.atr:
            pre_size = await self._current_position_size(product.product_id)

        async with SessionLocal() as session:
            order = Order(
                symbol=req.symbol,
                mode="live",
                side=req.side,
                order_type=req.order_type,
                size=req.size,
                unfilled_size=req.size,
                limit_price=req.limit_price,
                reduce_only=req.reduce_only,
                state="pending",
                strategy=req.strategy,
                client_order_id=req.client_order_id,
            )
            session.add(order)
            await session.flush()

            try:
                result = await self.client.place_order(body)
            except DeltaApiError as exc:
                order.state = "cancelled"
                await session.commit()
                logger.error("Live order rejected for %s: %s", req.symbol, exc)
                return OrderResult(order.id, None, "rejected", 0, None, 0.0, reason=str(exc))

            order.broker_order_id = str(result.get("id"))
            order.state = result.get("state", "open")
            order.unfilled_size = result.get("unfilled_size", req.size)
            await session.commit()
            await session.refresh(order)

            filled = req.size - order.unfilled_size
            avg_price = float(result["limit_price"]) if result.get("limit_price") else None
            order_result = OrderResult(order.id, order.broker_order_id, order.state, filled, avg_price, 0.0)

        if not req.reduce_only and req.atr and pre_size == 0 and order_result.filled_size > 0:
            await self._attach_bracket(product, req)
        elif req.reduce_only and order_result.filled_size > 0:
            await self._cleanup_bracket_if_flat(product)

        return order_result

    async def cancel_order(self, order_id: int) -> None:
        async with SessionLocal() as session:
            order = await session.get(Order, order_id)
            if not order or not order.broker_order_id:
                return
            product = self.registry.get(order.symbol)
            await self.client.cancel_order({"id": int(order.broker_order_id), "product_id": product.product_id})
            order.state = "cancelled"
            await session.commit()

    async def sync_fills_and_positions(self) -> None:
        """Pulls authoritative state from Delta and reconciles local rows. In a
        full deployment this would be driven by the private WebSocket (orders/
        positions/v2_user_trades channels); this REST-based sync is the
        dependable fallback used until that channel is wired in.
        """
        for product in self.registry.all():
            try:
                positions = await self.client.get_positions_margined(product_ids=str(product.product_id))
            except DeltaApiError:
                logger.exception("Failed to sync position for %s", product.symbol)
                continue
            async with SessionLocal() as session:
                from sqlalchemy import select

                result = await session.execute(
                    select(Position).where(Position.symbol == product.symbol, Position.mode == "live")
                )
                pos = result.scalar_one_or_none()
                if pos is None:
                    pos = Position(symbol=product.symbol, mode="live")
                    session.add(pos)
                remote = positions[0] if positions else None
                if remote:
                    pos.size = int(remote.get("size", 0))
                    pos.entry_price = float(remote.get("entry_price", 0) or 0)
                    pos.margin = float(remote.get("margin", 0) or 0)
                    pos.liquidation_price = float(remote.get("liquidation_price", 0) or 0)
                    new_realized = float(remote.get("realized_pnl", 0) or 0)
                    if self.risk_manager is not None:
                        last = self._last_synced_realized_pnl.get(product.symbol, new_realized)
                        # A drop means the position fully closed and Delta
                        # reset its counter for the new cycle -- treat the
                        # new value as the delta rather than going negative.
                        delta = new_realized - last if new_realized >= last else new_realized
                        if delta != 0:
                            await self.risk_manager.record_realized_pnl(delta)
                        self._last_synced_realized_pnl[product.symbol] = new_realized
                    pos.realized_pnl = new_realized
                await session.commit()

    async def get_positions(self) -> dict[str, PositionView]:
        from sqlalchemy import select

        async with SessionLocal() as session:
            result = await session.execute(select(Position).where(Position.mode == "live"))
            out = {}
            for p in result.scalars().all():
                if p.size == 0:
                    continue
                price = self.cache.get_last_price(p.symbol) or p.entry_price
                product = self.registry.get(p.symbol)
                unrealized = p.size * product.contract_value * (price - p.entry_price)
                notional_usd = abs(p.size) * product.contract_value * price
                margin_usd = notional_usd / p.leverage if p.leverage else notional_usd
                out[p.symbol] = PositionView(
                    p.symbol, p.size, p.entry_price, unrealized, p.realized_pnl,
                    strategy=p.strategy, entry_time=p.entry_time, leverage=p.leverage,
                    fees_paid=p.fees_paid, notional_usd=notional_usd, margin_usd=margin_usd,
                )
            return out

    async def get_equity(self) -> float:
        try:
            balances = await self.client.get_balances()
        except DeltaApiError:
            logger.exception("Failed to fetch live balances")
            return 0.0
        usd_like = [b for b in balances if b.get("asset_symbol") in ("USD", "USDT", "USDC")]
        balance = sum(float(b.get("balance", 0) or 0) for b in usd_like)
        positions = await self.get_positions()
        return balance + sum(p.unrealized_pnl for p in positions.values())
