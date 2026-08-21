from __future__ import annotations

import logging
from typing import Optional

from app.core.delta_client import DeltaApiError, DeltaRestClient
from app.core.symbols import SymbolRegistry
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import Fill, Order, Position
from app.execution.broker_base import Broker, OrderRequest, OrderResult, PositionView
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
    ) -> None:
        self.client = client
        self.registry = registry
        self.cache = cache
        self.risk_manager = risk_manager
        # Delta's per-symbol realized_pnl accumulates for the life of a
        # position and resets when it fully closes -- tracked here so
        # sync_fills_and_positions can feed the daily-loss kill switch a
        # delta each poll instead of the raw (non-monotonic) total.
        self._last_synced_realized_pnl: dict[str, float] = {}

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
            return OrderResult(order.id, order.broker_order_id, order.state, filled, avg_price, 0.0)

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
