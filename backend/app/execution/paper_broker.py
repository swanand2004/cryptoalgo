from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.config import Settings, get_settings
from app.core.symbols import SymbolRegistry
from app.data.app_state import get_state
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import AppState, ClosedTrade, Fill, Order, Position
from app.execution.broker_base import Broker, OrderRequest, OrderResult, PositionView
from app.risk.position_sizing import required_margin, select_leverage, stop_and_target

logger = logging.getLogger(__name__)

_SLIPPAGE_BPS = 2.0  # simulated market-order slippage beyond best bid/ask


class PaperBroker(Broker):
    """Simulates fills against real live market data so paper-mode PnL is a
    realistic preview of live-mode behaviour: market orders fill at best
    bid/ask plus a small slippage allowance, limit orders fill immediately if
    they cross the book or rest until `check_resting_orders` sees price trade
    through them, and fees are simulated using the product's real maker/taker
    commission rates.
    """

    mode = "paper"

    def __init__(
        self,
        registry: SymbolRegistry,
        cache: MarketDataCache,
        settings: Optional[Settings] = None,
    ) -> None:
        self.registry = registry
        self.cache = cache
        self.settings = settings or get_settings()

    async def _get_balance(self) -> float:
        # Standalone read (own session) -- safe to use outside an open transaction.
        raw = await get_state("paper_balance")
        if raw is None:
            async with SessionLocal() as session:
                await self._set_balance_tx(session, self.settings.paper_starting_balance_usd)
                await session.commit()
            return self.settings.paper_starting_balance_usd
        return float(raw)

    async def _get_balance_tx(self, session) -> float:
        result = await session.execute(select(AppState).where(AppState.key == "paper_balance"))
        row = result.scalar_one_or_none()
        if row is None:
            await self._set_balance_tx(session, self.settings.paper_starting_balance_usd)
            return self.settings.paper_starting_balance_usd
        return float(row.value)

    async def _set_balance_tx(self, session, value: float) -> None:
        stmt = sqlite_insert(AppState).values(key="paper_balance", value=str(value))
        stmt = stmt.on_conflict_do_update(index_elements=["key"], set_={"value": str(value)})
        await session.execute(stmt)

    def _reference_price(self, symbol: str, side: str) -> Optional[float]:
        ob = self.cache.get_orderbook(symbol)
        if ob and ob.best_bid and ob.best_ask:
            base = ob.best_ask if side == "buy" else ob.best_bid
            slip = base * (_SLIPPAGE_BPS / 10_000)
            return base + slip if side == "buy" else base - slip
        return self.cache.get_last_price(symbol)

    async def place_order(self, req: OrderRequest, exit_reason: str = "signal") -> OrderResult:
        product = self.registry.get(req.symbol)
        price = req.limit_price if req.order_type == "limit" else self._reference_price(req.symbol, req.side)
        if price is None:
            return OrderResult(0, None, "rejected", 0, None, 0.0, reason="no market price available")

        ob = self.cache.get_orderbook(req.symbol)
        will_fill_now = req.order_type == "market" or self._crosses_book(req, ob)
        fee_rate = product.taker_commission_rate if will_fill_now else product.maker_commission_rate
        strategy_key = req.strategy or "manual"

        async with SessionLocal() as session:
            order = Order(
                symbol=req.symbol,
                mode="paper",
                side=req.side,
                order_type=req.order_type,
                size=req.size,
                unfilled_size=0 if will_fill_now else req.size,
                limit_price=req.limit_price,
                reduce_only=req.reduce_only,
                state="closed" if will_fill_now else "open",
                strategy=strategy_key,
                client_order_id=req.client_order_id,
            )
            session.add(order)
            await session.flush()

            fee = 0.0
            realized_pnl = 0.0
            if will_fill_now:
                leverage = select_leverage(req.quality, self.settings.min_leverage, self.settings.max_leverage)
                fee, realized_pnl = await self._apply_fill(
                    session, order, price, fee_rate, atr=req.atr, exit_reason=exit_reason, leverage=leverage
                )

            await session.commit()
            await session.refresh(order)
            return OrderResult(
                order_id=order.id,
                broker_order_id=None,
                state=order.state,
                # order.size may have been clamped down inside _apply_fill
                # (reduce-only orders can never overshoot the position) --
                # report what actually filled, not what was requested.
                filled_size=order.size if will_fill_now else 0,
                avg_fill_price=price if will_fill_now else None,
                fee=fee,
                realized_pnl=realized_pnl,
            )

    @staticmethod
    def _crosses_book(req: OrderRequest, ob) -> bool:
        if req.order_type != "limit" or req.limit_price is None or ob is None:
            return False
        if req.side == "buy" and ob.best_ask:
            return req.limit_price >= ob.best_ask
        if req.side == "sell" and ob.best_bid:
            return req.limit_price <= ob.best_bid
        return False

    def _stop_and_target(self, direction: int, entry_price: float, atr: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        """direction: +1 for long, -1 for short. Returns (stop_loss, take_profit)."""
        return stop_and_target(
            direction, entry_price, atr, self.settings.atr_stop_multiple, self.settings.atr_take_profit_multiple
        )

    async def _apply_fill(
        self,
        session,
        order: Order,
        price: float,
        fee_rate: float,
        atr: Optional[float] = None,
        exit_reason: str = "signal",
        leverage: Optional[float] = None,
    ) -> tuple[float, float]:
        """Returns (fee, realized_pnl) for this fill."""
        # No quality/leverage was supplied (e.g. a resting limit order fill,
        # which has no live signal at fill time) -- fall back to the floor
        # tier rather than guessing at conviction.
        if leverage is None:
            leverage = self.settings.min_leverage
        product = self.registry.get(order.symbol)
        now = dt.datetime.now(dt.timezone.utc)
        strategy_key = order.strategy or "manual"

        result = await session.execute(
            select(Position).where(
                Position.symbol == order.symbol, Position.mode == "paper", Position.strategy == strategy_key
            )
        )
        pos = result.scalar_one_or_none()
        if pos is None:
            pos = Position(symbol=order.symbol, mode="paper", strategy=strategy_key, size=0, entry_price=0.0)
            session.add(pos)
            await session.flush()

        if order.reduce_only:
            # A reduce-only order must never grow the position, hold it
            # steady, or overshoot through flat into a brand-new opposite-
            # side position -- it may only shrink toward zero. Clamped here
            # (against pos.size read fresh, within this same transaction)
            # rather than trusted from whatever size the caller computed,
            # which could be stale if it was sized before an intervening
            # fill on this same position.
            max_reduce_size = abs(pos.size)
            if order.size > max_reduce_size:
                order.size = max_reduce_size
            same_direction_as_order = (pos.size > 0 and order.side == "sell") or (pos.size < 0 and order.side == "buy")
            if pos.size == 0 or not same_direction_as_order:
                order.size = 0

        product_notional = order.size * product.contract_value * price
        fee = product_notional * fee_rate

        balance_before = await self._get_balance_tx(session)
        signed_size = order.size if order.side == "buy" else -order.size
        realized_pnl = 0.0

        def open_or_extend(add_size: int, is_new_position: bool) -> None:
            new_size = pos.size + add_size
            if new_size != 0:
                pos.entry_price = (pos.entry_price * abs(pos.size) + price * abs(add_size)) / abs(new_size)
            pos.size = new_size
            if is_new_position:
                pos.entry_time = now
                # Leverage is a genuine tier selected once at entry (see
                # select_leverage) -- like a real "cross Nx" choice, it's
                # fixed for the life of the position, not recomputed as the
                # position resizes.
                pos.leverage = leverage
                # Stop-loss/take-profit are set ONCE, on genuine entry from
                # flat, and deliberately NOT recomputed on later same-direction
                # resizes -- a protective stop that keeps sliding around every
                # time the position grows isn't protecting anything. From here
                # the stop-loss trails favorably via peak_price/stop_distance
                # (see check_stop_loss_take_profit); take-profit stays fixed.
                if atr:
                    direction = 1 if new_size > 0 else -1
                    pos.stop_loss_price, pos.take_profit_price = self._stop_and_target(direction, pos.entry_price, atr)
                    pos.peak_price = pos.entry_price
                    pos.stop_distance = self.settings.atr_stop_multiple * atr
            pos.fees_paid += fee

        if pos.size == 0 or (pos.size > 0) == (signed_size > 0):
            open_or_extend(signed_size, is_new_position=(pos.size == 0))
        else:
            closing_qty = min(abs(signed_size), abs(pos.size))
            direction = 1 if pos.size > 0 else -1
            realized_pnl = closing_qty * product.contract_value * (price - pos.entry_price) * direction
            pos.realized_pnl += realized_pnl
            closing_fee = fee * (closing_qty / order.size) if order.size else 0.0
            remaining_fee = fee - closing_fee

            session.add(
                ClosedTrade(
                    symbol=order.symbol,
                    mode="paper",
                    side="long" if direction > 0 else "short",
                    strategy=pos.strategy,
                    size=closing_qty,
                    leverage=pos.leverage,
                    entry_price=pos.entry_price,
                    entry_time=pos.entry_time or now,
                    exit_price=price,
                    exit_time=now,
                    exit_reason=exit_reason,
                    realized_pnl=realized_pnl,
                    fees=closing_fee + pos.fees_paid * (closing_qty / abs(pos.size)) if pos.size else closing_fee,
                )
            )

            remainder = pos.size + signed_size
            pos.size = remainder
            if remainder == 0:
                pos.entry_price = 0.0
                pos.entry_time = None
                pos.leverage = None
                pos.fees_paid = 0.0
                pos.stop_loss_price = None
                pos.take_profit_price = None
                pos.peak_price = None
                pos.stop_distance = None
            elif (remainder > 0) != (direction > 0):
                # Flipped straight through flat to the opposite side within
                # this one fill -- the leftover quantity is a brand new entry,
                # so it selects its own fresh leverage tier just like any
                # other genuine entry.
                pos.entry_price = price
                pos.entry_time = now
                pos.leverage = leverage
                pos.fees_paid = remaining_fee
                new_direction = 1 if remainder > 0 else -1
                pos.stop_loss_price, pos.take_profit_price = self._stop_and_target(new_direction, price, atr)
                pos.peak_price = price if atr else None
                pos.stop_distance = self.settings.atr_stop_multiple * atr if atr else None
            else:
                # Partial close, same direction remains -- entry price/time/
                # SL/TP/leverage all unchanged (leverage is fixed at entry,
                # like a real "cross Nx" selection, not recomputed as the
                # remaining position shrinks). Fees scale down proportionally
                # with the closed fraction.
                pos.fees_paid = pos.fees_paid * (abs(remainder) / (abs(remainder) + closing_qty))

        session.add(
            Fill(
                order_id=order.id,
                symbol=order.symbol,
                mode="paper",
                side=order.side,
                size=order.size,
                price=price,
                fee=fee,
                role="taker",
            )
        )

        balance = balance_before - fee + realized_pnl
        await self._set_balance_tx(session, balance)

        return fee, realized_pnl

    async def cancel_order(self, order_id: int) -> None:
        async with SessionLocal() as session:
            order = await session.get(Order, order_id)
            if order and order.state == "open":
                order.state = "cancelled"
                await session.commit()

    async def check_resting_orders(self) -> None:
        """Scans open paper limit orders and fills any that the market has
        traded through since they were placed. Call this periodically.
        """
        async with SessionLocal() as session:
            result = await session.execute(
                select(Order).where(Order.mode == "paper", Order.state == "open")
            )
            open_orders = result.scalars().all()
            for order in open_orders:
                ob = self.cache.get_orderbook(order.symbol)
                if ob is None or order.limit_price is None:
                    continue
                crosses = (
                    order.side == "buy" and ob.best_ask and order.limit_price >= ob.best_ask
                ) or (order.side == "sell" and ob.best_bid and order.limit_price <= ob.best_bid)
                if not crosses:
                    continue
                product = self.registry.get(order.symbol)
                order.state = "closed"
                order.unfilled_size = 0
                await self._apply_fill(session, order, order.limit_price, product.maker_commission_rate)
            await session.commit()

    async def get_cash_balance(self) -> float:
        return await self._get_balance()

    async def get_positions(self) -> dict[str, PositionView]:
        balance = await self._get_balance()
        async with SessionLocal() as session:
            result = await session.execute(select(Position).where(Position.mode == "paper"))
            positions = result.scalars().all()
            out = {}
            for p in positions:
                if p.size == 0:
                    continue
                price = self.cache.get_last_price(p.symbol) or p.entry_price
                product = self.registry.get(p.symbol)
                unrealized = p.size * product.contract_value * (price - p.entry_price)
                notional_usd = abs(p.size) * product.contract_value * price
                # Margin is notional / this position's genuinely selected
                # leverage tier (fixed at entry -- see select_leverage), not
                # a rounded display value -- so it's meaningful on its own.
                # required_margin is just a defensive clamp against ever
                # displaying more margin than funds actually on hand.
                margin_usd = required_margin(notional_usd / p.leverage, balance) if p.leverage else notional_usd
                out[f"{p.symbol}:{p.strategy}"] = PositionView(
                    symbol=p.symbol,
                    size=p.size,
                    entry_price=p.entry_price,
                    unrealized_pnl=unrealized,
                    realized_pnl=p.realized_pnl,
                    strategy=p.strategy,
                    entry_time=p.entry_time,
                    leverage=p.leverage,
                    fees_paid=p.fees_paid,
                    stop_loss_price=p.stop_loss_price,
                    take_profit_price=p.take_profit_price,
                    notional_usd=notional_usd,
                    margin_usd=margin_usd,
                )
            return out

    async def get_equity(self) -> float:
        balance = await self._get_balance()
        positions = await self.get_positions()
        unrealized = sum(p.unrealized_pnl for p in positions.values())
        return balance + unrealized

    async def check_stop_loss_take_profit(self) -> None:
        """Monitors every open paper position against its stored stop-loss/
        take-profit levels, trails the stop favorably as price moves in the
        position's favor, and closes (market, reduce-only) any that have been
        breached. This is the actual protective mechanism -- without it, a
        position only ever exits when the strategy's own signal changes,
        which could be many minutes after an adverse move given a 60s tick.
        """
        to_close: list[tuple[str, str, int, float, float | None, float | None, str]] = []

        async with SessionLocal() as session:
            result = await session.execute(
                select(Position).where(Position.mode == "paper", Position.size != 0)
            )
            open_positions = result.scalars().all()

            for pos in open_positions:
                if pos.stop_loss_price is None and pos.take_profit_price is None:
                    continue
                price = self.cache.get_last_price(pos.symbol)
                if price is None:
                    continue

                is_long = pos.size > 0

                # Trailing stop: as the best price seen since entry improves,
                # drag the stop behind it. It only ever tightens (moves
                # favorably) -- never loosens. Take-profit is a fixed target
                # and is never touched here.
                #
                # Two-stage distance, not a constant one: trailing at a flat
                # stop_distance (2 ATR) behind the peak means a position that
                # spikes to within a whisker of take-profit (3 ATR) and then
                # reverses can give back almost the entire move -- the stop
                # only reaches breakeven once the peak is 2 ATR above entry,
                # i.e. two-thirds of the way to target, and even then sits
                # barely above it. Once the peak has moved
                # atr_breakeven_trigger_multiple ATRs in the position's
                # favor, switch to a tighter atr_tightened_trail_multiple
                # distance AND floor the stop at breakeven, so a fast move
                # that almost hits target and reverses locks in a real
                # portion of that move instead of round-tripping to scratch.
                #
                # Recomputed on EVERY tick, not just when the peak itself
                # advances: the peak might already be past the trigger while
                # the stop it produced is still sitting at the wider,
                # untriggered distance from an earlier tick -- and if the
                # trail-distance settings change while a position is open,
                # a peak that hasn't moved since would otherwise freeze that
                # position's stop under the OLD settings forever, since nothing
                # would ever re-enter this block to apply the new ones. Always
                # recomputing and taking the more-favorable of (existing,
                # candidate) still preserves the ratchet -- it can only ever
                # tighten further, exactly like a genuine new peak would.
                if pos.stop_distance and pos.peak_price is not None:
                    new_peak = max(pos.peak_price, price) if is_long else min(pos.peak_price, price)
                    pos.peak_price = new_peak

                    atr_at_entry = pos.stop_distance / self.settings.atr_stop_multiple
                    favorable_move = (new_peak - pos.entry_price) if is_long else (pos.entry_price - new_peak)
                    triggered = favorable_move >= self.settings.atr_breakeven_trigger_multiple * atr_at_entry

                    trail_distance = (
                        self.settings.atr_tightened_trail_multiple * atr_at_entry if triggered else pos.stop_distance
                    )
                    candidate_stop = new_peak - trail_distance if is_long else new_peak + trail_distance
                    if triggered:
                        candidate_stop = (
                            max(candidate_stop, pos.entry_price) if is_long else min(candidate_stop, pos.entry_price)
                        )

                    if pos.stop_loss_price is None or (
                        (is_long and candidate_stop > pos.stop_loss_price)
                        or (not is_long and candidate_stop < pos.stop_loss_price)
                    ):
                        pos.stop_loss_price = candidate_stop

                hit_stop = pos.stop_loss_price is not None and (
                    (is_long and price <= pos.stop_loss_price) or (not is_long and price >= pos.stop_loss_price)
                )
                hit_target = pos.take_profit_price is not None and (
                    (is_long and price >= pos.take_profit_price) or (not is_long and price <= pos.take_profit_price)
                )
                if hit_stop or hit_target:
                    reason = "stop_loss" if hit_stop else "take_profit"
                    to_close.append(
                        (pos.symbol, pos.strategy, pos.size, price, pos.stop_loss_price, pos.take_profit_price)
                        + (reason,)  # type: ignore[arg-type]
                    )

            await session.commit()

        for symbol, strategy, size, price, stop, target, reason in to_close:
            side = "sell" if size > 0 else "buy"
            logger.info(
                "%s triggered for %s/%s at %.4f (stop=%s target=%s)",
                reason, symbol, strategy, price, stop, target,
            )
            await self.place_order(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    size=abs(size),
                    order_type="market",
                    reduce_only=True,
                    strategy=strategy,
                ),
                exit_reason=reason,
            )
