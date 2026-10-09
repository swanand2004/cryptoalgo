from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime as dt
import logging
import time
from typing import Optional

import httpx

from app.config import Settings, get_settings
from app.core.delta_client import DeltaApiError, DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import ClosedTrade, Fill, Order, Position
from app.execution.broker_base import Broker, OrderRequest, OrderResult, PositionView
from app.notify import telegram  # owner alerts: fire-and-forget, off unless configured
from app.risk.position_sizing import select_leverage, stop_and_target
from app.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Delta holds ONE net position per product, and the trading loop trades ONE
# blended position per symbol under this name (trading_loop.CONSOLIDATED_STRATEGY;
# a test pins the two together). Live rows used to carry no strategy at all:
# the NOT NULL column made every sync fail, and get_positions keyed rows by bare
# symbol while the loop looks up "SYMBOL:consolidated" -- so live mode always
# saw itself flat and would have bought again on every tick.
LIVE_POSITION_STRATEGY = "consolidated"
LAST_GOOD_BALANCE_MAX_AGE_S = 300
# Delta's wallet balance is shared for this long between callers. The 1 s
# dashboard feed, three 15 s supervisors, the trading tick and the Analyst
# each read it: ~0.55 calls/s, ~44k/day on 2026-10-06, many of them
# simultaneous (each one a fresh connection that could time out). Any order
# placed through this broker, or any fill the sync sees, invalidates it.
BALANCE_CACHE_TTL_S = 5.0
# Set by fresh_balance(): the risk supervisors read the balance and THEN the
# deposit list, and the order matters -- a cached pre-deposit balance paired
# with a fresh list that already holds the deposit would raise the day's
# anchor without the money, read as a loss of the deposit and trip the
# daily-loss halt. So their reads skip the cache (as every read did before).
_FRESH_BALANCE: contextvars.ContextVar[bool] = contextvars.ContextVar("fresh_balance", default=False)
# Delta's error codes (docs.delta.exchange "Errors"; ccxt's delta.py maps the
# same strings). Cancelling an order that is no longer on the book -- it
# filled, was already cancelled, or the deadman switch removed it -- returns
# open_order_not_found: for a cancel that is success, not failure.
ORDER_GONE_CODES = frozenset({"open_order_not_found"})
# A post_only order that would have traded immediately is refused, not filled.
POST_ONLY_CROSS_CODES = frozenset({"immediate_execution_post_only"})
_RESTING_STATES = ("open", "pending")


def delta_error_code(exc: DeltaApiError) -> str | None:
    """The `error.code` of a Delta error response, if it has one."""
    payload = getattr(exc, "payload", None)
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return error.get("code")
    return error if isinstance(error, str) else None


def _apply_remote_order(order: Order, remote: dict) -> None:
    """Copies Delta's state and unfilled_size onto our row (fields it omits stay)."""
    if remote.get("state"):
        order.state = str(remote["state"])
    if remote.get("unfilled_size") is not None:
        order.unfilled_size = int(remote["unfilled_size"])


@contextlib.contextmanager
def fresh_balance():
    """Within this block, live equity reads Delta's balance directly."""
    token = _FRESH_BALANCE.set(True)
    try:
        yield
    finally:
        _FRESH_BALANCE.reset(token)


class LiveEquityUnavailable(RuntimeError):
    """Delta's balance could not be read. Callers must skip, not assume zero."""


class LiveBroker(Broker):
    """Places real orders on Delta. Only ever reached when the trading mode is
    LIVE (chosen on the dashboard; it persists across restarts, and risk
    limits halt it with the kill switch rather than switching to paper). Deadman-switch heartbeat wiring lives in core/deadman_switch.py and
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
        self._leverage_set: dict[int, float] = {}
        self._last_balance: float | None = None
        self._last_balance_at: float = 0.0
        self._balance_cached_at: float | None = None  # None = must refetch
        self._balance_gen = 0  # bumped by invalidate_balance_cache()
        self._balance_inflight: asyncio.Future | None = None
        # Why the NEXT closing fill happens (kill switch, manual, signal) --
        # set by place_order, consumed when the sync records the closed trade.
        self._exit_reason: dict[str, str] = {}
        self._flows_cache: tuple[float, list] | None = None

    async def external_flows(self, max_age_s: float = 0.0) -> list:
        """Recent deposits (+) and withdrawals (-) on the USD wallet, from
        Delta's wallet transactions (types containing "deposit"/"withdraw";
        commissions, funding and trade cashflows are P&L and excluded).
        Not cached by default (max_age_s=0): each supervisor reads its balance
        and THEN this list, and a list cached up to 10 s earlier by another
        supervisor could miss a deposit that balance already holds -- the
        deposit then read as profit (the 2026-10-05 profit-halt incident class)."""
        from app.risk.risk_manager import ExternalFlow

        now = time.monotonic()
        if self._flows_cache is not None and now - self._flows_cache[0] < max_age_s:
            return self._flows_cache[1]
        rows = await self.client.request("GET", "/v2/wallet/transactions", params={"page_size": 50}, auth=True)
        flows = []
        for r in rows or []:
            kind = str(r.get("transaction_type") or "").lower()
            if "deposit" not in kind and "withdraw" not in kind:
                continue
            if (r.get("asset_symbol") or "USD") != "USD":
                continue
            amount = float(r.get("amount") or 0.0)
            if "withdraw" in kind and amount > 0:
                amount = -amount
            created = dt.datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
            uid = str(r.get("uuid") or (r.get("meta_data") or {}).get("transaction_id") or r["created_at"])
            flows.append(ExternalFlow(uid=uid, time=created, amount=amount))
        self._flows_cache = (now, flows)
        return flows

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

    # ---- Exchange-side stops for live Analyst trades (app/analyst/execution.py).
    # Primitive and never raising: each returns None/False on any Delta or
    # network error, and the caller keeps the old stop resting until a new one
    # is confirmed. These stops are NOT rows in the Order table, so the maker
    # and cancel_open_orders logic never touch them; they are reduce-only, so
    # they can only ever shrink a position, never open or flip one.
    _STOP_ERRORS = (DeltaApiError, httpx.HTTPError, asyncio.TimeoutError)

    async def place_stop_order(self, symbol: str, side: str, size: int, stop_price: float,
                               client_order_id: str) -> str | None:
        """Reduce-only stop-market order (Delta: stop_order_type
        "stop_loss_order", triggered on the mark price like the bracket).
        Returns Delta's order id, or None if it was not placed."""
        product = self.registry.get(symbol)
        body = {
            "product_id": product.product_id,
            "size": int(size),
            "side": side,
            "order_type": "market_order",
            "stop_order_type": "stop_loss_order",
            "stop_price": self._round_to_tick(stop_price, product.tick_size),
            "stop_trigger_method": "mark_price",
            "reduce_only": True,
            "client_order_id": client_order_id,
        }
        try:
            result = await self.client.place_order(body)
        except self._STOP_ERRORS as exc:
            logger.error("Exchange stop for %s at %s NOT placed: %r", symbol, body["stop_price"], exc)
            return None
        oid = result.get("id") if isinstance(result, dict) else None
        if oid is None:
            logger.error("Exchange stop for %s: Delta returned no order id (%r)", symbol, result)
            return None
        logger.info("Exchange stop placed for %s: %s %d @ %s (order %s)", symbol, side, size, body["stop_price"], oid)
        return str(oid)

    async def edit_stop_order(self, symbol: str, order_id: str, stop_price: float) -> str | None:
        """Moves a resting stop in place (PUT /v2/orders). Returns the order id
        on success, None if Delta refused (the stop then rests unchanged)."""
        product = self.registry.get(symbol)
        body = {"id": int(order_id), "product_id": product.product_id,
                "stop_price": self._round_to_tick(stop_price, product.tick_size)}
        try:
            result = await self.client.edit_order(body)
        except self._STOP_ERRORS as exc:
            logger.warning("Edit of exchange stop %s for %s refused: %r", order_id, symbol, exc)
            return None
        oid = result.get("id") if isinstance(result, dict) else None
        return str(oid) if oid is not None else str(order_id)

    async def cancel_stop_order(self, symbol: str, order_id: str) -> bool:
        """Cancels one stop. Already gone (filled, cancelled, deadman) = True."""
        product = self.registry.get(symbol)
        try:
            await self.client.cancel_order({"id": int(order_id), "product_id": product.product_id})
        except DeltaApiError as exc:
            if delta_error_code(exc) in ORDER_GONE_CODES:
                return True
            logger.error("Cancel of exchange stop %s for %s failed: %r", order_id, symbol, exc)
            return False
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            logger.error("Cancel of exchange stop %s for %s did not reach Delta: %r", order_id, symbol, exc)
            return False
        return True

    async def resting_orders(self, symbol: str) -> list[dict] | None:
        """Delta's open + pending (untriggered stop) orders on this product, or
        None if the read failed -- "unknown" must never be read as "missing"."""
        product = self.registry.get(symbol)
        try:
            rows = await self.client.get_open_orders(product_ids=str(product.product_id))
        except self._STOP_ERRORS as exc:
            logger.warning("Could not read resting orders for %s: %r", symbol, exc)
            return None
        return list(rows or [])

    async def place_order(self, req: OrderRequest, exit_reason: str = "signal", *, post_only: bool = False) -> OrderResult:
        product = self.registry.get(req.symbol)
        if req.reduce_only or exit_reason != "signal":
            self._exit_reason[req.symbol] = exit_reason
        body = {
            "product_id": product.product_id,
            "size": req.size,
            "side": req.side,
            "order_type": "market_order" if req.order_type == "market" else "limit_order",
            "reduce_only": req.reduce_only,
        }
        if req.order_type == "limit" and req.limit_price is not None:
            body["limit_price"] = str(req.limit_price)
            if post_only:
                # Maker orders only (app/execution/maker.py): Delta refuses the
                # order rather than let it cross and pay the taker fee.
                body["post_only"] = True
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

        if not req.reduce_only:
            leverage = req.leverage or select_leverage(
                req.quality, self.settings.min_leverage, self.settings.max_leverage
            )
            if not await self._ensure_leverage(product, leverage):
                return OrderResult(0, None, "rejected", 0, None, 0.0, reason="could not set leverage on Delta")

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
                self.invalidate_balance_cache()
                order.state = "cancelled"
                await session.commit()
                if post_only and delta_error_code(exc) in POST_ONLY_CROSS_CODES:
                    # The touch moved between our book read and Delta: nothing
                    # traded. maker.py re-prices next tick or falls back to market.
                    logger.info("Post-only order for %s would have crossed; not placed", req.symbol)
                    return OrderResult(order.id, None, "rejected", 0, None, 0.0, reason="post_only_would_cross")
                logger.error("Live order rejected for %s: %s", req.symbol, exc)
                return OrderResult(order.id, None, "rejected", 0, None, 0.0, reason=str(exc))

            self.invalidate_balance_cache()  # a fill moves fees/margin/P&L
            order.broker_order_id = str(result.get("id"))
            order.state = result.get("state", "open")
            order.unfilled_size = result.get("unfilled_size", req.size)
            await session.commit()
            await session.refresh(order)

            filled = req.size - order.unfilled_size
            # Delta's order object carries the real fill in average_fill_price;
            # limit_price on a market order is only its protection bound, so
            # it is a fallback for responses without a fill price.
            fill_px = result.get("average_fill_price") or result.get("limit_price")
            avg_price = float(fill_px) if fill_px else None
            order_result = OrderResult(order.id, order.broker_order_id, order.state, filled, avg_price, 0.0)

        if not req.reduce_only and req.atr and pre_size == 0 and order_result.filled_size > 0:
            await self._attach_bracket(product, req)
        elif req.reduce_only and order_result.filled_size > 0:
            await self._cleanup_bracket_if_flat(product)

        # Refresh the local row NOW rather than at the next 15 s sync, so the
        # next trading tick can never size against a position that predates
        # this fill.
        await self._sync_product(product)
        return order_result

    async def _ensure_leverage(self, product, leverage: float) -> bool:
        """Sets Delta's per-product order leverage (an account setting on the
        exchange, not a per-order field) when it differs from what we last set."""
        lev = max(1, int(round(leverage)))
        if self._leverage_set.get(product.product_id) == lev:
            return True
        try:
            await self.client.request(
                "POST", f"/v2/products/{product.product_id}/orders/leverage", json_body={"leverage": str(lev)}, auth=True
            )
        except DeltaApiError:
            logger.exception("Could not set %dx leverage on %s", lev, product.symbol)
            return False
        self._leverage_set[product.product_id] = lev
        logger.info("Delta leverage for %s set to %dx", product.symbol, lev)
        return True

    async def cancel_order(self, order_id: int) -> bool:
        """Takes the order off Delta's book and updates our row. An order that
        is already gone (filled, cancelled, or removed by the deadman switch)
        counts as cancelled: its final state is read back from Delta.

        Returns False only when Delta said the order is gone but its final
        state could not be read, so it MAY have filled: the caller must not
        trade on a position read from before this call. Other Delta errors
        still raise (the order may still be resting)."""
        async with SessionLocal() as session:
            order = await session.get(Order, order_id)
            if not order or not order.broker_order_id:
                return True
            if order.state not in _RESTING_STATES:
                return True  # already closed/cancelled per Delta: nothing rests
            product = self.registry.get(order.symbol)
            unfilled_before = order.unfilled_size
            confirmed = True
            try:
                result = await self.client.cancel_order(
                    {"id": int(order.broker_order_id), "product_id": product.product_id}
                )
            except DeltaApiError as exc:
                if delta_error_code(exc) not in ORDER_GONE_CODES:
                    raise
                remote = await self._fetch_order(order.broker_order_id)
                if remote and remote.get("state") in ("closed", "cancelled"):
                    _apply_remote_order(order, remote)
                elif remote and remote.get("state") in _RESTING_STATES:
                    # Contradicts the cancel reply: trust the read, keep it
                    # open (a later cancel retries), and report "unsure".
                    _apply_remote_order(order, remote)
                    confirmed = False
                else:
                    # Gone from the book either way; a fill, if any, reaches
                    # the position through _sync_product below.
                    order.state = "cancelled"
                    confirmed = False
                logger.info(
                    "Cancel of %s order %s: already off Delta's book (now %s, unfilled %s)",
                    order.symbol, order.broker_order_id, order.state, order.unfilled_size,
                )
            else:
                order.state = "cancelled"
                if isinstance(result, dict) and result.get("unfilled_size") is not None:
                    order.unfilled_size = int(result["unfilled_size"])
            traded = not confirmed or (order.unfilled_size or 0) < (unfilled_before or 0)
            await session.commit()
        if traded:
            # It (may have) traded since the last sync: refresh the position now.
            self.invalidate_balance_cache()
            await self._sync_product(product)
        return confirmed

    async def _fetch_order(self, broker_order_id: str) -> dict | None:
        """Delta's order by id (GET /v2/orders/{id}), or None if unreadable."""
        try:
            remote = await self.client.request("GET", f"/v2/orders/{broker_order_id}", auth=True)
        except (DeltaApiError, httpx.HTTPError):
            logger.warning("Could not read Delta order %s", broker_order_id, exc_info=True)
            return None
        return remote if isinstance(remote, dict) else None

    async def _sync_orders(self, product) -> None:
        """Brings our resting live orders up to date with Delta: a maker order
        that fills, part-fills, or is cancelled by Delta (deadman switch) after
        it was placed used to stay "open" here forever. Only called when we
        hold such an order, so with maker execution off it costs no request."""
        from sqlalchemy import select

        async with SessionLocal() as session:
            rows = list((await session.execute(
                select(Order).where(
                    Order.mode == "live", Order.symbol == product.symbol,
                    Order.state.in_(_RESTING_STATES), Order.broker_order_id.is_not(None),
                )
            )).scalars().all())
            if not rows:
                return
            # Rows are read BEFORE asking Delta, so every row read here was
            # already acknowledged by Delta when its active list is taken.
            try:
                active = await self.client.get_open_orders(product_ids=str(product.product_id))
            except (DeltaApiError, httpx.HTTPError):
                logger.warning("Could not read open orders for %s", product.symbol, exc_info=True)
                return
            by_id = {str(o.get("id")): o for o in (active or []) if isinstance(o, dict)}
            for order in rows:
                remote = by_id.get(order.broker_order_id)
                if remote is None:
                    # Not on the book any more: read how it ended.
                    remote = await self._fetch_order(order.broker_order_id)
                    if remote is None or not remote.get("state"):
                        # Unreadable, but absent from the active list: it is
                        # not resting, so stop treating it as working. Any fill
                        # reaches the position sync that follows. (If Delta's
                        # by-id read says it IS still open, that is trusted.)
                        remote = {"state": "cancelled"}
                if remote.get("state") != order.state or remote.get("unfilled_size") not in (None, order.unfilled_size):
                    logger.info("Live %s order %s on Delta: %s, unfilled %s", product.symbol,
                                order.broker_order_id, remote.get("state"), remote.get("unfilled_size"))
                _apply_remote_order(order, remote)
            await session.commit()

    async def sync_fills_and_positions(self) -> None:
        """Pulls authoritative state from Delta and reconciles local rows. In a
        full deployment this would be driven by the private WebSocket (orders/
        positions/v2_user_trades channels); this REST-based sync is the
        dependable fallback used until that channel is wired in.
        """
        for product in self.registry.all():
            await self._sync_product(product)

    async def _fills_since(self, product, since: dt.datetime) -> list[dict]:
        """Delta's own fills for this product since `since` (UTC) -- the source
        of truth for exit prices and the commission actually charged."""
        start_us = int(since.replace(tzinfo=since.tzinfo or dt.timezone.utc).timestamp() * 1_000_000)
        try:
            rows = await self.client.request(
                "GET", "/v2/fills", params={"product_ids": str(product.product_id), "start_time": start_us,
                                           "page_size": 100}, auth=True,
            )
        except (DeltaApiError, httpx.HTTPError):
            logger.exception("Could not read fills for %s", product.symbol)
            return []
        if not isinstance(rows, list):
            rows = []
        cutoff = since.replace(tzinfo=since.tzinfo or dt.timezone.utc)
        out = []
        for f in rows:
            try:
                when = dt.datetime.fromisoformat(str(f.get("created_at")).replace("Z", "+00:00"))
            except ValueError:
                continue
            if when >= cutoff:
                out.append({**f, "_when": when})
        return sorted(out, key=lambda f: f["_when"])

    async def _record_close(self, session, product, size: int, entry_price: float,
                            entry_time: Optional[dt.datetime], leverage: Optional[float]) -> None:
        """A live position just went flat (or flipped): log it as a closed
        trade with Delta's real exit price and fees, like paper trades are."""
        now = dt.datetime.now(dt.timezone.utc)
        opened = (entry_time or now - dt.timedelta(minutes=1))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=dt.timezone.utc)
        # entry_time is stamped by the sync just AFTER the opening fill, so
        # look a minute back to include it.
        fills = await self._fills_since(product, opened - dt.timedelta(seconds=60))
        closing_side = "sell" if size > 0 else "buy"
        closing = [f for f in fills if f.get("side") == closing_side]
        qty = sum(float(f.get("size") or 0) for f in closing)
        if qty > 0:
            exit_price = sum(float(f.get("price") or 0) * float(f.get("size") or 0) for f in closing) / qty
        else:
            exit_price = self.cache.get_last_price(product.symbol) or entry_price
        fees = sum(float(f.get("commission") or 0) for f in fills)
        direction = 1 if size > 0 else -1
        exit_reason = self._exit_reason.pop(product.symbol, "signal")
        realized = direction * (exit_price - entry_price) * abs(size) * product.contract_value
        session.add(ClosedTrade(
            symbol=product.symbol, mode="live", side="long" if size > 0 else "short",
            strategy=LIVE_POSITION_STRATEGY, size=abs(size), leverage=leverage,
            entry_price=entry_price, entry_time=opened.replace(tzinfo=None), exit_price=exit_price,
            exit_time=now.replace(tzinfo=None),
            exit_reason=exit_reason,
            realized_pnl=realized,
            fees=fees,
        ))
        telegram.trade_closed(product.symbol, "long" if size > 0 else "short", size, entry_price, exit_price,
                              realized, fees, exit_reason)
        logger.info("Live %s %s closed: %.2f -> %.2f, fees %.4f", product.symbol,
                    "long" if size > 0 else "short", entry_price, exit_price, fees)

    async def _sync_product(self, product) -> None:
        from sqlalchemy import select

        # Orders FIRST, then the position: a fill our order rows show is then
        # always already in the position read after it.
        try:
            await self._sync_orders(product)
        except Exception:  # never let order bookkeeping block the position sync
            logger.exception("Failed to sync open orders for %s", product.symbol)
        try:
            positions = await self.client.get_positions_margined(product_ids=str(product.product_id))
        except (DeltaApiError, httpx.HTTPError):
            logger.exception("Failed to sync position for %s", product.symbol)
            return
        async with SessionLocal() as session:
            result = await session.execute(
                select(Position).where(
                    Position.symbol == product.symbol, Position.mode == "live",
                    Position.strategy == LIVE_POSITION_STRATEGY,
                )
            )
            pos = result.scalar_one_or_none()
            if pos is None:
                pos = Position(symbol=product.symbol, mode="live", strategy=LIVE_POSITION_STRATEGY,
                               size=0, entry_price=0.0)
                session.add(pos)
            remote = positions[0] if positions else None
            previous_size = pos.size or 0
            prev_entry, prev_time, prev_lev = pos.entry_price, pos.entry_time, pos.leverage
            new_size = int(remote.get("size", 0)) if remote else 0
            if new_size != previous_size:
                self.invalidate_balance_cache()  # a fill we did not place (bracket, manual)
            if previous_size != 0 and (new_size == 0 or (previous_size > 0) != (new_size > 0)):
                await self._record_close(session, product, previous_size, prev_entry, prev_time, prev_lev)
            if not remote:
                # Flat on Delta: an absent position object means size 0, not "unknown".
                pos.size = 0
                pos.fees_paid = 0.0
            else:
                pos.size = int(remote.get("size", 0))
                pos.entry_price = float(remote.get("entry_price", 0) or 0)
                pos.margin = float(remote.get("margin", 0) or 0)
                pos.liquidation_price = float(remote.get("liquidation_price", 0) or 0)
                if pos.margin > 0 and pos.size:
                    pos.leverage = round(abs(pos.size) * product.contract_value * pos.entry_price / pos.margin, 2)
                if pos.size and (previous_size == 0 or (previous_size > 0) != (pos.size > 0)):
                    pos.entry_time = dt.datetime.now(dt.timezone.utc)
                    # Any fill that opens (or flips) a live position: the trend
                    # book, the Analyst, a bracket, or a manual trade on Delta.
                    telegram.trade_opened(product.symbol, "long" if pos.size > 0 else "short", pos.size,
                                          pos.entry_price)
                if pos.size and (pos.size != previous_size or not pos.fees_paid):
                    # Open-position fees from Delta's fills, refreshed only when
                    # the size changes (or once, to backfill) -- not every 15 s.
                    since = pos.entry_time or dt.datetime.now(dt.timezone.utc)
                    since = since if since.tzinfo else since.replace(tzinfo=dt.timezone.utc)
                    fills = await self._fills_since(product, since - dt.timedelta(seconds=60))
                    pos.fees_paid = sum(float(f.get("commission") or 0) for f in fills)
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
                out[f"{p.symbol}:{p.strategy or LIVE_POSITION_STRATEGY}"] = PositionView(
                    p.symbol, p.size, p.entry_price, unrealized, p.realized_pnl,
                    strategy=p.strategy, entry_time=p.entry_time, leverage=p.leverage,
                    fees_paid=p.fees_paid, notional_usd=notional_usd, margin_usd=margin_usd,
                    liquidation_price=p.liquidation_price or None,
                )
            return out

    async def get_equity(self) -> float:
        """Live equity = Delta USD balance (see _balance/_fetch_balance) + open
        P&L marked at the live price. Raises LiveEquityUnavailable rather than
        ever guessing a balance."""
        balance = await self._balance()
        positions = await self.get_positions()
        return balance + sum(p.unrealized_pnl for p in positions.values())

    def invalidate_balance_cache(self) -> None:
        """The next get_equity reads Delta again (and ignores any read already
        in flight, which may predate the change)."""
        self._balance_gen += 1
        self._balance_cached_at = None
        self._balance_inflight = None

    async def _balance(self) -> float:
        """Delta USD balance: cached for BALANCE_CACHE_TTL_S, and concurrent
        callers share ONE request instead of each opening their own."""
        if _FRESH_BALANCE.get():
            return await self._fetch_balance(self._balance_gen)
        if self._balance_cached_at is not None and time.monotonic() - self._balance_cached_at < BALANCE_CACHE_TTL_S:
            return self._last_balance
        if self._balance_inflight is None or self._balance_inflight.done():
            fut = asyncio.ensure_future(self._fetch_balance(self._balance_gen))
            # Read the outcome even if every waiter was cancelled, so a failure
            # is never reported as "Task exception was never retrieved".
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())
            self._balance_inflight = fut
        # Shielded: a caller that goes away (a closed dashboard tab) must not
        # cancel the read the other callers are waiting on.
        return await asyncio.shield(self._balance_inflight)

    async def _fetch_balance(self, gen: int) -> float:
        """A failed fetch used to return 0.0, and every circuit breaker
        believed it: on 2026-10-03 one expired-signature error read as a 100%
        loss, the daily-loss supervisor flattened a real position and halted
        trading on a false alarm. Now a failure falls back to the last good
        balance for a few minutes, then RAISES -- an unknown balance must make
        the checks skip, never make them fire."""
        started = time.monotonic()
        try:
            balances = await self.client.get_balances()
        except (DeltaApiError, httpx.HTTPError) as exc:
            # Aged from NOW, not from before the request: a concurrent call
            # could refresh it meanwhile (that logged "-2s ago"), and a slow
            # failure must not stretch the fallback window.
            age = time.monotonic() - self._last_balance_at if self._last_balance is not None else None
            # httpx timeouts stringify to "", so name the type as well.
            reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            telegram.equity_read_failed(type(exc).__name__)  # alerts once after 5 min of failures
            if age is None or age > LAST_GOOD_BALANCE_MAX_AGE_S:
                raise LiveEquityUnavailable(f"Delta balance unavailable: {reason}") from exc
            logger.warning("Live balance fetch failed (%s); using the balance from %.0fs ago", reason, max(age, 0.0))
            return self._last_balance
        usd_like = [b for b in balances if b.get("asset_symbol") in ("USD", "USDT", "USDC")]
        balance = sum(float(b.get("balance", 0) or 0) for b in usd_like)
        # Stamped with the request START, so the cache never claims fresher
        # data than it has.
        self._last_balance, self._last_balance_at = balance, started
        telegram.equity_read_ok()  # clears an equity-outage alert, if one was sent
        if gen == self._balance_gen:  # no order/fill since this read began
            self._balance_cached_at = started
        return balance
