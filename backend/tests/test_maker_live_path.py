"""Maker execution on the LIVE path: resting orders that Delta fills, part-
fills or cancels after placement, cancels of orders that are already gone,
post_only, and -- most important -- a kill-switch flatten that always closes
the positions even when cancelling resting orders fails."""
import datetime as dt

import httpx
import pytest
from sqlalchemy import select

from app.config import Settings
from app.core.delta_client import DeltaApiError
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.app_state import set_trading_mode
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import Order
from app.execution import maker
from app.execution.broker_base import OrderRequest, OrderResult, PositionView
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.risk.risk_manager import RiskManager

T0 = dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.timezone.utc)
GONE = {"success": False, "error": {"code": "open_order_not_found"}}


def _registry(settings) -> SymbolRegistry:
    reg = SymbolRegistry(settings)
    reg._by_symbol["TESTUSD"] = ProductInfo(
        product_id=1, symbol="TESTUSD", contract_type="perpetual_futures", tick_size=0.5,
        contract_value=0.01, underlying_asset="TEST", state="live", default_leverage=10,
        taker_commission_rate=0.0005, maker_commission_rate=0.0002, spot_index_symbol="",
    )
    return reg


class _FakeDelta:
    """Delta's order book for OUR orders: limit orders rest until the test
    fills or cancels them; market orders fill at once. No network."""

    def __init__(self) -> None:
        self.orders: dict[int, dict] = {}
        self.position = 0
        self.next_id = 100
        self.calls: list = []
        self.open_order_reads = 0
        self.cancel_error: Exception | None = None
        self.place_error: Exception | None = None
        self.by_id_error: Exception | None = None

    # -- test controls -------------------------------------------------
    def fill(self, oid: int, qty: int) -> None:
        o = self.orders[oid]
        o["unfilled_size"] -= qty
        self.position += qty if o["side"] == "buy" else -qty
        if o["unfilled_size"] == 0:
            o["state"] = "closed"

    def deadman_cancels_everything(self) -> None:
        for o in self.orders.values():
            if o["state"] in ("open", "pending"):
                o["state"] = "cancelled"

    def bodies(self, kind: str) -> list[dict]:
        return [c[1] for c in self.calls if c[0] == kind]

    # -- Delta API ----------------------------------------------------------
    async def get_balances(self):
        return [{"asset_symbol": "USD", "balance": "1000"}]

    async def get_positions_margined(self, product_ids=None):
        if self.position == 0:
            return []
        return [{"size": self.position, "entry_price": "1000", "margin": "10", "liquidation_price": "0",
                 "realized_pnl": "0"}]

    async def get_open_orders(self, product_ids=None):
        self.open_order_reads += 1
        return [dict(o) for o in self.orders.values() if o["state"] in ("open", "pending")]

    async def place_order(self, body):
        self.calls.append(("ORDER", body))
        if self.place_error is not None:
            raise self.place_error
        oid, self.next_id = self.next_id, self.next_id + 1
        o = {"id": oid, "side": body["side"], "size": body["size"], "unfilled_size": body["size"], "state": "open",
             "limit_price": body.get("limit_price")}
        self.orders[oid] = o
        if body["order_type"] == "market_order":
            o["average_fill_price"] = "1000"
            self.fill(oid, body["size"])
        return dict(o)

    async def cancel_order(self, body):
        self.calls.append(("CANCEL", body))
        if self.cancel_error is not None:
            raise self.cancel_error
        o = self.orders.get(body["id"])
        if o is None or o["state"] not in ("open", "pending"):
            raise DeltaApiError(400, GONE)
        o["state"] = "cancelled"
        return dict(o)

    async def cancel_all_orders(self, body=None):
        return {}

    async def request(self, method, path, *, json_body=None, auth=False, params=None):
        self.calls.append((method, path, json_body))
        if method == "GET" and path.startswith("/v2/orders/"):
            if self.by_id_error is not None:
                raise self.by_id_error
            o = self.orders.get(int(path.rsplit("/", 1)[1]))
            if o is None:
                raise DeltaApiError(404, {"success": False, "error": {"code": "not_found"}})
            return dict(o)
        if path == "/v2/fills":
            return []
        return {}


class _Runtime:
    def __init__(self, timeout_s: int = 300) -> None:
        self.settings = Settings(
            delta_api_key="k", delta_api_secret="s", maker_execution="all", maker_timeout_seconds=timeout_s,
            max_margin_per_trade_pct=1.0, max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85,
        )
        self.registry = _registry(self.settings)
        self.cache = MarketDataCache()
        self.book(999.0, 1001.0)
        self.risk_manager = RiskManager(self.settings)
        self.delta = _FakeDelta()
        self.paper = PaperBroker(self.registry, self.cache, self.settings)
        self.live = LiveBroker(self.delta, self.registry, self.cache, None, self.settings)
        self.order_manager = OrderManager(self.paper, self.live, self.risk_manager, self.registry, self.cache)

    def book(self, bid: float, ask: float) -> None:
        self.cache.update_orderbook("TESTUSD", bids=[[bid, 100]], asks=[[ask, 100]], timestamp=1)


def _buy(size: int = 10, **kw) -> OrderRequest:
    return OrderRequest(symbol="TESTUSD", side="buy", size=size, strategy="consolidated", quality=0.6, **kw)


async def _orders(state: str | None = None) -> list[Order]:
    async with SessionLocal() as session:
        stmt = select(Order).where(Order.mode == "live").order_by(Order.id)
        if state:
            stmt = stmt.where(Order.state == state)
        return list((await session.execute(stmt)).scalars().all())


async def _position(rt) -> int:
    pos = (await rt.live.get_positions()).get("TESTUSD:consolidated")
    return pos.size if pos else 0


@pytest.fixture(autouse=True)
async def _live_mode():
    maker._clock.clear()
    await set_trading_mode("live")
    yield
    maker._clock.clear()


# ---- post_only -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_maker_limit_order_is_sent_post_only_but_market_orders_are_not():
    rt = _Runtime()
    result = await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    assert result.state == "open" and result.filled_size == 0
    [body] = rt.delta.bodies("ORDER")
    assert body["order_type"] == "limit_order" and body["post_only"] is True and body["limit_price"] == "999.0"
    # The ordinary (maker off) path: a market order, exactly as before -- no post_only key at all.
    await rt.order_manager.submit(_buy(size=2))
    market = rt.delta.bodies("ORDER")[-1]
    assert market["order_type"] == "market_order" and "post_only" not in market


@pytest.mark.asyncio
async def test_post_only_refusal_is_retried_at_the_new_touch_then_falls_back_to_market():
    rt = _Runtime(timeout_s=300)
    rt.delta.place_error = DeltaApiError(400, {"success": False, "error": {"code": "immediate_execution_post_only"}})
    result = await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    assert result.state == "rejected" and result.reason == "post_only_would_cross" and result.filled_size == 0
    assert maker._clock["TESTUSD:consolidated"][1] == T0  # the clock keeps running
    rt.delta.place_error = None
    rt.book(1000.0, 1002.0)
    result = await maker.execute_as_maker(rt, "live", _buy(), now=T0 + dt.timedelta(minutes=1))
    assert result.state == "open" and rt.delta.bodies("ORDER")[-1]["limit_price"] == "1000.0"
    result = await maker.execute_as_maker(rt, "live", _buy(), now=T0 + dt.timedelta(minutes=6))
    assert result.state == "closed" and rt.delta.bodies("ORDER")[-1]["order_type"] == "market_order"
    assert await _position(rt) == 10


# ---- order sync -------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_keeps_unfilled_size_current_and_closes_a_filled_order():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    [order] = await _orders("open")
    oid = int(order.broker_order_id)
    rt.delta.fill(oid, 4)
    await rt.live.sync_fills_and_positions()
    [order] = await _orders()
    assert (order.state, order.unfilled_size) == ("open", 6)
    assert await _position(rt) == 4
    rt.delta.fill(oid, 6)
    await rt.live.sync_fills_and_positions()
    [order] = await _orders()
    assert (order.state, order.unfilled_size) == ("closed", 0)
    assert await _position(rt) == 10


@pytest.mark.asyncio
async def test_part_filled_order_still_at_the_touch_is_left_working():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    [order] = await _orders("open")
    rt.delta.fill(int(order.broker_order_id), 4)
    await rt.live.sync_fills_and_positions()
    # The loop now asks only for the 6 still missing: the 6 resting are exactly that.
    assert await maker.execute_as_maker(rt, "live", _buy(size=6), now=T0 + dt.timedelta(minutes=1)) is None
    assert rt.delta.bodies("CANCEL") == []


@pytest.mark.asyncio
async def test_no_resting_orders_means_no_extra_delta_requests():
    # Maker off (the live default): only market orders, which come back closed,
    # so the order sync never asks Delta for anything.
    rt = _Runtime()
    rt.settings.maker_execution = "paper"
    await rt.order_manager.submit(_buy(size=3))
    await rt.live.sync_fills_and_positions()
    assert rt.delta.open_order_reads == 0
    assert not [c for c in rt.delta.calls if c[0] == "GET" and str(c[1]).startswith("/v2/orders/")]


@pytest.mark.asyncio
async def test_order_cancelled_by_the_deadman_switch_is_unfilled_and_re_pegged_then_taken():
    rt = _Runtime(timeout_s=300)
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    rt.delta.deadman_cancels_everything()  # heartbeats lapsed: Delta wiped our resting orders
    await rt.live.sync_fills_and_positions()
    [order] = await _orders()
    assert (order.state, order.unfilled_size) == ("cancelled", 10)
    result = await maker.execute_as_maker(rt, "live", _buy(), now=T0 + dt.timedelta(minutes=1))
    assert result.state == "open" and rt.delta.bodies("ORDER")[-1]["post_only"] is True
    rt.delta.deadman_cancels_everything()  # gone again, and this time no sync before the timeout
    result = await maker.execute_as_maker(rt, "live", _buy(), now=T0 + dt.timedelta(minutes=6))
    assert result.state == "closed" and result.filled_size == 10
    assert await _position(rt) == 10


# ---- cancel_order ------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancelling_an_order_that_already_filled_is_success_and_refreshes_the_position():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    [order] = await _orders("open")
    rt.delta.fill(int(order.broker_order_id), 10)  # filled since the last sync
    assert await rt.live.cancel_order(order.id) is True
    [order] = await _orders()
    assert (order.state, order.unfilled_size) == ("closed", 0)
    assert await _position(rt) == 10  # synced at once, not 15 s later


@pytest.mark.asyncio
async def test_gone_order_whose_ending_cannot_be_read_is_reported_unsure():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    [order] = await _orders("open")
    rt.delta.deadman_cancels_everything()
    rt.delta.by_id_error = httpx.ConnectTimeout("boom")
    assert await rt.live.cancel_order(order.id) is False
    [order] = await _orders()
    assert order.state == "cancelled"


@pytest.mark.asyncio
async def test_other_cancel_errors_still_raise_and_leave_the_order_open():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    rt.delta.cancel_error = DeltaApiError(500, {"success": False, "error": {"code": "unavailable"}})
    [order] = await _orders("open")
    with pytest.raises(DeltaApiError):
        await rt.live.cancel_order(order.id)
    assert [o.state for o in await _orders()] == ["open"]


@pytest.mark.asyncio
async def test_successful_cancel_records_a_part_fill_from_delta_reply():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    [order] = await _orders("open")
    rt.delta.fill(int(order.broker_order_id), 3)
    assert await rt.live.cancel_order(order.id) is True
    [order] = await _orders()
    assert (order.state, order.unfilled_size) == ("cancelled", 7)
    assert await _position(rt) == 3


@pytest.mark.asyncio
async def test_timeout_does_not_double_buy_when_the_resting_order_filled_unseen():
    # The tick computed "buy 10" from a position read before the resting order
    # filled. Taking 10 more at market would double the trade.
    rt = _Runtime(timeout_s=300)
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    [order] = await _orders("open")
    rt.delta.fill(int(order.broker_order_id), 10)
    result = await maker.execute_as_maker(rt, "live", _buy(), now=T0 + dt.timedelta(minutes=6))
    assert result is None
    assert [b["order_type"] for b in rt.delta.bodies("ORDER")] == ["limit_order"]  # no market order
    assert await _position(rt) == 10


@pytest.mark.asyncio
async def test_repeg_after_an_unseen_part_fill_waits_for_the_position():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(), now=T0)
    [order] = await _orders("open")
    rt.delta.fill(int(order.broker_order_id), 4)
    rt.book(1000.0, 1002.0)  # touch moved: the order would be re-pegged
    assert await maker.execute_as_maker(rt, "live", _buy(), now=T0 + dt.timedelta(minutes=1)) is None
    assert len(rt.delta.bodies("ORDER")) == 1
    assert await _position(rt) == 4
    # Next tick works from the refreshed position: 6 more at the new touch.
    result = await maker.execute_as_maker(rt, "live", _buy(size=6), now=T0 + dt.timedelta(minutes=2))
    assert result.state == "open" and rt.delta.bodies("ORDER")[-1]["size"] == 6


# ---- flatten_all: the kill switch must always flatten -------------------------------

@pytest.mark.asyncio
async def test_kill_switch_flattens_even_when_cancelling_resting_orders_fails():
    rt = _Runtime()
    await rt.order_manager.submit(_buy(size=5))  # an open long
    await maker.execute_as_maker(rt, "live", _buy(size=3), now=T0)  # plus a resting buy
    assert await _position(rt) == 5
    rt.delta.cancel_error = httpx.ConnectTimeout("Delta unreachable for cancels")
    await rt.order_manager.flatten_all(mode_override="live", reason="kill_switch")
    assert await _position(rt) == 0
    assert rt.delta.position == 0
    close = rt.delta.bodies("ORDER")[-1]
    assert close["order_type"] == "market_order" and close["reduce_only"] is True and close["side"] == "sell"
    # The failed cancel was tried again after the flatten (twice in all).
    assert len(rt.delta.bodies("CANCEL")) == 2


@pytest.mark.asyncio
async def test_kill_switch_flattens_when_the_resting_order_already_filled():
    # The original blocker: the cancel hits an order Delta already filled.
    rt = _Runtime()
    await maker.execute_as_maker(rt, "live", _buy(size=3), now=T0)
    [order] = await _orders("open")
    rt.delta.fill(int(order.broker_order_id), 3)  # unseen fill: local position still flat
    await rt.order_manager.flatten_all(mode_override="live", reason="kill_switch")
    assert rt.delta.position == 0 and await _position(rt) == 0
    assert [o.state for o in await _orders()] == ["closed", "closed"]


class _StubBroker:
    mode = "live"

    def __init__(self, fail_symbol: str) -> None:
        self.fail_symbol = fail_symbol
        self.closed: list[str] = []

    async def get_positions(self):
        return {f"{s}:consolidated": PositionView(s, 1, 100.0, 0.0, 0.0, strategy="consolidated")
                for s in ("AAAUSD", "BBBUSD", "CCCUSD")}

    async def place_order(self, req, exit_reason="signal"):
        if req.symbol == self.fail_symbol:
            raise httpx.ReadTimeout("no reply")
        self.closed.append(req.symbol)
        return OrderResult(1, "1", "closed", req.size, 100.0, 0.0)

    async def cancel_order(self, order_id):
        raise AssertionError("no resting orders in this test")


@pytest.mark.asyncio
async def test_one_failed_close_does_not_leave_the_other_positions_open():
    rt = _Runtime()
    stub = _StubBroker(fail_symbol="BBBUSD")
    om = OrderManager(rt.paper, stub, rt.risk_manager, rt.registry, rt.cache)
    with pytest.raises(RuntimeError, match="BBBUSD"):
        await om.flatten_all(mode_override="live")
    assert stub.closed == ["AAAUSD", "CCCUSD"]
