import datetime as dt

import pytest
from sqlalchemy import select

from app.config import Settings
from app.core.delta_client import DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import Order
from app.execution import maker
from app.execution.broker_base import OrderRequest
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.risk.risk_manager import RiskManager

T0 = dt.datetime(2026, 10, 2, 12, 0, tzinfo=dt.timezone.utc)


def _registry() -> SymbolRegistry:
    reg = SymbolRegistry(Settings(delta_api_key="k", delta_api_secret="s"))
    reg._by_symbol["TESTUSD"] = ProductInfo(
        product_id=1, symbol="TESTUSD", contract_type="perpetual_futures", tick_size=0.5,
        contract_value=0.01, underlying_asset="TEST", state="live", default_leverage=10,
        taker_commission_rate=0.0005, maker_commission_rate=0.0002, spot_index_symbol="",
    )
    return reg


class _Runtime:
    def __init__(self, timeout_s: int = 900) -> None:
        self.settings = Settings(
            delta_api_key="k", delta_api_secret="s", maker_execution="paper", maker_timeout_seconds=timeout_s,
            max_margin_per_trade_pct=1.0, max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85,
        )
        self.registry = _registry()
        self.cache = MarketDataCache()
        self.book(999.0, 1001.0)
        self.risk_manager = RiskManager(self.settings)
        self.paper = PaperBroker(self.registry, self.cache, self.settings)
        live = LiveBroker(DeltaRestClient(self.settings), self.registry, self.cache)
        self.order_manager = OrderManager(self.paper, live, self.risk_manager, self.registry, self.cache)

    def book(self, bid: float, ask: float) -> None:
        self.cache.update_orderbook("TESTUSD", bids=[[bid, 100]], asks=[[ask, 100]], timestamp=1)


def _buy(size: int = 10, **kw) -> OrderRequest:
    return OrderRequest(symbol="TESTUSD", side="buy", size=size, strategy="consolidated", quality=0.6, **kw)


async def _orders(state: str | None = None) -> list[Order]:
    async with SessionLocal() as session:
        stmt = select(Order).order_by(Order.id)
        if state:
            stmt = stmt.where(Order.state == state)
        return list((await session.execute(stmt)).scalars().all())


@pytest.fixture(autouse=True)
def _fresh_clock():
    maker._clock.clear()
    yield
    maker._clock.clear()


def test_maker_is_on_everywhere_by_default_and_paper_mode_still_works():
    s = Settings(delta_api_key="k", delta_api_secret="s")
    assert s.maker_execution == "all"  # owner approved live limit orders, 2026-10-07
    s.maker_execution = "paper"
    assert maker.maker_enabled(s, "paper") and not maker.maker_enabled(s, "live")
    s.maker_execution = "all"
    assert maker.maker_enabled(s, "live")
    s.maker_execution = "off"
    assert not maker.maker_enabled(s, "paper")


@pytest.mark.asyncio
async def test_rebalance_rests_at_the_best_bid_instead_of_paying_the_spread():
    rt = _Runtime()
    result = await maker.execute_as_maker(rt, "paper", _buy(), now=T0)
    assert result.state == "open" and result.filled_size == 0
    [order] = await _orders("open")
    assert order.order_type == "limit" and order.limit_price == 999.0
    assert (await rt.order_manager.get_positions()) == {} or all(p.size == 0 for p in (await rt.order_manager.get_positions()).values())


@pytest.mark.asyncio
async def test_unchanged_touch_leaves_the_working_order_alone():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "paper", _buy(), now=T0)
    assert await maker.execute_as_maker(rt, "paper", _buy(), now=T0 + dt.timedelta(minutes=1)) is None
    assert len(await _orders("open")) == 1


@pytest.mark.asyncio
async def test_moving_touch_re_pegs_without_resetting_the_clock():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "paper", _buy(), now=T0)
    rt.book(1000.0, 1002.0)
    await maker.execute_as_maker(rt, "paper", _buy(), now=T0 + dt.timedelta(minutes=5))
    open_orders = await _orders("open")
    assert [o.limit_price for o in open_orders] == [1000.0]
    assert maker._clock["TESTUSD:consolidated"][1] == T0  # still counting from the first attempt


@pytest.mark.asyncio
async def test_resting_order_fills_at_its_limit_with_the_maker_fee():
    rt = _Runtime()
    await maker.execute_as_maker(rt, "paper", _buy(), now=T0)
    rt.book(998.0, 999.0)  # sellers come down to our bid
    await rt.order_manager.process_resting_fills()
    [filled] = await _orders("closed")
    assert filled.limit_price == 999.0
    pos = (await rt.order_manager.get_positions())["TESTUSD:consolidated"]
    assert pos.size == 10 and pos.entry_price == 999.0
    # 10 contracts * 0.01 * 999 = 99.9 notional -> maker 0.02% = 0.01998, not taker 0.04995
    assert pos.fees_paid == pytest.approx(99.9 * 0.0002)


@pytest.mark.asyncio
async def test_unfilled_after_timeout_falls_back_to_market():
    rt = _Runtime(timeout_s=900)
    await maker.execute_as_maker(rt, "paper", _buy(), now=T0)
    result = await maker.execute_as_maker(rt, "paper", _buy(), now=T0 + dt.timedelta(minutes=16))
    assert result.state == "closed" and result.filled_size == 10
    states = sorted(o.state for o in await _orders())
    assert states == ["cancelled", "closed"]  # the resting order never fills later
    assert "TESTUSD:consolidated" not in maker._clock


@pytest.mark.asyncio
async def test_no_order_book_means_market_not_a_blind_limit():
    rt = _Runtime()
    rt.cache.orderbook.pop("TESTUSD")
    rt.cache.mark_price["TESTUSD"] = 1000.0  # a price exists, just no book to rest in
    result = await maker.execute_as_maker(rt, "paper", _buy(), now=T0)
    assert result.state == "closed"


@pytest.mark.asyncio
async def test_forced_close_cancels_resting_orders_first():
    # A resting buy left behind would fill AFTER the close and silently reopen
    # the position the user (or kill switch) just closed.
    rt = _Runtime()
    rt.book(999.0, 1001.0)
    await rt.order_manager.submit(_buy(size=5))  # an open market long to close
    await maker.execute_as_maker(rt, "paper", _buy(size=3), now=T0)
    assert len(await _orders("open")) == 1
    await rt.order_manager.close_position("TESTUSD", "consolidated")
    assert await _orders("open") == []
    rt.book(998.0, 999.0)
    await rt.order_manager.process_resting_fills()
    positions = await rt.order_manager.get_positions()
    assert all(p.size == 0 for p in positions.values())


@pytest.mark.asyncio
async def test_resting_fill_pnl_reaches_the_daily_loss_kill_switch():
    rt = _Runtime()
    await rt.order_manager.submit(_buy(size=10))  # long 10 @ ~1001.2
    before = rt.risk_manager.daily_pnl.realized_pnl
    sell = OrderRequest(symbol="TESTUSD", side="sell", size=10, strategy="consolidated", reduce_only=True)
    rt.book(900.0, 901.0)
    await maker.execute_as_maker(rt, "paper", sell, now=T0)  # rests at the ask, 901
    rt.book(901.0, 902.0)  # buyers come up to our ask
    await rt.order_manager.process_resting_fills()
    assert rt.risk_manager.daily_pnl.realized_pnl < before  # the loss was recorded
