import pytest

from app.config import Settings
from app.core.delta_client import DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.app_state import get_state, get_trading_mode, is_kill_switch_on, set_kill_switch
from app.data.market_cache import MarketDataCache
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.execution.broker_base import OrderRequest
from app.risk.risk_manager import RiskManager
from app.scheduler import _daily_profit_target_supervisor


def _registry() -> SymbolRegistry:
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    reg = SymbolRegistry(settings)
    reg._by_symbol["TESTUSD"] = ProductInfo(
        product_id=1, symbol="TESTUSD", contract_type="perpetual_futures", tick_size=0.5,
        contract_value=0.01, underlying_asset="TEST", state="live", default_leverage=10,
        taker_commission_rate=0.0005, maker_commission_rate=0.0002, spot_index_symbol="",
    )
    return reg


class _FakeRuntime:
    def __init__(self, settings: Settings) -> None:
        self.registry = _registry()
        self.settings = settings
        self.cache = MarketDataCache()
        self.cache.update_orderbook("TESTUSD", bids=[[999.0, 100]], asks=[[1001.0, 100]], timestamp=1)
        self.risk_manager = RiskManager(settings)
        self.paper_broker = PaperBroker(self.registry, self.cache, settings)
        self.live_broker = LiveBroker(DeltaRestClient(settings), self.registry, self.cache)
        self.order_manager = OrderManager(
            self.paper_broker, self.live_broker, self.risk_manager, self.registry, self.cache
        )


def _settings(**overrides) -> Settings:
    base = dict(delta_api_key="k", delta_api_secret="s", daily_profit_target_pct=0.10)
    base.update(overrides)
    return Settings(**base)


@pytest.mark.asyncio
async def test_no_target_reached_leaves_kill_switch_off_and_nothing_flattened():
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )

    await _daily_profit_target_supervisor(runtime)  # establishes starting_equity, no target yet
    await _daily_profit_target_supervisor(runtime)  # price unchanged -- still no target

    assert not await is_kill_switch_on()
    positions = await runtime.order_manager.get_positions()
    assert positions["TESTUSD:manual"].size == 1000  # untouched


@pytest.mark.asyncio
async def test_target_reached_auto_engages_kill_switch_and_flattens():
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )

    await _daily_profit_target_supervisor(runtime)  # establishes today's starting equity
    assert not await is_kill_switch_on()

    # Price rallies hard -- a real mark-to-market gain on the still-open
    # position, well past the 10% daily target, with zero trades ever closed.
    runtime.cache.update_orderbook("TESTUSD", bids=[[1199.0, 100]], asks=[[1201.0, 100]], timestamp=2)

    await _daily_profit_target_supervisor(runtime)

    assert await is_kill_switch_on()
    assert await get_trading_mode() == "paper"
    positions = await runtime.order_manager.get_positions()
    assert "TESTUSD:manual" not in positions  # flattened to flat, locking in the gain


@pytest.mark.asyncio
async def test_manual_re_enable_after_auto_trigger_is_not_immediately_refought():
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )
    await _daily_profit_target_supervisor(runtime)
    runtime.cache.update_orderbook("TESTUSD", bids=[[1199.0, 100]], asks=[[1201.0, 100]], timestamp=2)
    await _daily_profit_target_supervisor(runtime)
    assert await is_kill_switch_on()

    # User deliberately turns it back off, choosing to keep trading today.
    await set_kill_switch(False)

    # Still technically past target (equity hasn't dropped) -- must NOT
    # immediately flip back on.
    await _daily_profit_target_supervisor(runtime)
    assert not await is_kill_switch_on()


@pytest.mark.asyncio
async def test_already_engaged_kill_switch_is_left_alone():
    runtime = _FakeRuntime(_settings())
    await set_kill_switch(True)

    await _daily_profit_target_supervisor(runtime)  # should no-op entirely (early return)

    assert await is_kill_switch_on()


@pytest.mark.asyncio
async def test_combined_realized_plus_unrealized_profit_triggers_the_target_even_when_neither_alone_would():
    """The exact scenario the mark-to-market design exists for: some of
    today's gain is already realized (a partial close booked into cash
    balance) and the rest is still unrealized (still open) -- NEITHER piece
    alone reaches the 10% target, but their SUM does, and the supervisor
    must catch that combined total, not just whichever piece it happens to
    look at.
    """
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )
    await _daily_profit_target_supervisor(runtime)  # establishes starting_equity (~10,000) at entry price
    assert not await is_kill_switch_on()

    # Price rallies +10% (1000 -> 1100). Realize just over half the
    # position here -- books a REALIZED gain of 600*0.01*100 = $600 (6% of
    # starting equity), well under the 10%/$1,000 target on its own.
    runtime.cache.update_orderbook("TESTUSD", bids=[[1099.0, 100]], asks=[[1101.0, 100]], timestamp=2)
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="sell", size=600, order_type="market", atr=1.0, quality=1.0),
    )
    await _daily_profit_target_supervisor(runtime)
    assert not await is_kill_switch_on()  # realized-only gain alone doesn't reach the target

    # Price rallies further to 1200. The remaining 400 contracts (entry
    # price unchanged at 1000 from the partial close) now carry an
    # UNREALIZED gain of 400*0.01*200 = $800 (8% of starting equity) --
    # also under the target on its own.
    runtime.cache.update_orderbook("TESTUSD", bids=[[1199.0, 100]], asks=[[1201.0, 100]], timestamp=3)

    positions = await runtime.order_manager.get_positions()
    assert positions["TESTUSD:manual"].size == 400  # confirms the partial-close bookkeeping above

    # Combined: $600 realized + $800 unrealized = $1,400 = 14% of starting
    # equity -- past the 10% target even though each piece alone was not.
    await _daily_profit_target_supervisor(runtime)

    assert await is_kill_switch_on()
    positions = await runtime.order_manager.get_positions()
    assert "TESTUSD:manual" not in positions  # flattened, locking in the combined gain


@pytest.mark.asyncio
async def test_profit_target_flag_is_independent_of_the_loss_flag():
    """The two supervisors persist DIFFERENT flags -- a day where the profit
    target auto-triggered must not be mistaken for a loss-breach auto-trigger
    (or vice versa) by whichever runs next.
    """
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )
    await _daily_profit_target_supervisor(runtime)
    runtime.cache.update_orderbook("TESTUSD", bids=[[1199.0, 100]], asks=[[1201.0, 100]], timestamp=2)
    await _daily_profit_target_supervisor(runtime)

    today = runtime.risk_manager.daily_pnl.day.isoformat()
    assert await get_state("daily_profit_target_auto_triggered_day") == today
    assert await get_state("daily_kill_switch_auto_triggered_day") is None
