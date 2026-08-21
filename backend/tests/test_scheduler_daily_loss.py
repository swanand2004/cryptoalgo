import pytest

from app.config import Settings
from app.core.delta_client import DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.app_state import get_trading_mode, is_kill_switch_on, set_kill_switch
from app.data.market_cache import MarketDataCache
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.execution.broker_base import OrderRequest
from app.risk.risk_manager import RiskManager
from app.scheduler import _daily_loss_supervisor


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
    base = dict(delta_api_key="k", delta_api_secret="s", max_daily_loss_usd=100)
    base.update(overrides)
    return Settings(**base)


@pytest.mark.asyncio
async def test_no_breach_leaves_kill_switch_off_and_nothing_flattened():
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )

    await _daily_loss_supervisor(runtime)  # establishes starting_equity, no breach yet
    await _daily_loss_supervisor(runtime)  # price unchanged -- still no breach

    assert not await is_kill_switch_on()
    positions = await runtime.order_manager.get_positions()
    assert positions["TESTUSD:manual"].size == 1000  # untouched


@pytest.mark.asyncio
async def test_breach_auto_engages_kill_switch_and_flattens():
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )

    await _daily_loss_supervisor(runtime)  # establishes today's starting equity
    assert not await is_kill_switch_on()

    # Price craters -- a real mark-to-market loss on the still-open position,
    # well past the 100 USD daily limit, with zero trades ever closed.
    runtime.cache.update_orderbook("TESTUSD", bids=[[899.0, 100]], asks=[[901.0, 100]], timestamp=2)

    await _daily_loss_supervisor(runtime)

    assert await is_kill_switch_on()
    assert await get_trading_mode() == "paper"
    positions = await runtime.order_manager.get_positions()
    assert "TESTUSD:manual" not in positions  # flattened to flat


@pytest.mark.asyncio
async def test_manual_re_enable_after_auto_trigger_is_not_immediately_refought():
    """The core fix: a deliberate manual re-enable later the same day must
    be respected, not re-halted on the very next 15s tick just because the
    account is still down for the day.
    """
    runtime = _FakeRuntime(_settings())
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )
    await _daily_loss_supervisor(runtime)
    runtime.cache.update_orderbook("TESTUSD", bids=[[899.0, 100]], asks=[[901.0, 100]], timestamp=2)
    await _daily_loss_supervisor(runtime)
    assert await is_kill_switch_on()

    # User deliberately turns it back off, choosing to keep trading today.
    await set_kill_switch(False)

    # Still technically breached (equity hasn't recovered) -- must NOT
    # immediately flip back on.
    await _daily_loss_supervisor(runtime)
    assert not await is_kill_switch_on()


@pytest.mark.asyncio
async def test_already_engaged_kill_switch_is_left_alone():
    runtime = _FakeRuntime(_settings())
    await set_kill_switch(True)

    await _daily_loss_supervisor(runtime)  # should no-op entirely (early return)

    assert await is_kill_switch_on()
