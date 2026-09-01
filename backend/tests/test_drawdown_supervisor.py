import pytest

from app.config import Settings
from app.core.delta_client import DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.app_state import get_state, is_kill_switch_on, set_kill_switch, set_state
from app.data.market_cache import MarketDataCache
from app.execution.broker_base import OrderRequest
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.risk.risk_manager import RiskManager
from app.scheduler import _daily_kill_switch_reset, _drawdown_supervisor


def _settings(**overrides) -> Settings:
    base = dict(delta_api_key="k", delta_api_secret="s", max_drawdown_pct=0.20)
    base.update(overrides)
    return Settings(**base)


def _registry() -> SymbolRegistry:
    reg = SymbolRegistry(_settings())
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


@pytest.mark.asyncio
async def test_high_water_mark_ratchets_up_and_never_down():
    rm = RiskManager(_settings())

    await rm.check_max_drawdown(10_000)
    assert float(await get_state("equity_high_water_mark")) == 10_000

    await rm.check_max_drawdown(12_000)  # new peak
    assert float(await get_state("equity_high_water_mark")) == 12_000

    await rm.check_max_drawdown(11_000)  # pullback -- peak must NOT follow it down
    assert float(await get_state("equity_high_water_mark")) == 12_000


@pytest.mark.asyncio
async def test_drawdown_under_the_limit_is_allowed():
    rm = RiskManager(_settings(max_drawdown_pct=0.20))
    await rm.check_max_drawdown(10_000)

    ok, reason = await rm.check_max_drawdown(8_500)  # -15%, under the 20% limit
    assert ok
    assert reason == ""


@pytest.mark.asyncio
async def test_drawdown_at_the_limit_breaches():
    rm = RiskManager(_settings(max_drawdown_pct=0.20))
    await rm.check_max_drawdown(10_000)

    ok, reason = await rm.check_max_drawdown(8_000)  # exactly -20%
    assert not ok
    assert "Max drawdown breached" in reason


@pytest.mark.asyncio
async def test_multi_day_bleed_that_no_daily_limit_would_ever_catch():
    """The exact hole this limit exists to close: three consecutive days
    each losing 9% -- every one of them individually UNDER a 10% daily loss
    limit, so check_daily_loss never fires once -- while the account ends up
    24.6% below its peak.
    """
    rm = RiskManager(_settings(max_drawdown_pct=0.20, daily_loss_limit_pct=0.10))
    equity = 10_000.0
    await rm.check_max_drawdown(equity)

    breached_daily = False
    for _ in range(3):
        start_of_day = equity
        equity *= 0.91  # -9% on the day
        daily_ok, _ = await rm.check_daily_loss(equity)
        rm.daily_pnl.starting_equity = start_of_day  # re-anchor as a real new day would
        daily_ok, _ = await rm.check_daily_loss(equity)
        if not daily_ok:
            breached_daily = True
        rm.daily_pnl.starting_equity = equity

    assert not breached_daily, "no single day should have breached the daily limit"

    ok, reason = await rm.check_max_drawdown(equity)
    assert not ok, "but the cumulative drawdown must breach"
    assert "Max drawdown breached" in reason


@pytest.mark.asyncio
async def test_supervisor_halts_and_flattens_on_breach():
    runtime = _FakeRuntime(_settings(max_drawdown_pct=0.10))
    await runtime.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=1000, order_type="market", atr=1.0, quality=1.0),
    )

    await _drawdown_supervisor(runtime)  # establishes the high-water mark
    assert not await is_kill_switch_on()

    # Price craters -- a real mark-to-market drawdown well past the 10% limit.
    runtime.cache.update_orderbook("TESTUSD", bids=[[500.0, 100]], asks=[[502.0, 100]], timestamp=2)
    await _drawdown_supervisor(runtime)

    assert await is_kill_switch_on()
    assert await get_state("drawdown_halt_active") == "on"
    positions = await runtime.order_manager.get_positions()
    assert "TESTUSD:manual" not in positions  # flattened


@pytest.mark.asyncio
async def test_drawdown_halt_survives_the_daily_kill_switch_reset():
    """A drawdown is peak-to-trough across days, so a new calendar day is
    NOT evidence it recovered -- _daily_kill_switch_reset must leave it
    alone, or the one multi-day control silently becomes a daily one.
    """
    await set_kill_switch(True)
    await set_state("kill_switch_engaged_day", "2020-01-01")  # stale: normally would be reset
    await set_state("drawdown_halt_active", "on")

    await _daily_kill_switch_reset()

    assert await is_kill_switch_on(), "drawdown halt must NOT be cleared by the daily reset"


@pytest.mark.asyncio
async def test_daily_reset_still_works_when_no_drawdown_halt_is_active():
    await set_kill_switch(True)
    await set_state("kill_switch_engaged_day", "2020-01-01")
    await set_state("drawdown_halt_active", "off")

    await _daily_kill_switch_reset()

    assert not await is_kill_switch_on()
