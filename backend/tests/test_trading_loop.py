import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.core.delta_client import DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import Candle
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.risk.risk_manager import RiskManager
from app.strategy.base import Strategy
from app.strategy.ensemble import SignalEnsemble, WeightedStrategy
from app.strategy.trend_momentum import TrendMomentumStrategy
from app.execution.broker_base import OrderRequest
from app.trading_loop import (
    CONSOLIDATED_STRATEGY,
    _compute_member_target,
    _tick_symbol,
    drop_incomplete_last_bar,
    get_live_signals,
)

RES = 900  # 15m


def _candles(bar_times: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": bar_times,
            "open": [1.0] * len(bar_times),
            "high": [1.0] * len(bar_times),
            "low": [1.0] * len(bar_times),
            "close": [1.0] * len(bar_times),
            "volume": [1.0] * len(bar_times),
        }
    )


def test_drops_the_last_bar_when_it_is_still_forming():
    """Regression test: the live case that motivated this -- the newest bar
    started only ~2 minutes ago, well inside its 15-minute window.
    """
    bar_times = [0, 900, 1800]
    now_ts = 1800 + 120  # last bar started 120s ago, needs 900s to close
    result = drop_incomplete_last_bar(_candles(bar_times), RES, now_ts)
    assert list(result["time"]) == [0, 900]


def test_keeps_the_last_bar_once_it_has_actually_closed():
    bar_times = [0, 900, 1800]
    now_ts = 1800 + 900  # exactly one full resolution period has elapsed
    result = drop_incomplete_last_bar(_candles(bar_times), RES, now_ts)
    assert list(result["time"]) == [0, 900, 1800]


def test_keeps_the_last_bar_well_after_it_closed():
    bar_times = [0, 900, 1800]
    now_ts = 1800 + 900 + 500  # long since closed
    result = drop_incomplete_last_bar(_candles(bar_times), RES, now_ts)
    assert list(result["time"]) == [0, 900, 1800]


def test_empty_input_is_a_no_op():
    empty = _candles([])
    result = drop_incomplete_last_bar(empty, RES, now_ts=12345)
    assert len(result) == 0


def test_defaults_to_wall_clock_now_when_not_given():
    import datetime as dt

    real_now = dt.datetime.now(dt.timezone.utc).timestamp()
    # A bar that "started" at the real current time is, by definition, still
    # forming -- must be dropped when now_ts is left to default.
    result = drop_incomplete_last_bar(_candles([0, int(real_now)]))
    assert list(result["time"]) == [0]


class _FakeRuntime:
    def __init__(self, registry, ensembles) -> None:
        self.registry = registry
        self.ensembles = ensembles
        self.cache = MarketDataCache()


def _synthetic_candle_rows(symbol: str, n: int = 300, seed: int = 3) -> list[Candle]:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.004, n)
    close = 1_000 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(10, 100, n)
    time = np.arange(n) * 900
    return [
        Candle(
            symbol=symbol, resolution="15m", time=int(time[i]),
            open=float(open_[i]), high=float(high[i]), low=float(low[i]),
            close=float(close[i]), volume=float(volume[i]),
        )
        for i in range(n)
    ]


def _registry_with_testusd() -> SymbolRegistry:
    reg = SymbolRegistry(Settings(delta_api_key="k", delta_api_secret="s"))
    reg._by_symbol["TESTUSD"] = ProductInfo(
        product_id=1, symbol="TESTUSD", contract_type="perpetual_futures", tick_size=0.5,
        contract_value=0.01, underlying_asset="TEST", state="live", default_leverage=10,
        taker_commission_rate=0.0005, maker_commission_rate=0.0002, spot_index_symbol="",
    )
    return reg


@pytest.mark.asyncio
async def test_get_live_signals_returns_every_member_including_disabled_but_not_the_ensemble():
    async with SessionLocal() as session:
        for row in _synthetic_candle_rows("TESTUSD"):
            session.add(row)
        await session.commit()

    registry = _registry_with_testusd()
    ensemble = SignalEnsemble(
        [
            WeightedStrategy(TrendMomentumStrategy(), weight=1.0, enabled=True),
            WeightedStrategy(TrendMomentumStrategy(), weight=0.0, enabled=False),
        ]
    )
    # Give the two members distinct names so the response is unambiguous --
    # WeightedStrategy doesn't rename the underlying strategy, so patch each
    # instance's `name` directly for this test.
    ensemble.members[0].strategy.name = "enabled_one"
    ensemble.members[1].strategy.name = "disabled_one"

    runtime = _FakeRuntime(registry, {"TESTUSD": ensemble})

    signals = await get_live_signals(runtime, "TESTUSD")

    by_name = {s["strategy"]: s for s in signals}
    # The blended ensemble no longer trades its own position (see
    # run_trading_tick's docstring), so it has no signal to show here.
    assert "ensemble" not in by_name

    assert by_name["enabled_one"]["enabled"] is True
    assert by_name["enabled_one"]["signal"] is not None

    assert by_name["disabled_one"]["enabled"] is False
    # Disabled strategies still get a real computed signal, not a stub --
    # the dashboard should be able to show "what would this say right now".
    assert by_name["disabled_one"]["signal"] is not None


@pytest.mark.asyncio
async def test_get_live_signals_returns_empty_list_for_unknown_symbol():
    runtime = _FakeRuntime(_registry_with_testusd(), {})
    signals = await get_live_signals(runtime, "TESTUSD")
    assert signals == []


class _ConstantSignalStrategy(Strategy):
    """Deterministic test double: always signals the same value, regardless
    of the feature data -- lets a test force two strategies into a
    guaranteed-coinciding fresh entry without depending on real market
    randomness lining up.
    """

    def __init__(self, name: str, signal_value: float) -> None:
        self.name = name
        self.signal_value = signal_value

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.signal_value, index=df.index)

    def required_warmup_bars(self) -> int:
        return 5


class _FullFakeRuntime:
    """Unlike _FakeRuntime above (used for the read-only get_live_signals
    tests), this one has a REAL order_manager wired to a real PaperBroker --
    needed to verify actual positions get created by _tick_symbol.
    """

    def __init__(self, registry, ensembles, settings) -> None:
        self.registry = registry
        self.ensembles = ensembles
        self.settings = settings
        self.cache = MarketDataCache()
        self.cache.update_orderbook("TESTUSD", bids=[[999.0, 100]], asks=[[1001.0, 100]], timestamp=1)
        risk_manager = RiskManager(settings)
        paper_broker = PaperBroker(registry, self.cache, settings)
        live_broker = LiveBroker(DeltaRestClient(settings), registry, self.cache)
        self.order_manager = OrderManager(paper_broker, live_broker, risk_manager, registry, self.cache)


def _consolidation_settings() -> Settings:
    return Settings(
        delta_api_key="k",
        delta_api_secret="s",
        max_leverage=50,
        max_portfolio_leverage=20,
        max_margin_utilization=0.85,
        # Loosened from the 0.10 default: these tests check the
        # CONSOLIDATION arithmetic (combined size > solo size), which the
        # normal per-trade cap would otherwise mask by clipping both
        # scenarios down to the same ceiling -- per-trade capping itself is
        # already covered by test_order_manager.py.
        max_margin_per_trade_pct=1.0,
    )


async def _seed_testusd_candles(n: int = 300, seed: int = 3) -> None:
    async with SessionLocal() as session:
        for row in _synthetic_candle_rows("TESTUSD", n=n, seed=seed):
            session.add(row)
        await session.commit()


@pytest.mark.asyncio
async def test_multiple_strategies_produce_exactly_one_blended_position():
    """The core of the 2026-08-31 consolidation: however many strategies are
    enabled, a symbol gets exactly ONE position, built from their weighted
    blend. Previously each member held its own -- which measured out at 0.733
    mean pairwise correlation, i.e. paying three sets of fees and three lots
    of margin for ~1.3 effective bets.
    """
    await _seed_testusd_candles()
    settings = _consolidation_settings()
    registry = _registry_with_testusd()

    weak_long = _ConstantSignalStrategy("weak_long", 0.5)
    strong_long = _ConstantSignalStrategy("strong_long", 0.9)
    ensemble = SignalEnsemble(
        [
            WeightedStrategy(weak_long, weight=1.0, enabled=True),
            WeightedStrategy(strong_long, weight=1.0, enabled=True),
        ]
    )

    runtime = _FullFakeRuntime(registry, {"TESTUSD": ensemble}, settings)
    await _tick_symbol(runtime, "TESTUSD")

    positions = await runtime.order_manager.get_positions()
    testusd = {k: v for k, v in positions.items() if k.startswith("TESTUSD:") and v.size != 0}

    assert len(testusd) == 1, f"expected exactly one blended position, got {list(testusd)}"
    assert f"TESTUSD:{CONSOLIDATED_STRATEGY}" in testusd
    assert "TESTUSD:weak_long" not in positions
    assert "TESTUSD:strong_long" not in positions


@pytest.mark.asyncio
async def test_legacy_per_member_positions_are_closed_on_consolidation():
    """Positions opened under the OLD per-member design are no longer managed
    by the loop, so they must be closed rather than orphaned to drift until
    their stop or target happens to fire.
    """
    await _seed_testusd_candles()
    settings = _consolidation_settings()
    registry = _registry_with_testusd()

    strat = _ConstantSignalStrategy("trend_momentum", 0.8)
    ensemble = SignalEnsemble([WeightedStrategy(strat, weight=1.0, enabled=True)])
    runtime = _FullFakeRuntime(registry, {"TESTUSD": ensemble}, settings)

    # A leftover position tagged with an individual member's name.
    await runtime.order_manager.paper_broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=50, order_type="market",
                     strategy="trend_momentum", atr=1.0, quality=1.0)
    )
    assert (await runtime.order_manager.get_positions())["TESTUSD:trend_momentum"].size == 50

    await _tick_symbol(runtime, "TESTUSD")

    positions = await runtime.order_manager.get_positions()
    assert "TESTUSD:trend_momentum" not in positions, "legacy member position was left orphaned"


@pytest.mark.asyncio
async def test_consolidated_position_is_sized_as_one_trade_not_a_sum_of_members():
    """Under consolidation the blended signal produces ONE ordinary trade, so
    it must be sized like one -- a single margin_per_trade_pct allocation at
    the blend's own conviction -- not the sum of what each member would have
    wanted separately (which is what the old per-member design produced, and
    why the per-trade margin cap used to bind on essentially every entry).
    """
    await _seed_testusd_candles()
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    registry = _registry_with_testusd()
    product = registry.get("TESTUSD")

    weak_long = _ConstantSignalStrategy("weak_long", 0.5)
    strong_long = _ConstantSignalStrategy("strong_long", 0.9)
    ensemble = SignalEnsemble(
        [
            WeightedStrategy(weak_long, weight=1.0, enabled=True),
            WeightedStrategy(strong_long, weight=1.0, enabled=True),
        ]
    )
    runtime = _FullFakeRuntime(registry, {"TESTUSD": ensemble}, settings)

    from app.data.query import load_candles_df
    from app.features.feature_pipeline import build_feature_frame

    candles = await load_candles_df("TESTUSD", "4h", limit=400)
    if len(candles) < 50:
        candles = await load_candles_df("TESTUSD", "15m", limit=400)
    features = build_feature_frame(candles)
    price = runtime.cache.get_last_price("TESTUSD")
    equity = await runtime.order_manager.get_equity()

    # What each member would have targeted ON ITS OWN under the old design.
    _, weak_target, _ = _compute_member_target(
        runtime, product, features, price, {}, equity, strategy_name="weak_long",
        signal_ensemble=SignalEnsemble([WeightedStrategy(weak_long, weight=1.0)]),
    )
    _, strong_target, _ = _compute_member_target(
        runtime, product, features, price, {}, equity, strategy_name="strong_long",
        signal_ensemble=SignalEnsemble([WeightedStrategy(strong_long, weight=1.0)]),
    )
    naive_sum = abs(weak_target) + abs(strong_target)

    await _tick_symbol(runtime, "TESTUSD")
    positions = await runtime.order_manager.get_positions()
    size = abs(positions[f"TESTUSD:{CONSOLIDATED_STRATEGY}"].size)

    assert size > 0
    # The whole point of consolidating: one bet's worth of exposure, not two.
    assert size < naive_sum, f"consolidated size {size} should be below the old summed size {naive_sum}"


@pytest.mark.asyncio
async def test_position_is_tagged_consolidated_not_by_member_name():
    """Every live position now carries the single CONSOLIDATED_STRATEGY tag,
    never an individual member's name -- even when only one strategy is
    enabled. Positions are keyed by "{symbol}:{strategy}", so a per-member
    tag here would mean the loop had opened a per-member position again.
    """
    async with SessionLocal() as session:
        for row in _synthetic_candle_rows("TESTUSD", n=300, seed=11):
            session.add(row)
        await session.commit()

    settings = Settings(delta_api_key="k", delta_api_secret="s", max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85)
    registry = _registry_with_testusd()

    only_strategy = _ConstantSignalStrategy("solo_long", 0.8)
    ensemble = SignalEnsemble([WeightedStrategy(only_strategy, weight=1.0, enabled=True)])
    runtime = _FullFakeRuntime(registry, {"TESTUSD": ensemble}, settings)

    await _tick_symbol(runtime, "TESTUSD")

    positions = await runtime.order_manager.get_positions()
    assert "TESTUSD:solo_long" not in positions
    assert f"TESTUSD:{CONSOLIDATED_STRATEGY}" in positions
    assert positions[f"TESTUSD:{CONSOLIDATED_STRATEGY}"].size > 0
