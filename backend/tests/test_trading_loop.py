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
from app.trading_loop import _tick_symbol, drop_incomplete_last_bar, get_live_signals

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
async def test_coinciding_fresh_entries_are_consolidated_into_one_position():
    """The exact scenario reported live: two strategies both open a fresh,
    same-direction position on the same symbol at the same tick (e.g. graha
    and trend_momentum both hitting full conviction on one clean SOL rally).
    Only ONE position should result -- owned by the higher-conviction
    strategy, sized as the SUM of what both independently wanted -- not two
    separate positions paying double margin and fees for one bet.
    """
    await _seed_testusd_candles()
    settings = _consolidation_settings()
    registry = _registry_with_testusd()

    weak_long = _ConstantSignalStrategy("weak_long", 0.5)  # clears entry threshold, lower conviction
    strong_long = _ConstantSignalStrategy("strong_long", 0.9)  # higher conviction -- should become primary
    ensemble = SignalEnsemble(
        [
            WeightedStrategy(weak_long, weight=1.0, enabled=True),
            WeightedStrategy(strong_long, weight=1.0, enabled=True),
        ]
    )

    runtime = _FullFakeRuntime(registry, {"TESTUSD": ensemble}, settings)
    await _tick_symbol(runtime, "TESTUSD")

    positions = await runtime.order_manager.get_positions()
    weak_key = "TESTUSD:weak_long"
    strong_key = "TESTUSD:strong_long"

    assert weak_key not in positions  # consolidated away -- no separate position
    assert strong_key in positions  # the higher-conviction strategy owns the combined position
    assert positions[strong_key].size > 0


@pytest.mark.asyncio
async def test_consolidated_size_matches_the_capped_sum_of_both_strategies_own_targets():
    """The consolidated order must be built from the SUM of what weak_long
    and strong_long each independently wanted -- not just the larger one's
    own size alone -- and THEN run through the exact same per-trade margin
    cap (risk_manager.cap_position_size) any single trade would be, at the
    higher-conviction (primary) strategy's leverage tier.

    Under fixed_margin_contracts, every trade -- including each one of these
    two BEFORE consolidation -- independently targets the full configured
    margin_per_trade_pct of equity. Two coinciding fresh entries summed
    together therefore naturally want roughly double a single trade's normal
    margin budget, so the per-trade cap now genuinely (and correctly) binds
    on the consolidated order every time, rather than being a rare ceiling
    like the old ATR-risk-targeted sizing left it. This test verifies the
    executed size against risk_manager.cap_position_size itself (already
    unit-tested in test_risk_manager.py) rather than re-deriving the cap
    formula by hand, so it can't drift from what production actually applies.
    """
    await _seed_testusd_candles()
    # Realistic (default) leverage bounds -- NOT an artificially huge range.
    # select_leverage rounds its result, and at an extreme max_leverage the
    # rounding granularity is so coarse that ordinary float noise between two
    # independently-run feature/signal pipelines (this test's hand-built one
    # vs _tick_symbol's own) shifts the resulting leverage tier enough to
    # throw off the expected contract count by a lot. At default bounds
    # (10-50x) that noise doesn't move the rounded tier at all.
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
    from app.trading_loop import _compute_member_target

    candles = await load_candles_df("TESTUSD", "15m", limit=400)
    features = build_feature_frame(candles)
    # _tick_symbol prices off runtime.cache.get_last_price (orderbook mid)
    # FIRST, falling back to the candle close only if the cache has no
    # orderbook -- _FullFakeRuntime seeds bids=999/asks=1001, so the real run
    # prices at that 1000.0 mid, not the candle close. Must match here too,
    # or this test's own target/cap computations silently price off a
    # different number than what actually executes.
    price = runtime.cache.get_last_price("TESTUSD")
    equity = await runtime.order_manager.get_equity()  # the real starting equity, before any trade

    _, weak_target, _ = _compute_member_target(
        runtime, product, features, price, {}, equity, strategy_name="weak_long",
        signal_ensemble=SignalEnsemble([WeightedStrategy(weak_long, weight=1.0)]),
    )
    _, strong_target, _ = _compute_member_target(
        runtime, product, features, price, {}, equity, strategy_name="strong_long",
        signal_ensemble=SignalEnsemble([WeightedStrategy(strong_long, weight=1.0)]),
    )
    assert weak_target > 0 and strong_target > 0  # both genuinely want to go long
    naive_sum = weak_target + strong_target

    expected_capped = runtime.order_manager.risk_manager.cap_position_size(
        naive_sum, "TESTUSD", price, product.contract_value, equity,
        current_portfolio_notional_usd=0.0, current_portfolio_margin_usd=0.0, quality=0.9,
    )

    await _tick_symbol(runtime, "TESTUSD")
    positions = await runtime.order_manager.get_positions()
    combined_size = positions["TESTUSD:strong_long"].size

    # The per-trade margin cap must actually be binding here (proving this
    # test exercises the capped path, not accidentally passing because
    # nothing needed clipping) -- otherwise a regression that dropped
    # capping entirely could still pass by coincidence.
    assert expected_capped < naive_sum
    # +-1 tolerance for independent int() truncation across the two paths.
    assert abs(combined_size - expected_capped) <= 1


@pytest.mark.asyncio
async def test_non_coinciding_entries_are_not_consolidated():
    """A strategy managing an EXISTING position must never be swept into
    consolidation just because another strategy opens fresh in the same
    direction at the same tick -- only simultaneous FROM-FLAT entries merge.
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
    assert "TESTUSD:solo_long" in positions
    assert positions["TESTUSD:solo_long"].size > 0
