import pytest

from app.config import Settings
from app.core.delta_client import DeltaRestClient
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.market_cache import MarketDataCache
from app.execution.broker_base import OrderRequest
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.risk.risk_manager import RiskManager


def _settings(**overrides) -> Settings:
    base = dict(delta_api_key="k", delta_api_secret="s")
    base.update(overrides)
    return Settings(**base)


def _registry(settings: Settings) -> SymbolRegistry:
    reg = SymbolRegistry(settings)
    reg._by_symbol["BTCUSD"] = ProductInfo(
        product_id=27,
        symbol="BTCUSD",
        contract_type="perpetual_futures",
        tick_size=0.5,
        contract_value=0.001,
        underlying_asset="BTC",
        state="live",
        default_leverage=100,
        taker_commission_rate=0.0005,
        maker_commission_rate=0.0002,
        spot_index_symbol=".DEXBTUSD",
    )
    reg._by_symbol["SOLUSD"] = ProductInfo(
        product_id=14823,
        symbol="SOLUSD",
        contract_type="perpetual_futures",
        tick_size=0.0001,
        contract_value=1.0,
        underlying_asset="SOL",
        state="live",
        default_leverage=100,
        taker_commission_rate=0.0005,
        maker_commission_rate=0.0002,
        spot_index_symbol=".DESOLUSD",
    )
    return reg


def _cache() -> MarketDataCache:
    cache = MarketDataCache()
    # BTC priced ~830x higher than SOL -- this ratio is what exposes the bug.
    cache.update_orderbook("BTCUSD", bids=[[64_800, 100]], asks=[[64_810, 100]], timestamp=1)
    cache.update_orderbook("SOLUSD", bids=[[78.0, 100]], asks=[[78.05, 100]], timestamp=1)
    return cache


def _order_manager(settings: Settings, cache: MarketDataCache) -> OrderManager:
    registry = _registry(settings)
    risk_manager = RiskManager(settings)
    paper_broker = PaperBroker(registry, cache, settings)
    live_broker = LiveBroker(DeltaRestClient(settings), registry, cache)
    return OrderManager(paper_broker, live_broker, risk_manager, registry, cache)


@pytest.mark.asyncio
async def test_sizing_a_high_priced_symbol_order_is_not_starved_by_a_low_priced_symbols_existing_position():
    """Regression test for the reported bug: BTC/ETH orders were being capped
    to 0 contracts whenever the account already held a position in a much
    lower-priced symbol (e.g. SOL). The portfolio-notional check was
    re-deriving each open position's notional using the NEW order's price
    instead of that position's own symbol's price -- so a SOL position's
    size, multiplied by BTC's ~830x-higher price, looked like a many-million-
    dollar position and instantly exhausted the portfolio notional budget for
    every subsequent BTC/ETH order, regardless of how much room was actually
    left.
    """
    settings = _settings(max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85)
    cache = _cache()
    om = _order_manager(settings, cache)

    # Open a real, sizeable SOL position first (independent strategy slot).
    sol_result = await om.submit(
        OrderRequest(symbol="SOLUSD", side="sell", size=500, order_type="market", strategy="mean_reversion", quality=0.5)
    )
    assert sol_result.state != "rejected"
    assert sol_result.filled_size > 0

    # Now size a BTC order against the SAME account -- it must be judged
    # against the real (correctly-priced) portfolio notional/margin, not an
    # artifact of BTC's price applied to the SOL position's contract count.
    btc_result = await om.submit(
        OrderRequest(symbol="BTCUSD", side="buy", size=900, order_type="market", strategy="trend_momentum", quality=0.3)
    )
    assert btc_result.state != "rejected"
    assert btc_result.filled_size > 0


@pytest.mark.asyncio
async def test_realized_pnl_from_a_closing_fill_feeds_the_daily_loss_tracker():
    """Regression test: RiskManager.record_realized_pnl existed but was never
    called anywhere in production, so the daily-loss kill switch could never
    trip regardless of real losses. OrderManager.submit() must feed every
    non-zero realized PnL from a fill into the tracker.
    """
    settings = _settings(max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85)
    cache = _cache()
    om = _order_manager(settings, cache)

    open_result = await om.submit(
        OrderRequest(symbol="SOLUSD", side="buy", size=100, order_type="market", strategy="trend_momentum", quality=0.5)
    )
    assert open_result.state != "rejected"
    assert om.risk_manager.daily_pnl.realized_pnl == 0.0  # opening a position realizes nothing

    # Price rallies, then the position is closed at a profit.
    cache.update_orderbook("SOLUSD", bids=[[85.0, 100]], asks=[[85.05, 100]], timestamp=2)
    close_result = await om.submit(
        OrderRequest(symbol="SOLUSD", side="sell", size=100, order_type="market", strategy="trend_momentum", quality=0.5)
    )
    assert close_result.state != "rejected"
    assert close_result.realized_pnl > 0
    assert om.risk_manager.daily_pnl.realized_pnl == pytest.approx(close_result.realized_pnl)


@pytest.mark.asyncio
async def test_max_concurrent_positions_blocks_a_new_entry_past_the_cap():
    """Account-wide hard cap on simultaneously open positions -- with 7
    strategies x 3 symbols each trading its own independent position, the
    margin/leverage caps alone still allow up to ~8-9 concurrent trades.
    This is a separate, smaller, easier-to-reason-about limit on top of that.
    """
    settings = _settings(max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85, max_concurrent_positions=2)
    cache = _cache()
    om = _order_manager(settings, cache)

    first = await om.submit(
        OrderRequest(symbol="SOLUSD", side="buy", size=50, order_type="market", strategy="strategy_a", quality=0.5)
    )
    assert first.state != "rejected"

    second = await om.submit(
        OrderRequest(symbol="BTCUSD", side="buy", size=50, order_type="market", strategy="strategy_b", quality=0.5)
    )
    assert second.state != "rejected"

    # A THIRD distinct (symbol, strategy) position, with the cap already at 2.
    third = await om.submit(
        OrderRequest(symbol="BTCUSD", side="buy", size=50, order_type="market", strategy="strategy_c", quality=0.5)
    )
    assert third.state == "rejected"
    assert "max concurrent positions" in third.reason


@pytest.mark.asyncio
async def test_extending_an_existing_position_is_not_blocked_by_the_concurrent_position_cap():
    """Adding to (or flipping) an ALREADY-open position must never be treated
    as a new slot against the cap -- only a genuine open-from-flat entry
    should be blocked once the cap is reached.
    """
    settings = _settings(max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85, max_concurrent_positions=1)
    cache = _cache()
    om = _order_manager(settings, cache)

    opened = await om.submit(
        OrderRequest(symbol="SOLUSD", side="buy", size=50, order_type="market", strategy="strategy_a", quality=0.5)
    )
    assert opened.state != "rejected"

    # Cap is already at its limit (1), but this EXTENDS the same existing
    # position rather than opening a new one -- must still go through.
    extended = await om.submit(
        OrderRequest(symbol="SOLUSD", side="buy", size=25, order_type="market", strategy="strategy_a", quality=0.5)
    )
    assert extended.state != "rejected"


@pytest.mark.asyncio
async def test_closing_a_position_frees_a_slot_for_a_new_one_under_the_cap():
    settings = _settings(max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85, max_concurrent_positions=1)
    cache = _cache()
    om = _order_manager(settings, cache)

    await om.submit(
        OrderRequest(symbol="SOLUSD", side="buy", size=50, order_type="market", strategy="strategy_a", quality=0.5)
    )

    # At the cap -- a new, distinct position is rejected.
    blocked = await om.submit(
        OrderRequest(symbol="BTCUSD", side="buy", size=50, order_type="market", strategy="strategy_b", quality=0.5)
    )
    assert blocked.state == "rejected"

    # Close the SOL position (reduce-only -- never blocked by the cap).
    closed = await om.submit(
        OrderRequest(
            symbol="SOLUSD", side="sell", size=50, order_type="market", strategy="strategy_a", reduce_only=True
        )
    )
    assert closed.state != "rejected"

    # The slot is free again -- the BTC entry now succeeds.
    now_allowed = await om.submit(
        OrderRequest(symbol="BTCUSD", side="buy", size=50, order_type="market", strategy="strategy_b", quality=0.5)
    )
    assert now_allowed.state != "rejected"


@pytest.mark.asyncio
async def test_close_position_flattens_exactly_one_symbol_strategy_pair():
    """Regression test: disabling a strategy orphans whatever position it was
    holding (the trading loop skips disabled members entirely), so there was
    no way to deliberately close just that one position without a full
    flatten_all() of the whole account.
    """
    settings = _settings(max_leverage=50, max_portfolio_leverage=20, max_margin_utilization=0.85)
    cache = _cache()
    om = _order_manager(settings, cache)

    await om.submit(
        OrderRequest(symbol="SOLUSD", side="buy", size=50, order_type="market", strategy="strategy_a", quality=0.5)
    )
    await om.submit(
        OrderRequest(symbol="BTCUSD", side="buy", size=50, order_type="market", strategy="strategy_b", quality=0.5)
    )

    result = await om.close_position("SOLUSD", "strategy_a")
    assert result is not None
    assert result.state != "rejected"

    positions = await om.get_positions()
    assert "SOLUSD:strategy_a" not in positions
    assert "BTCUSD:strategy_b" in positions  # untouched


@pytest.mark.asyncio
async def test_close_position_returns_none_when_nothing_is_open():
    settings = _settings()
    cache = _cache()
    om = _order_manager(settings, cache)

    result = await om.close_position("SOLUSD", "strategy_a")
    assert result is None
