import pytest
from sqlalchemy import select

from app.config import Settings
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import ClosedTrade, Position
from app.execution.broker_base import OrderRequest
from app.execution.paper_broker import PaperBroker


@pytest.fixture(autouse=True)
def _enable_atr_brackets(monkeypatch):
    """This file tests the broker's stop-loss/take-profit machinery, which is
    disabled by default from 2026-09-07 (see Settings.atr_brackets_enabled:
    with the live trend_ladder strategy it backtests at -100% against +968%
    on signal exits alone, because a stop closes a position the unchanged
    signal immediately re-opens).

    That is a STRATEGY-level policy, not a defect in the machinery, and the
    machinery must keep working for whenever a strategy wants it. So the whole
    module opts in, rather than each test asserting against a disabled
    feature.
    """
    monkeypatch.setenv("ATR_BRACKETS_ENABLED", "true")


def _registry() -> SymbolRegistry:
    reg = SymbolRegistry(Settings(delta_api_key="k", delta_api_secret="s"))
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
    return reg


def _cache_with_book(bid: float, ask: float) -> MarketDataCache:
    cache = MarketDataCache()
    cache.update_orderbook("BTCUSD", bids=[[bid, 100]], asks=[[ask, 100]], timestamp=1)
    return cache


@pytest.mark.asyncio
async def test_market_buy_then_sell_realizes_expected_pnl():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    buy_result = await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market"))
    assert buy_result.state == "closed"
    assert buy_result.filled_size == 10

    positions = await broker.get_positions()
    assert positions["BTCUSD:manual"].size == 10

    # Price rallies -- move the whole book up before closing.
    cache.update_orderbook("BTCUSD", bids=[[51_990, 100]], asks=[[52_010, 100]], timestamp=2)

    sell_result = await broker.place_order(OrderRequest(symbol="BTCUSD", side="sell", size=10, order_type="market"))
    assert sell_result.state == "closed"

    positions = await broker.get_positions()
    assert "BTCUSD:manual" not in positions  # flat positions are omitted

    equity = await broker.get_equity()
    # Started at 10_000, bought near 50_010(+slip), sold near 51_990(-slip): should be net positive.
    assert equity > 10_000


@pytest.mark.asyncio
async def test_resting_limit_order_fills_when_price_touches():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    result = await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=5, order_type="limit", limit_price=49_500)
    )
    assert result.state == "open"

    positions = await broker.get_positions()
    assert "BTCUSD:manual" not in positions

    # Market trades down through our resting limit price.
    cache.update_orderbook("BTCUSD", bids=[[49_400, 100]], asks=[[49_450, 100]], timestamp=2)
    await broker.check_resting_orders()

    positions = await broker.get_positions()
    assert positions["BTCUSD:manual"].size == 5


@pytest.mark.asyncio
async def test_equity_reflects_unrealized_pnl():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market"))
    equity_before = await broker.get_equity()

    cache.update_orderbook("BTCUSD", bids=[[52_000, 100]], asks=[[52_020, 100]], timestamp=2)
    equity_after = await broker.get_equity()

    assert equity_after > equity_before


async def _closed_trades() -> list[ClosedTrade]:
    async with SessionLocal() as session:
        result = await session.execute(select(ClosedTrade).order_by(ClosedTrade.id))
        return list(result.scalars().all())


async def _position(strategy: str = "manual") -> Position:
    async with SessionLocal() as session:
        result = await session.execute(
            select(Position).where(
                Position.symbol == "BTCUSD", Position.mode == "paper", Position.strategy == strategy
            )
        )
        return result.scalar_one()


@pytest.mark.asyncio
async def test_full_close_emits_a_closed_trade_with_entry_and_exit_metadata():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum")
    )
    pos = await _position("trend_momentum")
    assert pos.strategy == "trend_momentum"
    assert pos.entry_time is not None
    # OrderRequest.quality defaults to 1.0 (full conviction) when unspecified,
    # which selects the leverage ceiling (max_leverage, 50x by default).
    assert pos.leverage == 50.0

    cache.update_orderbook("BTCUSD", bids=[[51_990, 100]], asks=[[52_010, 100]], timestamp=2)
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="sell", size=10, order_type="market", strategy="trend_momentum")
    )

    trades = await _closed_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade.symbol == "BTCUSD"
    assert trade.side == "long"
    assert trade.strategy == "trend_momentum"
    assert trade.size == 10
    assert trade.entry_time is not None and trade.exit_time is not None
    assert trade.exit_time >= trade.entry_time
    assert trade.exit_reason == "signal"
    assert trade.realized_pnl > 0  # price rallied while long

    pos_after = await _position("trend_momentum")
    assert pos_after.size == 0
    assert pos_after.strategy == "trend_momentum"  # strategy is the row's identity, never cleared
    assert pos_after.entry_time is None


@pytest.mark.asyncio
async def test_partial_close_emits_closed_trade_and_keeps_remainder_open():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market"))
    await broker.place_order(OrderRequest(symbol="BTCUSD", side="sell", size=4, order_type="market"))

    trades = await _closed_trades()
    assert len(trades) == 1
    assert trades[0].size == 4

    pos = await _position()
    assert pos.size == 6
    assert pos.entry_time is not None  # remainder keeps original entry metadata


@pytest.mark.asyncio
async def test_flip_through_flat_closes_old_side_and_opens_new_entry():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="mean_reversion")
    )
    # Sell 15 (same strategy): closes the 10 long AND opens a new 5-short position in one fill.
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="sell", size=15, order_type="market", strategy="mean_reversion")
    )

    trades = await _closed_trades()
    assert len(trades) == 1
    assert trades[0].side == "long"
    assert trades[0].size == 10
    assert trades[0].strategy == "mean_reversion"

    pos = await _position("mean_reversion")
    assert pos.size == -5
    assert pos.strategy == "mean_reversion"
    assert pos.entry_time is not None


@pytest.mark.asyncio
async def test_different_strategies_hold_fully_independent_positions_on_same_symbol():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum")
    )
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="sell", size=6, order_type="market", strategy="mean_reversion")
    )

    positions = await broker.get_positions()
    assert positions["BTCUSD:trend_momentum"].size == 10
    assert positions["BTCUSD:mean_reversion"].size == -6

    # Closing one doesn't touch the other.
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="sell", size=10, order_type="market", strategy="trend_momentum")
    )
    positions = await broker.get_positions()
    assert "BTCUSD:trend_momentum" not in positions
    assert positions["BTCUSD:mean_reversion"].size == -6


@pytest.mark.asyncio
async def test_leverage_scales_with_trade_quality_not_notional():
    """Leverage is a genuine tier picked by conviction (quality), NOT derived
    from how big the position happens to be -- a tiny position at full
    conviction gets the ceiling tier, and a huge one at minimal conviction
    still only gets the floor tier.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    broker = PaperBroker(_registry(), cache, settings)

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=1, order_type="market", strategy="a", quality=1.0)
    )
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=400, order_type="market", strategy="b", quality=0.0)
    )
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="c", quality=0.5)
    )

    full_conviction = await _position("a")
    zero_conviction = await _position("b")
    mid_conviction = await _position("c")

    assert full_conviction.leverage == settings.max_leverage
    assert zero_conviction.leverage == settings.min_leverage
    assert settings.min_leverage < mid_conviction.leverage < settings.max_leverage


@pytest.mark.asyncio
async def test_fees_accumulate_while_position_open_and_reset_on_full_close():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    result1 = await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market"))
    pos = await _position()
    assert pos.fees_paid == pytest.approx(result1.fee)

    result2 = await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=5, order_type="market"))
    pos = await _position()
    assert pos.fees_paid == pytest.approx(result1.fee + result2.fee)

    await broker.place_order(OrderRequest(symbol="BTCUSD", side="sell", size=15, order_type="market"))
    pos = await _position()
    assert pos.fees_paid == 0.0

    trades = await _closed_trades()
    assert len(trades) == 1
    # Closed trade's fee total includes both the accumulated entry fees and the exit fee.
    assert trades[0].fees > result1.fee + result2.fee


@pytest.mark.asyncio
async def test_fees_split_correctly_on_partial_close():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    entry_result = await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market"))
    # Close 40% of the position -- 60% of the accumulated entry fee should remain.
    await broker.place_order(OrderRequest(symbol="BTCUSD", side="sell", size=4, order_type="market"))

    pos = await _position()
    assert pos.size == 6
    assert pos.fees_paid == pytest.approx(entry_result.fee * 0.6, rel=1e-6)


@pytest.mark.asyncio
async def test_stop_loss_and_take_profit_computed_on_entry_when_atr_given():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos = await _position("trend_momentum")
    # Derived from config, so this tests that the broker APPLIES the configured
    # multiples rather than pinning whatever they currently happen to be.
    cfg = Settings(delta_api_key="k", delta_api_secret="s")
    assert pos.stop_loss_price == pytest.approx(pos.entry_price - cfg.atr_stop_multiple * 200.0)
    assert pos.take_profit_price == pytest.approx(pos.entry_price + cfg.atr_take_profit_multiple * 200.0)


@pytest.mark.asyncio
async def test_no_stop_loss_take_profit_without_atr():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market"))
    pos = await _position()
    assert pos.stop_loss_price is None
    assert pos.take_profit_price is None


@pytest.mark.asyncio
async def test_stop_loss_monitor_closes_long_position_when_price_drops_through_stop():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos = await _position("trend_momentum")
    stop = pos.stop_loss_price

    # Price collapses through the stop.
    crash_price = stop - 50
    cache.update_orderbook("BTCUSD", bids=[[crash_price - 5, 100]], asks=[[crash_price + 5, 100]], timestamp=2)

    await broker.check_stop_loss_take_profit()

    pos_after = await _position("trend_momentum")
    assert pos_after.size == 0

    trades = await _closed_trades()
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop_loss"


@pytest.mark.asyncio
async def test_take_profit_monitor_closes_short_position_when_price_falls_through_target():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="sell", size=10, order_type="market", strategy="mean_reversion", atr=200.0)
    )
    pos = await _position("mean_reversion")
    target = pos.take_profit_price  # below entry for a short

    rally_down_price = target - 50
    cache.update_orderbook(
        "BTCUSD", bids=[[rally_down_price - 5, 100]], asks=[[rally_down_price + 5, 100]], timestamp=2
    )

    await broker.check_stop_loss_take_profit()

    pos_after = await _position("mean_reversion")
    assert pos_after.size == 0

    trades = await _closed_trades()
    assert len(trades) == 1
    assert trades[0].exit_reason == "take_profit"


@pytest.mark.asyncio
async def test_stop_loss_monitor_does_nothing_when_price_has_not_breached_levels():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    await broker.check_stop_loss_take_profit()

    pos = await _position("trend_momentum")
    assert pos.size == 10  # untouched, price hasn't moved
    assert (await _closed_trades()) == []


@pytest.mark.asyncio
async def test_position_view_reports_notional_and_margin_in_usd():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    broker = PaperBroker(_registry(), cache, settings)

    # notional ~= 400 * 0.001 * ~50020 =~ $20,008. Default quality (1.0,
    # unspecified) selects the leverage ceiling (50x by default).
    await broker.place_order(OrderRequest(symbol="BTCUSD", side="buy", size=400, order_type="market"))

    positions = await broker.get_positions()
    view = positions["BTCUSD:manual"]
    expected_notional = 400 * 0.001 * cache.get_orderbook("BTCUSD").mid  # approx, price includes slippage
    assert view.notional_usd == pytest.approx(expected_notional, rel=0.01)
    assert view.leverage == settings.max_leverage
    # Margin is genuinely notional / this position's selected leverage tier,
    # not a rounded display artifact -- and can never exceed cash on hand.
    balance = await broker._get_balance()
    assert view.margin_usd == pytest.approx(view.notional_usd / settings.max_leverage, rel=1e-6)
    assert view.margin_usd <= balance


@pytest.mark.asyncio
async def test_margin_never_exceeds_available_funds_even_at_the_floor_leverage_tier():
    """Regression test for the reported bug: a position's margin showing as
    MORE than total account funds. PaperBroker.place_order doesn't itself run
    the risk manager's per-trade margin cap (that happens in OrderManager,
    upstream of the broker) -- so if something ever lets a huge notional
    through at the floor leverage tier (worst case for margin: notional /
    min_leverage), margin_usd must still never exceed cash on hand.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    broker = PaperBroker(_registry(), cache, settings)

    # ~3000 contracts * 0.001 * ~50020 =~ $150,060 notional at the floor 10x
    # tier (quality=0.0) -> notional/leverage =~ $15,006, well past the
    # $10,000 starting balance.
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=3000, order_type="market", quality=0.0)
    )

    positions = await broker.get_positions()
    view = positions["BTCUSD:manual"]
    balance = await broker._get_balance()

    assert view.leverage == settings.min_leverage
    assert view.notional_usd / settings.min_leverage > balance  # confirms the overshoot scenario is in play
    assert view.margin_usd <= balance
    assert view.margin_usd <= view.notional_usd


@pytest.mark.asyncio
async def test_stop_loss_take_profit_stay_fixed_when_position_is_extended():
    """Regression test: SL/TP must be set once on genuine entry and NOT
    recomputed on later same-direction resizes -- a protective stop that
    keeps sliding to a new level every time the position grows isn't actually
    protecting the position.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos_before = await _position("trend_momentum")
    original_stop = pos_before.stop_loss_price
    original_target = pos_before.take_profit_price

    # Price moves and volatility changes, then the position is extended
    # (same direction) with a very different ATR -- if the bug were still
    # present, this would recompute SL/TP off the new blended entry price.
    cache.update_orderbook("BTCUSD", bids=[[51_990, 100]], asks=[[52_010, 100]], timestamp=2)
    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=5, order_type="market", strategy="trend_momentum", atr=900.0)
    )

    pos_after = await _position("trend_momentum")
    assert pos_after.size == 15
    assert pos_after.entry_price != pos_before.entry_price  # entry price DID reblend
    assert pos_after.stop_loss_price == original_stop  # but SL/TP did NOT move
    assert pos_after.take_profit_price == original_target


@pytest.mark.asyncio
async def test_default_settings_do_not_cut_winners_short():
    """Locks in the live default, REVERSED on 2026-09-03 after measurement.

    This test previously asserted the opposite -- that a mere 0.6-ATR
    favourable move tightened the stop and floored it at breakeven, per an
    explicit "make the trailing more aggressive" request that was never
    backtested. Measured over 9 years and 9,411 trades, that policy was the
    single largest destroyer of the strategy: winners were cut at ~0.4-0.5
    ATR while losers ran the full stop, giving average loss 1.6-2x average
    win. The strategy won 58-60% of trades and still went to -99%, because
    it needed 61-66% just to break even.

    With the trigger raised to 3.0 ATR (>= the take-profit distance) the
    trail cannot engage before the target is hit, so a winner is allowed to
    run. Out-of-sample effect: payoff ratio 0.59 -> 1.53, return -75.2% ->
    +1.4%.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos_before = await _position("trend_momentum")
    original_stop = pos_before.stop_loss_price

    # A solid 1.5-ATR favourable move -- which under the OLD 0.5-ATR trigger
    # would have tightened the stop and floored it at breakeven.
    new_peak = pos_before.entry_price + 300
    cache.update_orderbook("BTCUSD", bids=[[new_peak - 5, 100]], asks=[[new_peak + 5, 100]], timestamp=2)
    await broker.check_stop_loss_take_profit()

    pos_after = await _position("trend_momentum")
    # The stop may RATCHET UP behind the peak -- that is risk reduction and is
    # fine. What must NOT happen is it being floored at breakeven, which is
    # what forces a winner out on the first pullback.
    assert pos_after.stop_loss_price >= original_stop  # ratcheting up is allowed
    assert pos_after.stop_loss_price < pos_before.entry_price  # but NOT to breakeven
    # It must still sit a full stop-distance below the peak, not hug it.
    assert (new_peak - pos_after.stop_loss_price) == pytest.approx(pos_before.stop_distance, rel=1e-6)


@pytest.mark.asyncio
async def test_trailing_stop_tightens_as_long_position_moves_favorably():
    """Pins the trigger/trail multiples explicitly (1.0/0.8) rather than
    relying on Settings' defaults -- this test illustrates the MECHANISM at
    those specific reference numbers, independent of whatever the live
    default is currently tuned to (see config.py)."""
    cache = _cache_with_book(bid=49_990, ask=50_010)
    settings = Settings(
        delta_api_key="k", delta_api_secret="s", atr_stop_multiple=2.0,
        atr_breakeven_trigger_multiple=1.0, atr_tightened_trail_multiple=0.8
    )
    broker = PaperBroker(_registry(), cache, settings)

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos_before = await _position("trend_momentum")
    original_stop = pos_before.stop_loss_price
    original_target = pos_before.take_profit_price
    stop_distance = pos_before.stop_distance
    assert stop_distance == pytest.approx(2.0 * 200.0)

    # Price rallies well above entry, past the breakeven-trigger threshold
    # (1 ATR = 200) -- the stop should switch to the tightened trail
    # (0.8 ATR) AND floor at breakeven, not the wide 2-ATR entry stop_distance.
    new_peak = pos_before.entry_price + 300  # rallies, but stays under the +600 take-profit target
    cache.update_orderbook("BTCUSD", bids=[[new_peak - 5, 100]], asks=[[new_peak + 5, 100]], timestamp=2)
    await broker.check_stop_loss_take_profit()

    pos_after = await _position("trend_momentum")
    assert pos_after.size == 10  # not closed, price is nowhere near either level
    assert pos_after.stop_loss_price > original_stop
    assert pos_after.stop_loss_price > pos_before.entry_price  # floored at breakeven, and past it
    assert pos_after.stop_loss_price == pytest.approx(new_peak - 0.8 * 200.0)  # tightened 0.8-ATR trail
    assert pos_after.take_profit_price == original_target  # target never trails


@pytest.mark.asyncio
async def test_trailing_stop_never_loosens_when_price_pulls_back():
    """Pins the trigger/trail multiples explicitly -- the 150-point pullback
    below is sized specifically to stay above a 0.8-ATR trail (160), which a
    tighter live default would no longer guarantee."""
    cache = _cache_with_book(bid=49_990, ask=50_010)
    settings = Settings(
        delta_api_key="k", delta_api_secret="s", atr_stop_multiple=2.0,
        atr_breakeven_trigger_multiple=1.0, atr_tightened_trail_multiple=0.8
    )
    broker = PaperBroker(_registry(), cache, settings)

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos_before = await _position("trend_momentum")

    # Rally, so the stop trails up.
    new_peak = pos_before.entry_price + 300  # rallies, but stays under the +600 take-profit target
    cache.update_orderbook("BTCUSD", bids=[[new_peak - 5, 100]], asks=[[new_peak + 5, 100]], timestamp=2)
    await broker.check_stop_loss_take_profit()
    trailed_stop = (await _position("trend_momentum")).stop_loss_price

    # Price pulls back (but stays above the trailed, tightened stop of
    # new_peak - 0.8*200 = new_peak - 160) -- the stop must NOT loosen back
    # down with it.
    pullback_price = new_peak - 150
    cache.update_orderbook(
        "BTCUSD", bids=[[pullback_price - 5, 100]], asks=[[pullback_price + 5, 100]], timestamp=3
    )
    await broker.check_stop_loss_take_profit()

    pos_after = await _position("trend_momentum")
    assert pos_after.size == 10
    assert pos_after.stop_loss_price == trailed_stop


@pytest.mark.asyncio
async def test_trailing_stop_eventually_closes_position_once_price_reverses_through_it():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos_before = await _position("trend_momentum")

    # Rally well past entry so the trailed stop rises above the original
    # entry-based stop (i.e. it's now protecting locked-in profit).
    new_peak = pos_before.entry_price + 300  # rallies, but stays under the +600 take-profit target
    cache.update_orderbook("BTCUSD", bids=[[new_peak - 5, 100]], asks=[[new_peak + 5, 100]], timestamp=2)
    await broker.check_stop_loss_take_profit()
    trailed_stop = (await _position("trend_momentum")).stop_loss_price
    assert trailed_stop > pos_before.stop_loss_price

    # Now reverse hard through the trailed (tightened) stop.
    crash_price = trailed_stop - 50
    cache.update_orderbook("BTCUSD", bids=[[crash_price - 5, 100]], asks=[[crash_price + 5, 100]], timestamp=3)
    await broker.check_stop_loss_take_profit()

    pos_after = await _position("trend_momentum")
    assert pos_after.size == 0

    trades = await _closed_trades()
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].exit_price == pytest.approx(crash_price, rel=0.01)


@pytest.mark.asyncio
async def test_small_favorable_move_below_trigger_still_uses_wide_stop():
    """Below the 1-ATR breakeven trigger, the trail must stay at the original
    wide 2-ATR distance and is NOT floored at breakeven -- the tightened trail
    is deliberately reserved for a real, committed move, not any tiny wiggle
    in the position's favor. Pins the trigger explicitly (1.0 ATR = 200) --
    the 150-point rally below is sized specifically to stay under it.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    settings = Settings(delta_api_key="k", delta_api_secret="s", atr_stop_multiple=2.0,
                        atr_breakeven_trigger_multiple=1.0)
    broker = PaperBroker(_registry(), cache, settings)

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos_before = await _position("trend_momentum")

    # Only 150 favorable (< 1 ATR = 200 trigger).
    new_peak = pos_before.entry_price + 150
    cache.update_orderbook("BTCUSD", bids=[[new_peak - 5, 100]], asks=[[new_peak + 5, 100]], timestamp=2)
    await broker.check_stop_loss_take_profit()

    pos_after = await _position("trend_momentum")
    assert pos_after.stop_loss_price == pytest.approx(new_peak - 400.0)  # still the wide 2-ATR trail
    assert pos_after.stop_loss_price < pos_before.entry_price  # still below breakeven


@pytest.mark.asyncio
async def test_fast_spike_short_of_take_profit_then_reversal_locks_in_real_profit():
    """The exact scenario a flat 2-ATR trail handles badly: price spikes to
    within a whisker of take-profit (3 ATR) without touching it, then
    reverses. The tightened, breakeven-floored trail should close the
    position for a solid chunk of that move -- not a round-trip to scratch.
    Pins the trigger/trail multiples explicitly (1.0/0.8) so this test's own
    math in the comments below stays accurate regardless of the live default.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    settings = Settings(
        delta_api_key="k", delta_api_secret="s", atr_stop_multiple=2.0,
        atr_breakeven_trigger_multiple=1.0, atr_tightened_trail_multiple=0.8
    )
    broker = PaperBroker(_registry(), cache, settings)

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )
    pos_before = await _position("trend_momentum")

    # Spikes to entry + 2.5 ATR (500) -- close to the +3 ATR (600) target,
    # but never touches it.
    spike_peak = pos_before.entry_price + 500
    cache.update_orderbook("BTCUSD", bids=[[spike_peak - 5, 100]], asks=[[spike_peak + 5, 100]], timestamp=2)
    await broker.check_stop_loss_take_profit()
    pos_mid = await _position("trend_momentum")
    assert pos_mid.size == 10  # take-profit (600) was never reached

    # Reverses hard, straight through the trailed stop.
    crash_price = pos_mid.stop_loss_price - 50
    cache.update_orderbook("BTCUSD", bids=[[crash_price - 5, 100]], asks=[[crash_price + 5, 100]], timestamp=3)
    await broker.check_stop_loss_take_profit()

    pos_after = await _position("trend_momentum")
    assert pos_after.size == 0

    trades = await _closed_trades()
    assert len(trades) == 1
    # Old flat 2-ATR trail would have exited at spike_peak - 400 = entry+100.
    # The tightened 0.8-ATR trail exits at spike_peak - 160 = entry+340
    # instead -- meaningfully more profit banked from the same spike.
    assert trades[0].realized_pnl > 0
    assert trades[0].exit_price > pos_before.entry_price + 200


@pytest.mark.asyncio
async def test_trailing_stop_tightens_further_when_settings_change_even_without_a_new_peak():
    """Regression test: the stop used to only recompute when a genuinely NEW
    peak was set. If the trail-distance settings change while a position is
    open and price has already pulled back off its peak (no new peak to
    trigger a recompute), the stop would freeze forever under the OLD
    settings -- exactly what happened live when atr_tightened_trail_multiple
    was tightened from 1.0 to 0.8 ATR: open, already-triggered positions kept
    their stop computed under the old 1.0 multiple indefinitely. Recomputing
    on every tick (not just on a new peak) fixes this -- and can only ever
    tighten further, never loosen, since the candidate is still compared
    against the existing stop before being applied.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    loose_settings = Settings(
        delta_api_key="k", delta_api_secret="s", atr_breakeven_trigger_multiple=1.0, atr_tightened_trail_multiple=1.0
    )
    broker = PaperBroker(_registry(), cache, loose_settings)

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum", atr=200.0)
    )

    # Rally past the breakeven trigger (1 ATR = 200) under the OLD, wider
    # 1.0-ATR trail multiple.
    peak = (await _position("trend_momentum")).entry_price + 300
    cache.update_orderbook("BTCUSD", bids=[[peak - 5, 100]], asks=[[peak + 5, 100]], timestamp=2)
    await broker.check_stop_loss_take_profit()
    stop_under_old_settings = (await _position("trend_momentum")).stop_loss_price
    assert stop_under_old_settings == pytest.approx(peak - 1.0 * 200.0)

    # Settings tighten (0.8 ATR) -- price does NOT make a new peak, it just
    # sits at the same level (or pulls back slightly, still above the stop).
    broker.settings.atr_tightened_trail_multiple = 0.8
    cache.update_orderbook("BTCUSD", bids=[[peak - 6, 100]], asks=[[peak + 4, 100]], timestamp=3)
    await broker.check_stop_loss_take_profit()

    stop_after_settings_change = (await _position("trend_momentum")).stop_loss_price
    assert stop_after_settings_change > stop_under_old_settings  # tightened further
    assert stop_after_settings_change == pytest.approx(peak - 0.8 * 200.0)


@pytest.mark.asyncio
async def test_reduce_only_order_is_clamped_to_the_current_position_size():
    """Regression test: a reduce-only order sized against a stale view of the
    position (e.g. computed before an intervening fill shrank it) must be
    clamped to what's actually open, never allowed to overshoot.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum")
    )

    # Request a reduce-only close for MORE than the actual 10-contract
    # position (as if sized from a stale snapshot).
    result = await broker.place_order(
        OrderRequest(
            symbol="BTCUSD", side="sell", size=25, order_type="market", reduce_only=True, strategy="trend_momentum"
        )
    )

    assert result.filled_size == 10  # clamped down to the real position size
    pos = await _position("trend_momentum")
    assert pos.size == 0  # closed to flat, not flipped short


@pytest.mark.asyncio
async def test_reduce_only_order_cannot_flip_the_position_through_flat():
    """A reduce-only order in the WRONG direction for the current position
    (would extend it, or flip it through flat to the opposite side) must be
    zeroed out entirely, not partially applied.
    """
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    await broker.place_order(
        OrderRequest(symbol="BTCUSD", side="buy", size=10, order_type="market", strategy="trend_momentum")
    )

    # A reduce-only BUY against a LONG position would only ever extend it --
    # never a legitimate reduction.
    result = await broker.place_order(
        OrderRequest(
            symbol="BTCUSD", side="buy", size=5, order_type="market", reduce_only=True, strategy="trend_momentum"
        )
    )

    assert result.filled_size == 0
    pos = await _position("trend_momentum")
    assert pos.size == 10  # unchanged


@pytest.mark.asyncio
async def test_reduce_only_order_on_a_flat_position_is_a_no_op():
    cache = _cache_with_book(bid=49_990, ask=50_010)
    broker = PaperBroker(_registry(), cache, Settings(delta_api_key="k", delta_api_secret="s"))

    result = await broker.place_order(
        OrderRequest(
            symbol="BTCUSD", side="sell", size=5, order_type="market", reduce_only=True, strategy="trend_momentum"
        )
    )

    assert result.filled_size == 0
