import datetime as dt

import pytest

from app.config import Settings
from app.data.app_state import is_kill_switch_on
from app.data.market_cache import MarketDataCache
from app.risk.risk_manager import RiskManager


def _settings(**overrides) -> Settings:
    base = dict(delta_api_key="k", delta_api_secret="s", daily_loss_limit_pct=0.10, max_leverage=5)
    base.update(overrides)
    return Settings(**base)


def _today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


@pytest.mark.asyncio
async def test_trading_allowed_by_default():
    rm = RiskManager(_settings())
    cache = MarketDataCache()
    cache.system_status = {"status": "live"}

    allowed, reason = await rm.trading_allowed(cache, equity=10_000)
    assert allowed
    assert reason == ""


@pytest.mark.asyncio
async def test_daily_loss_breach_alone_does_not_auto_engage_kill_switch():
    """A daily-loss breach is logged (visible via check_daily_loss's own
    reason string) but `trading_allowed`/`evaluate_kill_switch` alone don't
    flip the kill switch flag or block new orders on their own -- that's
    now the job of a separate, proactive scheduler check (see
    scheduler.py's _daily_loss_supervisor), which sets the kill switch flag
    explicitly (and flattens) the first time a breach is seen each day,
    while still respecting a deliberate manual re-enable afterward rather
    than fighting it every tick for as long as the breach persists.
    """
    rm = RiskManager(_settings(daily_loss_limit_pct=0.01))
    cache = MarketDataCache()
    cache.system_status = {"status": "live"}

    # Establishes today's starting equity at 10_000 (first call of the day).
    await rm.trading_allowed(cache, equity=10_000)

    # Mark-to-market equity has now dropped 150 USD -- a real breach against
    # the 1% (100 USD) limit -- but trading_allowed/evaluate_kill_switch alone
    # still don't flip the flag or block orders themselves.
    allowed, reason = await rm.trading_allowed(cache, equity=9_850)
    assert allowed
    assert reason == ""
    assert not await is_kill_switch_on()

    ok, breach_reason = await rm.check_daily_loss(9_850)
    assert not ok
    assert "Daily loss limit breached" in breach_reason


@pytest.mark.asyncio
async def test_daily_loss_check_is_mark_to_market_not_realized_only():
    """Regression test: check_daily_loss used to look only at REALIZED PnL,
    which meant a position sitting deep underwater but not yet closed --
    exactly the case a daily-loss circuit breaker exists to catch -- never
    registered until a stop eventually fired. `equity` already reflects
    realized + unrealized (see PaperBroker.get_equity), so comparing it
    directly against the day's starting equity catches an still-open,
    still-losing position immediately, with zero realized trades at all.
    """
    rm = RiskManager(_settings(daily_loss_limit_pct=0.01))

    ok, _ = await rm.check_daily_loss(10_000)  # establishes starting_equity
    assert ok

    # No trades ever closed (record_realized_pnl never called) -- purely a
    # mark-to-market drawdown on a still-open position.
    ok, reason = await rm.check_daily_loss(9_850)
    assert not ok
    assert "Daily loss limit breached" in reason


@pytest.mark.asyncio
async def test_daily_profit_target_not_reached_below_the_threshold():
    rm = RiskManager(_settings(daily_profit_target_pct=0.10))

    ok, _ = await rm.check_daily_profit_target(10_000)  # establishes starting_equity
    assert ok

    ok, reason = await rm.check_daily_profit_target(10_500)  # only +5%, target is +10%
    assert ok
    assert reason == ""


@pytest.mark.asyncio
async def test_daily_profit_target_reached_at_the_threshold():
    rm = RiskManager(_settings(daily_profit_target_pct=0.10))

    await rm.check_daily_profit_target(10_000)  # establishes starting_equity

    ok, reason = await rm.check_daily_profit_target(11_000)  # exactly +10%
    assert not ok
    assert "Daily profit target reached" in reason


@pytest.mark.asyncio
async def test_daily_profit_target_is_mark_to_market_not_realized_only():
    """Same principle as the loss-limit check: an open, still-unrealized
    gain counts immediately, not only once a trade actually closes."""
    rm = RiskManager(_settings(daily_profit_target_pct=0.10))

    await rm.check_daily_profit_target(10_000)  # establishes starting_equity

    # No trades ever closed (record_realized_pnl never called) -- purely a
    # mark-to-market gain on a still-open position.
    ok, reason = await rm.check_daily_profit_target(11_200)
    assert not ok
    assert "Daily profit target reached" in reason


@pytest.mark.asyncio
async def test_daily_profit_target_and_daily_loss_share_the_same_starting_equity():
    """Both checks anchor to the SAME DailyPnLTracker.starting_equity -- one
    "today" concept, two symmetric limits on it -- so whichever check runs
    first establishes it for both."""
    rm = RiskManager(_settings(daily_profit_target_pct=0.10, daily_loss_limit_pct=0.01))

    await rm.check_daily_loss(10_000)  # establishes starting_equity via the OTHER check

    ok, reason = await rm.check_daily_profit_target(11_000)
    assert not ok
    assert "started 10000.00" in reason


@pytest.mark.asyncio
async def test_kill_switch_status_is_none_when_switch_is_off():
    rm = RiskManager(_settings())
    assert await rm.kill_switch_status(10_000) is None


@pytest.mark.asyncio
async def test_kill_switch_status_reports_profit_target_when_that_flag_is_set_today():
    from app.data.app_state import set_kill_switch, set_state

    rm = RiskManager(_settings())
    await rm.check_daily_profit_target(10_000)  # establishes starting_equity via risk_manager's own tracker
    today = _today_iso()
    await set_state("daily_profit_target_auto_triggered_day", today)
    await set_kill_switch(True)

    status = await rm.kill_switch_status(11_000)

    assert status["type"] == "profit_target"
    assert status["pnl_usd"] == pytest.approx(1_000.0)
    assert status["pnl_pct"] == pytest.approx(0.10)
    assert "profit target" in status["message"].lower()


@pytest.mark.asyncio
async def test_kill_switch_status_reports_daily_loss_when_that_flag_is_set_today():
    from app.data.app_state import set_kill_switch, set_state

    rm = RiskManager(_settings())
    await rm.check_daily_loss(10_000)
    today = _today_iso()
    await set_state("daily_kill_switch_auto_triggered_day", today)
    await set_kill_switch(True)

    status = await rm.kill_switch_status(9_000)

    assert status["type"] == "daily_loss"
    assert status["pnl_usd"] == pytest.approx(-1_000.0)
    assert "loss limit" in status["message"].lower()


@pytest.mark.asyncio
async def test_kill_switch_status_falls_back_to_manual_when_neither_todays_flag_is_set():
    from app.data.app_state import set_kill_switch

    rm = RiskManager(_settings())
    await rm.check_daily_loss(10_000)  # establishes starting_equity, no breach
    await set_kill_switch(True)  # e.g. the dashboard button, not an auto-trigger

    status = await rm.kill_switch_status(10_050)

    assert status["type"] == "manual"
    assert status["message"] == "Kill switch manually engaged."


@pytest.mark.asyncio
async def test_manually_engaged_kill_switch_still_blocks_trading():
    from app.data.app_state import set_kill_switch

    rm = RiskManager(_settings())
    cache = MarketDataCache()
    cache.system_status = {"status": "live"}

    await set_kill_switch(True)

    allowed, reason = await rm.trading_allowed(cache, equity=10_000)
    assert not allowed
    assert "Kill switch" in reason


@pytest.mark.asyncio
async def test_system_maintenance_blocks_trading():
    rm = RiskManager(_settings())
    cache = MarketDataCache()
    cache.system_status = {"status": "maintenance"}

    allowed, reason = await rm.trading_allowed(cache, equity=10_000)
    assert not allowed
    assert "maintenance" in reason.lower()


def test_cap_position_size_respects_leverage_and_portfolio_limits():
    rm = RiskManager(_settings(max_leverage=2, max_portfolio_leverage=0.1))
    # price=50_000, contract_value=0.001 -> $50 notional/contract.
    capped = rm.cap_position_size(
        proposed_contracts=100,
        symbol="BTCUSD",
        price=50_000,
        contract_value=0.001,
        equity=10_000,
        current_portfolio_notional_usd=0,
    )
    # Leverage cap allows up to 2x equity = $20,000 notional (400 contracts),
    # but the 0.1x portfolio leverage cap ($1,000 notional) is tighter -> 20 contracts.
    assert capped == 20


def test_cap_position_size_preserves_sign_for_shorts():
    rm = RiskManager(_settings(max_leverage=5, max_portfolio_leverage=10))
    capped = rm.cap_position_size(
        proposed_contracts=-50,
        symbol="BTCUSD",
        price=50_000,
        contract_value=0.001,
        equity=10_000,
        current_portfolio_notional_usd=0,
    )
    assert capped < 0


def test_cap_position_size_shrinks_order_when_existing_margin_usage_is_near_the_cap():
    """The exact scenario that motivated this check: the account is almost at
    its 85% aggregate margin budget already, so a new order -- even one that
    would pass the notional/leverage caps and comfortably fit under the
    per-trade cap on its own -- must be shrunk to whatever margin is
    actually still free, not allowed through at full size.
    """
    rm = RiskManager(_settings(max_leverage=50, max_portfolio_leverage=100, max_margin_utilization=0.85))
    capped = rm.cap_position_size(
        proposed_contracts=200,
        symbol="BTCUSD",
        price=50_000,
        contract_value=0.001,
        equity=10_000,
        current_portfolio_notional_usd=8_000,
        current_portfolio_margin_usd=8_450,
    )
    # Only $50 of margin budget remains ($8,500 cap - $8,450 used). Quality
    # defaults to 1.0 (full conviction) -> leverage = max_leverage = 50, so
    # the requested 200 contracts ($10,000 notional) would need $200 margin
    # at that tier -- far more than the $50 left, so it must be shrunk hard.
    assert 0 < capped < 200


def test_cap_position_size_unaffected_by_margin_check_when_budget_is_ample():
    rm = RiskManager(_settings(max_leverage=15, max_portfolio_leverage=20, max_margin_utilization=0.85))
    capped = rm.cap_position_size(
        proposed_contracts=10,
        symbol="BTCUSD",
        price=50_000,
        contract_value=0.001,
        equity=10_000,
        current_portfolio_notional_usd=0,
        current_portfolio_margin_usd=0,
    )
    assert capped == 10


def test_cap_position_size_limits_a_single_trades_margin_to_a_fraction_of_equity():
    """Regression test: a single trade sized aggressively enough to need more
    margin (at its own selected leverage tier) than max_margin_per_trade_pct
    of equity must be shrunk to fit that budget, not allowed to eat the
    entire max_margin_utilization budget by itself as the first order in.
    """
    rm = RiskManager(
        _settings(max_leverage=15, max_portfolio_leverage=20, max_margin_utilization=0.85, max_margin_per_trade_pct=0.15)
    )
    # notional = 600 * 0.001 * 50_000 = $30,000. Quality defaults to 1.0
    # (full conviction) -> leverage = max_leverage = 15, so this needs
    # $30,000 / 15 = $2,000 margin -- past the $1,500 (15% of $10,000)
    # per-trade budget.
    capped = rm.cap_position_size(
        proposed_contracts=600,
        symbol="BTCUSD",
        price=50_000,
        contract_value=0.001,
        equity=10_000,
        current_portfolio_notional_usd=0,
        current_portfolio_margin_usd=0,
    )
    notional_per_contract = 0.001 * 50_000
    leverage = 15  # quality=1.0 (default) selects the ceiling
    resulting_margin = (capped * notional_per_contract) / leverage
    assert resulting_margin == pytest.approx(10_000 * 0.15, rel=0.02)


def test_cap_position_size_leaves_room_for_several_concurrent_trades():
    """The scenario the per-trade cap exists for: with 5 strategies trading
    independently, several similarly-sized trades opening back to back
    (each one folding the last into `current_portfolio_margin_usd` before
    the next is sized) should ALL get through with a real, non-zero, EQUAL
    size -- not have the first one swallow the whole margin budget and
    starve the rest.
    """
    rm = RiskManager(
        _settings(max_leverage=15, max_portfolio_leverage=20, max_margin_utilization=0.85, max_margin_per_trade_pct=0.15)
    )
    equity = 10_000
    price = 50_000
    contract_value = 0.001
    notional_per_contract = contract_value * price
    leverage = 15  # quality=1.0 (default) selects the ceiling (max_leverage)

    cumulative_notional = 0.0
    cumulative_margin = 0.0
    sizes: list[int] = []
    for _ in range(5):
        capped = rm.cap_position_size(
            proposed_contracts=1500,  # each strategy independently wants an aggressive $75,000 notional
            symbol="BTCUSD",
            price=price,
            contract_value=contract_value,
            equity=equity,
            current_portfolio_notional_usd=cumulative_notional,
            current_portfolio_margin_usd=cumulative_margin,
        )
        sizes.append(capped)
        cumulative_notional += capped * notional_per_contract
        cumulative_margin += (capped * notional_per_contract) / leverage

    # All 5 independently-sized trades got a real, equal allocation -- each
    # one capped at the same 15%-of-equity margin budget, not the first one
    # grabbing everything and leaving the rest at zero.
    assert all(s > 0 for s in sizes)
    assert len(set(sizes)) == 1
    assert cumulative_margin <= equity * 0.85 + 1e-6


def test_default_margin_settings_support_four_concurrent_trades():
    """Locks in the current business requirement: with the DEFAULT
    max_margin_per_trade_pct (0.10, not overridden here), the 40% aggregate
    margin budget should support 4 concurrently open trades of similar size
    (0.40 / 0.10 = 4).

    max_margin_utilization dropped from 0.85 to 0.40 on 2026-08-31 after
    directly measuring the 3 live strategies' actual return correlation
    (mean pairwise 0.733 across all 9 strategy/symbol combinations, not the
    near-independence originally assumed) -- see config.py's comment on
    max_margin_utilization for the full derivation. This is a real,
    intentional reduction in how much aggregate risk the book is allowed to
    carry, not a regression.
    """
    # max_portfolio_leverage deliberately NOT overridden -- this test exists
    # to lock in the real, default-settings behavior, including the fact
    # that the portfolio notional cap is sized to match the margin cap (see
    # config.py) so it doesn't silently become the binding constraint before
    # margin does.
    settings = _settings(max_leverage=50, max_margin_utilization=0.40)
    rm = RiskManager(settings)
    equity = 10_000
    price = 50_000
    contract_value = 0.001
    notional_per_contract = contract_value * price
    leverage = 50  # quality=1.0 (default) selects the ceiling (max_leverage)

    cumulative_notional = 0.0
    cumulative_margin = 0.0
    sizes: list[int] = []
    for _ in range(4):
        capped = rm.cap_position_size(
            proposed_contracts=1500,  # each strategy independently wants an aggressive notional
            symbol="BTCUSD",
            price=price,
            contract_value=contract_value,
            equity=equity,
            current_portfolio_notional_usd=cumulative_notional,
            current_portfolio_margin_usd=cumulative_margin,
        )
        sizes.append(capped)
        cumulative_notional += capped * notional_per_contract
        cumulative_margin += (capped * notional_per_contract) / leverage

    # All 4 get a real, full-sized allocation -- not the first one or two
    # eating the whole budget.
    assert all(s > 0 for s in sizes)
    assert sum(1 for s in sizes if s == sizes[0]) >= 4
    assert cumulative_margin <= equity * 0.40 + 1e-6
