"""Team test agent, cycle 1: daily loss/profit limits, deposits/withdrawals kept out
of P&L (and the lock), and the scheduler supervisors halting BEFORE they flatten.
Tests marked xfail(strict=True) pin real bugs."""
import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.data.app_state import get_state, get_trading_mode, is_kill_switch_on, set_kill_switch, set_trading_mode
from app.risk.risk_manager import ExternalFlow, RiskManager

UTC = dt.timezone.utc


def _rm(**kw) -> RiskManager:
    base = dict(delta_api_key="k", delta_api_secret="s", daily_profit_target_pct=0.10, daily_loss_limit_pct=0.20)
    base.update(kw)
    return RiskManager(Settings(**base))


def _now():
    return dt.datetime.now(UTC)


async def _v(x):
    return x


@pytest.fixture(autouse=True)
async def _switch_off():
    await set_kill_switch(False)
    yield
    await set_kill_switch(False)


# ================================================================ daily limits
@pytest.mark.asyncio
async def test_daily_loss_breaches_exactly_at_the_limit_and_not_a_cent_before():
    rm = _rm()
    assert (await rm.check_daily_loss(100.0))[0]
    assert (await rm.check_daily_loss(80.01))[0]
    ok, reason = await rm.check_daily_loss(80.0)
    assert not ok and "Daily loss limit breached" in reason


@pytest.mark.asyncio
async def test_profit_target_breaches_exactly_at_the_target_and_zero_means_off():
    rm = _rm()
    assert (await rm.check_daily_profit_target(100.0))[0]
    assert (await rm.check_daily_profit_target(109.99))[0]
    assert not (await rm.check_daily_profit_target(110.0))[0]
    off = _rm(daily_profit_target_pct=0.0)
    assert (await off.check_daily_profit_target(100.0))[0]
    assert (await off.check_daily_profit_target(1000.0))[0]  # +900% and still not halted: the target is off


@pytest.mark.asyncio
async def test_a_new_utc_day_re_anchors_on_current_equity():
    rm = _rm()
    assert (await rm.check_daily_loss(100.0))[0]
    rm.daily_pnl.day = rm.daily_pnl.day - dt.timedelta(days=1)  # yesterday's anchor
    rm.daily_pnl.realized_pnl = -30.0
    assert (await rm.check_daily_loss(50.0))[0]  # -50% vs yesterday, but today starts at 50
    assert rm.daily_pnl.starting_equity == 50.0 and rm.daily_pnl.realized_pnl == 0.0
    assert rm.daily_pnl.day == _now().date()


@pytest.mark.asyncio
async def test_a_restart_keeps_todays_anchor_including_an_applied_deposit():
    rm = _rm()
    assert (await rm.check_daily_profit_target(3.24))[0]
    await rm.apply_external_flows([ExternalFlow("dep", _now() + dt.timedelta(seconds=1), 11.76)])
    restarted = _rm()
    await restarted.load_daily_pnl_state()
    assert restarted.daily_pnl.starting_equity == pytest.approx(15.0)
    assert (await restarted.check_daily_profit_target(15.2))[0]  # +1.3%, not +369%
    # and the deposit is not re-applied after the restart
    await restarted.apply_external_flows([ExternalFlow("dep", _now(), 11.76)])
    assert restarted.daily_pnl.starting_equity == pytest.approx(15.0)


# ================================================================ external flows
@pytest.mark.asyncio
async def test_a_flow_before_the_day_is_anchored_is_not_double_counted():
    rm = _rm()
    # balance read already contains the deposit; no anchor exists yet (fresh start)
    await rm.apply_external_flows([ExternalFlow("dep", _now(), 10.0)])
    assert rm.daily_pnl.starting_equity == 0.0
    assert (await rm.check_daily_profit_target(13.0))[0]  # anchors at 13, which includes the deposit
    assert rm.daily_pnl.starting_equity == 13.0


@pytest.mark.asyncio
async def test_a_flow_from_a_previous_utc_day_never_moves_todays_anchor():
    rm = _rm()
    assert (await rm.check_daily_loss(100.0))[0]
    yesterday = dt.datetime.combine(_now().date(), dt.time(), tzinfo=UTC) - dt.timedelta(minutes=1)
    await rm.apply_external_flows([ExternalFlow("old", yesterday, -40.0)])
    assert rm.daily_pnl.starting_equity == 100.0


@pytest.mark.asyncio
async def test_a_withdrawal_bigger_than_the_day_start_floors_at_zero_then_re_anchors():
    rm = _rm()
    assert (await rm.check_daily_loss(10.0))[0]
    await rm.apply_external_flows([ExternalFlow("wd", _now() + dt.timedelta(seconds=1), -12.0)])
    assert rm.daily_pnl.starting_equity == 0.0
    assert (await rm.check_daily_loss(0.5))[0]  # re-anchored on what is left, not a -95% "loss"


@pytest.mark.asyncio
async def test_concurrent_calls_with_overlapping_lists_apply_each_flow_once():
    rm = _rm()
    assert (await rm.check_daily_profit_target(10.0))[0]
    t = _now() + dt.timedelta(seconds=1)
    d1, d2 = ExternalFlow("d1", t, 5.0), ExternalFlow("d2", t, 3.0)
    await asyncio.gather(rm.apply_external_flows([d1]), rm.apply_external_flows([d1, d2]),
                         rm.apply_external_flows([d2]), rm.apply_external_flows([d2, d1]))
    assert rm.daily_pnl.starting_equity == pytest.approx(18.0)
    assert sorted(__import__("json").loads(await get_state("external_flows_applied"))) == ["d1", "d2"]


@pytest.mark.asyncio
async def test_a_deposit_after_the_peak_moves_the_peak_so_a_later_loss_is_measured_fairly():
    rm = _rm(max_drawdown_pct=0.5)
    assert (await rm.check_max_drawdown(10.0))[0]  # peak 10, stamped now
    await rm.apply_external_flows([ExternalFlow("dep", _now() + dt.timedelta(seconds=1), 90.0)])
    assert float(await get_state("equity_high_water_mark")) == pytest.approx(100.0)
    assert not (await rm.check_max_drawdown(45.0))[0]  # -55% of the real $100 peak IS a drawdown


# ================================================================ supervisors: halt BEFORE flatten
class _OM:
    def __init__(self, equity, log, raise_on_flatten=False, live_broker=None):
        self.equity, self.log, self.raise_on_flatten = equity, log, raise_on_flatten
        self.live_broker = live_broker

    async def get_equity(self):
        return self.equity

    async def get_positions(self):
        return {}

    async def flatten_all(self, mode_override=None, reason="kill_switch"):
        self.log.append(("flatten", reason, mode_override, await is_kill_switch_on()))
        if self.raise_on_flatten:
            raise RuntimeError("Delta 502 mid-flatten")


@pytest.mark.asyncio
async def test_daily_loss_supervisor_halts_before_it_flattens_and_stays_live():
    from app.scheduler import _daily_loss_supervisor

    await set_trading_mode("live")
    log, rm = [], _rm()
    assert (await rm.check_daily_loss(100.0))[0]
    await _daily_loss_supervisor(SimpleNamespace(order_manager=_OM(79.0, log), risk_manager=rm))
    assert log == [("flatten", "daily_loss", "live", True)]
    assert await is_kill_switch_on() and await get_trading_mode() == "live"
    assert await get_state("daily_kill_switch_auto_triggered_day") == _now().date().isoformat()


@pytest.mark.asyncio
async def test_drawdown_supervisor_halts_before_it_flattens():
    from app.scheduler import _drawdown_supervisor

    await set_trading_mode("live")
    log, rm = [], _rm(max_drawdown_pct=0.6)
    await rm.set_high_water_mark(100.0)
    await _drawdown_supervisor(SimpleNamespace(order_manager=_OM(39.0, log), risk_manager=rm))
    assert log == [("flatten", "max_drawdown", "live", True)]
    assert await get_state("drawdown_halt_active") == "on"


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["loss", "profit", "drawdown"])
async def test_a_flatten_that_raises_still_leaves_the_halt_and_its_flag(which):
    from app import scheduler

    await set_trading_mode("live")
    log, rm = [], _rm(max_drawdown_pct=0.6)
    if which == "drawdown":
        await rm.set_high_water_mark(100.0)
        sup, equity, flag = scheduler._drawdown_supervisor, 39.0, ("drawdown_halt_active", "on")
    else:
        assert (await rm.check_daily_loss(100.0))[0]
        today = _now().date().isoformat()
        if which == "loss":
            sup, equity, flag = scheduler._daily_loss_supervisor, 79.0, ("daily_kill_switch_auto_triggered_day", today)
        else:
            sup, equity, flag = (scheduler._daily_profit_target_supervisor, 111.0,
                                 ("daily_profit_target_auto_triggered_day", today))
    with pytest.raises(RuntimeError):
        await sup(SimpleNamespace(order_manager=_OM(equity, log, raise_on_flatten=True), risk_manager=rm))
    assert await is_kill_switch_on()
    assert await get_state(flag[0]) == flag[1]
    assert log and log[0][3] is True  # the halt was already on when the flatten began


class _FlowsBroker:
    """A live broker whose deposit list the test controls (no cache)."""

    def __init__(self):
        self.flows = []

    async def external_flows(self):
        return list(self.flows)


@pytest.mark.asyncio
async def test_profit_supervisor_does_not_halt_on_a_deposit_it_can_see():
    from app.scheduler import _daily_profit_target_supervisor

    await set_trading_mode("live")
    log, rm, broker = [], _rm(), _FlowsBroker()
    assert (await rm.check_daily_profit_target(3.24))[0]
    broker.flows.append(ExternalFlow("dep", _now() + dt.timedelta(seconds=1), 11.76))
    await _daily_profit_target_supervisor(
        SimpleNamespace(order_manager=_OM(15.0, log, live_broker=broker), risk_manager=rm))
    assert log == [] and not await is_kill_switch_on()


@pytest.mark.asyncio
async def test_a_deposit_landing_inside_the_flow_cache_window_does_not_trip_the_profit_halt():
    from app.execution.live_broker import LiveBroker
    from app.scheduler import _daily_profit_target_supervisor

    await set_trading_mode("live")
    rows = []

    class _Client:
        async def request(self, method, path, params=None, auth=False):
            return list(rows)

    broker = LiveBroker.__new__(LiveBroker)
    broker.client = _Client()
    broker._flows_cache = None
    log, rm = [], _rm()
    om = _OM(3.24, log, live_broker=broker)
    runtime = SimpleNamespace(order_manager=om, risk_manager=rm)

    assert (await rm.check_daily_profit_target(3.24))[0]
    await _daily_profit_target_supervisor(runtime)  # t=0: reads (and caches) an empty deposit list
    rows.append({"transaction_type": "deposit", "amount": "11.76", "asset_symbol": "USD",
                 "created_at": (_now() + dt.timedelta(seconds=1)).isoformat(), "uuid": "dep-1"})
    om.equity = 15.0  # t=5s: the deposit is in the balance
    await _daily_profit_target_supervisor(runtime)
    assert log == [] and not await is_kill_switch_on()
