"""Deposits/withdrawals are not P&L; a halt is set before flattening; a halted
account is kept flat (2026-10-05 incident: a Rs 1,000 deposit tripped the +10%
profit target, and a BNB buy slipped in between the flatten and the halt)."""
import datetime as dt
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.data.app_state import get_state, is_kill_switch_on, set_kill_switch, set_state
from app.risk.risk_manager import ExternalFlow, RiskManager

UTC = dt.timezone.utc


def _rm(**kw) -> RiskManager:
    return RiskManager(Settings(delta_api_key="k", delta_api_secret="s", daily_profit_target_pct=0.10, **kw))


def _now():
    return dt.datetime.now(UTC)


@pytest.mark.asyncio
async def test_a_deposit_after_the_day_started_is_not_profit():
    rm = _rm()
    assert (await rm.check_daily_profit_target(3.24))[0]  # anchors today's start at 3.24
    deposit = ExternalFlow("dep-1", _now() + dt.timedelta(seconds=1), 11.76)

    applied = await rm.apply_external_flows([deposit])

    assert applied == [deposit]
    assert rm.daily_pnl.starting_equity == pytest.approx(15.00)
    ok, _ = await rm.check_daily_profit_target(14.98)  # the exact incident numbers
    assert ok


@pytest.mark.asyncio
async def test_a_withdrawal_after_the_day_started_is_not_a_loss():
    rm = _rm(daily_loss_limit_pct=0.20)
    assert (await rm.check_daily_loss(100.0))[0]
    await rm.apply_external_flows([ExternalFlow("wd-1", _now() + dt.timedelta(seconds=1), -50.0)])
    assert rm.daily_pnl.starting_equity == pytest.approx(50.0)
    ok, _ = await rm.check_daily_loss(49.0)  # -2%, not -51%
    assert ok


@pytest.mark.asyncio
async def test_a_flow_before_the_anchor_is_already_inside_it_and_each_flow_applies_once():
    rm = _rm()
    early = ExternalFlow("dep-early", _now() - dt.timedelta(minutes=1), 10.0)
    assert (await rm.check_daily_profit_target(20.0))[0]
    await rm.apply_external_flows([early])
    assert rm.daily_pnl.starting_equity == pytest.approx(20.0)

    late = ExternalFlow("dep-late", _now() + dt.timedelta(seconds=1), 5.0)
    await rm.apply_external_flows([late])
    await rm.apply_external_flows([late, early])  # seen again: no double count
    assert rm.daily_pnl.starting_equity == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_peak_without_a_stamp_is_taken_to_contain_past_flows_then_later_ones_move_it():
    rm = _rm()
    await set_state("equity_high_water_mark", "14.98")  # set after the deposit, before stamps existed
    await rm.apply_external_flows([ExternalFlow("dep-old", _now() - dt.timedelta(minutes=5), 11.76)])
    assert float(await get_state("equity_high_water_mark")) == pytest.approx(14.98)

    await rm.apply_external_flows([ExternalFlow("wd-new", _now() + dt.timedelta(seconds=1), -4.98)])
    assert float(await get_state("equity_high_water_mark")) == pytest.approx(10.0)
    ok, _ = await rm.check_max_drawdown(10.0)
    assert ok  # a withdrawal is not a drawdown


@pytest.mark.asyncio
async def test_live_broker_reads_only_deposits_and_withdrawals():
    from app.execution.live_broker import LiveBroker

    rows = [
        {"transaction_type": "deposit", "amount": "11.76470588", "asset_symbol": "USD",
         "created_at": "2026-10-05T07:23:26.015456Z", "uuid": "a"},
        {"transaction_type": "withdrawal", "amount": "5", "asset_symbol": "USD",
         "created_at": "2026-10-05T08:00:00Z", "uuid": "b"},
        {"transaction_type": "commission", "amount": "-0.01", "asset_symbol": "USD",
         "created_at": "2026-10-05T07:23:34Z", "uuid": "c"},
        {"transaction_type": "cashflow", "amount": "0.2", "asset_symbol": "USD",
         "created_at": "2026-10-05T07:23:34Z", "uuid": "d"},
    ]

    class _Client:
        async def request(self, method, path, params=None, auth=False):
            assert path == "/v2/wallet/transactions" and auth
            return rows

    broker = LiveBroker.__new__(LiveBroker)
    broker.client = _Client()
    broker._flows_cache = None
    flows = await broker.external_flows()
    assert [(f.uid, round(f.amount, 2)) for f in flows] == [("a", 11.76), ("b", -5.0)]
    assert flows[0].time == dt.datetime(2026, 10, 5, 7, 23, 26, 15456, tzinfo=UTC)


class _OrderManager:
    def __init__(self, positions, log):
        self._positions = positions
        self.log = log

    async def get_positions(self):
        return self._positions

    async def get_equity(self):
        return 100.0

    async def flatten_all(self, mode_override=None, reason="kill_switch"):
        self.log.append(("flatten", reason, await is_kill_switch_on()))


@pytest.mark.asyncio
async def test_halted_account_is_flattened_and_a_running_one_is_left_alone():
    from app.scheduler import _halted_means_flat

    log = []
    pos = {"BNBUSD:consolidated": SimpleNamespace(symbol="BNBUSD", size=1)}
    runtime = SimpleNamespace(order_manager=_OrderManager(pos, log))

    await set_kill_switch(False)
    await _halted_means_flat(runtime)
    assert log == []

    await set_kill_switch(True)
    await _halted_means_flat(runtime)
    assert log == [("flatten", "kill_switch", True)]


@pytest.mark.asyncio
async def test_profit_target_halt_is_set_before_the_flatten():
    from app.scheduler import _daily_profit_target_supervisor

    log = []
    rm = _rm()
    await set_kill_switch(False)
    assert (await rm.check_daily_profit_target(100.0))[0]
    runtime = SimpleNamespace(order_manager=_OrderManager({}, log), risk_manager=rm)
    runtime.order_manager.get_equity = lambda: _async(111.0)

    await _daily_profit_target_supervisor(runtime)

    assert log == [("flatten", "profit_target", True)]  # the halt was already on when it flattened


async def _async(v):
    return v


@pytest.mark.asyncio
async def test_concurrent_supervisors_apply_a_deposit_once():
    import asyncio

    rm = _rm()
    assert (await rm.check_daily_profit_target(3.24))[0]
    dep = ExternalFlow("dep-race", _now() + dt.timedelta(seconds=1), 11.76)
    await asyncio.gather(*(rm.apply_external_flows([dep]) for _ in range(3)))
    assert rm.daily_pnl.starting_equity == pytest.approx(15.00)
