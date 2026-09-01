import datetime as dt

import pytest

from app.data.app_state import get_state, is_kill_switch_on, set_kill_switch, set_state
from app.scheduler import _daily_kill_switch_reset


def _today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


@pytest.mark.asyncio
async def test_switch_already_off_is_a_harmless_noop():
    await _daily_kill_switch_reset()
    assert not await is_kill_switch_on()


@pytest.mark.asyncio
async def test_switch_engaged_today_is_left_alone():
    await set_kill_switch(True)  # stamps kill_switch_engaged_day = today, via app_state itself

    await _daily_kill_switch_reset()

    assert await is_kill_switch_on()


@pytest.mark.asyncio
async def test_switch_engaged_on_an_earlier_day_gets_reset_off():
    await set_kill_switch(True)
    await set_state("kill_switch_engaged_day", "2020-01-01")  # simulate it having been on since a past day

    await _daily_kill_switch_reset()

    assert not await is_kill_switch_on()


@pytest.mark.asyncio
async def test_switch_on_with_no_engaged_day_recorded_at_all_gets_reset_off():
    """Regression test: the FIRST version of this job used its own
    "have I run before" bookkeeping instead of asking when the switch
    itself was engaged, so a switch already on (and already stale) the very
    first time the job ever ran was wrongly left alone -- exactly what
    happened live on 2026-08-23. Simulated here by turning the switch on
    directly via set_state (bypassing app_state.set_kill_switch, the same
    as any kill_switch value written before kill_switch_engaged_day
    tracking existed) so no engaged-day record exists at all.
    """
    await set_state("kill_switch", "on")
    assert await get_state("kill_switch_engaged_day") is None

    await _daily_kill_switch_reset()

    assert not await is_kill_switch_on()


@pytest.mark.asyncio
async def test_does_not_touch_the_daily_supervisors_own_trigger_flags():
    """The two daily-breach supervisors' own auto-triggered-day flags
    already self-invalidate once the date no longer matches -- this job
    must not also try to manage them (that would be redundant, and a bug
    here could otherwise interfere with their own logic).
    """
    await set_state("daily_kill_switch_auto_triggered_day", "2020-01-01")
    await set_state("daily_profit_target_auto_triggered_day", "2020-01-01")
    await set_kill_switch(True)
    await set_state("kill_switch_engaged_day", "2020-01-01")

    await _daily_kill_switch_reset()

    assert not await is_kill_switch_on()
    assert await get_state("daily_kill_switch_auto_triggered_day") == "2020-01-01"
    assert await get_state("daily_profit_target_auto_triggered_day") == "2020-01-01"
