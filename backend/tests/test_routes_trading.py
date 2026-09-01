import datetime as dt

import pytest

from app.api.routes_trading import KillSwitchUpdate, get_closed_trades, get_mode, update_kill_switch
from app.data.app_state import get_trading_mode, set_kill_switch, set_trading_mode


class _FlattenSpy:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def flatten_all(self, mode_override: str | None = None) -> None:
        self.calls.append(mode_override)


class _FakeRuntime:
    def __init__(self) -> None:
        self.order_manager = _FlattenSpy()


@pytest.mark.asyncio
async def test_kill_switch_flattens_the_mode_that_was_actually_live_before_switching_to_paper():
    """Regression test: update_kill_switch used to call set_trading_mode("paper")
    BEFORE flatten_all(), and flatten_all() with no override reads the
    *current* mode -- so a kill-switch hit while genuinely LIVE-trading would
    flatten paper positions (a no-op) while real Delta positions stayed open
    and unmanaged, defeating the one thing the kill switch exists to
    guarantee. It must flatten whichever mode was active BEFORE the switch.
    """
    await set_trading_mode("live")
    runtime = _FakeRuntime()

    await update_kill_switch(KillSwitchUpdate(on=True), runtime=runtime)

    assert runtime.order_manager.calls == ["live"]
    assert await get_trading_mode() == "paper"


@pytest.mark.asyncio
async def test_kill_switch_flattens_paper_when_already_in_paper_mode():
    await set_trading_mode("paper")
    runtime = _FakeRuntime()

    await update_kill_switch(KillSwitchUpdate(on=True), runtime=runtime)

    assert runtime.order_manager.calls == ["paper"]
    assert await get_trading_mode() == "paper"


@pytest.mark.asyncio
async def test_turning_kill_switch_off_does_not_flatten_anything():
    runtime = _FakeRuntime()

    await update_kill_switch(KillSwitchUpdate(on=False), runtime=runtime)

    assert runtime.order_manager.calls == []


class _FlattenSpyWithEquity(_FlattenSpy):
    async def get_equity(self) -> float:
        return 8_000.0


class _FakeRuntimeWithEquity:
    def __init__(self) -> None:
        self.order_manager = _FlattenSpyWithEquity()


@pytest.mark.asyncio
async def test_turning_kill_switch_off_acknowledges_a_drawdown_halt_and_rebases_the_peak():
    """Turning the switch off is the ONLY thing that clears a max-drawdown
    halt, and it must re-base the high-water mark at the same moment --
    otherwise _drawdown_supervisor re-halts on its very next tick, since
    equity is still exactly as far below the old peak as it was a second
    ago.
    """
    from app.data.app_state import get_state, set_state

    await set_state("drawdown_halt_active", "on")
    await set_state("equity_high_water_mark", "10000.0")

    await update_kill_switch(KillSwitchUpdate(on=False), runtime=_FakeRuntimeWithEquity())

    assert await get_state("drawdown_halt_active") == "off"
    assert float(await get_state("equity_high_water_mark")) == 8_000.0


@pytest.mark.asyncio
async def test_turning_kill_switch_off_leaves_the_peak_alone_when_no_halt_is_active():
    """A routine manual toggle must NOT quietly reset the high-water mark --
    that would erase the drawdown measurement for anyone who happens to use
    the kill switch during an ordinary drawdown.
    """
    from app.data.app_state import get_state, set_state

    await set_state("drawdown_halt_active", "off")
    await set_state("equity_high_water_mark", "10000.0")

    await update_kill_switch(KillSwitchUpdate(on=False), runtime=_FakeRuntimeWithEquity())

    assert float(await get_state("equity_high_water_mark")) == 10_000.0


class _FakePaperBroker:
    async def get_cash_balance(self) -> float:
        return 10_000.0


class _FakeRuntimeWithBroker:
    def __init__(self) -> None:
        self.paper_broker = _FakePaperBroker()


@pytest.mark.asyncio
async def test_closed_trades_route_forwards_date_range_to_the_query_layer(monkeypatch):
    """Regression test: the /closed_trades route must actually pass
    date_from/date_to through to load_closed_trades, not silently drop them
    -- the query-layer filtering itself is covered in
    test_query_closed_trades.py.
    """
    import app.api.routes_trading as routes_mod

    captured = {}

    async def _fake_load_closed_trades(mode, symbol=None, date_from=None, date_to=None, limit=100):
        captured["mode"] = mode
        captured["symbol"] = symbol
        captured["date_from"] = date_from
        captured["date_to"] = date_to
        return []

    monkeypatch.setattr(routes_mod, "load_closed_trades", _fake_load_closed_trades)

    date_from = dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)
    date_to = dt.datetime(2026, 8, 21, tzinfo=dt.timezone.utc)
    result = await get_closed_trades(
        mode="paper", date_from=date_from, date_to=date_to, runtime=_FakeRuntimeWithBroker()
    )

    assert result == []
    assert captured["date_from"] == date_from
    assert captured["date_to"] == date_to


class _FakeOrderManagerForMode:
    async def get_equity(self) -> float:
        return 11_000.0


class _FakeRiskManagerForMode:
    def __init__(self, status: dict | None) -> None:
        self._status = status
        self.calls: list[float] = []

    async def kill_switch_status(self, equity: float) -> dict | None:
        self.calls.append(equity)
        return self._status


class _FakeRuntimeForMode:
    def __init__(self, status: dict | None) -> None:
        self.order_manager = _FakeOrderManagerForMode()
        self.risk_manager = _FakeRiskManagerForMode(status)


@pytest.mark.asyncio
async def test_get_mode_omits_reason_and_skips_the_equity_lookup_when_switch_is_off():
    await set_kill_switch(False)
    runtime = _FakeRuntimeForMode(status=None)

    result = await get_mode(runtime=runtime)

    assert result["kill_switch"] is False
    assert result["kill_switch_reason"] is None
    assert runtime.risk_manager.calls == []  # never even asked -- nothing to explain


@pytest.mark.asyncio
async def test_get_mode_surfaces_the_risk_managers_explanation_when_switch_is_on():
    """Regression test: the dashboard used to show an engaged kill switch
    with zero explanation of why -- /mode must forward RiskManager.
    kill_switch_status's reason through so the frontend can show a real
    banner instead of a bare ON/OFF toggle.
    """
    await set_kill_switch(True)
    status = {"type": "profit_target", "message": "Daily profit target reached (+10.0%)."}
    runtime = _FakeRuntimeForMode(status=status)

    result = await get_mode(runtime=runtime)

    assert result["kill_switch"] is True
    assert result["kill_switch_reason"] == status
    assert runtime.risk_manager.calls == [11_000.0]  # the CURRENT equity, not a stale figure
