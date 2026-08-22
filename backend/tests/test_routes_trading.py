import datetime as dt

import pytest

from app.api.routes_trading import KillSwitchUpdate, get_closed_trades, update_kill_switch
from app.data.app_state import get_trading_mode, set_trading_mode


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
