"""Exchange-side stops for live Analyst trades (Settings.analyst_exchange_stops,
2026-10-07): a reduce-only stop-market order on Delta at the trade's stop,
placed after the fill, moved (edit, else place-then-cancel) at most once a
minute and only on a >= 0.1 ATR move, re-placed when it goes missing (deadman
switch), and cancelled when the trade closes. False = software stop only."""
import types

import pytest
from sqlalchemy import select

from app.analyst import execution as ex
from app.config import Settings
from app.core.delta_client import DeltaApiError
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.app_state import set_kill_switch, set_trading_mode
from app.data.db import SessionLocal
from app.data.models import AnalystTrade
from app.execution.broker_base import OrderResult, PositionView
from app.execution.live_broker import LiveBroker


def _s(**kw):
    kw.setdefault("analyst_risk_cap", 0)
    kw.setdefault("analyst_min_liq_to_stop", 0)
    kw.setdefault("analyst_target_leverage", 0)
    kw.setdefault("analyst_trail_mode", "live")  # so the software stop moves and the exchange stop must follow
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


class _StopBroker:
    """The LiveBroker stop helpers, recorded."""

    def __init__(self):
        self.calls, self.resting, self.n, self.edit_ok, self.read_ok = [], {}, 0, True, True

    async def place_stop_order(self, symbol, side, size, stop_price, client_order_id):
        self.n += 1
        oid = str(100 + self.n)
        self.calls.append(("place", side, size, stop_price))
        self.resting[oid] = {"id": int(oid), "state": "pending", "client_order_id": client_order_id}
        return oid

    async def edit_stop_order(self, symbol, order_id, stop_price):
        self.calls.append(("edit", order_id, stop_price))
        return order_id if self.edit_ok else None

    async def cancel_stop_order(self, symbol, order_id):
        self.calls.append(("cancel", order_id))
        self.resting.pop(order_id, None)
        return True

    async def resting_orders(self, symbol):
        return list(self.resting.values()) if self.read_ok else None

    def kinds(self):
        return [c[0] for c in self.calls]


class _OM:
    def __init__(self):
        self.positions, self.closed, self.live_broker = {}, [], _StopBroker()

    async def get_equity(self):
        return 14.0

    async def get_positions(self):
        return self.positions

    async def submit(self, req):
        self.positions[f"{req.symbol}:consolidated"] = PositionView(req.symbol, req.size, 2701.0, 0, 0,
                                                                    strategy="consolidated")
        return OrderResult(1, "x", "closed", req.size, 2701.0, 0.0)

    async def close_position(self, symbol, strategy):
        self.closed.append(symbol)
        self.positions.pop(f"{symbol}:{strategy}", None)
        return OrderResult(2, "y", "closed", 1, 2600.0, 0.0)


def _call(score=0.6):  # ETH BUY 2700, stop 2550 (ATR fallback = 150/1.5 = 100), TP 2850
    return {"symbol": "ETHUSD", "decision": "BUY", "conviction": int(round(score * 10)), "score": score,
            "size_factor": 1.0, "entry": 2700.0, "stop_loss": 2550.0, "take_profit": 2850.0, "families": {},
            "notes": [], "factors": [], "time": "2026-10-07T00:00:00+00:00"}


@pytest.fixture
def clock(monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(ex, "time", types.SimpleNamespace(time=lambda: now[0]))
    return now


async def _open(mode="live", **kw):
    await set_trading_mode(mode)
    await set_kill_switch(False)
    om = _OM()
    reg = types.SimpleNamespace(get=lambda s: types.SimpleNamespace(contract_value=0.01))
    prices = {"ETHUSD": 2700.0}
    rt = types.SimpleNamespace(settings=_s(trading_engine="combined", **kw), order_manager=om, registry=reg,
                               live_signals={"ETHUSD": 0.4}, cache=types.SimpleNamespace(get_last_price=prices.get))
    trade = await ex.open_best_trade(rt, {"ETHUSD": _call()})
    return rt, om, om.live_broker, prices, trade


async def _trade():
    async with SessionLocal() as s:
        return (await s.execute(select(AnalystTrade))).scalars().one()


@pytest.mark.asyncio
async def test_a_live_fill_places_a_reduce_only_stop_at_the_trade_stop(clock):
    rt, om, b, prices, trade = await _open()
    assert trade.live and b.calls == [("place", "sell", 1, pytest.approx(2551.0))]
    assert ex._stop_state(await _trade())["id"] == "101"


@pytest.mark.asyncio
async def test_setting_off_is_the_old_behaviour_no_exchange_calls_at_all(clock):
    rt, om, b, prices, trade = await _open(analyst_exchange_stops=False)
    assert trade.live and "exchange_stop" not in (trade.reasons or {})
    for px in (2720.0, 2760.0):
        clock[0] += 120
        prices["ETHUSD"] = px
        await ex.manage_open_trades(rt, {})
    prices["ETHUSD"] = 2500.0
    await ex.manage_open_trades(rt, {})
    t = await _trade()
    assert b.calls == [] and om.closed == ["ETHUSD"] and t.result == "TRAIL" and not t.is_open


@pytest.mark.asyncio
async def test_paper_trades_get_no_exchange_stop(clock):
    rt, om, b, prices, trade = await _open(mode="paper")
    assert not trade.live and b.calls == []


@pytest.mark.asyncio
async def test_moves_are_throttled_to_once_a_minute_and_a_tenth_of_an_atr(clock):
    rt, om, b, prices, _ = await _open()
    prices["ETHUSD"] = 2720.0  # live trail: stop 2570 (+19 > 0.1 ATR = 10) ...
    await ex.manage_open_trades(rt, {})
    assert b.kinds() == ["place"]  # ... but within the first minute: no call
    clock[0] += 61
    await ex.manage_open_trades(rt, {})
    assert b.calls[-1] == ("edit", "101", pytest.approx(2570.0))
    prices["ETHUSD"] = 2725.0  # stop 2575: only +5 < 10
    clock[0] += 61
    await ex.manage_open_trades(rt, {})
    assert b.kinds() == ["place", "edit"]
    assert ex._stop_state(await _trade())["price"] == pytest.approx(2570.0)


@pytest.mark.asyncio
async def test_a_refused_edit_places_the_new_stop_before_cancelling_the_old(clock):
    rt, om, b, prices, _ = await _open()
    b.edit_ok = False
    prices["ETHUSD"] = 2740.0
    clock[0] += 61
    await ex.manage_open_trades(rt, {})
    assert b.kinds() == ["place", "edit", "place", "cancel"] and b.calls[-1] == ("cancel", "101")
    assert list(b.resting) == ["102"] and ex._stop_state(await _trade())["id"] == "102"


@pytest.mark.asyncio
async def test_a_stop_removed_by_the_deadman_switch_is_re_placed(clock):
    rt, om, b, prices, _ = await _open()
    b.resting.clear()  # heartbeats lapsed: Delta cancelled our orders
    b.read_ok = False  # an unreadable book is NOT "missing": no duplicate
    clock[0] += 61
    await ex.manage_open_trades(rt, {})
    assert b.kinds() == ["place"]
    b.read_ok = True
    clock[0] += 61
    await ex.manage_open_trades(rt, {})
    assert b.kinds() == ["place", "place"] and list(b.resting) == ["102"]


@pytest.mark.asyncio
async def test_a_software_close_cancels_the_stop_and_sweeps_tagged_duplicates(clock):
    rt, om, b, prices, _ = await _open()
    b.resting["555"] = {"id": 555, "state": "pending", "client_order_id": ex.STOP_TAG + "ETHUSD-1"}  # lost reply
    b.resting["777"] = {"id": 777, "state": "open", "client_order_id": "book-order"}  # not ours: untouched
    prices["ETHUSD"] = 2860.0  # TP1, call not strong -> TP
    await ex.manage_open_trades(rt, {})
    t = await _trade()
    assert t.result == "TP" and om.closed == ["ETHUSD"]
    assert sorted(c[1] for c in b.calls if c[0] == "cancel") == ["101", "555"] and list(b.resting) == ["777"]


@pytest.mark.asyncio
async def test_an_exchange_stop_fill_closes_the_record_through_sync(clock):
    rt, om, b, prices, _ = await _open()
    om.positions.clear()  # Delta's stop fired: the position is flat
    b.resting.clear()
    prices["ETHUSD"] = 2548.0
    await ex.manage_open_trades(rt, {})
    t = await _trade()
    assert not t.is_open and t.result == "SYNC" and om.closed == []
    assert b.calls[-1] == ("cancel", "101")  # gone already: a harmless, idempotent cancel


# ------------------------------------------------ the LiveBroker helpers
def _broker(client):
    reg = SymbolRegistry(Settings(delta_api_key="k", delta_api_secret="s"))
    reg._by_symbol["ETHUSD"] = ProductInfo(
        product_id=3, symbol="ETHUSD", contract_type="perpetual_futures", tick_size=0.05, contract_value=0.01,
        underlying_asset="ETH", state="live", default_leverage=10, taker_commission_rate=0.0005,
        maker_commission_rate=0.0002, spot_index_symbol="",
    )
    return LiveBroker(client, reg, None, settings=Settings(delta_api_key="k", delta_api_secret="s"))


class _Client:
    def __init__(self):
        self.bodies = []

    async def place_order(self, body):
        self.bodies.append(("POST", body))
        return {"id": 42, "state": "pending"}

    async def edit_order(self, body):
        self.bodies.append(("PUT", body))
        raise DeltaApiError(400, {"error": {"code": "invalid_stop_price"}})

    async def cancel_order(self, body):
        self.bodies.append(("DELETE", body))
        raise DeltaApiError(400, {"error": {"code": "open_order_not_found"}})


@pytest.mark.asyncio
async def test_live_broker_stop_order_fields():
    c = _Client()
    lb = _broker(c)
    assert await lb.place_stop_order("ETHUSD", "sell", 1, 2551.03, "an-stop-ETHUSD-1") == "42"
    body = c.bodies[0][1]
    assert body == {"product_id": 3, "size": 1, "side": "sell", "order_type": "market_order",
                    "stop_order_type": "stop_loss_order", "stop_price": "2551.05", "stop_trigger_method": "mark_price",
                    "reduce_only": True, "client_order_id": "an-stop-ETHUSD-1"}
    assert await lb.edit_stop_order("ETHUSD", "42", 2570.0) is None  # refused: caller places-then-cancels
    assert await lb.cancel_stop_order("ETHUSD", "42") is True  # already gone counts as cancelled
