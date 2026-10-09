"""Team test agent, cycle 1: the Analyst's real-money paths (app/analyst/execution.py).

Covers what tests/test_analyst_execution.py does not: plan_size edges, SYNC/FLIP/TP
extension/one-way stop across ticks, the live vs paper close price, gold priced
from stored candles, the open-trade gates (kill switch, one per market, book
agreement, coins the book holds, paper_only, rejected orders, equity unavailable)
and the hourly paper-trade walker's interplay with the tick-by-tick exits.
Tests marked xfail(strict=True) pin a real bug (see the reason)."""
import datetime as dt
import time
import types

import pytest
from sqlalchemy import select

from app.analyst import execution as ex
from app.config import Settings
from app.data.app_state import set_kill_switch, set_trading_mode
from app.data.db import SessionLocal
from app.data.models import AnalystTrade, Candle
from app.execution.broker_base import OrderResult, PositionView

UTC = dt.timezone.utc


def _s(**kw):
    # These tests pin the base sizing mechanics (analyst_max_trade_risk); the
    # CRO's tighter caps (analyst_risk_cap, analyst_min_liq_to_stop -- only
    # ever skip a trade) are switched off here and tested in test_risk_policy.py.
    kw.setdefault("analyst_risk_cap", 0)
    kw.setdefault("analyst_min_liq_to_stop", 0)
    kw.setdefault("analyst_target_leverage", 0)  # leverage target: tested in test_analyst_leverage.py
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


def _call(sym, decision, score, entry=2700.0, stop=2550.0, tp=2850.0):
    return {"symbol": sym, "decision": decision, "conviction": int(round(abs(score) * 10)), "score": score,
            "size_factor": 1.0, "entry": entry, "stop_loss": stop, "take_profit": tp, "families": {}, "notes": [],
            "factors": [], "time": "2026-10-06T00:00:00+00:00"}


class _OM:
    """Order manager fake: fills at the cache price + 1, can reject chosen
    symbols, and returns a configurable result for close_position."""

    def __init__(self, prices, equity=14.0, positions=None, reject=(), close_result="fill"):
        self.prices, self.equity = prices, equity
        self.positions = positions if positions is not None else {}
        self.reject, self.close_result = set(reject), close_result
        self.orders, self.closed = [], []

    async def get_equity(self):
        if isinstance(self.equity, Exception):
            raise self.equity
        return self.equity

    async def get_positions(self):
        return self.positions

    async def submit(self, req):
        self.orders.append(req)
        if req.symbol in self.reject:
            return OrderResult(0, None, "rejected", 0, None, 0.0, reason="exchange said no")
        fill = self.prices[req.symbol] + 1.0
        size = req.size if req.side == "buy" else -req.size
        self.positions[f"{req.symbol}:consolidated"] = PositionView(req.symbol, size, fill, 0, 0, strategy="consolidated")
        return OrderResult(1, "x", "closed", req.size, fill, 0.0)

    async def close_position(self, symbol, strategy):
        self.closed.append(symbol)
        if self.close_result == "reject":
            return OrderResult(0, None, "rejected", 0, None, 0.0, reason="Delta unreachable")
        self.positions.pop(f"{symbol}:{strategy}", None)
        return OrderResult(2, "y", "closed", 1, 2600.0, 0.0)


def _runtime(om, signals=None, engine="combined", **kw):
    cvs = {"ETHUSD": 0.01, "BTCUSD": 0.001, "SOLUSD": 1.0, "BNBUSD": 0.1}
    reg = types.SimpleNamespace(get=lambda s: types.SimpleNamespace(contract_value=cvs.get(s, 1.0)))
    return types.SimpleNamespace(settings=_s(trading_engine=engine, **kw), order_manager=om, registry=reg,
                                 live_signals=signals or {}, cache=types.SimpleNamespace(get_last_price=om.prices.get))


async def _add_trade(**kw) -> int:
    base = dict(symbol="ETHUSD", side="BUY", conviction=7, score=0.7, size_factor=1.0, entry=2700.0,
                stop_loss=2550.0, initial_stop=2550.0, take_profit=2850.0, atr=100.0, peak=2700.0,
                contracts=1, live=True, entry_time=dt.datetime.now(UTC), reasons={})
    base.update(kw)
    async with SessionLocal() as s:
        t = AnalystTrade(**base)
        s.add(t)
        await s.commit()
        return t.id


async def _get(trade_id) -> AnalystTrade:
    async with SessionLocal() as s:
        return await s.get(AnalystTrade, trade_id)


async def _all_trades():
    async with SessionLocal() as s:
        return (await s.execute(select(AnalystTrade).order_by(AnalystTrade.id))).scalars().all()


@pytest.fixture(autouse=True)
async def _live_and_running():
    await set_trading_mode("live")
    await set_kill_switch(False)
    yield
    await set_kill_switch(False)


def _held(sym="ETHUSD", size=1):
    return {f"{sym}:consolidated": PositionView(sym, size, 2700.0, 0, 0, strategy="consolidated")}


# ================================================================ plan_size
def test_plan_size_refuses_nonsense_inputs():
    s = _s()
    assert ex.plan_size(equity=0.0, price=2700, stop=2550, contract_value=0.01, settings=s) is None
    assert ex.plan_size(equity=-5.0, price=2700, stop=2550, contract_value=0.01, settings=s) is None
    assert ex.plan_size(equity=14.0, price=2700, stop=2700, contract_value=0.01, settings=s) is None  # no risk = no size
    assert ex.plan_size(equity=14.0, price=0.0, stop=-1.0, contract_value=0.01, settings=s) is None
    assert ex.plan_size(equity=14.0, price=2700, stop=2550, contract_value=0.0, settings=s) is None


def test_tiny_equity_skips_a_contract_whose_margin_needs_more_than_max_leverage():
    # $0.50: one $27 ETH contract needs ceil(27 / 0.425) = 64x > 50x -> skipped, not opened at 50x
    assert ex.plan_size(equity=0.5, price=2700, stop=2699, contract_value=0.01, settings=_s()) is None


def test_leverage_is_the_ceiling_and_exactly_max_leverage_is_allowed():
    # notional $27 on $0.6417: 27 / (0.85 * 0.6417) = 49.5 -> 50x, the ceiling, which is still allowed
    assert ex.plan_size(equity=0.6417, price=2700, stop=2699, contract_value=0.01, settings=_s()) == (1, 50.0)
    assert ex.plan_size(equity=0.6417, price=2700, stop=2699, contract_value=0.01,
                        settings=_s(max_leverage=49.0)) is None


def test_one_contract_rule_at_exactly_the_risk_cap_and_leverage_never_below_1x():
    # one contract risks $7 = exactly 70% of $10 -> allowed; $100 notional / $8.50 -> 12x
    assert ex.plan_size(equity=10.0, price=100, stop=93, contract_value=1.0, settings=_s()) == (1, 12.0)
    assert ex.plan_size(equity=10.0, price=100, stop=92.9, contract_value=1.0, settings=_s()) is None
    n, lev = ex.plan_size(equity=1_000_000.0, price=100, stop=99, contract_value=1.0, settings=_s())
    assert n == 20_000 and lev == 3.0  # 2% of $1M / $1 -> 20k contracts, $2M / $850k -> 3x
    n, lev = ex.plan_size(equity=10_000.0, price=100, stop=50, contract_value=1.0, settings=_s())
    assert n == 4 and lev == 1.0


def test_short_side_sizes_off_the_distance_not_the_sign():
    assert ex.plan_size(equity=14.0, price=2700, stop=2850, contract_value=0.01, settings=_s()) == (1, 3.0)


# ================================================================ open_best_trade gates
@pytest.mark.asyncio
async def test_kill_switch_blocks_even_a_paper_only_open():
    await set_kill_switch(True)
    om = _OM({"XAUTUSD": 4150.0})
    gold = _call("XAUTUSD", "BUY", 0.6, 4150.0, 4100.0, 4200.0)
    assert await ex.open_best_trade(_runtime(om), {"XAUTUSD": gold}, paper_only=True, require_book=False) is None
    assert await _all_trades() == [] and om.orders == []


@pytest.mark.asyncio
async def test_book_agreement_needs_the_entry_line_and_a_known_signal():
    om = _OM({"ETHUSD": 2700.0})
    results = {"ETHUSD": _call("ETHUSD", "BUY", 0.7)}
    assert await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.10}), results) is None  # below the 0.15 line
    assert await ex.open_best_trade(_runtime(om, {}), results) is None  # book has no view on the coin
    assert om.orders == []
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.15}), results)  # exactly on the line
    assert t is not None and t.live


@pytest.mark.asyncio
async def test_a_coin_the_book_holds_is_skipped_and_the_next_best_one_trades():
    om = _OM({"ETHUSD": 2700.0, "BTCUSD": 85000.0}, positions=_held("BTCUSD", -1))
    results = {"BTCUSD": _call("BTCUSD", "BUY", 0.9, 85000, 81850, 88150), "ETHUSD": _call("ETHUSD", "BUY", 0.6)}
    t = await ex.open_best_trade(_runtime(om, {"BTCUSD": 0.5, "ETHUSD": 0.5}), results)
    assert t.symbol == "ETHUSD" and [o.symbol for o in om.orders] == ["ETHUSD"]


@pytest.mark.asyncio
async def test_a_rejected_order_leaves_no_record_and_the_next_call_is_tried():
    om = _OM({"ETHUSD": 2700.0, "BTCUSD": 85000.0}, reject={"ETHUSD"})
    results = {"ETHUSD": _call("ETHUSD", "BUY", 0.8), "BTCUSD": _call("BTCUSD", "BUY", 0.6, 85000, 81850, 88150)}
    t = await ex.open_best_trade(_runtime(om, {"BTCUSD": 0.5, "ETHUSD": 0.5}), results)
    assert [o.symbol for o in om.orders] == ["ETHUSD", "BTCUSD"]
    assert t.symbol == "BTCUSD" and t.contracts == 1 and t.entry == 85001.0
    assert [x.symbol for x in await _all_trades()] == ["BTCUSD"]


@pytest.mark.asyncio
async def test_equity_unavailable_opens_nothing():
    om = _OM({"ETHUSD": 2700.0}, equity=RuntimeError("balance unavailable"))
    assert await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}), {"ETHUSD": _call("ETHUSD", "BUY", 0.7)}) is None
    assert om.orders == [] and await _all_trades() == []


@pytest.mark.asyncio
async def test_strongest_absolute_score_wins_and_a_short_is_placed_as_a_sell():
    om = _OM({"ETHUSD": 2700.0, "BTCUSD": 85000.0}, equity=1000.0)
    results = {"ETHUSD": _call("ETHUSD", "SELL", -0.9, 2700, 2850, 2550),
               "BTCUSD": _call("BTCUSD", "BUY", 0.6, 85000, 81850, 88150)}
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": -0.5, "BTCUSD": 0.5}), results)
    assert t.symbol == "ETHUSD" and t.side == "SELL" and om.orders[0].side == "sell"
    assert t.stop_loss == pytest.approx(2851.0) and t.take_profit == pytest.approx(2551.0)  # re-centred on 2701
    assert t.initial_stop == t.stop_loss and t.peak == t.entry


@pytest.mark.asyncio
async def test_wait_and_low_conviction_calls_are_never_traded():
    om = _OM({"ETHUSD": 2700.0, "BTCUSD": 85000.0})
    wait = _call("ETHUSD", "WAIT", 0.9)
    weak = _call("BTCUSD", "BUY", 0.15, 85000, 81850, 88150)  # |score| 0.15 < analyst_min_score 0.20
    assert await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5, "BTCUSD": 0.5}), {"ETHUSD": wait, "BTCUSD": weak}) is None
    assert om.orders == []


@pytest.mark.asyncio
async def test_paper_only_never_sends_an_order_even_under_the_combined_engine():
    om = _OM({"ETHUSD": 2700.0})
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}), {"ETHUSD": _call("ETHUSD", "BUY", 0.7)}, paper_only=True)
    assert t is not None and not t.live and t.contracts == 0 and om.orders == []


@pytest.mark.asyncio
async def test_paper_mode_under_the_combined_engine_routes_to_the_broker_but_is_not_live():
    await set_trading_mode("paper")
    om = _OM({"ETHUSD": 2700.0})
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}), {"ETHUSD": _call("ETHUSD", "BUY", 0.7)})
    assert t.contracts == 1 and not t.live and len(om.orders) == 1


@pytest.mark.asyncio
async def test_an_open_trade_on_the_same_coin_blocks_a_second_one():
    await _add_trade(symbol="ETHUSD")
    om = _OM({"ETHUSD": 2700.0}, positions=_held("ETHUSD"))
    assert await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}), {"ETHUSD": _call("ETHUSD", "BUY", 0.7)}) is None
    assert om.orders == []


@pytest.mark.asyncio
async def test_one_trade_per_market_even_when_the_open_coin_dropped_out_of_this_run():
    await _add_trade(symbol="ETHUSD")  # live ETH position open; ETH's analysis failed this minute
    om = _OM({"ETHUSD": 2700.0, "BTCUSD": 85000.0}, equity=1000.0, positions=_held("ETHUSD"))
    results = {"BTCUSD": _call("BTCUSD", "BUY", 0.8, 85000, 81850, 88150)}
    assert await ex.open_best_trade(_runtime(om, {"BTCUSD": 0.5}), results) is None
    assert om.orders == []


# ================================================================ manage_open_trades
@pytest.mark.asyncio
async def test_sync_when_the_position_is_flat_on_the_exchange_but_still_keyed():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2650.0}, positions=_held("ETHUSD", 0))  # key present, size 0
    await ex.manage_open_trades(_runtime(om), {})
    t = await _get(tid)
    assert not t.is_open and t.result == "SYNC" and t.exit_price == 2650.0 and om.closed == []
    assert t.r_multiple == pytest.approx(-50 / 150 - 2 * 0.0007 * 2700 / 150)


@pytest.mark.asyncio
async def test_a_strong_opposite_call_flips_out_on_the_exchange_at_the_fill():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2710.0}, positions=_held("ETHUSD"))
    await ex.manage_open_trades(_runtime(om), {"ETHUSD": _call("ETHUSD", "SELL", -0.7)})
    t = await _get(tid)
    assert om.closed == ["ETHUSD"] and not t.is_open and t.result == "FLIP" and t.exit_price == 2600.0


@pytest.mark.asyncio
async def test_a_weak_opposite_call_does_not_flip():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2710.0}, positions=_held("ETHUSD"))
    await ex.manage_open_trades(_runtime(om), {"ETHUSD": _call("ETHUSD", "SELL", -0.3)})  # conviction 3 < 4
    t = await _get(tid)
    assert t.is_open and om.closed == []


@pytest.mark.asyncio
async def test_tp1_with_a_still_strong_call_extends_and_persists_without_an_order():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2860.0}, positions=_held("ETHUSD"))
    await ex.manage_open_trades(_runtime(om), {"ETHUSD": _call("ETHUSD", "BUY", 0.7)})
    t = await _get(tid)
    assert t.is_open and t.extended and om.closed == []
    assert t.take_profit == pytest.approx(3000.0)  # one more R
    assert t.stop_loss == pytest.approx(2775.0)  # the +0.5R lock (above the 2760 tight trail)
    assert t.peak == 2860.0


@pytest.mark.asyncio
async def test_tp1_without_a_strong_call_closes_live_at_the_fill():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2860.0}, positions=_held("ETHUSD"))
    await ex.manage_open_trades(_runtime(om), {"ETHUSD": _call("ETHUSD", "BUY", 0.3)})
    t = await _get(tid)
    assert om.closed == ["ETHUSD"] and t.result == "TP" and t.exit_price == 2600.0  # the fill, not the level


@pytest.mark.asyncio
async def test_the_stop_never_moves_back_across_ticks():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2800.0}, positions=_held("ETHUSD"))
    rt = _runtime(om, analyst_trail_mode="live")  # the old ATR trail ("after_tp1": test_analyst_trail_mode.py)
    await ex.manage_open_trades(rt, {})
    first = await _get(tid)
    assert first.stop_loss == pytest.approx(2650.0) and first.peak == 2800.0  # 2800 - 1.5 x 100
    om.prices["ETHUSD"] = 2700.0
    await ex.manage_open_trades(rt, {})
    second = await _get(tid)
    assert second.is_open and second.stop_loss == pytest.approx(2650.0) and second.peak == 2800.0
    om.prices["ETHUSD"] = 2640.0
    await ex.manage_open_trades(rt, {})
    third = await _get(tid)
    assert not third.is_open and third.result == "TRAIL"


@pytest.mark.asyncio
async def test_paper_target_fills_at_its_level_even_on_a_gap():
    tid = await _add_trade(contracts=0, live=False)
    om = _OM({"ETHUSD": 2900.0})
    await ex.manage_open_trades(_runtime(om, engine="book"), {})
    t = await _get(tid)
    assert t.result == "TP" and t.exit_price == 2850.0 and om.closed == []
    assert t.r_multiple == pytest.approx(1.0 - 2 * 0.0007 * 2700 / 150)


@pytest.mark.asyncio
async def test_paper_stop_fills_where_the_tick_crossed_it():
    tid = await _add_trade(contracts=0, live=False)
    om = _OM({"ETHUSD": 2500.0})
    await ex.manage_open_trades(_runtime(om, engine="combined"), {})  # paper record even under combined
    t = await _get(tid)
    assert t.result == "SL" and t.exit_price == 2500.0 and om.closed == []


@pytest.mark.asyncio
async def test_gold_is_priced_from_the_stored_15m_close_when_the_feed_has_none():
    tid = await _add_trade(symbol="XAUTUSD", entry=4150.0, stop_loss=4100.0, initial_stop=4100.0,
                           take_profit=4200.0, atr=40.0, peak=4150.0, contracts=0, live=False)
    om = _OM({})  # the websocket streams no XAUTUSD price
    rt = _runtime(om)
    await ex.manage_open_trades(rt, {})
    assert (await _get(tid)).is_open  # no price at all: left alone
    async with SessionLocal() as s:
        s.add(Candle(symbol="XAUTUSD", resolution="15m", time=int(time.time()) - 900,
                     open=4110.0, high=4112.0, low=4085.0, close=4090.0, volume=0.0))
        await s.commit()
    await ex.manage_open_trades(rt, {})
    t = await _get(tid)
    assert not t.is_open and t.result == "SL" and t.exit_price == 4090.0 and om.closed == []


@pytest.mark.asyncio
async def test_a_rejected_exchange_close_keeps_the_record_open():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2540.0}, positions=_held("ETHUSD"), close_result="reject")
    await ex.manage_open_trades(_runtime(om), {})  # stop crossed, close attempted, Delta refuses
    assert om.closed == ["ETHUSD"]
    assert (await _get(tid)).is_open


@pytest.mark.asyncio
async def test_a_live_record_still_closes_on_the_exchange_after_the_engine_is_switched():
    tid = await _add_trade()
    om = _OM({"ETHUSD": 2540.0}, positions=_held("ETHUSD"))
    await ex.manage_open_trades(_runtime(om, engine="book"), {})
    assert om.closed == ["ETHUSD"]
    assert (await _get(tid)).result == "SL"


# ================================================================ the hourly paper walker vs tick exits
@pytest.mark.asyncio
async def test_the_hourly_paper_walker_does_not_retro_stop_a_trailed_trade():
    from app.analyst.service import update_paper_trades

    entry_ts = (int(time.time()) // 3600 - 3) * 3600
    tid = await _add_trade(contracts=0, live=False, entry=100.0, initial_stop=97.0, stop_loss=101.0,
                           take_profit=106.0, peak=103.5, atr=2.0, extended=True,
                           entry_time=dt.datetime.fromtimestamp(entry_ts, UTC))
    async with SessionLocal() as s:  # bar 1 dipped to 99.5 (stop was 97 then), bar 2 ran to 103.5
        s.add(Candle(symbol="BINANCE:ETHUSD", resolution="1h", time=entry_ts, open=100, high=101, low=99.5,
                     close=100.8, volume=1))
        s.add(Candle(symbol="BINANCE:ETHUSD", resolution="1h", time=entry_ts + 3600, open=100.8, high=103.5,
                     low=101.6, close=103.2, volume=1))
        await s.commit()
    await update_paper_trades()
    assert (await _get(tid)).is_open
