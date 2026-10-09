"""The Analyst as a trader: sizing, the one-way trailing stop, the adaptive TP1,
the >50% confidence gate and the trend-book agreement rule (2026-10-06)."""
import types

import pytest
from sqlalchemy import select

from app.analyst import execution as ex
from app.config import Settings
from app.data.app_state import set_kill_switch, set_trading_mode
from app.data.db import SessionLocal
from app.data.models import AnalystTrade
from app.execution.broker_base import OrderResult, PositionView


def _s(**kw):
    # These tests pin the base sizing mechanics (analyst_max_trade_risk); the
    # CRO's tighter caps (analyst_risk_cap, analyst_min_liq_to_stop -- only
    # ever skip a trade) are switched off here and tested in test_risk_policy.py.
    kw.setdefault("analyst_risk_cap", 0)
    kw.setdefault("analyst_min_liq_to_stop", 0)
    kw.setdefault("analyst_target_leverage", 0)  # leverage target: tested in test_analyst_leverage.py
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


# ------------------------------------------------------------- sizing
def test_risk_based_size_and_the_one_contract_rule_for_a_small_account():
    s = _s()
    # $14: ETH 0.01 contract, stop $150 away -> $1.50 risk (10.7%) -> 1 contract at 3x
    assert ex.plan_size(equity=14.0, price=2700, stop=2550, contract_value=0.01, settings=s) == (1, 3.0)
    # BTC 0.001 contract, stop $3,150 away -> $3.15 risk (22%) -> allowed now (cap 70%), 8x
    assert ex.plan_size(equity=14.0, price=85000, stop=81850, contract_value=0.001, settings=s) == (1, 8.0)
    # one contract risking more than the 70% cap is still refused
    assert ex.plan_size(equity=14.0, price=122, stop=110, contract_value=1.0, settings=s) is None
    assert ex.plan_size(equity=14.0, price=85000, stop=81850, contract_value=0.001,
                        settings=_s(analyst_max_trade_risk=0.12)) is None
    # $1,000: 2% = $20 of risk / $1.50 per ETH contract -> 13 contracts
    n, lev = ex.plan_size(equity=1000.0, price=2700, stop=2550, contract_value=0.01, settings=s)
    assert n == 13 and lev == 1.0


# ------------------------------------------------------------- dynamic exits
def _exit(**kw):
    base = dict(side="BUY", entry=100.0, stop=97.0, initial_stop=97.0, atr=2.0, peak=100.0, take_profit=103.0,
                extended=False, still_strong=False,
                settings=_s(analyst_trail_mode="live"))  # the old trail; "after_tp1": test_analyst_trail_mode.py
    base.update(kw)
    return ex.update_exit(**base)


def test_stop_only_ever_moves_forward():
    u = _exit(price=102.0)  # peak 102 -> trail 1.5 ATR = 3 -> 99
    assert u.stop == pytest.approx(99.0) and u.action is None
    back = _exit(price=100.5, stop=u.stop, peak=u.peak)  # price falls back: the stop stays at 99
    assert back.stop == pytest.approx(99.0) and back.peak == 102.0


def test_after_plus_one_r_the_trail_tightens_and_the_target_moves_on_a_strong_call():
    u = _exit(price=103.5, still_strong=True)  # +1.17R: tight trail 1 ATR from 103.5 = 101.5
    assert u.action is None and u.extended and u.take_profit == pytest.approx(106.0)
    assert u.stop == pytest.approx(101.5)  # also the +0.5R lock


def test_tp1_closes_unless_the_call_is_still_strong():
    assert _exit(price=103.2).action == "TP"
    assert _exit(price=103.2, still_strong=True).action is None


def test_stop_hit_reports_trail_once_the_stop_has_moved():
    assert _exit(price=96.9).action == "SL"
    assert _exit(price=100.9, stop=101.0, peak=104.0).action == "TRAIL"


def test_short_trade_mirrors():
    u = _exit(side="SELL", entry=100.0, stop=103.0, initial_stop=103.0, peak=100.0, take_profit=97.0, price=98.0)
    assert u.stop == pytest.approx(101.0) and u.action is None  # 98 + 3


# ------------------------------------------------------------- gates and the book
def test_agreement_with_the_trend_book():
    assert ex.agrees_with_book("BUY", 0.4) and ex.agrees_with_book("SELL", -0.3)
    assert not ex.agrees_with_book("BUY", 0.05) and not ex.agrees_with_book("BUY", -0.4)
    assert not ex.agrees_with_book("SELL", None)


def _call(sym, decision, score, entry=2700.0, stop=2550.0, tp=2850.0):
    return {"symbol": sym, "decision": decision, "conviction": int(round(abs(score) * 10)), "score": score,
            "size_factor": 1.0, "entry": entry, "stop_loss": stop, "take_profit": tp, "families": {}, "notes": [],
            "factors": [], "time": "2026-10-06T00:00:00+00:00"}


class _OM:
    def __init__(self, equity=14.0, positions=None):
        self.equity, self.positions, self.orders, self.closed = equity, positions or {}, [], []

    async def get_equity(self):
        return self.equity

    async def get_positions(self):
        return self.positions

    async def submit(self, req):
        self.orders.append(req)
        self.positions[f"{req.symbol}:consolidated"] = PositionView(req.symbol, req.size, 2701.0, 0, 0, strategy="consolidated")
        return OrderResult(1, "x", "closed", req.size, 2701.0, 0.0)

    async def close_position(self, symbol, strategy):
        self.closed.append(symbol)
        self.positions.pop(f"{symbol}:{strategy}", None)
        return OrderResult(2, "y", "closed", 1, 2600.0, 0.0)


def _runtime(om, signals, engine="combined", **kw):
    reg = types.SimpleNamespace(get=lambda s: types.SimpleNamespace(contract_value=0.01 if s == "ETHUSD" else 0.001))
    prices = {"ETHUSD": 2700.0, "BTCUSD": 85000.0}
    return types.SimpleNamespace(settings=_s(trading_engine=engine, **kw), order_manager=om, registry=reg,
                                 live_signals=signals, cache=types.SimpleNamespace(get_last_price=prices.get), _prices=prices)


@pytest.mark.asyncio
async def test_combined_engine_trades_only_confident_calls_with_the_book_and_skips_unaffordable():
    await set_trading_mode("live")
    await set_kill_switch(False)
    om = _OM()
    results = {
        "BTCUSD": _call("BTCUSD", "BUY", 0.8, 85000, 81850, 88150),  # strongest; one contract risks 22% here
        "ETHUSD": _call("ETHUSD", "BUY", 0.6),
        "SOLUSD": _call("SOLUSD", "BUY", 0.3),  # below the 40% gate
    }
    rt = _runtime(om, {"BTCUSD": 0.5, "ETHUSD": 0.4, "SOLUSD": 0.5}, analyst_max_trade_risk=0.12)
    trade = await ex.open_best_trade(rt, results)
    assert trade.symbol == "ETHUSD" and trade.live and trade.contracts == 1
    assert [o.symbol for o in om.orders] == ["ETHUSD"] and om.orders[0].leverage == 3.0
    assert trade.entry == 2701.0 and trade.stop_loss == pytest.approx(2551.0)  # re-centred on the fill


@pytest.mark.asyncio
async def test_no_trade_against_the_book_or_on_a_coin_the_book_holds():
    await set_trading_mode("live")
    await set_kill_switch(False)
    results = {"ETHUSD": _call("ETHUSD", "BUY", 0.7)}
    assert await ex.open_best_trade(_runtime(_OM(), {"ETHUSD": -0.4}), results) is None  # book is short ETH
    held = {"ETHUSD:consolidated": PositionView("ETHUSD", 2, 2700, 0, 0, strategy="consolidated")}
    assert await ex.open_best_trade(_runtime(_OM(positions=held), {"ETHUSD": 0.4}), results) is None


@pytest.mark.asyncio
async def test_kill_switch_blocks_new_trades():
    await set_trading_mode("live")
    await set_kill_switch(True)
    assert await ex.open_best_trade(_runtime(_OM(), {"ETHUSD": 0.4}), {"ETHUSD": _call("ETHUSD", "BUY", 0.7)}) is None
    await set_kill_switch(False)


@pytest.mark.asyncio
async def test_live_trade_is_closed_on_the_exchange_when_its_stop_is_crossed():
    await set_trading_mode("live")
    await set_kill_switch(False)
    om = _OM()
    rt = _runtime(om, {"ETHUSD": 0.4})
    await ex.open_best_trade(rt, {"ETHUSD": _call("ETHUSD", "BUY", 0.6)})
    rt._prices["ETHUSD"] = 2540.0  # below the 2551 stop
    await ex.manage_open_trades(rt, {})
    async with SessionLocal() as s:
        t = (await s.execute(select(AnalystTrade))).scalars().one()
    assert om.closed == ["ETHUSD"] and not t.is_open and t.result == "SL" and t.exit_price == 2600.0  # the fill


@pytest.mark.asyncio
async def test_position_closed_elsewhere_is_reconciled():
    await set_trading_mode("live")
    await set_kill_switch(False)
    om = _OM()
    rt = _runtime(om, {"ETHUSD": 0.4})
    await ex.open_best_trade(rt, {"ETHUSD": _call("ETHUSD", "BUY", 0.6)})
    om.positions.clear()  # e.g. the kill switch flattened it
    await ex.manage_open_trades(rt, {})
    async with SessionLocal() as s:
        t = (await s.execute(select(AnalystTrade))).scalars().one()
    assert not t.is_open and t.result == "SYNC" and om.closed == []


@pytest.mark.asyncio
async def test_book_engine_keeps_the_analyst_on_paper():
    await set_trading_mode("live")
    await set_kill_switch(False)
    om = _OM()
    trade = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.4}, engine="book"), {"ETHUSD": _call("ETHUSD", "BUY", 0.6)})
    assert trade is not None and not trade.live and trade.contracts == 0 and om.orders == []


@pytest.mark.asyncio
async def test_gold_is_paper_only_and_never_blocks_the_crypto_trade():
    await set_trading_mode("live")
    await set_kill_switch(False)
    om = _OM()
    rt = _runtime(om, {"ETHUSD": 0.4})
    gold = _call("XAUTUSD", "BUY", 0.5, 4150.0, 4100.0, 4200.0)
    g = await ex.open_best_trade(rt, {"XAUTUSD": gold}, paper_only=True, require_book=False)
    assert g is not None and not g.live and g.contracts == 0 and om.orders == []  # no book needed, no order
    eth = await ex.open_best_trade(rt, {"ETHUSD": _call("ETHUSD", "BUY", 0.6)})
    assert eth is not None and eth.symbol == "ETHUSD" and eth.live  # gold's open trade did not block it
    assert await ex.open_best_trade(rt, {"XAUTUSD": gold}, paper_only=True, require_book=False) is None  # one gold at a time
