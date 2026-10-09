"""CRO risk policy, cycle 3 (2026-10-07): the Analyst's per-trade risk cap and
the liquidation-beyond-the-stop check. Both may only ever SKIP a trade: with
them on, plan_size returns either None or exactly what it returned without them."""
import itertools
import types

import pytest

from app.analyst import execution as ex
from app.config import Settings
from app.data.app_state import set_kill_switch, set_trading_mode
from app.execution.broker_base import OrderResult, PositionView


def _s(**kw):
    # The risk cap is pinned at the lowest-fitting leverage; the leverage
    # target (default 20, clamped) is tested in test_analyst_leverage.py.
    kw.setdefault("analyst_target_leverage", 0)
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


OLD = dict(analyst_risk_cap=0.0, analyst_min_liq_to_stop=0.0)  # the behaviour before this policy


# ------------------------------------------------------------- defaults
def test_defaults_are_the_recommended_policy():
    s = _s()
    assert s.analyst_risk_cap == 0.15 and s.analyst_min_liq_to_stop == 1.5
    assert ex.trade_risk_cap(s) == 0.15  # tighter than the user's 0.70
    assert ex.trade_risk_cap(_s(**OLD)) == 0.70


def test_live_numbers_2026_10_06_eth_trades_btc_is_skipped():
    """GET /api/analyst on the VPS, equity $13.905: BTC 1 contract risks $2.86
    (20.6%), ETH $1.69 (12.2%)."""
    s = _s()
    assert ex.plan_size(equity=13.905, price=2709.675, stop=2540.406, contract_value=0.01, settings=s) == (1, 3.0)
    assert ex.plan_size(equity=13.905, price=85981.75, stop=83123.67, contract_value=0.001, settings=s) is None
    # the old rules took BTC at 8x
    assert ex.plan_size(equity=13.905, price=85981.75, stop=83123.67, contract_value=0.001,
                        settings=_s(**OLD)) == (1, 8.0)


# ------------------------------------------------------------- risk cap
def test_the_cap_is_the_tighter_of_the_two_and_never_loosens():
    # user cap 0.10 is tighter than the CRO's 0.50: ETH at 12% stays refused
    s = _s(analyst_max_trade_risk=0.10, analyst_risk_cap=0.50)
    assert ex.trade_risk_cap(s) == 0.10
    assert ex.plan_size(equity=13.9, price=2709.675, stop=2540.406, contract_value=0.01, settings=s) is None


def test_exactly_at_the_cap_is_allowed_just_above_is_not():
    s = _s(analyst_risk_cap=0.15, analyst_min_liq_to_stop=0.0)
    # $1.50 risk on $10 = 15.0% exactly
    assert ex.plan_size(equity=10.0, price=100, stop=98.5, contract_value=1.0, settings=s) == (1, 12.0)
    assert ex.plan_size(equity=10.0, price=100, stop=98.49, contract_value=1.0, settings=s) is None


def test_the_cap_does_not_touch_normally_sized_trades():
    # $1,000: 2% risk = 13 ETH contracts, far inside every cap
    n, lev = ex.plan_size(equity=1000.0, price=2700, stop=2550, contract_value=0.01, settings=_s())
    assert (n, lev) == (13, 1.0)


# ------------------------------------------------------------- liquidation check
def test_liquidation_estimate_is_conservative():
    # 10x: 10% minus the 0.5% maintenance margin and 0.12% fee allowance
    assert ex.liquidation_distance(10.0) == pytest.approx(0.1 - 0.005 - 0.0012)
    assert ex.liquidation_distance(1.0) == pytest.approx(0.9938)


def test_trade_whose_liquidation_sits_inside_the_stop_is_skipped():
    # $10 equity, one $100 SOL-like contract -> 12x -> liquidation ~7.7% away;
    # a 9% stop would never be reached -- liquidation fires first
    risky = dict(analyst_max_trade_risk=0.95, analyst_risk_cap=0.0)
    assert ex.plan_size(equity=10.0, price=100, stop=91, contract_value=1.0,
                        settings=_s(**risky, analyst_min_liq_to_stop=0.0)) == (1, 12.0)
    assert ex.plan_size(equity=10.0, price=100, stop=91, contract_value=1.0,
                        settings=_s(**risky, analyst_min_liq_to_stop=1.0)) is None
    # same for a short: the distance, not the sign
    assert ex.plan_size(equity=10.0, price=100, stop=109, contract_value=1.0,
                        settings=_s(**risky, analyst_min_liq_to_stop=1.0)) is None


def test_the_buffer_ratio_needs_liquidation_well_beyond_the_stop():
    # 12x -> 7.71% to liquidation. A 6% stop is beyond 1.0x but not 1.5x (9%)
    base = dict(analyst_max_trade_risk=0.95, analyst_risk_cap=0.0)
    assert ex.plan_size(equity=10.0, price=100, stop=94, contract_value=1.0,
                        settings=_s(**base, analyst_min_liq_to_stop=1.0)) == (1, 12.0)
    assert ex.plan_size(equity=10.0, price=100, stop=94, contract_value=1.0,
                        settings=_s(**base, analyst_min_liq_to_stop=1.5)) is None


# ------------------------------------------------------------- only ever skips
def test_the_policy_only_ever_skips_never_changes_a_size_or_leverage():
    new, old = _s(), _s(**OLD)
    seen_skip = seen_same = 0
    for equity, price, stop_pct, cv in itertools.product(
            (0.7, 3.0, 9.0, 13.9, 30.0, 150.0, 5000.0),
            (2.0, 120.0, 785.0, 2700.0, 86000.0),
            (0.002, 0.01, 0.03, 0.06, 0.12, 0.3),
            (0.001, 0.01, 0.1, 1.0)):
        for d in (1, -1):
            kw = dict(equity=equity, price=price, stop=price * (1 - d * stop_pct), contract_value=cv)
            a, b = ex.plan_size(**kw, settings=new), ex.plan_size(**kw, settings=old)
            assert a is None or a == b, (kw, a, b)
            if a is None and b is not None:
                seen_skip += 1
            elif a is not None:
                seen_same += 1
                # every trade it lets through has liquidation 1.5x beyond the stop
                assert ex.liquidation_distance(a[1]) >= 1.5 * stop_pct - 1e-12
                # and risks no more than the cap (or the 2% target when several contracts)
                assert a[0] * price * stop_pct * cv <= max(0.15, 0.02) * equity + 1e-9
    assert seen_skip > 0 and seen_same > 0


# ------------------------------------------------------------- end to end
class _OM:
    def __init__(self, prices, equity=13.9):
        self.prices, self.equity, self.positions, self.orders = prices, equity, {}, []

    async def get_equity(self):
        return self.equity

    async def get_positions(self):
        return self.positions

    async def submit(self, req):
        self.orders.append(req)
        fill = self.prices[req.symbol]
        self.positions[f"{req.symbol}:consolidated"] = PositionView(req.symbol, req.size, fill, 0, 0,
                                                                     strategy="consolidated")
        return OrderResult(1, "x", "closed", req.size, fill, 0.0)


def _call(sym, score, entry, stop, tp):
    return {"symbol": sym, "decision": "BUY", "conviction": int(round(abs(score) * 10)), "score": score,
            "size_factor": 1.0, "entry": entry, "stop_loss": stop, "take_profit": tp, "families": {}, "notes": [],
            "factors": [], "time": "2026-10-06T12:00:00+00:00"}


@pytest.fixture(autouse=True)
async def _live():
    await set_trading_mode("live")
    await set_kill_switch(False)
    yield


@pytest.mark.asyncio
async def test_open_best_trade_skips_the_too_risky_best_call_and_takes_eth():
    cvs = {"BTCUSD": 0.001, "ETHUSD": 0.01}
    om = _OM({"BTCUSD": 85981.75, "ETHUSD": 2709.675})
    rt = types.SimpleNamespace(
        settings=_s(trading_engine="combined"), order_manager=om,
        registry=types.SimpleNamespace(get=lambda s: types.SimpleNamespace(contract_value=cvs[s])),
        live_signals={"BTCUSD": 0.5, "ETHUSD": 0.5}, cache=types.SimpleNamespace(get_last_price=om.prices.get))
    results = {"BTCUSD": _call("BTCUSD", 0.6, 85981.75, 83123.67, 88839.83),
               "ETHUSD": _call("ETHUSD", 0.5, 2709.675, 2540.406, 2878.944)}
    gates = {}
    t = await ex.open_best_trade(rt, results, gates=gates)
    assert gates["BTCUSD"] == "too_risky_to_size"
    assert t.symbol == "ETHUSD" and [o.symbol for o in om.orders] == ["ETHUSD"] and om.orders[0].leverage == 3.0
