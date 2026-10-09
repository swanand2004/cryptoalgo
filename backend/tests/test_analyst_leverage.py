"""analyst_target_leverage (owner, 2026-10-07: "use 15-20 leverage").

0 = the old behaviour exactly. Above the computed leverage the order uses the
target, but never a leverage whose estimated liquidation sits closer than
analyst_min_liq_to_stop x the stop distance -- then the highest safe whole one."""
import itertools

import pytest

from app.analyst import execution as ex
from app.config import Settings


def _s(**kw):
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


ETH = dict(equity=13.905, price=2709.675, stop=2540.406, contract_value=0.01)  # live numbers, 2026-10-06


def test_default_targets_20x_clamped_to_10x_on_eth_today_and_0_keeps_3x():
    s = _s()
    assert s.analyst_target_leverage == 20.0  # clamped target, owner-approved 2026-10-07
    assert ex.plan_size(**ETH, settings=s) == (1, 10.0)  # liquidation >= 1.5x the stop away
    assert ex.plan_size(**ETH, settings=_s(analyst_target_leverage=0)) == (1, 3.0)


def test_eth_today_target_20_is_clamped_to_10x_by_the_liq_rule():
    # stop 6.25% away; 1.5x that = 9.37%; liquidation_distance(10) = 9.38%, (11) = 8.47%
    assert ex.plan_size(**ETH, settings=_s(analyst_target_leverage=20)) == (1, 10.0)
    assert ex.plan_size(**ETH, settings=_s(analyst_target_leverage=15)) == (1, 10.0)
    assert ex.plan_size(**ETH, settings=_s(analyst_target_leverage=8)) == (1, 8.0)


def test_tight_stop_allows_the_full_target():
    # 2% stop: 1.5x = 3% -> liquidation_distance(20) = 4.38% is safe
    assert ex.plan_size(equity=13.905, price=2700, stop=2646, contract_value=0.01,
                        settings=_s(analyst_target_leverage=20)) == (1, 20.0)
    # short side is symmetric
    assert ex.plan_size(equity=13.905, price=2700, stop=2754, contract_value=0.01,
                        settings=_s(analyst_target_leverage=20)) == (1, 20.0)


def test_target_is_capped_by_max_leverage_and_rounded_down():
    assert ex.plan_size(equity=13.905, price=2700, stop=2680, contract_value=0.01,
                        settings=_s(analyst_target_leverage=80, max_leverage=50)) == (1, 50.0)
    assert ex.plan_size(equity=13.905, price=2700, stop=2646, contract_value=0.01,
                        settings=_s(analyst_target_leverage=15.7)) == (1, 15.0)


def test_never_lowers_the_computed_leverage():
    # computed is 3x; a target below it changes nothing
    assert ex.plan_size(**ETH, settings=_s(analyst_target_leverage=2)) == (1, 3.0)
    # a wide stop where no leverage above the computed one is safe keeps the computed one
    wide = dict(equity=13.905, price=2700, stop=2700 * (1 - 0.20), contract_value=0.01)
    base = ex.plan_size(**wide, settings=_s(analyst_risk_cap=0.0))
    assert base is not None
    assert ex.plan_size(**wide, settings=_s(analyst_risk_cap=0.0, analyst_target_leverage=20)) == base


def test_with_the_ratio_off_liquidation_still_stays_beyond_the_stop():
    s = _s(analyst_min_liq_to_stop=0.0, analyst_target_leverage=20)
    n, lev = ex.plan_size(**ETH, settings=s)
    assert ex.liquidation_distance(lev) >= (2709.675 - 2540.406) / 2709.675
    assert ex.liquidation_distance(lev + 1) < (2709.675 - 2540.406) / 2709.675  # the highest such
    assert lev == 14.0


def test_target_never_skips_and_never_changes_contracts():
    """Over a grid: same None-ness and same contracts as target=0; leverage >= the
    old one; and the liquidation rule holds whenever the target raised it."""
    prices = [100.0, 2700.0, 85000.0]
    stops = [0.01, 0.03, 0.0625, 0.09, 0.15]
    equities = [5.0, 13.905, 100.0, 5000.0]
    cvs = [0.001, 0.01, 1.0]
    for p, sp, eq, cv, tgt, side in itertools.product(prices, stops, equities, cvs, (10, 15, 20), (1, -1)):
        kw = dict(equity=eq, price=p, stop=p * (1 - side * sp), contract_value=cv)
        old = ex.plan_size(**kw, settings=_s(analyst_target_leverage=0))
        new = ex.plan_size(**kw, settings=_s(analyst_target_leverage=tgt))
        assert (old is None) == (new is None)
        if old is None:
            continue
        assert new[0] == old[0] and new[1] >= old[1] and new[1] <= tgt
        if new[1] > old[1]:
            assert ex.liquidation_distance(new[1]) >= 1.5 * sp - 1e-12


@pytest.mark.parametrize("lev,dist", [(15, 1 / 15 - 0.0062), (20, 0.05 - 0.0062)])
def test_liquidation_distance_at_owner_leverages(lev, dist):
    assert ex.liquidation_distance(lev) == pytest.approx(dist)
