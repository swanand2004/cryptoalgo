"""analyst_trail_mode (2026-10-07): "after_tp1" = the loss doctor's FIX A
(scratchpad/team/doctor_analyst/PREREG.txt): no ATR trail and no breakeven
floor; the stop holds at the initial level stop until TP1, then only the TP1
+0.5R lock moves it, and it never moves backwards. "live" = the old trail."""
import random

import pytest

from app.analyst import execution as ex
from app.config import Settings


def _s(**kw):
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


def _exit(mode="after_tp1", **kw):
    # BUY at 100, level stop 97 (1R = 3), TP1 103, ATR 2.
    base = dict(side="BUY", entry=100.0, stop=97.0, initial_stop=97.0, atr=2.0, peak=100.0, take_profit=103.0,
                extended=False, still_strong=False, settings=_s(analyst_trail_mode=mode))
    base.update(kw)
    return ex.update_exit(**base)


def test_default_is_after_tp1_and_live_is_the_rollback():
    assert _s().analyst_trail_mode == "after_tp1"
    assert _exit(mode="live", price=102.0).stop == pytest.approx(99.0)  # the old 1.5-ATR trail


def test_stop_holds_at_the_level_stop_until_tp1():
    u = _exit(price=102.9)  # +0.97R, just short of TP1: "live" would trail to 100.9
    assert u.stop == 97.0 and u.action is None and u.peak == 102.9
    back = _exit(price=100.0, peak=u.peak)  # a pull-back after the run: still the level stop
    assert back.stop == 97.0 and back.action is None
    assert _exit(price=96.9).action == "SL"  # the level stop itself still closes it


def test_tp1_closes_unless_still_strong_then_locks_half_r_and_never_trails():
    assert _exit(price=103.2).action == "TP"
    u = _exit(price=103.2, still_strong=True)
    assert u.action is None and u.extended and u.take_profit == pytest.approx(106.0)
    assert u.stop == pytest.approx(101.5)  # the +0.5R lock, nothing more
    run = _exit(price=105.5, stop=u.stop, peak=u.peak, take_profit=u.take_profit, extended=True, still_strong=True)
    assert run.stop == pytest.approx(101.5)  # no ATR trail after TP1 either ("live" would sit at 104.5)
    dip = _exit(price=102.0, stop=run.stop, peak=run.peak, take_profit=run.take_profit, extended=True)
    assert dip.stop == pytest.approx(101.5) and dip.action is None  # never backwards
    hit = _exit(price=101.4, stop=dip.stop, peak=dip.peak, take_profit=dip.take_profit, extended=True)
    assert hit.action == "TRAIL"  # stop had moved off the level stop


def test_short_trade_mirrors():
    kw = dict(side="SELL", entry=100.0, stop=103.0, initial_stop=103.0, peak=100.0, take_profit=97.0)
    assert _exit(price=97.5, **kw).stop == 103.0
    u = _exit(price=96.8, still_strong=True, **kw)
    assert u.extended and u.stop == pytest.approx(98.5) and u.take_profit == pytest.approx(94.0)


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_stop_never_moves_backwards_on_a_random_walk(side):
    rng = random.Random(7)
    d = 1 if side == "BUY" else -1
    stop = init = 100.0 - d * 3.0
    st = dict(peak=100.0, take_profit=100.0 + d * 3.0, extended=False)
    px = 100.0
    for _ in range(2000):
        px += rng.gauss(0, 0.6)
        u = ex.update_exit(side=side, entry=100.0, stop=stop, initial_stop=init, atr=2.0, price=px,
                           still_strong=rng.random() < 0.5, settings=_s(), **st)
        assert d * (u.stop - stop) >= 0 and d * (u.stop - init) >= 0
        if not st["extended"] and not u.extended:
            assert u.stop == init  # before TP1 nothing moves it
        if u.action:
            break
        stop = u.stop
        st = dict(peak=u.peak, take_profit=u.take_profit, extended=u.extended)
