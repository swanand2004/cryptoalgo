import pytest

from app.gold.engine import (
    MIN_VIABLE_RR,
    TRADE_MAX_STOP_ATR,
    TRADE_STOP_ATR,
    _trade_plan,
)


def _levels(support=90.0, resistance=110.0, atr=2.0):
    return {"support": support, "resistance": resistance, "atr": atr}


def test_long_stop_sits_below_entry_and_target_above():
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] < plan["entry"] < plan["take_profit"]


def test_short_stop_sits_above_entry_and_target_below():
    plan = _trade_plan(_levels(), "SELL", price=100.0, target_price=94.0)
    assert plan["take_profit"] < plan["entry"] < plan["stop_loss"]


def test_stop_is_the_documented_atr_multiple_when_structure_is_far_away():
    """Must agree with the `invalidation` line the levels panel already shows.
    Two different stop distances on the same card would be a bug the user
    reads as the system contradicting itself."""
    plan = _trade_plan(_levels(support=10.0), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] == pytest.approx(100.0 - TRADE_STOP_ATR * 2.0)


def test_stop_steps_past_a_nearby_structural_level_rather_than_sitting_on_it():
    """A stop parked right at an obvious low gets taken by a wick and then the
    trade works without you. Support at 95.5 is 0.25 ATR under the flat
    96.0 ATR stop, so the stop goes below the level instead."""
    plan = _trade_plan(_levels(support=95.5), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] < 95.5
    assert "beyond structure" in plan["note"]


def test_a_distant_structural_level_does_not_drag_the_stop_out():
    """Only NEARBY structure justifies widening. Support 20 ATR away is not
    this trade's invalidation, it is a different trade."""
    plan = _trade_plan(_levels(support=60.0), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] == pytest.approx(100.0 - TRADE_STOP_ATR * 2.0)


def test_stop_width_is_capped():
    """Even structure cannot push the stop past the cap -- past that width it
    is no longer the same trade, whatever the chart says."""
    plan = _trade_plan(_levels(support=95.0, atr=2.0), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_atr_multiple"] <= TRADE_MAX_STOP_ATR + 1e-9


def test_a_target_that_does_not_cover_the_stop_is_marked_not_viable():
    """The live long_term gold call is exactly this: R:R 0.41. Showing it as a
    tradeable plan would be the dishonest option."""
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=101.0)
    assert plan["risk_reward"] < MIN_VIABLE_RR
    assert plan["viable"] is False
    assert "does not cover" in plan["note"]


def test_the_structural_level_is_never_substituted_for_the_model_target():
    """The crux. Resistance would give a flattering ratio here; the model's own
    target gives a poor one. The plan must report the poor one, because only
    the model's number has a measured track record behind it."""
    plan = _trade_plan(_levels(resistance=140.0), "BUY", price=100.0, target_price=101.0)
    assert plan["take_profit"] == pytest.approx(101.0)
    assert plan["structural_risk_reward"] > plan["risk_reward"]  # tempting, and unused
    assert plan["viable"] is False


def test_a_model_that_contradicts_its_own_signal_is_surfaced_not_hidden():
    """The gated classifier says BUY while the return regression predicts a
    fall. Inventing a target from structure would paper over a genuine
    internal disagreement."""
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=97.0)
    assert plan["model_conflict"] is True
    assert plan["viable"] is False
    assert plan["risk_reward"] is None


def test_hold_gives_no_entry_but_keeps_the_levels_that_would_start_a_trade():
    plan = _trade_plan(_levels(), "HOLD", price=100.0, target_price=101.0)
    assert plan["entry"] is None and plan["stop_loss"] is None
    assert plan["trigger_long_above"] == 110.0
    assert plan["trigger_short_below"] == 90.0


def test_no_atr_means_no_plan_rather_than_a_guessed_stop():
    """Without a volatility estimate any stop distance is invented."""
    plan = _trade_plan({"support": 90.0, "resistance": 110.0, "atr": None},
                       "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] is None
    assert "cannot place a stop" in plan["note"]


def test_conviction_scales_size_only_and_never_moves_the_prices():
    """Micro conviction is a risk gate. If it could move the stop it would be
    changing where the call is wrong, which it has no information about."""
    full = _trade_plan(_levels(), "BUY", 100.0, 106.0, conviction=1.0)
    damped = _trade_plan(_levels(), "BUY", 100.0, 106.0, conviction=0.21)
    assert damped["size_multiplier"] == pytest.approx(0.21)
    for key in ("entry", "stop_loss", "take_profit", "risk_reward"):
        assert damped[key] == pytest.approx(full[key])


def test_a_good_target_is_marked_takeable():
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=110.0)
    assert plan["risk_reward"] == pytest.approx(10.0 / 4.0)
    assert plan["viable"] is True
