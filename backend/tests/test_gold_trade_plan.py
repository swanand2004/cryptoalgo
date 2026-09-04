import pytest

from app.gold.engine import (
    MAX_ENTRY_DISTANCE_ATR,
    MIN_VIABLE_RR,
    TARGET_RR,
    TRADE_MAX_STOP_ATR,
    TRADE_STOP_ATR,
    _entry_for_rr,
    _trade_plan,
)


def _levels(support=90.0, resistance=110.0, atr=2.0):
    return {"support": support, "resistance": resistance, "atr": atr}


# --- stop placement --------------------------------------------------------

def test_long_stop_sits_below_entry_and_target_above():
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] < plan["entry"] < plan["take_profit"]


def test_short_stop_sits_above_entry_and_target_below():
    plan = _trade_plan(_levels(), "SELL", price=100.0, target_price=94.0)
    assert plan["take_profit"] < plan["entry"] < plan["stop_loss"]


def test_stop_is_the_documented_atr_multiple_when_structure_is_far_away():
    """Must agree with the `invalidation` line the levels panel already shows.
    Two different stop distances on one card reads as the system contradicting
    itself."""
    plan = _trade_plan(_levels(support=10.0), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] == pytest.approx(100.0 - TRADE_STOP_ATR * 2.0)


def test_stop_steps_past_a_nearby_structural_level_rather_than_sitting_on_it():
    """A stop parked at an obvious low gets taken by a wick and then the trade
    works without you. Support at 95.5 is 0.25 ATR under the flat 96.0 ATR
    stop, so the stop goes below the level instead."""
    plan = _trade_plan(_levels(support=95.5), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] < 95.5
    assert "beyond structure" in plan["note"]


def test_a_distant_structural_level_does_not_drag_the_stop_out():
    plan = _trade_plan(_levels(support=60.0), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] == pytest.approx(100.0 - TRADE_STOP_ATR * 2.0)


def test_stop_width_is_capped():
    plan = _trade_plan(_levels(support=95.0, atr=2.0), "BUY", price=100.0, target_price=106.0)
    assert plan["stop_atr_multiple"] <= TRADE_MAX_STOP_ATR + 1e-9


# --- the entry price -------------------------------------------------------

def test_entry_for_rr_solves_the_ratio_exactly():
    """The algebra the whole entry policy rests on: with stop and target fixed
    as LEVELS, the entry is the only free variable."""
    entry = _entry_for_rr(target=120.0, stop=90.0, rr=2.0, long_side=True)
    assert (120.0 - entry) / (entry - 90.0) == pytest.approx(2.0)


def test_a_market_entry_that_already_pays_well_is_left_alone():
    """Do not invent a limit order when the trade is good right now -- an
    unnecessary limit is a trade you simply never get filled on."""
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=115.0)
    assert plan["entry_type"] == "market"
    assert plan["entry"] == pytest.approx(100.0)
    assert plan["risk_reward"] >= MIN_VIABLE_RR


def test_a_poor_market_ratio_produces_the_limit_price_that_fixes_it():
    """The live long_term gold call was exactly this: 0.34x at market. The
    useful answer is the price at which it becomes a real trade, not a
    refusal and certainly not a nicer-looking target."""
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=102.0)
    assert plan["risk_reward_at_market"] < MIN_VIABLE_RR
    assert plan["entry_type"] == "limit"
    assert plan["entry"] < 100.0                       # a better price than chasing
    assert plan["risk_reward"] == pytest.approx(TARGET_RR)
    assert "waiting for" in plan["note"]


def test_the_limit_entry_is_on_the_correct_side_for_a_short():
    plan = _trade_plan(_levels(), "SELL", price=100.0, target_price=98.0)
    assert plan["entry_type"] == "limit"
    assert plan["entry"] > 100.0                       # sell higher, not lower
    assert plan["risk_reward"] == pytest.approx(TARGET_RR)


def test_an_unreachable_limit_is_flagged_and_not_called_takeable():
    """A limit that far away may simply never trade before the forecast
    expires, and quoting a ratio for a fill that never happens is worse than
    quoting none.

    Reaching this state needs a WIDENED stop: with the default 2-ATR stop and
    TARGET_RR of 2, the solved entry is at most ~1.33 ATR away, so the guard
    can only bind once structure has pushed the stop out. Support at 95.0 sits
    exactly 0.5 ATR beyond the flat stop, which widens it to 2.75 ATR."""
    plan = _trade_plan(_levels(support=95.0, atr=2.0), "BUY", price=100.0, target_price=100.05)
    assert plan["entry_type"] == "limit"
    assert plan["entry_distance_atr"] > MAX_ENTRY_DISTANCE_ATR
    assert plan["entry_reachable"] is False
    assert plan["viable"] is False
    assert "may not trade" in plan["note"]


def test_the_cost_of_chasing_stays_visible():
    """Both ratios are reported: what waiting pays, and what chasing pays."""
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=102.0)
    assert plan["risk_reward"] > plan["risk_reward_at_market"]


# --- distances measured from where the reader is ---------------------------

def test_everything_is_measured_from_the_current_price():
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=104.0)
    assert plan["current_price"] == pytest.approx(100.0)
    assert plan["target_remaining_usd"] == pytest.approx(4.0)
    assert plan["target_remaining_pct"] == pytest.approx(4.0)
    # 2 ATR of 2.0 below 100 -> stop 96, i.e. 4% away from HERE, not from entry
    assert plan["stop_distance_pct"] == pytest.approx(4.0)


def test_target_remaining_is_positive_for_a_short_that_is_working():
    """"Remaining" means distance still to travel in the trade's favour, so it
    must not flip sign just because a short's target is below the price."""
    plan = _trade_plan(_levels(), "SELL", price=100.0, target_price=96.0)
    assert plan["target_remaining_usd"] == pytest.approx(4.0)


# --- honesty about the model ------------------------------------------------

def test_the_structural_level_is_never_substituted_for_the_model_target():
    """Resistance would give a flattering ratio here; the model's own target
    gives a poor one. Only the model's number has a track record behind it."""
    plan = _trade_plan(_levels(resistance=140.0), "BUY", price=100.0, target_price=101.0)
    assert plan["take_profit"] == pytest.approx(101.0)
    assert plan["structural_risk_reward"] > plan["risk_reward_at_market"]


def test_a_model_that_contradicts_its_own_signal_is_surfaced_not_hidden():
    """Gated classifier says BUY, return regression predicts a fall. Inventing
    a target from structure would paper over a real internal disagreement."""
    plan = _trade_plan(_levels(), "BUY", price=100.0, target_price=97.0)
    assert plan["model_conflict"] is True
    assert plan["viable"] is False
    assert plan["risk_reward"] is None
    assert plan["entry_type"] == "market"   # no limit invented for a broken call


def test_hold_gives_no_entry_but_keeps_the_levels_that_would_start_a_trade():
    plan = _trade_plan(_levels(), "HOLD", price=100.0, target_price=101.0)
    assert plan["entry"] is None and plan["stop_loss"] is None
    assert plan["trigger_long_above"] == 110.0
    assert plan["trigger_short_below"] == 90.0


def test_no_atr_means_no_plan_rather_than_a_guessed_stop():
    plan = _trade_plan({"support": 90.0, "resistance": 110.0, "atr": None},
                       "BUY", price=100.0, target_price=106.0)
    assert plan["stop_loss"] is None
    assert "cannot place a stop" in plan["note"]


def test_conviction_scales_size_only_and_never_moves_the_prices():
    """Micro conviction is a risk gate. Letting it move the stop would be
    changing where the call is wrong, which it has no information about."""
    full = _trade_plan(_levels(), "BUY", 100.0, 106.0, conviction=1.0)
    damped = _trade_plan(_levels(), "BUY", 100.0, 106.0, conviction=0.21)
    assert damped["size_multiplier"] == pytest.approx(0.21)
    for key in ("entry", "stop_loss", "take_profit", "risk_reward"):
        assert damped[key] == pytest.approx(full[key])
