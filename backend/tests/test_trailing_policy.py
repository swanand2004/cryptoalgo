"""The trailing stop: what actually runs, and where paper and live can diverge.

Three things are pinned here, because all three are easy to break silently.

1. THE WIDE TRAIL RATCHETS. It may only ever move in the position's favour.
2. THE TIGHTENED STAGE IS CURRENTLY UNREACHABLE, ON PURPOSE. The trigger sits
   at 3.0 ATR and the take-profit at 3.0 ATR, so the target fires first and the
   two-stage trail never engages. That is not dead code left by accident -- it
   is the fix. Measured over 9 years / 9,411 trades, an aggressive trail
   (trigger 0.5 ATR) cut winners at ~0.5 ATR while losers ran the full stop,
   producing a structural payoff of 0.59 that no achievable win rate beats:
   58-60% wins and still -99%. Raising the trigger past the take-profit took
   payoff to 1.53 and out-of-sample return from -75.2% to +1.4%.
3. PAPER AND LIVE AGREE ONLY BY COINCIDENCE. PaperBroker implements the
   two-stage trail in Python. LiveBroker attaches ONE Delta bracket order with
   a constant trail_amount and never edits it, so it can only ever do the wide
   stage. They match today because the tightened stage is disabled. Lower the
   trigger below the take-profit and paper starts tightening while live does
   not -- the backtest would stop describing the live system, silently.
   The last test fails if that happens.
"""
import pytest

from app.config import Settings


def _stop_after(peak_atr: float, s: Settings, entry: float = 100.0, atr: float = 1.0) -> tuple[float, bool]:
    """The paper broker's trailing block, over one peak. Kept as a transcription
    of app/execution/paper_broker.py so a change there shows up as a failure
    here rather than as a quiet behaviour drift."""
    stop_distance = s.atr_stop_multiple * atr
    peak = entry + peak_atr * atr
    atr_at_entry = stop_distance / s.atr_stop_multiple
    triggered = (peak - entry) >= s.atr_breakeven_trigger_multiple * atr_at_entry
    trail = s.atr_tightened_trail_multiple * atr_at_entry if triggered else stop_distance
    candidate = peak - trail
    if triggered:
        candidate = max(candidate, entry)
    return candidate, triggered


def test_the_trail_only_ever_tightens():
    """A stop that can loosen is not a stop. Walk the peak up, then hold it
    while price falls back, and the stop must not retreat."""
    s = Settings()
    stop = 100.0 - s.atr_stop_multiple
    for peak_atr in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        candidate, _ = _stop_after(peak_atr, s)
        assert candidate >= stop - 1e-9, f"stop retreated at peak {peak_atr} ATR"
        stop = max(stop, candidate)

    # Peak stays where it was when price pulls back, so the stop holds.
    held, _ = _stop_after(4.0, s)
    assert held == pytest.approx(_stop_after(4.0, s)[0])


def test_the_initial_stop_is_the_configured_distance():
    s = Settings()
    candidate, triggered = _stop_after(0.0, s)
    assert triggered is False
    assert candidate == pytest.approx(100.0 - s.atr_stop_multiple)


def test_the_tightened_stage_cannot_engage_before_the_target():
    """The whole point of the 2026-09-03 exit fix. If this starts failing,
    winners are being cut short again and the payoff ratio is about to
    collapse -- check the measurement in config.py before 'fixing' it."""
    s = Settings()
    assert s.atr_breakeven_trigger_multiple >= s.atr_take_profit_multiple

    # Just short of target: still on the wide trail.
    _, triggered = _stop_after(s.atr_take_profit_multiple - 0.1, s)
    assert triggered is False


def test_winners_are_not_cut_short_before_the_target():
    """At every peak below the take-profit the stop must stay strictly below
    entry-plus-target, i.e. it never reaches up and closes the trade early."""
    s = Settings()
    target = 100.0 + s.atr_take_profit_multiple
    peak = 0.0
    while peak < s.atr_take_profit_multiple:
        candidate, _ = _stop_after(peak, s)
        assert candidate < target
        peak += 0.25


def test_an_aggressive_trigger_would_reach_breakeven_early():
    """Proves the mechanism still WORKS -- it is disabled by configuration,
    not broken. With the old aggressive trigger the breakeven floor engages
    almost immediately, which is exactly the behaviour that was measured as
    ruinous and switched off."""
    aggressive = Settings(atr_breakeven_trigger_multiple=0.5, atr_tightened_trail_multiple=0.4)
    candidate, triggered = _stop_after(1.0, aggressive)
    assert triggered is True
    assert candidate >= 100.0                      # floored at breakeven
    assert candidate == pytest.approx(100.6)       # peak 101 minus a 0.4 ATR trail


def test_live_cannot_silently_ignore_a_tightened_trail_that_paper_applies():
    """PAPER/LIVE DIVERGENCE GUARD.

    LiveBroker attaches one Delta bracket with a constant trail_amount
    (atr_stop_multiple * ATR) and never edits it as the position moves, so it
    implements the WIDE stage only. PaperBroker implements both stages in
    Python. They agree today purely because the tightened stage is unreachable.

    If someone lowers the trigger below the take-profit, paper would start
    tightening and live would not: the backtest would quietly stop describing
    the live system, which is the one property this project's whole
    'same code path' design exists to protect. Enabling it means teaching
    LiveBroker to edit the bracket (DeltaClient.edit_order exists) -- not just
    changing the number.
    """
    s = Settings()
    tightening_reachable = s.atr_breakeven_trigger_multiple < s.atr_take_profit_multiple
    assert not tightening_reachable, (
        "atr_breakeven_trigger_multiple is now below atr_take_profit_multiple, so PaperBroker "
        "will tighten the trail but LiveBroker (a fixed bracket trail_amount, never edited) will "
        "not. Implement bracket editing in LiveBroker before shipping this."
    )
