import numpy as np
import pandas as pd
import pytest

from app.gold.macro_factors import (
    CFTC_SYMBOL,
    COMPOSITE_FACTORS,
    FACTOR_Z_CAP,
    blend_score,
    compute_macro_factors,
)


def _series(values, start=1_600_000_000, step=86_400):
    n = len(values)
    return pd.DataFrame({
        "time": [start + i * step for i in range(n)],
        "close": values,
    })


def _flat_then_rising(n=400, base=1.0, step=0.02, tail=40):
    """Flat, then a sustained move. A PERFECTLY linear series is useless for
    testing rate-of-change factors: its diff is constant, so the trailing
    z-score is undefined and the factor correctly returns neutral. The change
    itself has to vary for there to be anything to score."""
    flat = [base] * (n - tail)
    return _series(flat + [base + (i + 1) * step for i in range(tail)])


def _flat_then_spike(n=400, base=2.0, spike=4.0):
    return _series([base] * (n - 30) + [spike] * 30)


def _frames(**kw):
    return {
        "MACRO:REALRATE10Y": kw.get("real", _series([2.0] * 400)),
        "MACRO:DXY": kw.get("dxy", _series([100.0] * 400)),
        "MACRO:BREAKEVEN10Y": kw.get("be", _series([2.0] * 400)),
        CFTC_SYMBOL: kw.get("cftc", _series([30.0] * 200)),
    }


def _by_name(factors):
    return {f.name: f for f in factors}


def test_rising_real_rates_are_bearish_for_gold():
    """Gold yields nothing, so a rising real yield raises the opportunity
    cost of holding it. This sign was committed before any measurement."""
    f = _by_name(compute_macro_factors(_frames(real=_flat_then_rising())))["real_rate_change"]
    assert f.score < 0
    assert f.direction == "bearish"


def test_falling_real_rates_are_bullish_for_gold():
    f = _by_name(compute_macro_factors(_frames(real=_flat_then_rising(step=-0.02))))["real_rate_change"]
    assert f.score > 0
    assert f.direction == "bullish"


def test_a_strengthening_dollar_is_bearish_for_gold():
    """Gold is quoted in USD -- a stronger dollar mechanically lowers the
    price for the same ounce of demand."""
    f = _by_name(compute_macro_factors(_frames(dxy=_flat_then_rising(base=100.0, step=0.5))))["dollar_change"]
    assert f.score < 0


def test_rising_breakevens_are_bullish_for_gold():
    f = _by_name(compute_macro_factors(_frames(be=_flat_then_spike())))["breakeven_inflation"]
    assert f.score > 0


def test_crowded_speculative_longs_read_as_bearish_not_bullish():
    """Contrarian by design: the same 'crowded trade is fragile' logic this
    project already applies to funding and long/short ratios."""
    f = _by_name(compute_macro_factors(_frames(cftc=_flat_then_spike(n=200, base=20.0, spike=60.0))))
    assert f["cftc_positioning"].score < 0


def test_scores_are_bounded_to_plus_minus_one():
    """A single wild factor must not be able to dominate the blend."""
    violent = _series([1.0] * 370 + [500.0] * 30)
    for f in compute_macro_factors(_frames(real=violent, dxy=violent, be=violent, cftc=violent)):
        assert -1.0 <= f.score <= 1.0


def test_missing_series_contribute_zero_rather_than_being_dropped():
    """Degrading gracefully matters more than silently re-weighting: if FRED
    is unreachable the blend must lean on what remains, not quietly change
    what the other factors mean."""
    factors = compute_macro_factors({})
    assert len(factors) == 4
    assert all(f.score == 0.0 and f.z_score is None for f in factors)
    assert all("unavailable" in f.detail for f in factors)


def test_only_measured_viable_factors_enter_the_composite():
    """Measured 2026-09-04 on 2,196 daily bars: the equal-weighted composite
    of all four scored 44.6% at 21d -- WORSE than its own best component
    (real rates, 54.4%). Averaging one real signal with three noisy ones
    destroys it, so only the two whose committed sign held up contribute."""
    assert COMPOSITE_FACTORS == ("real_rate_change", "dollar_change")
    blend = blend_score(0.0, compute_macro_factors(_frames(be=_flat_then_spike())))
    contributing = [f for f in blend["factors"] if f["contributes"]]
    assert {f["name"] for f in contributing} == set(COMPOSITE_FACTORS)
    # A big breakeven reading is reported but must not move the composite.
    assert blend["factor_score"] == pytest.approx(0.0, abs=1e-9)


def test_non_contributing_factors_are_still_reported():
    """Zero weight, not hidden -- their reading is useful context and
    concealing them would make the composite look more informed than it is."""
    blend = blend_score(0.5, compute_macro_factors(_frames()))
    names = {f["name"] for f in blend["factors"]}
    assert "breakeven_inflation" in names and "cftc_positioning" in names
    assert blend["factors_total"] == 4


def test_blend_is_the_requested_fifty_fifty():
    factors = compute_macro_factors(_frames(real=_flat_then_rising(step=-0.02)))
    blend = blend_score(1.0, factors, model_weight=0.5)
    assert blend["model_weight"] == 0.5
    assert blend["factor_weight"] == 0.5
    expected = 0.5 * 1.0 + 0.5 * blend["factor_score"]
    assert blend["blended_score"] == pytest.approx(round(min(expected, 1.0), 4))


def test_model_weight_of_one_ignores_the_factors_entirely():
    factors = compute_macro_factors(_frames(real=_flat_then_rising()))
    blend = blend_score(0.8, factors, model_weight=1.0)
    assert blend["blended_score"] == pytest.approx(0.8)


def test_disagreement_between_the_two_halves_is_surfaced():
    """When the model and the macro picture point opposite ways that is the
    single most useful thing to show a human, so it must not be averaged
    away silently."""
    bearish_macro = compute_macro_factors(_frames(real=_flat_then_rising()))
    disagreeing = blend_score(0.9, bearish_macro)      # model bullish, macro bearish
    agreeing = blend_score(-0.9, bearish_macro)        # both bearish
    assert disagreeing["sides_agree"] is False
    assert agreeing["sides_agree"] is True


def test_a_missing_model_score_is_treated_as_no_opinion():
    blend = blend_score(None, compute_macro_factors(_frames()))
    assert blend["model_score"] == 0.0
    assert np.isfinite(blend["blended_score"])
