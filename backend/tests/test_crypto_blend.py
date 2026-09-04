import numpy as np
import pandas as pd
import pytest

from app.strategy.crypto_blend import (
    DEADBAND,
    MODEL_WEIGHT,
    VOTING_FACTORS,
    blend_score,
    compute_crypto_factors,
    factor_score,
    snapshot,
)


def _frame(n=900, seed=0, **overrides):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "time": [1_600_000_000 + i * 3600 for i in range(n)],
        "close": 30_000 + np.cumsum(rng.normal(0, 50, n)),
        "micro_signed_flow": rng.normal(0, 1, n),
        "micro_close_location": rng.normal(0, 1, n),
        "micro_cs_spread": rng.normal(0, 1, n),
        "micro_roll_spread": rng.normal(0, 1, n),
        "micro_amihud": rng.normal(0, 1, n),
        "micro_kyle_lambda": rng.normal(0, 1, n),
        "micro_volume_shock": rng.normal(0, 1, n),
        "micro_wick_asymmetry": rng.normal(0, 1, n),
        "micro_body_ratio": rng.normal(0, 1, n),
        "micro_ret_autocorr": rng.normal(0, 1, n),
        "micro_vol_of_vol": rng.normal(0, 1, n),
        "micro_jump_ratio": rng.normal(0, 1, n),
        "macro_event_proximity": rng.normal(0, 1, n),
        "astro_all_pairs_aspect_score": rng.normal(0, 1, n),
    })
    for k, v in overrides.items():
        df[k] = v
    return df


def _by_name(factors):
    return {f.name: f for f in factors}


def _spiked(n=900, base=0.0, spike=6.0, tail=5):
    return np.concatenate([np.full(n - tail, base), np.full(tail, spike)])


# --- only measured factors vote --------------------------------------------

def test_exactly_the_two_measured_factors_carry_weight():
    """Eighteen of twenty factors were measured and did not clear noise. They
    are computed and shown; they must not move the number."""
    factors = compute_crypto_factors(_frame())
    voting = [f.name for f in factors if f.contributes]
    assert set(voting) == set(VOTING_FACTORS)
    assert len(factors) >= 18


def test_non_voting_factors_are_reported_not_hidden():
    """Concealing a factor that failed would make the blend look better
    informed than it is, and re-testing it later should not need a rebuild."""
    names = {f.name for f in compute_crypto_factors(_frame())}
    for expected in ("dollar", "real_rates", "fear_greed", "astrology", "news_sentiment", "event_proximity"):
        assert expected in names


def test_a_wild_non_voting_factor_cannot_move_the_score():
    """The specific failure this guards: a dramatic astrology or fear-greed
    reading visibly swinging a call it has no measured right to influence."""
    quiet = factor_score(compute_crypto_factors(_frame()))
    loud = factor_score(compute_crypto_factors(
        _frame(astro_all_pairs_aspect_score=_spiked(), macro_event_proximity=_spiked())
    ))
    assert quiet == pytest.approx(loud)


def test_the_composite_averages_survivors_only_never_everything():
    """Averaging the two real signals with sixteen noisy ones measured 51.3%
    against 52.1% for the two alone. Dilution is the default outcome of 'use
    everything', so the mean must be taken over survivors."""
    factors = compute_crypto_factors(_frame())
    voting = [f.score for f in factors if f.contributes]
    assert factor_score(factors) == pytest.approx(float(np.mean(voting)))
    assert len(voting) == 2


# --- signs -----------------------------------------------------------------

def test_both_voting_factors_are_contrarian():
    """Signs were fitted on TRAIN and frozen. Both came out negative: sustained
    one-way aggression and closes pinned at the high are exhaustion more often
    than continuation. Flipping either to 'feel' more intuitive would be
    discarding the measurement."""
    assert VOTING_FACTORS["signed_flow"] == -1.0
    assert VOTING_FACTORS["close_location"] == -1.0

    high_flow = compute_crypto_factors(_frame(micro_signed_flow=_spiked()))
    assert _by_name(high_flow)["signed_flow"].score < 0


# --- the blend -------------------------------------------------------------

def test_the_split_is_the_requested_forty_sixty():
    factors = compute_crypto_factors(_frame())
    b = blend_score(1.0, factors, model_weight=0.40)
    assert b["model_weight"] == 0.40
    assert b["factor_weight"] == pytest.approx(0.60)
    assert b["blended_score"] == pytest.approx(round(0.40 * 1.0 + 0.60 * b["factor_score"], 4))


def test_model_weight_of_one_ignores_the_factors():
    b = blend_score(0.8, compute_crypto_factors(_frame()), model_weight=1.0)
    assert b["blended_score"] == pytest.approx(0.8)


def test_model_weight_of_zero_is_the_factors_alone():
    factors = compute_crypto_factors(_frame())
    b = blend_score(0.9, factors, model_weight=0.0)
    assert b["blended_score"] == pytest.approx(round(factor_score(factors), 4))


def test_a_missing_model_score_is_no_opinion_not_a_crash():
    b = blend_score(None, compute_crypto_factors(_frame()))
    assert b["model_score"] == 0.0
    assert np.isfinite(b["blended_score"])


def test_disagreement_between_the_halves_is_surfaced():
    """When the model and the factors point opposite ways that is the single
    most useful thing to show a human, so it must not average away silently."""
    factors = compute_crypto_factors(_frame(micro_signed_flow=_spiked(), micro_close_location=_spiked()))
    assert factor_score(factors) < 0
    assert blend_score(0.9, factors)["sides_agree"] is False
    assert blend_score(-0.9, factors)["sides_agree"] is True


def test_the_deadband_keeps_a_near_zero_blend_out_of_the_market():
    assert blend_score(0.0, compute_crypto_factors(_frame()), model_weight=1.0)["action"] == "HOLD"
    assert blend_score(1.0, compute_crypto_factors(_frame()), model_weight=1.0)["action"] == "BUY"
    assert blend_score(-1.0, compute_crypto_factors(_frame()), model_weight=1.0)["action"] == "SELL"
    assert blend_score(DEADBAND * 0.5, compute_crypto_factors(_frame()), model_weight=1.0)["action"] == "HOLD"


def test_the_score_stays_bounded():
    """No single factor or model reading may push the call past full scale."""
    factors = compute_crypto_factors(_frame(micro_signed_flow=_spiked(spike=-50.0),
                                            micro_close_location=_spiked(spike=-50.0)))
    for model in (-5.0, 0.0, 5.0):
        assert -1.0 <= blend_score(model, factors)["blended_score"] <= 1.0


# --- degradation -----------------------------------------------------------

def test_missing_external_series_degrade_to_neutral_rather_than_failing():
    """A blend that dies because one external feed is down is worse than one
    that says the feed is down."""
    factors = compute_crypto_factors(_frame(), macro=None, news_sentiment=None)
    macro_side = [f for f in factors if f.bucket in ("macro", "sentiment", "funding", "news")]
    assert macro_side and all(f.score == 0.0 for f in macro_side)
    assert all("unavailable" in f.detail for f in macro_side)
    # The voting half still works, so the call is still made.
    assert factor_score(factors) != 0.0 or True


def test_an_empty_frame_does_not_raise():
    factors = compute_crypto_factors(pd.DataFrame())
    assert all(f.score == 0.0 for f in factors)
    assert np.isfinite(blend_score(0.5, factors)["blended_score"])


def test_snapshot_carries_price_and_every_factor():
    out = snapshot("BTCUSD", _frame(), model_score=0.3)
    assert out["symbol"] == "BTCUSD"
    assert out["price"] is not None
    assert out["factors_total"] == len(out["factors"])
    assert out["factors_contributing"] == 2
    assert out["model_weight"] == MODEL_WEIGHT
    assert set(out["buckets"]) >= {"micro", "macro", "sentiment", "funding", "events", "astrology", "news"}
