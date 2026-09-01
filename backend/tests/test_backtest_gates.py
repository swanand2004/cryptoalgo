import numpy as np
import pytest

from app.backtest.gates import (
    DEPLOY_GATES,
    deflated_sharpe,
    evaluate_gates,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def test_expected_max_sharpe_rises_with_the_number_of_things_tried():
    """The whole point: trying more configurations raises the bar a real
    strategy must clear, because the best of many coin flips looks good."""
    one = expected_max_sharpe(n_trials=1, trial_sharpe_std=0.5)
    few = expected_max_sharpe(n_trials=10, trial_sharpe_std=0.5)
    many = expected_max_sharpe(n_trials=1000, trial_sharpe_std=0.5)
    assert one == 0.0  # nothing to correct for when only one thing was tried
    assert 0.0 < few < many


def test_expected_max_sharpe_scales_with_trial_dispersion():
    """Uses the DISPERSION of trial Sharpes -- a published variant of this
    substitutes 1/sqrt(freq), which is dimensionally wrong and lets weak
    strategies through."""
    tight = expected_max_sharpe(n_trials=100, trial_sharpe_std=0.1)
    wide = expected_max_sharpe(n_trials=100, trial_sharpe_std=1.0)
    assert wide > tight * 5


def test_deflated_sharpe_rejects_a_lone_good_looking_result_from_many_tries():
    """A Sharpe of 1.5 found after 500 attempts is far weaker evidence than
    the same 1.5 found on the first try."""
    first_try = deflated_sharpe(1.5, n_obs=1000, n_trials=1, trial_sharpe_std=0.5)
    after_many = deflated_sharpe(1.5, n_obs=1000, n_trials=500, trial_sharpe_std=0.5)
    assert first_try > 0.95
    assert after_many < first_try


def test_deflated_sharpe_penalises_fat_tails_and_negative_skew():
    """Crypto returns are skewed and fat-tailed, which flatters a naive
    Sharpe -- the correction must give back some of that."""
    normal = deflated_sharpe(1.2, n_obs=500, n_trials=10, trial_sharpe_std=0.3,
                             skew=0.0, kurtosis=3.0)
    nasty = deflated_sharpe(1.2, n_obs=500, n_trials=10, trial_sharpe_std=0.3,
                            skew=-1.5, kurtosis=9.0)
    assert nasty < normal


def test_deflated_sharpe_needs_enough_observations():
    thin = deflated_sharpe(2.0, n_obs=10, n_trials=5, trial_sharpe_std=0.3)
    thick = deflated_sharpe(2.0, n_obs=5000, n_trials=5, trial_sharpe_std=0.3)
    assert thin < thick


def test_pbo_is_low_when_one_config_genuinely_dominates():
    """A config with real, persistent edge should be selected correctly, so
    the probability of overfitting is low."""
    rng = np.random.default_rng(0)
    n = 2000
    noise = rng.normal(0, 0.01, size=(n, 5))
    noise[:, 0] += 0.004  # config 0 has a genuine, stable edge
    assert probability_of_backtest_overfitting(noise, n_splits=8) < 0.35


def test_pbo_is_high_when_every_config_is_pure_noise():
    """With no real edge, the in-sample winner is arbitrary and should fail
    out-of-sample about half the time or worse."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, size=(2000, 8))
    assert probability_of_backtest_overfitting(noise, n_splits=8) > 0.35


def test_pbo_requires_something_to_choose_between():
    """PBO is a statement about SELECTION -- undefined with one config."""
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 1)))


def test_sharpe_ratio_handles_degenerate_input():
    assert sharpe_ratio(np.array([])) == 0.0
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0  # zero variance


def test_gates_reject_this_projects_own_headline_numbers():
    """The concrete case these gates exist for: the pre-fix engine reported a
    'sum of returns' that looked spectacular while resting on a lookahead
    bug, and today's honest re-runs produce Sharpes around 1-3 on a few
    dozen trades. A result like that must NOT pass.
    """
    passed, failures = evaluate_gates(
        oos_sharpe=3.47,          # residual momentum on ETH, out-of-sample
        deflated=0.40,            # weak once multiple-testing is accounted for
        pbo_value=0.55,           # in-sample winner does not generalise
        n_trades=46,              # far too few
        edge_after_costs_bps=0.5,  # order-book imbalance's real edge
    )
    assert not passed
    assert len(failures) >= 3


def test_gates_pass_a_genuinely_strong_result():
    passed, failures = evaluate_gates(
        oos_sharpe=1.6, deflated=0.97, pbo_value=0.15,
        n_trades=800, edge_after_costs_bps=3.0,
    )
    assert passed
    assert failures == []


def test_deploy_gates_are_strict_enough_to_matter():
    """Guards against someone quietly loosening these until everything passes."""
    assert DEPLOY_GATES["min_deflated_sharpe"] >= 0.95
    assert DEPLOY_GATES["max_pbo"] <= 0.35
    assert DEPLOY_GATES["min_trades"] >= 100
