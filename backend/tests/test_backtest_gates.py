import numpy as np
import pandas as pd
import pytest

from app.backtest.gates import (
    DEPLOY_GATES,
    TAIL_GATES,
    deflated_sharpe,
    evaluate_gates,
    evaluate_tail_gates,
    expected_max_sharpe,
    find_vol_events,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    tail_metrics,
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


def test_tail_metrics_cvar_ratio_is_not_sign_inverted_on_a_losing_strategy():
    """The pasted version divided by returns.mean(), which is NEGATIVE for a
    losing strategy -- that flips the ratio's sign and lets a catastrophic
    tail loser slide under a "< 15" style gate. Normalising by mean WIN
    instead keeps the ratio positive and meaningful regardless of whether the
    strategy is net profitable."""
    rng = np.random.default_rng(1)
    small_wins = np.abs(rng.normal(0.001, 0.0005, size=300))
    one_blowup = np.array([-0.35])
    returns = np.concatenate([small_wins, one_blowup])
    metrics = tail_metrics(pd.Series(returns))
    assert metrics["cvar95_over_mean_win"] > 0
    assert metrics["worst_over_mean_win"] > 50  # the -35% bar dwarfs the small wins
    passed, failures = evaluate_tail_gates(metrics)
    assert not passed
    assert any("worst loss" in f or "CVaR95" in f for f in failures)


def test_tail_metrics_calmar_does_not_divide_by_zero_with_no_drawdown():
    """A strategy that never has a single down bar has dd == 0 -- calmar must
    report +inf rather than raising or returning NaN."""
    returns = pd.Series(np.full(50, 0.001))
    metrics = tail_metrics(returns)
    assert metrics["max_drawdown"] == 0.0
    assert metrics["calmar"] == np.inf


def test_tail_metrics_calmar_uses_this_projects_bar_frequency_not_252():
    """Default bars_per_year must reflect this project's actual live
    resolution (4h bars, crypto trades every day), not the equities 252
    trading-day convention the pasted version hardcoded."""
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(-0.0002, 0.01, size=500))
    metrics_4h = tail_metrics(returns, bars_per_year=365 * 6)
    metrics_daily = tail_metrics(returns, bars_per_year=252)
    assert metrics_4h["calmar"] != metrics_daily["calmar"]


def test_tail_metrics_vol_events_matched_to_nearest_bar_not_exact_timestamp():
    """Reindexing on exact timestamps silently produces NaN for any event
    that doesn't land precisely on a bar, and a NaN .sum() reads as 0.0 --
    making a strategy that bled through every crisis look untouched. Nearest-
    bar matching must actually find the bar."""
    idx = pd.date_range("2026-01-01", periods=100, freq="4h")
    returns = pd.Series(np.full(100, 0.001), index=idx)
    returns.iloc[50] = -0.20  # a real crash bar
    # event timestamp deliberately offset from the bar grid -- exact-match
    # reindexing would miss it entirely
    event_ts = idx[50] + pd.Timedelta(minutes=7)
    metrics = tail_metrics(returns, vol_events=[event_ts])
    assert metrics["n_vol_events"] == 1
    assert metrics["return_in_events"] == pytest.approx(-0.20)


def test_tail_metrics_reports_insufficient_data_below_ten_observations():
    metrics = tail_metrics(pd.Series([0.01, -0.01, 0.02]))
    assert metrics["insufficient_data"] is True


def test_evaluate_tail_gates_fails_a_strategy_never_seen_through_a_stress_event():
    """Untested through a real shock is not the same as safe -- must fail
    the gate outright rather than passing by default."""
    returns = pd.Series(np.abs(np.random.default_rng(3).normal(0.002, 0.0005, size=200)))
    metrics = tail_metrics(returns)  # no vol_events supplied
    passed, failures = evaluate_tail_gates(metrics)
    assert not passed
    assert any("stress events" in f for f in failures)


def test_evaluate_tail_gates_passes_a_genuinely_well_behaved_strategy():
    idx = pd.date_range("2026-01-01", periods=400, freq="4h")
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.0006, 0.004, size=400), index=idx)
    events = [idx[i] for i in (40, 130, 260, 350)]
    for e in events:
        returns.loc[e] -= 0.01  # survives some drawdown during the "stress events"
    metrics = tail_metrics(returns, vol_events=events)
    passed, failures = evaluate_tail_gates(metrics)
    assert passed
    assert failures == []


def test_evaluate_tail_gates_insufficient_data_fails_outright():
    metrics = tail_metrics(pd.Series([0.01, -0.01]))
    passed, failures = evaluate_tail_gates(metrics)
    assert not passed
    assert "not enough" in failures[0]


def test_find_vol_events_detects_a_genuine_shock_and_ignores_normal_noise():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2026-01-01", periods=200, freq="4h")
    prices = pd.Series(100.0, index=idx)
    noise = rng.normal(0, 0.002, size=200)
    log_prices = np.cumsum(noise)
    shock_bar = 120
    log_prices[shock_bar:] += -0.15  # a sharp one-bar crash, sustained after
    prices = pd.Series(100.0 * np.exp(log_prices), index=idx)
    events = find_vol_events(prices, window=30, z=3.0)
    assert len(events) >= 1
    assert idx[shock_bar] in events or idx[shock_bar - 1] in events or idx[shock_bar + 1] in events


def test_find_vol_events_finds_nothing_in_pure_low_vol_noise():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2026-01-01", periods=200, freq="4h")
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, size=200))), index=idx)
    events = find_vol_events(prices, window=30, z=4.0)
    assert len(events) == 0


def test_tail_gates_are_strict_enough_to_matter():
    """Guards against someone quietly loosening these until everything passes."""
    assert TAIL_GATES["min_vol_events"] >= 3
    assert TAIL_GATES["max_worst_over_mean_win"] <= 20.0
    assert TAIL_GATES["max_cvar95_over_mean_win"] <= 15.0
