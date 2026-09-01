from __future__ import annotations

"""Formal statistical gates for backtest results.

Today (2026-08-31) two bugs were found in app/backtest/engine.py that had
inflated every backtest this project ever produced by roughly 30-50x. They
were caught by NOTICING a pattern by eye -- an ever-tighter trailing stop
improving monotonically with no optimum. That worked, but it is not a
process. These gates make it one.

Two independent questions, deliberately kept separate:

  deflated_sharpe -- "is this Sharpe real, or is it the best of many tries?"
      Corrects an observed Sharpe for (a) how many configurations were
      tried, (b) how many observations back it, and (c) the non-normality
      of the return distribution (crypto returns are fat-tailed and skewed,
      which flatters a naive Sharpe). Bailey & Lopez de Prado (2014).

  probability_of_backtest_overfitting -- "does picking the in-sample winner
      actually work?" Via CSCV: repeatedly split the return history in half,
      pick the best config on one half, and see where it ranks on the other.
      If the in-sample winner lands in the bottom half out-of-sample more
      often than not, selection itself is overfitting.

Both are cheap. Run them before believing any result.
"""

from itertools import combinations

import numpy as np
from scipy.stats import norm


def annualisation_factor(bars_per_year: int) -> float:
    return float(np.sqrt(bars_per_year))


def sharpe_ratio(returns: np.ndarray, bars_per_year: int = 365 * 6) -> float:
    """Annualised Sharpe. Default assumes 4h bars (6 per day)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * annualisation_factor(bars_per_year))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected MAXIMUM Sharpe from `n_trials` strategies that all genuinely
    have zero edge -- i.e. the bar a real strategy must clear.

    Note this needs the DISPERSION of Sharpes across the trials, not a
    frequency constant. A version of this that substitutes 1/sqrt(freq) for
    `trial_sharpe_std` (as some published snippets do) is dimensionally
    wrong and will silently pass weak strategies.
    """
    if n_trials <= 1 or trial_sharpe_std <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(trial_sharpe_std * ((1.0 - gamma) * a + gamma * b))


def deflated_sharpe(
    observed_sharpe: float,
    n_obs: int,
    n_trials: int,
    trial_sharpe_std: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the observed Sharpe reflects genuine skill.

    `observed_sharpe` and `trial_sharpe_std` must be in the SAME units
    (both annualised, or both per-bar). `kurtosis` is the raw fourth moment
    (3.0 for a normal distribution), not excess kurtosis.

    Returns a probability in [0, 1]; treat <0.95 as "not established".
    """
    if n_obs < 2:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std)
    # Variance of the Sharpe estimator under non-normal returns.
    denom_sq = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    denom = np.sqrt(max(denom_sq, 1e-12))
    z = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def probability_of_backtest_overfitting(perf: np.ndarray, n_splits: int = 10) -> float:
    """CSCV probability that selecting the in-sample best config leads to
    below-median out-of-sample performance.

    `perf`: (n_observations, n_configs) matrix of per-bar returns, one column
    per configuration tried. Needs >= 2 configs -- PBO is a statement about
    SELECTION, so it is undefined when there was no choice to make.

    Interpretation: 0.0 = selection generalises perfectly; 0.5 = selecting
    the winner is a coin flip; >0.5 = the in-sample winner is systematically
    the WRONG choice. Reject above ~0.35.
    """
    perf = np.asarray(perf, dtype=float)
    if perf.ndim != 2 or perf.shape[1] < 2:
        raise ValueError("perf must be (n_obs, n_configs) with at least 2 configs")
    n_splits = max(2, n_splits - (n_splits % 2))  # must be even to halve
    blocks = np.array_split(np.arange(perf.shape[0]), n_splits)

    logits = []
    n_configs = perf.shape[1]
    for combo in combinations(range(n_splits), n_splits // 2):
        tr = np.concatenate([blocks[i] for i in combo])
        te = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        best = int(np.nanmean(perf[tr], axis=0).argmax())
        oos = np.nanmean(perf[te], axis=0)
        # rank of the in-sample winner among out-of-sample results
        rank = float((oos.argsort().argsort()[best] + 1) / (n_configs + 1))
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    return float(np.mean(np.asarray(logits) <= 0.0))


# Thresholds a result must clear before it is allowed near live capital.
# These are deliberately strict: this project's entire backtest history was
# fabricated by an engine bug until 2026-08-31, and the live account is
# 45.6% below its starting balance. The burden of proof sits high.
DEPLOY_GATES = {
    "min_oos_sharpe": 1.0,
    "min_deflated_sharpe": 0.95,
    "max_pbo": 0.35,
    "min_trades": 100,
    "min_edge_after_costs_bps": 1.0,
}


def evaluate_gates(
    oos_sharpe: float,
    deflated: float,
    pbo_value: float,
    n_trades: int,
    edge_after_costs_bps: float,
    gates: dict | None = None,
) -> tuple[bool, list[str]]:
    """Returns (passed, reasons_for_failure). Empty reasons means it passed."""
    g = gates or DEPLOY_GATES
    failures = []
    if oos_sharpe < g["min_oos_sharpe"]:
        failures.append(f"OOS Sharpe {oos_sharpe:.2f} < {g['min_oos_sharpe']}")
    if deflated < g["min_deflated_sharpe"]:
        failures.append(f"deflated Sharpe {deflated:.3f} < {g['min_deflated_sharpe']}")
    if pbo_value > g["max_pbo"]:
        failures.append(f"PBO {pbo_value:.3f} > {g['max_pbo']}")
    if n_trades < g["min_trades"]:
        failures.append(f"only {n_trades} trades < {g['min_trades']}")
    if edge_after_costs_bps < g["min_edge_after_costs_bps"]:
        failures.append(
            f"edge after costs {edge_after_costs_bps:.2f}bps < {g['min_edge_after_costs_bps']}bps"
        )
    return (not failures), failures
