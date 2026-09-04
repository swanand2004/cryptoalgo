"""Flat bars must not be counted as prediction errors.

This is a regression test for a measurement bug that mattered more than any
strategy change made this session. PAXG is thinly traded and prints a lot of
exactly-flat bars -- 10.7% of 15m bars have a zero 4-bar forward return.
`np.sign(0)` is 0, so a flat bar compared against any non-zero prediction
scores as WRONG, and crucially it scores as wrong for a prediction AND for
that prediction's exact opposite at the same time.

The effect was a near-constant deduction from every accuracy figure, which
silently moved the coin-flip line from 50% down to ~46% and made every
"below 50%, therefore no edge" conclusion meaningless. The intraday gold model
was reported at 50.89% -- an apparent coin flip -- when it was actually 55.6%.

There is also nothing to get right on a bar that did not move, so excluding
them hides no losses: they are untradeable in either direction.
"""
import numpy as np
import pandas as pd

from app.gold.train import _evaluate_fold


class _StubModel:
    """Returns predictions handed to it, so the metric is tested rather than
    LightGBM."""

    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X):
        return self._preds


def _run(preds, truth):
    X = pd.DataFrame({"f": np.zeros(len(preds))})
    y = pd.Series(np.asarray(truth, dtype=float))
    return _evaluate_fold(_StubModel(preds), X, y)


def test_flat_outcomes_are_excluded_not_counted_as_wrong():
    """Four calls, all correct on the bars that moved, plus four flat bars.
    Accuracy must be 100%, not 50%."""
    preds = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    truth = [0.01, 0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0]
    acc = _run(preds, truth)[0]
    assert acc == 1.0


def test_a_signal_and_its_exact_opposite_cannot_both_be_wrong():
    """The property that exposed the bug. On bars that moved, flipping every
    prediction must flip the accuracy about 50% -- the two have to sum to 1.
    With flat bars counted as errors they summed to less, which is not a
    property any accuracy measure can have."""
    rng = np.random.default_rng(7)
    preds = rng.normal(size=400)
    truth = rng.normal(size=400) * 0.01
    truth[::5] = 0.0  # 20% flat bars

    forward = _run(preds, truth)[0]
    reversed_ = _run(-preds, truth)[0]
    assert forward + reversed_ == 1.0


def test_all_flat_is_reported_as_zero_rather_than_dividing_by_zero():
    acc = _run([1.0, -1.0, 1.0], [0.0, 0.0, 0.0])[0]
    assert acc == 0.0


def test_decidable_accuracy_also_excludes_flat_bars():
    """The confident-subset metric had the same flaw and the same fix. The
    confident bars here are the large-magnitude predictions; the flat outcomes
    among them must not drag the number down."""
    preds = [5.0, 5.0, -5.0, 5.0, 0.001, 0.001]
    truth = [0.01, 0.0, -0.01, 0.01, -0.5, 0.5]
    decidable_acc = _run(preds, truth)[4]
    # Of the confident bars, three moved and all three were called correctly.
    assert decidable_acc == 1.0


def test_decidable_fraction_still_counts_every_bar():
    """Coverage is a property of the MODEL's confidence, not of whether the
    market happened to move, so this one must keep its full denominator."""
    preds = [5.0, 5.0, 0.0, 0.0]
    truth = [0.01, 0.0, 0.01, 0.01]
    decidable_fraction = _run(preds, truth)[3]
    assert decidable_fraction == 0.5
