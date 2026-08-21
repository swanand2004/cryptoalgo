import numpy as np
import pandas as pd
import pytest

from app.ml.train import CONFIDENCE_THRESHOLD_MULTIPLE, _evaluate_fold


class _FakeBooster:
    def __init__(self, predictions: list[float]) -> None:
        self._predictions = np.array(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predictions[: len(X)]


def test_decidable_accuracy_isolates_the_high_confidence_subset():
    """Constructed so the low-|prediction| half is WRONG every time and the
    high-|prediction| half is RIGHT every time -- overall accuracy lands
    exactly at a coin flip (0.5) while the decidable (high-confidence)
    subset is a clean 1.0, with decidable_fraction landing on exactly the
    high-magnitude half. Verifies the threshold is computed from the
    predictions' OWN spread (never the labels) and correctly separates the
    two halves.
    """
    preds = [0.1, -0.1, 0.1, -0.1, 5.0, 5.0, -5.0, -5.0]
    y_test = pd.Series([-0.01, 0.01, -0.01, 0.01, 0.02, 0.02, -0.02, -0.02])
    X_test = pd.DataFrame({"f1": [0.0] * len(preds)})

    acc, ic, rmse, decidable_fraction, decidable_acc, returned_preds = _evaluate_fold(
        _FakeBooster(preds), X_test, y_test
    )

    assert acc == pytest.approx(0.5)
    assert decidable_fraction == pytest.approx(0.5)
    assert decidable_acc == pytest.approx(1.0)
    assert list(returned_preds) == preds


def test_zero_dispersion_predictions_all_count_as_decidable():
    # All predictions identical -> std is 0 -> threshold is 0 -> every
    # |pred| (>= 0 always) clears it. Degenerate, but must resolve to "no
    # filtering happened" (decidable_fraction=1.0) rather than dividing by
    # zero or treating a flat prediction set as having no confident bars.
    preds = [0.01] * 6
    y_test = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    X_test = pd.DataFrame({"f1": [0.0] * len(preds)})

    acc, ic, rmse, decidable_fraction, decidable_acc, _ = _evaluate_fold(_FakeBooster(preds), X_test, y_test)
    assert decidable_fraction == pytest.approx(1.0)
    assert decidable_acc == pytest.approx(acc)  # no filtering happened -- decidable == overall


def test_confidence_threshold_multiple_is_a_single_shared_constant():
    # Documents the empirical finding: this is one constant, not a
    # per-symbol lookup table (see train.py's docstring for why).
    assert 0.0 < CONFIDENCE_THRESHOLD_MULTIPLE < 2.0
