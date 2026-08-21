import numpy as np
import pandas as pd
import pytest

from app.ml.predict import MLPredictor


class _FakeBooster:
    def __init__(self, predictions: list[float]) -> None:
        self._predictions = np.array(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predictions[: len(X)]


def _predictor_with(predictions: list[float], pred_std: float, confidence_multiple: float = 0.95) -> MLPredictor:
    predictor = MLPredictor(symbol="TESTUSD")  # no model on disk -- _try_load leaves it not-ready
    predictor.model = _FakeBooster(predictions)
    predictor.meta = {
        "feature_columns": ["f1"],
        "pred_std": pred_std,
        "confidence_threshold_multiple": confidence_multiple,
    }
    return predictor


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"f1": [1.0] * n})


def test_not_ready_returns_all_zero():
    predictor = MLPredictor(symbol="NOPE_DOES_NOT_EXIST")
    assert not predictor.is_ready
    result = predictor.predict_score_batch(_df(3))
    assert (result == 0.0).all()


def test_confidence_gate_zeroes_low_magnitude_predictions():
    # pred_std=1.0, confidence_multiple=0.95 -> threshold |raw| >= 0.95.
    predictions = [0.5, 0.94, 0.95, 2.0]  # below, below, at, well above threshold
    predictor = _predictor_with(predictions, pred_std=1.0, confidence_multiple=0.95)

    gated = predictor.predict_score_batch(_df(4), require_confidence=True)
    assert gated.iloc[0] == 0.0  # 0.5 < 0.95 -> abstain
    assert gated.iloc[1] == 0.0  # 0.94 < 0.95 -> abstain
    assert gated.iloc[2] != 0.0  # 0.95 >= 0.95 -> decides
    assert gated.iloc[3] != 0.0  # 2.0 >= 0.95 -> decides


def test_require_confidence_false_returns_the_always_call_score():
    predictions = [0.1, 0.2]  # both well below any reasonable confidence threshold
    predictor = _predictor_with(predictions, pred_std=1.0, confidence_multiple=0.95)

    ungated = predictor.predict_score_batch(_df(2), require_confidence=False)
    assert ungated.iloc[0] != 0.0
    assert ungated.iloc[1] != 0.0
    # And matches the raw scaling exactly (raw / (2*pred_std), clipped).
    assert ungated.iloc[0] == pytest.approx(0.1 / 2.0)
    assert ungated.iloc[1] == pytest.approx(0.2 / 2.0)


def test_gated_and_ungated_agree_above_the_threshold():
    predictions = [5.0]  # comfortably above threshold either way
    predictor = _predictor_with(predictions, pred_std=1.0, confidence_multiple=0.95)

    gated = predictor.predict_score_batch(_df(1), require_confidence=True)
    ungated = predictor.predict_score_batch(_df(1), require_confidence=False)
    assert gated.iloc[0] == ungated.iloc[0]


def test_missing_confidence_multiple_in_meta_defaults_to_always_call():
    """Backward compatibility: a model trained before this feature existed
    has no confidence_threshold_multiple in its meta.json -- must not
    crash, and must behave as "never gate" (multiple=0.0) rather than
    silently zeroing every prediction.
    """
    predictor = _predictor_with([0.01], pred_std=1.0)
    del predictor.meta["confidence_threshold_multiple"]

    gated = predictor.predict_score_batch(_df(1), require_confidence=True)
    assert gated.iloc[0] != 0.0
