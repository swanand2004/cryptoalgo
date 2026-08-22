import numpy as np
import pandas as pd
import pytest

from app.gold.predict import GoldPredictor


class _FakeBooster:
    def __init__(self, predictions: list[float]) -> None:
        self._predictions = np.array(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predictions[: len(X)]


def _predictor_with(predictions: list[float], pred_std: float, confidence_multiple: float = 0.95) -> GoldPredictor:
    predictor = GoldPredictor(horizon="short_term")  # no model on disk -- _try_load leaves it not-ready
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
    predictor = GoldPredictor(horizon="nonexistent_horizon")
    assert not predictor.is_ready
    result = predictor.predict_score_batch(_df(3))
    assert (result == 0.0).all()


def test_confidence_gate_zeroes_low_magnitude_predictions():
    predictions = [0.5, 0.94, 0.95, 2.0]
    predictor = _predictor_with(predictions, pred_std=1.0, confidence_multiple=0.95)

    gated = predictor.predict_score_batch(_df(4), require_confidence=True)
    assert gated.iloc[0] == 0.0
    assert gated.iloc[1] == 0.0
    assert gated.iloc[2] != 0.0
    assert gated.iloc[3] != 0.0


def test_require_confidence_false_returns_the_always_call_score():
    predictions = [0.1, 0.2]
    predictor = _predictor_with(predictions, pred_std=1.0, confidence_multiple=0.95)

    ungated = predictor.predict_score_batch(_df(2), require_confidence=False)
    assert ungated.iloc[0] == pytest.approx(0.1 / 2.0)
    assert ungated.iloc[1] == pytest.approx(0.2 / 2.0)


def test_missing_confidence_multiple_in_meta_defaults_to_always_call():
    predictor = _predictor_with([0.01], pred_std=1.0)
    del predictor.meta["confidence_threshold_multiple"]

    gated = predictor.predict_score_batch(_df(1), require_confidence=True)
    assert gated.iloc[0] != 0.0


def test_two_horizons_are_independent_predictor_instances():
    """short_term and long_term GoldPredictors must never accidentally
    share loaded state -- each looks at its OWN gold_{horizon}_latest.json
    pointer."""
    short = GoldPredictor(horizon="short_term")
    long = GoldPredictor(horizon="long_term")
    assert short.horizon != long.horizon


def test_predict_raw_return_last_returns_the_unnormalized_prediction():
    """Price targets need the RAW predicted return (current_price * (1 +
    raw)), not the [-1, 1] normalized score predict_score_batch returns."""
    predictor = _predictor_with([0.0123, 999.0], pred_std=1.0)  # second value never reached (only last row used)
    raw = predictor.predict_raw_return_last(_df(1))
    assert raw == pytest.approx(0.0123)


def test_predict_raw_return_last_is_none_when_not_ready():
    predictor = GoldPredictor(horizon="nonexistent_horizon")
    assert predictor.predict_raw_return_last(_df(1)) is None


def test_predict_raw_return_last_is_none_for_missing_features():
    predictor = _predictor_with([0.05], pred_std=1.0)
    df = pd.DataFrame({"f1": [np.nan]})
    assert predictor.predict_raw_return_last(df) is None
