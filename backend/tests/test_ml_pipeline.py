import json

import numpy as np
import pandas as pd
import pytest

from app.features.feature_pipeline import build_feature_frame
from app.ml.dataset import build_training_frame, purged_walk_forward_splits
from app.ml.train import train_and_save, walk_forward_validate
from app.ml.predict import MLPredictor


def _synthetic_candles(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.004, n)
    close = 3_000 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(10, 100, n)
    time = np.arange(n) * 900  # 15m bars

    return pd.DataFrame(
        {"time": time, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_purged_walk_forward_splits_respect_embargo():
    splits = purged_walk_forward_splits(1000, n_splits=4, embargo=15)
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        # embargo enforced: no train sample within `embargo` bars of test start
        assert test_idx.min() - train_idx.max() >= 15


def test_build_training_frame_drops_warmup_and_label_tail():
    candles = _synthetic_candles(n=500)
    features = build_feature_frame(candles)
    X, y = build_training_frame(features, horizon_bars=15, warmup_bars=60)

    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) < len(features)


def test_train_and_predict_round_trip(tmp_path, monkeypatch):
    import app.ml.train as train_mod
    import app.ml.predict as predict_mod

    monkeypatch.setattr(train_mod, "MODEL_STORE_DIR", tmp_path)
    monkeypatch.setattr(predict_mod, "MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_candles(n=1200)
    features = build_feature_frame(candles)

    report = train_and_save(features, symbol="TESTUSD", resolution="15m", horizon_bars=15)
    assert report.n_samples > 0
    assert (tmp_path / f"TESTUSD_15m_latest.json").exists()

    predictor = MLPredictor("TESTUSD", "15m")
    assert predictor.is_ready

    scores = predictor.predict_score_batch(features)
    assert len(scores) == len(features)
    assert scores.abs().max() <= 1.0


def test_walk_forward_validate_returns_out_of_sample_predictions():
    """Regression test: pred_std used to be calibrated from the final model's
    predictions on its OWN full training set (in-sample), which understates
    the model's true out-of-sample prediction spread and made live signal
    magnitudes systematically overconfident. walk_forward_validate must
    expose the held-out fold predictions so train_and_save can calibrate
    against those instead.
    """
    candles = _synthetic_candles(n=1200)
    features = build_feature_frame(candles)
    X, y = build_training_frame(features, horizon_bars=15)

    fold_metrics, oos_preds = walk_forward_validate(X, y, horizon_bars=15)

    assert len(fold_metrics) > 0
    assert len(oos_preds) > 0
    assert len(oos_preds) == sum(f.n_test for f in fold_metrics)


def test_trained_model_pred_std_is_calibrated_out_of_sample(tmp_path, monkeypatch):
    import app.ml.train as train_mod

    monkeypatch.setattr(train_mod, "MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_candles(n=1200)
    features = build_feature_frame(candles)
    X, y = build_training_frame(features, horizon_bars=15)
    _, oos_preds = walk_forward_validate(X, y, horizon_bars=15)

    report = train_and_save(features, symbol="TESTUSD2", resolution="15m", horizon_bars=15)

    assert report.prediction_std == pytest.approx(float(np.std(oos_preds)), rel=1e-6)


def test_training_report_is_persisted_not_just_logged(tmp_path, monkeypatch):
    """Regression test: the purged walk-forward fold metrics (directional
    accuracy, information coefficient, rmse) computed on EVERY retrain --
    the only honest, out-of-sample accuracy number this model has -- were
    computed, logged once, and then discarded. The daily retrain scheduler
    job never persisted or surfaced them anywhere, so there was no way to
    check a live model's real skill except an isolated backtest that
    unavoidably overlaps its own training data.
    """
    import app.ml.train as train_mod

    monkeypatch.setattr(train_mod, "MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_candles(n=1200)
    features = build_feature_frame(candles)

    report = train_and_save(features, symbol="TESTUSD3", resolution="15m", horizon_bars=15)

    latest_report_path = tmp_path / "TESTUSD3_15m_latest_report.json"
    assert latest_report_path.exists()

    persisted = json.loads(latest_report_path.read_text())
    assert persisted["symbol"] == "TESTUSD3"
    assert persisted["mean_directional_accuracy"] == pytest.approx(report.mean_directional_accuracy)
    assert persisted["mean_information_coefficient"] == pytest.approx(report.mean_information_coefficient)
    assert persisted["mean_decidable_directional_accuracy"] == pytest.approx(report.mean_decidable_directional_accuracy)
    assert persisted["mean_decidable_fraction"] == pytest.approx(report.mean_decidable_fraction)
    assert len(persisted["fold_metrics"]) == len(report.fold_metrics)
    assert len(persisted["fold_metrics"]) > 0
    for fold in persisted["fold_metrics"]:
        assert set(fold.keys()) == {
            "fold", "directional_accuracy", "information_coefficient", "rmse", "n_test",
            "decidable_fraction", "decidable_directional_accuracy",
        }

    # The versioned copy (not just the "latest" pointer) must also exist, so
    # historical accuracy can be compared across retrains later.
    version = json.loads((tmp_path / "TESTUSD3_15m_latest.json").read_text())["version"]
    versioned_report_path = tmp_path / f"TESTUSD3_15m_{version}.report.json"
    assert versioned_report_path.exists()
