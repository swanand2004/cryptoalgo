import json

import numpy as np
import pandas as pd
import pytest

from app.gold.features import build_gold_feature_frame
from app.gold.train import (
    GOLD_HORIZONS,
    build_gold_training_frame,
    gold_walk_forward_validate,
    train_and_save_gold,
)


def _synthetic_gold_candles(n: int = 1000, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.003, n)
    close = 4_500 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    time = np.arange(n) * 3600

    return pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close})


def test_build_gold_training_frame_drops_warmup_and_label_tail():
    candles = _synthetic_gold_candles(n=1000)
    features = build_gold_feature_frame(candles)
    X, y = build_gold_training_frame(features, horizon_bars=24)

    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) < len(features)


def test_gold_walk_forward_validate_returns_out_of_sample_predictions():
    candles = _synthetic_gold_candles(n=1000)
    features = build_gold_feature_frame(candles)
    X, y = build_gold_training_frame(features, horizon_bars=24)

    fold_metrics, oos_preds = gold_walk_forward_validate(X, y, horizon_bars=24)

    assert len(fold_metrics) > 0
    assert len(oos_preds) > 0
    assert len(oos_preds) == sum(f.n_test for f in fold_metrics)


def test_train_and_predict_round_trip(tmp_path, monkeypatch):
    import app.gold.predict as predict_mod
    import app.gold.train as train_mod

    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)
    monkeypatch.setattr(predict_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_gold_candles(n=1000)
    features = build_gold_feature_frame(candles)

    report = train_and_save_gold(features, horizon="short_term", horizon_bars=24, resolution="1h")
    assert report.n_samples > 0
    assert (tmp_path / "gold_short_term_latest.json").exists()

    from app.gold.predict import GoldPredictor

    predictor = GoldPredictor("short_term")
    assert predictor.is_ready

    scores = predictor.predict_score_batch(features)
    assert len(scores) == len(features)
    assert scores.abs().max() <= 1.0


def test_training_report_is_persisted_with_gold_specific_fields(tmp_path, monkeypatch):
    import app.gold.train as train_mod

    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_gold_candles(n=1000)
    features = build_gold_feature_frame(candles)
    cfg = GOLD_HORIZONS["long_term"]

    report = train_and_save_gold(
        features, horizon="long_term", horizon_bars=cfg["horizon_bars"], resolution=cfg["resolution"],
        feature_columns=cfg["feature_columns"], warmup_bars=cfg["warmup_bars"], n_splits=cfg["n_splits"],
    )

    latest_report_path = tmp_path / "gold_long_term_latest_report.json"
    assert latest_report_path.exists()

    persisted = json.loads(latest_report_path.read_text())
    assert persisted["horizon"] == "long_term"
    assert persisted["resolution"] == "1d"
    assert persisted["mean_directional_accuracy"] == pytest.approx(report.mean_directional_accuracy)
    assert len(persisted["fold_metrics"]) == len(report.fold_metrics)

    version = json.loads((tmp_path / "gold_long_term_latest.json").read_text())["version"]
    assert (tmp_path / f"gold_long_term_{version}.report.json").exists()


def test_two_horizons_never_collide_in_the_same_model_store(tmp_path, monkeypatch):
    """short_term and long_term must be fully independent models -- training
    one must not overwrite or interfere with the other's pointer/artifacts.
    """
    import app.gold.train as train_mod

    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_gold_candles(n=1000)
    features = build_gold_feature_frame(candles)

    for horizon in ("short_term", "long_term"):
        cfg = GOLD_HORIZONS[horizon]
        train_and_save_gold(
            features, horizon=horizon, horizon_bars=cfg["horizon_bars"], resolution=cfg["resolution"],
            feature_columns=cfg["feature_columns"], warmup_bars=cfg["warmup_bars"], n_splits=cfg["n_splits"],
        )

    assert (tmp_path / "gold_short_term_latest.json").exists()
    assert (tmp_path / "gold_long_term_latest.json").exists()

    short_version = json.loads((tmp_path / "gold_short_term_latest.json").read_text())["version"]
    long_version = json.loads((tmp_path / "gold_long_term_latest.json").read_text())["version"]
    assert (tmp_path / f"gold_short_term_{short_version}.txt").exists()
    assert (tmp_path / f"gold_long_term_{long_version}.txt").exists()


def test_long_term_trains_on_realistically_thin_daily_history(tmp_path, monkeypatch):
    """Regression test for a real bug caught manually (2026-08-21): with the
    FULL feature set's 210-bar warmup, `.DEXAUTUSD`'s real ~127-day daily
    history left ZERO usable training rows (100/200-bar-lookback features
    like hurst_200/ma_z_200 can never be non-NaN with under 200 total rows).
    GOLD_HORIZONS["long_term"] must use a reduced feature set/warmup/
    n_splits that actually fits this little data.
    """
    import app.gold.train as train_mod

    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_gold_candles(n=127, seed=19)  # matches real .DEXAUTUSD daily history depth
    features = build_gold_feature_frame(candles)
    cfg = GOLD_HORIZONS["long_term"]

    report = train_and_save_gold(
        features, horizon="long_term", horizon_bars=cfg["horizon_bars"], resolution=cfg["resolution"],
        feature_columns=cfg["feature_columns"], warmup_bars=cfg["warmup_bars"], n_splits=cfg["n_splits"],
    )
    assert report.n_samples > 0
