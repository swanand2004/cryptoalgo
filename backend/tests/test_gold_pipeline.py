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


def test_long_term_now_uses_the_full_feature_set_at_realistic_paxg_scale(tmp_path, monkeypatch):
    """long_term switched from Delta's `.DEXAUTUSD` (~127 real daily bars)
    to PAXG/USDT via Binance (~2200 real daily bars) on 2026-08-25 -- see
    app/gold/ingest.py's docstring. GOLD_HORIZONS["long_term"] now uses the
    SAME full feature set/210-bar warmup/5-fold split as the other two
    horizons (no more special-cased reduced set), which this realistically-
    sized sample must train cleanly with a real (not degenerate) 5-fold
    walk-forward.
    """
    import app.gold.train as train_mod

    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_gold_candles(n=2185, seed=19)  # matches PAXG/USDT's real daily history depth
    features = build_gold_feature_frame(candles)
    cfg = GOLD_HORIZONS["long_term"]
    assert cfg["n_splits"] == 5
    assert cfg["warmup_bars"] == 210

    report = train_and_save_gold(
        features, horizon="long_term", horizon_bars=cfg["horizon_bars"], resolution=cfg["resolution"],
        feature_columns=cfg["feature_columns"], warmup_bars=cfg["warmup_bars"], n_splits=cfg["n_splits"],
    )
    assert report.n_samples > 1900
    assert len(report.fold_metrics) == 5


def test_legacy_reduced_feature_set_still_works_on_a_genuinely_thin_sample(tmp_path, monkeypatch):
    """GOLD_HORIZONS["long_term"] no longer uses this (see the test above),
    but the reduced feature set/warmup (GOLD_FEATURE_COLUMNS_LONG_TERM /
    GOLD_WARMUP_BARS_LONG_TERM in features.py) is kept as a real, working
    capability for a daily data source that's genuinely this thin -- the
    exact scenario that motivated it originally (Delta's `.DEXAUTUSD`, ~127
    real daily bars, 2026-08-21).
    """
    import app.gold.train as train_mod
    from app.gold.features import GOLD_FEATURE_COLUMNS_LONG_TERM, GOLD_WARMUP_BARS_LONG_TERM

    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    candles = _synthetic_gold_candles(n=127, seed=19)
    features = build_gold_feature_frame(candles)

    report = train_and_save_gold(
        features, horizon="long_term", horizon_bars=5, resolution="1d",
        feature_columns=GOLD_FEATURE_COLUMNS_LONG_TERM, warmup_bars=GOLD_WARMUP_BARS_LONG_TERM, n_splits=1,
    )
    assert report.n_samples > 0
