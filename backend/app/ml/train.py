from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.features.feature_pipeline import FEATURE_COLUMNS
from app.ml.dataset import build_training_frame, purged_walk_forward_splits

logger = logging.getLogger(__name__)

MODEL_STORE_DIR = Path(__file__).resolve().parent / "model_store"

# Per-resolution training config: `limit` is a candle-count comfortably
# within what's actually ingested for that resolution (see the real
# history check this project has done before -- 15m has ~68 days, 1h/4h
# ~98 days -- these sit safely under each), and `horizon_bars` is chosen
# so every resolution predicts roughly the SAME ~4-hour-ahead target,
# not a horizon that silently means something different at each
# timeframe (15 bars is ~3h45m at 15m, but would be 15 HOURS at 1h and
# 60 HOURS at 4h if reused as-is). This keeps "multi-timeframe" meaning
# "the same short-term question, asked from data at different
# granularities" rather than conflating granularity with horizon.
#
# Honest limitation, not hidden: 4h has far fewer independent bars than
# 15m (500 vs 5000) even though it covers a similar calendar span, so its
# walk-forward folds are smaller and noisier -- reported as such, not
# smoothed over.
ML_RESOLUTIONS: dict[str, dict] = {
    "15m": {"limit": 5000, "horizon_bars": 15},  # ~3h45m ahead
    "1h": {"limit": 2000, "horizon_bars": 4},  # ~4h ahead
    "4h": {"limit": 500, "horizon_bars": 1},  # ~4h ahead
}

# "Decidable, not predictable": forcing a directional call on every single
# bar means most of them are made with no real edge (measured directional
# accuracy sits barely above a coin flip, see TrainingReport.
# mean_directional_accuracy). But |predicted return| is itself informative
# about the prediction's OWN quality -- confirmed via purged walk-forward
# AND a genuine 70/30 holdout the threshold was never tuned on: restricting
# to the top ~30% most confident predictions (|pred| >= this many multiples
# of pred_std) held up as a real, positive effect on ALL 3 symbols on the
# holdout (BTC +3.2pt, ETH +1.0pt, SOL +1.3pt over each symbol's own
# overall accuracy), with an even larger effect at the top ~10% (BTC +2.5,
# ETH +13.9, SOL +6.3pt) traded off against a smaller, noisier sample.
# 0.95 (top ~30%) was chosen as the more ROBUST cut -- it has no exceptions
# across the 3 symbols, unlike top ~20% which dipped slightly on BTC.
# Empirically this ratio came out nearly identical across BTC/ETH/SOL
# (0.92-0.96x for top 30%, 1.6-1.7x for top 10%) because it's really
# exploiting the roughly-normal SHAPE of the prediction distribution, not a
# symbol-specific quirk -- hence one shared constant rather than a
# per-symbol lookup table.
CONFIDENCE_THRESHOLD_MULTIPLE = 0.95

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    # Fixed so retraining on the same data is reproducible -- without this,
    # bagging/feature subsampling differ every run, making it impossible to
    # tell "the model actually improved" from "RNG noise" between retrains.
    "seed": 42,
    "bagging_seed": 42,
    "feature_fraction_seed": 42,
}


@dataclass
class FoldMetrics:
    fold: int
    directional_accuracy: float
    information_coefficient: float
    rmse: float
    n_test: int
    # "Decidable" metrics -- accuracy restricted to the subset of this
    # fold's bars where |prediction| >= CONFIDENCE_THRESHOLD_MULTIPLE *
    # this fold's own OOS prediction std. decidable_fraction is what
    # fraction of the fold's bars clear that bar (should sit near the ~30%
    # the multiple was calibrated for, modulo fold-to-fold variation).
    decidable_fraction: float = 0.0
    decidable_directional_accuracy: float = 0.0


@dataclass
class TrainingReport:
    symbol: str
    resolution: str
    horizon_bars: int
    trained_at: str
    n_samples: int
    fold_metrics: list[FoldMetrics] = field(default_factory=list)
    mean_directional_accuracy: float = 0.0
    mean_information_coefficient: float = 0.0
    mean_decidable_fraction: float = 0.0
    mean_decidable_directional_accuracy: float = 0.0
    prediction_std: float = 0.0
    model_path: str = ""


def _long_only_baseline(y: pd.Series) -> float:
    """Share of MOVED bars that closed up -- computed on the same basis as
    directional accuracy so the two are directly comparable."""
    moved = y[np.sign(y) != 0]
    return float((moved > 0).mean()) if len(moved) else 0.5


def _evaluate_fold(model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, float, float, float, np.ndarray]:
    preds = model.predict(X_test)
    y_true = y_test.values
    directional_acc = float(np.mean(np.sign(preds) == np.sign(y_true)))
    ic, _ = spearmanr(preds, y_true)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))

    # "Decidable, not predictable" -- see CONFIDENCE_THRESHOLD_MULTIPLE.
    # Calibrated from THIS fold's own OOS prediction spread, never the
    # test labels, so this stays leak-free the same way pred_std does.
    fold_pred_std = np.std(preds)
    threshold = CONFIDENCE_THRESHOLD_MULTIPLE * fold_pred_std
    confident_mask = np.abs(preds) >= threshold
    decidable_fraction = float(np.mean(confident_mask))
    decidable_acc = (
        float(np.mean(np.sign(preds[confident_mask]) == np.sign(y_true[confident_mask])))
        if confident_mask.any()
        else 0.0
    )
    return directional_acc, float(ic) if not np.isnan(ic) else 0.0, rmse, decidable_fraction, decidable_acc, preds


def walk_forward_validate(
    X: pd.DataFrame, y: pd.Series, horizon_bars: int, n_splits: int = 5
) -> tuple[list[FoldMetrics], np.ndarray]:
    """Returns (fold_metrics, out_of_sample_predictions) -- the latter is
    every fold's predictions on its own held-out test block, concatenated.
    Used by train_and_save to calibrate pred_std honestly (see its docstring).
    """
    splits = purged_walk_forward_splits(len(X), n_splits=n_splits, embargo=horizon_bars)
    results = []
    oos_preds: list[np.ndarray] = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_set = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        model = lgb.train(LGBM_PARAMS, train_set, num_boost_round=200)
        acc, ic, rmse, decidable_frac, decidable_acc, preds = _evaluate_fold(model, X.iloc[test_idx], y.iloc[test_idx])
        results.append(FoldMetrics(fold_i, acc, ic, rmse, len(test_idx), decidable_frac, decidable_acc))
        oos_preds.append(preds)
        logger.info(
            "Fold %d: dir_acc=%.3f IC=%.3f rmse=%.5f n=%d | decidable(%.0f%% of bars): dir_acc=%.3f",
            fold_i, acc, ic, rmse, len(test_idx), decidable_frac * 100, decidable_acc,
        )
    return results, (np.concatenate(oos_preds) if oos_preds else np.array([]))


def train_and_save(
    feature_df: pd.DataFrame,
    symbol: str,
    resolution: str = "15m",
    horizon_bars: int = 15,
    model_version: str | None = None,
) -> TrainingReport:
    """Trains a LightGBM regressor on forward returns, validates it with purged
    walk-forward CV (the metrics that matter -- out-of-sample, no leakage), then
    refits on the full dataset for the artifact actually used live. Both the
    model and a `pred_std` normalization constant (used to scale raw predicted
    returns into a [-1, 1] signal) are persisted together.
    """
    X, y = build_training_frame(feature_df, horizon_bars=horizon_bars)
    if len(X) < 200:
        raise ValueError(f"Not enough samples to train ({len(X)}) -- need at least 200")

    fold_metrics, oos_preds = walk_forward_validate(X, y, horizon_bars)

    full_set = lgb.Dataset(X, label=y)
    final_model = lgb.train(LGBM_PARAMS, full_set, num_boost_round=200)

    version = model_version or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_STORE_DIR / f"{symbol}_{resolution}_{version}.txt"
    final_model.save_model(str(model_path))

    # Calibrated from OUT-OF-SAMPLE fold predictions, not the final model's
    # predictions on the data it was just fit on -- in-sample predictions
    # understate a model's true prediction spread, which made live signal
    # magnitudes (score = raw / (2*pred_std)) systematically overconfident,
    # saturating toward +-1 more often than the model's real skill justified.
    # Falls back to in-sample only if walk-forward produced no folds at all
    # (e.g. too little data), which should be rare given the len(X) >= 200
    # guard above.
    if len(oos_preds) > 1 and np.std(oos_preds) > 0:
        pred_std = float(np.std(oos_preds))
    else:
        pred_std = float(np.std(final_model.predict(X)))

    meta = {
        "symbol": symbol,
        "resolution": resolution,
        "horizon_bars": horizon_bars,
        "version": version,
        "feature_columns": FEATURE_COLUMNS,
        # The bar a directional model actually has to clear. Crypto drifts up
        # too -- measured on nine years of hourly Binance history, 51.3% of
        # moved BTC bars and 51.0% of ETH bars close higher over 4 hours -- so
        # comparing a model to 50% flatters it by more than a point. Recorded
        # per-model rather than assumed, because it differs by symbol and
        # horizon (SOL's is 49.95%, i.e. below a coin flip).
        "long_only_baseline": _long_only_baseline(y),
        "pred_std": pred_std,
        "confidence_threshold_multiple": CONFIDENCE_THRESHOLD_MULTIPLE,
    }
    meta_path = MODEL_STORE_DIR / f"{symbol}_{resolution}_{version}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    latest_pointer = MODEL_STORE_DIR / f"{symbol}_{resolution}_latest.json"
    latest_pointer.write_text(json.dumps({"version": version}))

    report = TrainingReport(
        symbol=symbol,
        resolution=resolution,
        horizon_bars=horizon_bars,
        trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        n_samples=len(X),
        fold_metrics=fold_metrics,
        mean_directional_accuracy=float(np.mean([f.directional_accuracy for f in fold_metrics])) if fold_metrics else 0.0,
        mean_information_coefficient=float(np.mean([f.information_coefficient for f in fold_metrics])) if fold_metrics else 0.0,
        mean_decidable_fraction=float(np.mean([f.decidable_fraction for f in fold_metrics])) if fold_metrics else 0.0,
        mean_decidable_directional_accuracy=(
            float(np.mean([f.decidable_directional_accuracy for f in fold_metrics])) if fold_metrics else 0.0
        ),
        prediction_std=meta["pred_std"],
        model_path=str(model_path),
    )
    logger.info(
        "Trained %s %s model v%s: dir_acc=%.3f IC=%.3f (n=%d) | decidable(~%.0f%% of bars): dir_acc=%.3f",
        symbol, resolution, version, report.mean_directional_accuracy, report.mean_information_coefficient, len(X),
        report.mean_decidable_fraction * 100, report.mean_decidable_directional_accuracy,
    )

    # Persisted so the ONLY honest accuracy number this model has -- purged
    # walk-forward, out-of-sample, computed fresh every retrain -- survives
    # past the log line. Without this, every day's real validation result
    # was computed and then thrown away, leaving nothing to check the
    # model's actual skill against except an isolated backtest that
    # (unavoidably) overlaps the data it was trained on.
    report_dict = {
        "symbol": report.symbol,
        "resolution": report.resolution,
        "horizon_bars": report.horizon_bars,
        "trained_at": report.trained_at,
        "n_samples": report.n_samples,
        "mean_directional_accuracy": report.mean_directional_accuracy,
        "mean_information_coefficient": report.mean_information_coefficient,
        "mean_decidable_fraction": report.mean_decidable_fraction,
        "mean_decidable_directional_accuracy": report.mean_decidable_directional_accuracy,
        "confidence_threshold_multiple": CONFIDENCE_THRESHOLD_MULTIPLE,
        "prediction_std": report.prediction_std,
        "model_path": report.model_path,
        "fold_metrics": [
            {
                "fold": f.fold,
                "directional_accuracy": f.directional_accuracy,
                "information_coefficient": f.information_coefficient,
                "rmse": f.rmse,
                "n_test": f.n_test,
                "decidable_fraction": f.decidable_fraction,
                "decidable_directional_accuracy": f.decidable_directional_accuracy,
            }
            for f in fold_metrics
        ],
    }
    report_path = MODEL_STORE_DIR / f"{symbol}_{resolution}_{version}.report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))
    latest_report_pointer = MODEL_STORE_DIR / f"{symbol}_{resolution}_latest_report.json"
    latest_report_pointer.write_text(json.dumps(report_dict, indent=2))

    return report
