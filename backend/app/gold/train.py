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

from app.gold.features import (
    GOLD_FEATURE_COLUMNS,
    GOLD_FEATURE_COLUMNS_LONG_TERM,
    GOLD_WARMUP_BARS,
    GOLD_WARMUP_BARS_LONG_TERM,
)
from app.ml.dataset import make_forward_return_labels, purged_walk_forward_splits

logger = logging.getLogger(__name__)

GOLD_MODEL_STORE_DIR = Path(__file__).resolve().parent / "model_store"

# Two horizons, deliberately far apart -- "short term" and "long term" as
# genuinely different questions, not the same target reused at a different
# resolution (contrast with crypto's ML_RESOLUTIONS, which keeps one ~4h
# target constant across resolutions). `.DEXAUTUSD`'s real available history
# (checked live against Delta's own candles endpoint, 2026-08-21): ~3000
# hourly bars (~126 days) and only ~127 daily bars -- both limits below sit
# inside what's actually there.
#
# Honest limitation, not hidden: the long-term/1d model trains on roughly
# 60 usable rows after its (smaller) warmup -- nowhere near enough for a
# real 5-fold purged walk-forward (each fold needs ~20+ rows on its own), so
# it uses n_splits=1 (a single train/holdout split) and a reduced feature
# set (GOLD_FEATURE_COLUMNS_LONG_TERM -- see features.py) that doesn't need
# a 100/200-bar warmup it can't afford. Report the numbers as-is rather than
# smoothing over how little data backs them, and revisit (restore the full
# feature set and 5-fold validation) once more daily history accumulates.
GOLD_HORIZONS: dict[str, dict] = {
    # `.DEXAUTUSD` 15m history is shorter than 1h's (~4000 bars / ~42 days,
    # checked live 2026-08-21 -- a narrower retention window at this finer
    # granularity), but still comfortably covers the full feature set.
    "intraday": {
        "resolution": "15m", "limit": 4000, "horizon_bars": 4,  # ~1 hour ahead
        "feature_columns": GOLD_FEATURE_COLUMNS, "warmup_bars": GOLD_WARMUP_BARS, "n_splits": 5,
    },
    "short_term": {
        "resolution": "1h", "limit": 3000, "horizon_bars": 24,  # ~1 day ahead
        "feature_columns": GOLD_FEATURE_COLUMNS, "warmup_bars": GOLD_WARMUP_BARS, "n_splits": 5,
    },
    "long_term": {
        "resolution": "1d", "limit": 500, "horizon_bars": 5,  # ~1 week ahead
        "feature_columns": GOLD_FEATURE_COLUMNS_LONG_TERM, "warmup_bars": GOLD_WARMUP_BARS_LONG_TERM, "n_splits": 1,
    },
}

CONFIDENCE_THRESHOLD_MULTIPLE = 0.95

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 15,
    "learning_rate": 0.03,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "bagging_seed": 42,
    "feature_fraction_seed": 42,
}


@dataclass
class GoldFoldMetrics:
    fold: int
    directional_accuracy: float
    information_coefficient: float
    rmse: float
    n_test: int
    decidable_fraction: float = 0.0
    decidable_directional_accuracy: float = 0.0


@dataclass
class GoldTrainingReport:
    horizon: str
    resolution: str
    horizon_bars: int
    trained_at: str
    n_samples: int
    fold_metrics: list[GoldFoldMetrics] = field(default_factory=list)
    mean_directional_accuracy: float = 0.0
    mean_information_coefficient: float = 0.0
    mean_decidable_fraction: float = 0.0
    mean_decidable_directional_accuracy: float = 0.0
    prediction_std: float = 0.0
    model_path: str = ""


def build_gold_training_frame(
    feature_df: pd.DataFrame,
    horizon_bars: int,
    warmup_bars: int = GOLD_WARMUP_BARS,
    feature_columns: list[str] = GOLD_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    labels = make_forward_return_labels(feature_df, horizon_bars)
    df = feature_df.iloc[warmup_bars:].copy()
    labels = labels.iloc[warmup_bars:]

    valid = labels.notna() & df[feature_columns].notna().all(axis=1)
    X = df.loc[valid, feature_columns].reset_index(drop=True)
    y = labels.loc[valid].reset_index(drop=True)
    return X, y


def _evaluate_fold(
    model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, float, float, float, float, np.ndarray]:
    preds = model.predict(X_test)
    y_true = y_test.values
    directional_acc = float(np.mean(np.sign(preds) == np.sign(y_true)))
    ic, _ = spearmanr(preds, y_true)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))

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


def gold_walk_forward_validate(
    X: pd.DataFrame, y: pd.Series, horizon_bars: int, n_splits: int = 5
) -> tuple[list[GoldFoldMetrics], np.ndarray]:
    splits = purged_walk_forward_splits(len(X), n_splits=n_splits, embargo=horizon_bars)
    results = []
    oos_preds: list[np.ndarray] = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_set = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        model = lgb.train(LGBM_PARAMS, train_set, num_boost_round=200)
        acc, ic, rmse, decidable_frac, decidable_acc, preds = _evaluate_fold(model, X.iloc[test_idx], y.iloc[test_idx])
        results.append(GoldFoldMetrics(fold_i, acc, ic, rmse, len(test_idx), decidable_frac, decidable_acc))
        oos_preds.append(preds)
        logger.info(
            "Gold fold %d: dir_acc=%.3f IC=%.3f rmse=%.5f n=%d | decidable(%.0f%% of bars): dir_acc=%.3f",
            fold_i, acc, ic, rmse, len(test_idx), decidable_frac * 100, decidable_acc,
        )
    return results, (np.concatenate(oos_preds) if oos_preds else np.array([]))


def train_and_save_gold(
    feature_df: pd.DataFrame,
    horizon: str,
    horizon_bars: int,
    resolution: str,
    feature_columns: list[str] = GOLD_FEATURE_COLUMNS,
    warmup_bars: int = GOLD_WARMUP_BARS,
    n_splits: int = 5,
    model_version: str | None = None,
) -> GoldTrainingReport:
    """Trains gold's dedicated LightGBM regressor for one horizon
    ("short_term" or "long_term"), validated with the same purged walk-
    forward discipline used for the crypto models -- but a fully separate
    model/feature set/store directory, per explicit request that this be a
    genuinely different system, not the crypto strategy reapplied to a new
    symbol. Callers should pass the `feature_columns`/`warmup_bars`/
    `n_splits` from GOLD_HORIZONS[horizon] rather than relying on the
    (short-term-sized) defaults for every horizon -- see that dict's
    docstring for why long_term needs smaller values of all three.
    """
    X, y = build_gold_training_frame(
        feature_df, horizon_bars=horizon_bars, warmup_bars=warmup_bars, feature_columns=feature_columns
    )
    # 40, not the crypto pipeline's 200: gold's long-term/1d horizon has a
    # genuinely tiny total history (see GOLD_HORIZONS' docstring) -- this
    # floor is deliberately just enough to fit a single train/holdout split,
    # not a real statistical comfort threshold. n=40-60 is a real, honestly-
    # thin sample, not a bug.
    if len(X) < 40:
        raise ValueError(f"Not enough gold samples to train {horizon} ({len(X)}) -- need at least 40")

    fold_metrics, oos_preds = gold_walk_forward_validate(X, y, horizon_bars, n_splits=n_splits)

    full_set = lgb.Dataset(X, label=y)
    final_model = lgb.train(LGBM_PARAMS, full_set, num_boost_round=200)

    version = model_version or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    GOLD_MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = GOLD_MODEL_STORE_DIR / f"gold_{horizon}_{version}.txt"
    final_model.save_model(str(model_path))

    if len(oos_preds) > 1 and np.std(oos_preds) > 0:
        pred_std = float(np.std(oos_preds))
    else:
        pred_std = float(np.std(final_model.predict(X)))

    meta = {
        "horizon": horizon,
        "resolution": resolution,
        "horizon_bars": horizon_bars,
        "version": version,
        "feature_columns": feature_columns,
        "pred_std": pred_std,
        "confidence_threshold_multiple": CONFIDENCE_THRESHOLD_MULTIPLE,
    }
    meta_path = GOLD_MODEL_STORE_DIR / f"gold_{horizon}_{version}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    latest_pointer = GOLD_MODEL_STORE_DIR / f"gold_{horizon}_latest.json"
    latest_pointer.write_text(json.dumps({"version": version}))

    report = GoldTrainingReport(
        horizon=horizon,
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
        prediction_std=pred_std,
        model_path=str(model_path),
    )
    logger.info(
        "Trained gold %s model v%s: dir_acc=%.3f IC=%.3f (n=%d) | decidable(~%.0f%% of bars): dir_acc=%.3f",
        horizon, version, report.mean_directional_accuracy, report.mean_information_coefficient, len(X),
        report.mean_decidable_fraction * 100, report.mean_decidable_directional_accuracy,
    )

    report_dict = {
        "horizon": report.horizon,
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
    report_path = GOLD_MODEL_STORE_DIR / f"gold_{horizon}_{version}.report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))
    latest_report_pointer = GOLD_MODEL_STORE_DIR / f"gold_{horizon}_latest_report.json"
    latest_report_pointer.write_text(json.dumps(report_dict, indent=2))

    return report
