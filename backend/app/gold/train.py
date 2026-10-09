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

from app.gold.features import GOLD_FEATURE_COLUMNS_NO_MICRO  # noqa: F401
from app.gold.features import GOLD_FEATURE_COLUMNS, GOLD_WARMUP_BARS
from app.ml.dataset import make_forward_return_labels, purged_walk_forward_splits

logger = logging.getLogger(__name__)

GOLD_MODEL_STORE_DIR = Path(__file__).resolve().parent / "model_store"

# Two horizons, deliberately far apart -- "short term" and "long term" as
# genuinely different questions, not the same target reused at a different
# resolution (contrast with crypto's ML_RESOLUTIONS, which keeps one ~4h
# target constant across resolutions).
#
# Data source switched from Delta's `.DEXAUTUSD` to PAXG/USDT via Binance on
# 2026-08-25 (see app/gold/ingest.py's docstring for the full validation) --
# real available history is now ~127k 15m bars (since 2023-01-01), ~52k
# hourly bars, and ~2200 daily bars, vs the old source's ~127-150 days at
# every resolution. Only INTRADAY's accuracy improvement was statistically
# validated (holdout z=4.80); short_term/long_term switched for one
# consistent price source across the page, not because their accuracy
# improved -- it didn't, even with this much more data (z=0.68 and z=-0.22).
# The limits below intentionally sit inside what PAXG actually has, with
# headroom for it to keep growing daily.
#
# `limit` (training) vs `serving_limit` (live snapshot) are DELIBERATELY
# different scales -- a real bug caught live (2026-08-25): engine.py's
# per-tick snapshot used to load the same huge `limit` training uses, and
# rebuilding the full feature bank (in particular hurst_exponent's O(rows x
# window) rolling calculation) over 127k rows EVERY 3 minutes blocked the
# single asyncio event loop for 30+ minutes solid -- freezing the entire
# app, including crypto trading's own risk supervisors and HTTP routes, not
# just gold. `serving_limit` only needs enough rows for the LAST row's
# features to be fully warmed up (a few thousand, comfortably above
# `warmup_bars`), since live serving only ever reads the tail; the full
# `limit` is still used once a day by the retrain job, which now also runs
# off the main event loop (see scheduler._gold_retrain/asyncio.to_thread).
GOLD_HORIZONS: dict[str, dict] = {
    "intraday": {
        "resolution": "15m", "limit": 150_000, "serving_limit": 2_000, "horizon_bars": 4,  # ~1 hour ahead
        "feature_columns": GOLD_FEATURE_COLUMNS, "warmup_bars": GOLD_WARMUP_BARS, "n_splits": 5,
    },
    "short_term": {
        "resolution": "1h", "limit": 60_000, "serving_limit": 2_000, "horizon_bars": 24,  # ~1 day ahead
        "feature_columns": GOLD_FEATURE_COLUMNS, "warmup_bars": GOLD_WARMUP_BARS, "n_splits": 5,
    },
    # Formerly a reduced feature set/warmup/single-split (GOLD_FEATURE_
    # COLUMNS_LONG_TERM in features.py, still defined there for anyone
    # training on a genuinely thin sample) -- PAXG's ~2200 real daily bars
    # comfortably support the FULL feature set (210-bar warmup) and a real
    # 5-fold purged walk-forward, so this now matches short_term/intraday
    # exactly rather than needing a special case for thin data.
    "long_term": {
        # Micro columns excluded here and only here -- 1,981 daily rows cannot
        # carry 79 features, and the ablation measured a 3.3-point LOSS from
        # adding them (see GOLD_FEATURE_COLUMNS_NO_MICRO in features.py).
        "resolution": "1d", "limit": 3_000, "serving_limit": 500, "horizon_bars": 5,  # ~1 week ahead
        "feature_columns": GOLD_FEATURE_COLUMNS_NO_MICRO, "warmup_bars": GOLD_WARMUP_BARS, "n_splits": 5,
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
    ic, _ = spearmanr(preds, y_true)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))

    # FLAT bars are excluded from directional accuracy. PAXG prints a lot of
    # them -- 10.7% of 15m bars have an exactly-zero 4-bar forward return, and
    # 2.7% of hourly bars over 24 -- because it is a thinly traded token with
    # stale quotes. np.sign(0) is 0, so a flat bar scores as WRONG for a
    # prediction AND for its exact opposite simultaneously, which is not a
    # measure of anything. Counting them subtracted a near-constant ~10 points
    # from every intraday accuracy figure and made the model look like a coin
    # flip when it was not. There is also no direction to get right on a bar
    # that did not move, so they are not a loss being hidden -- they are
    # untradeable either way.
    moved = np.sign(y_true) != 0
    n_moved = int(moved.sum())
    directional_acc = (
        float(np.mean(np.sign(preds[moved]) == np.sign(y_true[moved]))) if n_moved else 0.0
    )

    fold_pred_std = np.std(preds)
    threshold = CONFIDENCE_THRESHOLD_MULTIPLE * fold_pred_std
    confident_mask = np.abs(preds) >= threshold
    decidable_fraction = float(np.mean(confident_mask))
    decidable_and_moved = confident_mask & moved
    decidable_acc = (
        float(np.mean(np.sign(preds[decidable_and_moved]) == np.sign(y_true[decidable_and_moved])))
        if decidable_and_moved.any()
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
    ("intraday", "short_term", or "long_term"), validated with the same
    purged walk-forward discipline used for the crypto models -- but a
    fully separate model/feature set/store directory, per explicit request
    that this be a genuinely different system, not the crypto strategy
    reapplied to a new symbol. Callers should pass the `feature_columns`/
    `warmup_bars`/`n_splits` from GOLD_HORIZONS[horizon] rather than relying
    on the defaults.
    """
    X, y = build_gold_training_frame(
        feature_df, horizon_bars=horizon_bars, warmup_bars=warmup_bars, feature_columns=feature_columns
    )
    # 40, not the crypto pipeline's 200: still supports training a REDUCED
    # feature set (GOLD_FEATURE_COLUMNS_LONG_TERM in features.py) on a
    # genuinely thin real sample if one is ever passed in directly -- not
    # what GOLD_HORIZONS itself uses live any more (see its docstring), but
    # a capability worth keeping for a data source that ever regresses.
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
    # The bar that actually matters. Gold drifts up, so 50% is the wrong
    # reference: a model only earns the right to pick sides if it beats simply
    # being long. Computed on the same moved-bars basis as directional
    # accuracy so the two are directly comparable.
    moved_labels = y[np.sign(y) != 0]
    long_only_baseline = float((moved_labels > 0).mean()) if len(moved_labels) else 0.5
    flat_bar_fraction = float(1.0 - len(moved_labels) / len(y)) if len(y) else 0.0
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
        "long_only_baseline": long_only_baseline,
        "flat_bar_fraction": flat_bar_fraction,
        "n_features": len(feature_columns),
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
