from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.feature_pipeline import FEATURE_COLUMNS


def make_forward_return_labels(df: pd.DataFrame, horizon_bars: int = 15) -> pd.Series:
    """Label = forward return over `horizon_bars`. Uses `close` shifted into the
    future, so the LAST `horizon_bars` rows of any dataframe will have NaN
    labels -- callers must drop them before training (never impute a label).
    """
    future_close = df["close"].shift(-horizon_bars)
    return (future_close / df["close"]) - 1


def build_training_frame(
    feature_df: pd.DataFrame, horizon_bars: int = 15, warmup_bars: int = 60
) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) with warmup rows (indicator NaNs) and label-horizon tail
    rows (label NaNs) dropped -- the only two legitimate reasons to drop rows
    in a time series dataset; everything else is kept in original order.
    """
    labels = make_forward_return_labels(feature_df, horizon_bars)
    df = feature_df.iloc[warmup_bars:].copy()
    labels = labels.iloc[warmup_bars:]

    valid = labels.notna() & df[FEATURE_COLUMNS].notna().all(axis=1)
    X = df.loc[valid, FEATURE_COLUMNS].reset_index(drop=True)
    y = labels.loc[valid].reset_index(drop=True)
    return X, y


def purged_walk_forward_splits(
    n_samples: int, n_splits: int = 5, embargo: int = 15
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward CV with an embargo: the `embargo` samples
    immediately before each test block are dropped from that fold's training
    set. This is required (not optional) here because our label at row i
    depends on close[i + horizon_bars] -- without the embargo, training rows
    just before a test block would leak information about the test block's
    early labels.
    """
    fold_size = n_samples // (n_splits + 1)
    if fold_size < 1:
        return []

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_start = train_end
        test_end = min(fold_size * (i + 1), n_samples)
        train_idx = np.arange(0, max(0, train_end - embargo))
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < 20 or len(test_idx) < 5:
            continue
        splits.append((train_idx, test_idx))
    return splits
