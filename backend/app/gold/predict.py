from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.gold.train import GOLD_MODEL_STORE_DIR

logger = logging.getLogger(__name__)


class GoldPredictor:
    """Loads one of gold's dedicated trained models ("short_term" or
    "long_term") and turns its raw predicted forward-return into a
    normalized [-1, 1] score using the `pred_std` captured at training time
    -- same confidence-gating design as the crypto MLPredictor
    (app/ml/predict.py), but reading gold's own model store/feature columns.
    """

    def __init__(self, horizon: str) -> None:
        self.horizon = horizon
        self.model: lgb.Booster | None = None
        self.meta: dict | None = None
        self._try_load()

    def _try_load(self) -> None:
        pointer_path = GOLD_MODEL_STORE_DIR / f"gold_{self.horizon}_latest.json"
        if not pointer_path.exists():
            logger.info("No trained gold model yet for horizon %s", self.horizon)
            return
        version = json.loads(pointer_path.read_text())["version"]
        model_path = GOLD_MODEL_STORE_DIR / f"gold_{self.horizon}_{version}.txt"
        meta_path = GOLD_MODEL_STORE_DIR / f"gold_{self.horizon}_{version}.meta.json"
        if not model_path.exists() or not meta_path.exists():
            logger.warning("Gold model pointer for %s references missing files", self.horizon)
            return
        self.model = lgb.Booster(model_file=str(model_path))
        self.meta = json.loads(meta_path.read_text())

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.meta is not None

    def reload(self) -> None:
        self._try_load()

    def predict_score_batch(self, feature_df: pd.DataFrame, require_confidence: bool = True) -> pd.Series:
        if not self.is_ready:
            return pd.Series(0.0, index=feature_df.index)

        cols = self.meta["feature_columns"]
        X = feature_df.reindex(columns=cols)
        valid_mask = X.notna().all(axis=1)

        raw = np.zeros(len(X))
        if valid_mask.any():
            raw[valid_mask.values] = self.model.predict(X.loc[valid_mask])

        pred_std = max(self.meta["pred_std"], 1e-9)
        score = raw / (2 * pred_std)
        score = np.clip(score, -1, 1)
        score[~valid_mask.values] = 0.0

        if require_confidence:
            confidence_multiple = self.meta.get("confidence_threshold_multiple", 0.0)
            below_threshold = np.abs(raw) < (confidence_multiple * pred_std)
            score[below_threshold] = 0.0

        return pd.Series(score, index=feature_df.index)

    def predict_raw_return_last(self, feature_df: pd.DataFrame) -> float | None:
        """The model's raw (un-normalized) predicted forward return for just
        the LAST row -- what a price TARGET actually needs (current_price *
        (1 + raw_return)), as opposed to predict_score_batch's [-1, 1]
        direction/conviction score. Returns None if not ready or the last
        row's features aren't valid.
        """
        if not self.is_ready:
            return None
        cols = self.meta["feature_columns"]
        X = feature_df.reindex(columns=cols).iloc[[-1]]
        if X.isna().any(axis=None):
            return None
        return float(self.model.predict(X)[0])
