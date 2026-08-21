from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.ml.train import MODEL_STORE_DIR

logger = logging.getLogger(__name__)


class MLPredictor:
    """Loads a trained per-symbol model and turns raw predicted forward-returns
    into a normalized [-1, 1] ensemble signal using the `pred_std` captured at
    training time (so the scaling reflects that model's own prediction spread,
    not an arbitrary constant).
    """

    def __init__(self, symbol: str, resolution: str = "15m") -> None:
        self.symbol = symbol
        self.resolution = resolution
        self.model: lgb.Booster | None = None
        self.meta: dict | None = None
        self._try_load()

    def _try_load(self) -> None:
        pointer_path = MODEL_STORE_DIR / f"{self.symbol}_{self.resolution}_latest.json"
        if not pointer_path.exists():
            logger.info("No trained model yet for %s %s", self.symbol, self.resolution)
            return
        version = json.loads(pointer_path.read_text())["version"]
        model_path = MODEL_STORE_DIR / f"{self.symbol}_{self.resolution}_{version}.txt"
        meta_path = MODEL_STORE_DIR / f"{self.symbol}_{self.resolution}_{version}.meta.json"
        if not model_path.exists() or not meta_path.exists():
            logger.warning("Model pointer for %s %s references missing files", self.symbol, self.resolution)
            return
        self.model = lgb.Booster(model_file=str(model_path))
        self.meta = json.loads(meta_path.read_text())

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.meta is not None

    def reload(self) -> None:
        self._try_load()

    def predict_score_batch(self, feature_df: pd.DataFrame, require_confidence: bool = True) -> pd.Series:
        """`require_confidence=True` (default): "decidable, not predictable"
        -- bars where |raw prediction| falls below
        confidence_threshold_multiple * pred_std are zeroed out (abstain)
        rather than forced into a directional call. Backtested via purged
        walk-forward AND a genuine holdout (see app/ml/train.py's
        CONFIDENCE_THRESHOLD_MULTIPLE docstring): restricting to the ~30%
        most confident bars held up as a real accuracy improvement on all 3
        symbols, unlike forcing every bar to guess. Pass False to get the
        raw always-call score (e.g. for offline analysis that wants every
        bar, not just the confident ones).
        """
        if not self.is_ready:
            return pd.Series(0.0, index=feature_df.index)

        cols = self.meta["feature_columns"]
        X = feature_df.reindex(columns=cols)
        valid_mask = X.notna().all(axis=1)

        raw = np.zeros(len(X))
        if valid_mask.any():
            raw[valid_mask.values] = self.model.predict(X.loc[valid_mask])

        pred_std = max(self.meta["pred_std"], 1e-9)
        score = raw / (2 * pred_std)  # ~95% of predictions land within [-1, 1]
        score = np.clip(score, -1, 1)
        score[~valid_mask.values] = 0.0

        if require_confidence:
            confidence_multiple = self.meta.get("confidence_threshold_multiple", 0.0)
            below_threshold = np.abs(raw) < (confidence_multiple * pred_std)
            score[below_threshold] = 0.0

        return pd.Series(score, index=feature_df.index)
