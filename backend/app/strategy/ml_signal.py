from __future__ import annotations

import pandas as pd

from app.ml.predict import MLPredictor
from app.strategy.base import Strategy


class MLSignalStrategy(Strategy):
    """Wraps a trained LightGBM model as one ensemble member. Contributes 0
    (no opinion) until a model has actually been trained for this symbol, so
    it's safe to enable before the first training run completes.
    """

    name = "ml_signal"

    def __init__(self, symbol: str, resolution: str = "15m") -> None:
        self.symbol = symbol
        self.predictor = MLPredictor(symbol, resolution)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if not self.predictor.is_ready:
            return pd.Series(0.0, index=df.index)
        return self.predictor.predict_score_batch(df)

    def required_warmup_bars(self) -> int:
        return 60
