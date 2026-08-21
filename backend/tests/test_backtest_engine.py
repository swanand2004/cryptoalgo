import numpy as np
import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.backtest.report import build_report
from app.features.feature_pipeline import build_feature_frame
from app.strategy.ensemble import SignalEnsemble, WeightedStrategy
from app.strategy.trend_momentum import TrendMomentumStrategy


def _synthetic_trending_candles(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = 0.0006
    noise = rng.normal(0, 0.004, n)
    log_returns = drift + noise
    close = 50_000 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(10, 100, n)
    time = np.arange(n) * 3600

    return pd.DataFrame(
        {"time": time, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_backtest_runs_and_produces_sane_report():
    candles = _synthetic_trending_candles()
    features = build_feature_frame(candles)

    ensemble = SignalEnsemble([WeightedStrategy(TrendMomentumStrategy(), weight=1.0)])
    result = run_backtest(features, ensemble, BacktestConfig(starting_equity=10_000))
    report = build_report(result, resolution="1h")

    assert len(result.equity_curve) == len(features)
    assert result.equity_curve.iloc[0] == 10_000
    assert np.isfinite(report.sharpe)
    assert report.final_equity > 0
    assert report.max_drawdown_pct <= 0


def test_flat_signal_produces_no_trades():
    candles = _synthetic_trending_candles(n=200)
    features = build_feature_frame(candles)

    class ZeroStrategy(TrendMomentumStrategy):
        def generate_signal(self, df):  # noqa: D401
            return pd.Series(0.0, index=df.index)

    ensemble = SignalEnsemble([WeightedStrategy(ZeroStrategy(), weight=1.0)])
    result = run_backtest(features, ensemble)

    assert len(result.trades) == 0
    assert (result.equity_curve == 10_000.0).all()
