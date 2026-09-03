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


def _always_long_ensemble() -> SignalEnsemble:
    class AlwaysLong(TrendMomentumStrategy):
        def generate_signal(self, df):  # noqa: D401
            return pd.Series(1.0, index=df.index)

    return SignalEnsemble([WeightedStrategy(AlwaysLong(), weight=1.0)])


def _flat_then_spike_candles(n: int = 120) -> pd.DataFrame:
    """Flat price to warm indicators up, then ONE bar with a big high and a
    low back near the open -- the exact shape that exposes intrabar
    lookahead in a trailing stop.
    """
    close = np.full(n, 50_000.0)
    high = close * 1.0005
    low = close * 0.9995
    open_ = close.copy()

    spike = n - 2
    high[spike] = 52_000.0  # big favourable extreme within the bar
    low[spike] = 49_900.0  # ...and it comes right back down within the SAME bar
    close[spike] = 50_000.0

    return pd.DataFrame(
        {
            "time": np.arange(n) * 3600,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 50.0),
        }
    )


def test_trailing_stop_cannot_use_the_same_bars_high_it_is_then_tested_against():
    """Regression test for a real lookahead bug found 2026-08-31: the engine
    trailed the stop up to THIS bar's high, then checked THIS bar's low
    against that freshly-raised stop -- i.e. it assumed the position caught
    the bar's exact extreme and exited a hair below it, harvesting each
    bar's intrabar range as free profit.

    Its signature was an ever-tighter trail looking monotonically better with
    no optimum (sum of returns across the 9 live strategy/symbol combos:
    548% at 1.0/0.8 -> 1187% at 0.5/0.4 -> 2298% at 0.2/0.15). With the
    ordering corrected, a single spike-and-retrace bar must NOT be able to
    produce a profitable exit near that bar's own high.
    """
    candles = _flat_then_spike_candles()
    features = build_feature_frame(candles)
    cfg = BacktestConfig(
        starting_equity=10_000,
        atr_stop_multiple=2.0,  # pinned: this test is about lookahead, not stop width
        atr_breakeven_trigger_multiple=0.2,  # deliberately hair-trigger
        atr_tightened_trail_multiple=0.15,
        slippage_bps=0.0,  # isolate the lookahead effect from slippage
    )

    result = run_backtest(features, _always_long_ensemble(), cfg)

    # Any exit that filled ABOVE the spike bar's open would only be reachable
    # by trailing off that same bar's high before testing its low.
    spike_open = float(features["open"].iloc[len(features) - 2])
    exits_above_open = [t for t in result.trades if t.realized_pnl != 0 and t.price > spike_open * 1.001]
    assert not exits_above_open, (
        f"exited at {[round(t.price, 2) for t in exits_above_open]} -- above the spike bar's own "
        f"open ({spike_open:.2f}), which requires seeing that bar's high before its low"
    )


def test_stop_exits_pay_slippage_and_fill_at_the_open_when_the_bar_gaps_through():
    """A stop is a MARKET order into an adverse move: it must not fill at
    exactly the stop level for free. Until 2026-08-31 it did, with no
    slippage and no gap handling -- which directly subsidised ever-tighter
    trailing (a tighter trail fires far more stops, each getting a perfect
    fill). Here the bar gaps hard THROUGH the stop, so the fill must be at
    the (much worse) open, not the untouched stop price.
    """
    n = 120
    close = np.full(n, 50_000.0)
    high = close * 1.0005
    low = close * 0.9995
    open_ = close.copy()

    gap = n - 2
    open_[gap] = 45_000.0  # gaps far below any plausible stop
    high[gap] = 45_100.0
    low[gap] = 44_000.0
    close[gap] = 44_500.0

    candles = pd.DataFrame(
        {
            "time": np.arange(n) * 3600,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 50.0),
        }
    )
    features = build_feature_frame(candles)
    cfg = BacktestConfig(starting_equity=10_000, slippage_bps=2.0)

    result = run_backtest(features, _always_long_ensemble(), cfg)

    stop_exits = [t for t in result.trades if t.realized_pnl < 0]
    assert stop_exits, "the gap should have stopped the long out"
    worst = min(stop_exits, key=lambda t: t.price)
    # Must fill at/below the gapped open (45,000) -- never back up at a stop
    # level the market never actually traded at on the way down.
    assert worst.price <= 45_000.0, f"filled at {worst.price:.2f}, above the gapped open of 45,000"


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
