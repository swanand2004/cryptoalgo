import pytest
import numpy as np
import pandas as pd

from app.backtest.engine import BacktestConfig, run_backtest
from app.risk.position_sizing import SizingInputs, fixed_margin_contracts
from app.strategy.regime import VolRegime
from app.trading_loop import _regime_size_multiplier


def _inputs(**kw):
    base = dict(equity_usd=10_000.0, signal=1.0, price=100.0, contract_value=1.0,
                margin_per_trade_pct=0.10, min_leverage=10.0, max_leverage=10.0)
    base.update(kw)
    return SizingInputs(**base)


def test_size_multiplier_defaults_to_a_no_op():
    """Every pre-existing caller must size exactly as before."""
    assert fixed_margin_contracts(_inputs()) == fixed_margin_contracts(_inputs(size_multiplier=1.0))


def test_half_multiplier_halves_the_position():
    full = fixed_margin_contracts(_inputs(size_multiplier=1.0))
    half = fixed_margin_contracts(_inputs(size_multiplier=0.5))
    assert half == full // 2
    assert half > 0


def test_multiplier_scales_margin_not_direction():
    """A short must stay a short -- scaling must never flip the sign."""
    short = fixed_margin_contracts(_inputs(signal=-1.0, size_multiplier=0.5))
    assert short < 0


def test_negative_multiplier_is_clamped_rather_than_inverting_the_trade():
    assert fixed_margin_contracts(_inputs(size_multiplier=-2.0)) == 0


def test_zero_multiplier_means_no_position():
    assert fixed_margin_contracts(_inputs(size_multiplier=0.0)) == 0


class _Settings:
    regime_sizing_enabled = True
    regime_adx_trend_threshold = 25.0
    regime_vol_lookback_bars = 200
    regime_low_vol_percentile = 0.25
    regime_high_vol_percentile = 0.75
    regime_high_vol_size_multiplier = 0.5
    regime_low_vol_size_multiplier = 1.0


class _Runtime:
    def __init__(self, settings=None):
        self.settings = settings or _Settings()


def _vol_frame(atr_pct):
    n = len(atr_pct)
    return pd.DataFrame({
        "adx_14": np.full(n, 30.0),
        "ema_9": np.full(n, 101.0),
        "ema_21": np.full(n, 100.0),
        "atr_pct": atr_pct,
    })


def test_live_loop_halves_size_in_a_high_volatility_regime():
    atr = np.concatenate([np.full(200, 0.005), np.full(20, 0.05)])
    mult, label = _regime_size_multiplier(_Runtime(), _vol_frame(atr))
    assert label == VolRegime.HIGH.value
    assert mult == 0.5


def test_live_loop_uses_normal_size_in_a_steady_market():
    mult, label = _regime_size_multiplier(_Runtime(), _vol_frame(np.full(250, 0.01)))
    assert label == VolRegime.NORMAL.value
    assert mult == 1.0


def test_regime_sizing_can_be_switched_off_in_config():
    settings = _Settings()
    settings.regime_sizing_enabled = False
    atr = np.concatenate([np.full(200, 0.005), np.full(20, 0.05)])
    mult, label = _regime_size_multiplier(_Runtime(settings), _vol_frame(atr))
    assert mult == 1.0 and label == "disabled"


def test_regime_failure_fails_open_at_normal_size():
    """A regime classifier is a risk refinement, not a safety gate -- if it
    throws, trading must continue at normal size rather than halting."""
    mult, label = _regime_size_multiplier(_Runtime(), "not a dataframe")
    assert mult == 1.0 and label == "error"


def _backtest_frame(n=400):
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, n)))
    atr = np.concatenate([np.full(n // 2, 0.5), np.full(n - n // 2, 5.0)])
    return pd.DataFrame({
        "time": 1_700_000_000 + np.arange(n) * 3600,
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 1.0, "atr_14": atr, "atr_pct": atr / close,
        "adx_14": np.full(n, 30.0), "ema_9": close + 1.0, "ema_21": close,
    })


class _AlwaysLong:
    deadband = 0.15
    exit_deadband = 0.06

    def warmup_bars(self):
        return 20

    def smoothed_signal(self, df):
        return pd.Series(1.0, index=df.index)


def test_backtest_regime_sizing_is_off_by_default_so_baselines_are_unchanged():
    df = _backtest_frame()
    off = run_backtest(df, _AlwaysLong(), BacktestConfig())
    explicit_off = run_backtest(df, _AlwaysLong(), BacktestConfig(regime_sizing_enabled=False))
    assert off.equity_curve.iloc[-1] == explicit_off.equity_curve.iloc[-1]


def test_backtest_regime_sizing_actually_reduces_exposure_when_enabled():
    df = _backtest_frame()
    off = run_backtest(df, _AlwaysLong(), BacktestConfig(regime_sizing_enabled=False))
    on = run_backtest(df, _AlwaysLong(), BacktestConfig(regime_sizing_enabled=True))

    # Compare during the VOLATILE half specifically. Peak exposure over the
    # whole run is identical by construction -- it happens in the calm first
    # half, where the multiplier is 1.0 in both runs. The reduction is
    # supposed to appear only once volatility actually spikes, so asserting
    # on the overall peak would test the wrong thing.
    half = len(df) // 2
    assert on.position.abs().iloc[half:].max() < off.position.abs().iloc[half:].max()
    # ...and the calm half must be untouched, or this is just a blanket
    # size cut wearing a regime label.
    assert on.position.abs().iloc[:half].max() == off.position.abs().iloc[:half].max()


def test_signal_thresholds_come_from_config_not_hardcoded_defaults(monkeypatch):
    """The line that decides whether anything trades at all must be
    configurable. It was a SignalEnsemble constructor default until
    2026-09-03, so it could not be changed without editing code."""
    import asyncio

    from app.config import Settings, get_settings
    import app.strategy.registry as registry

    custom = Settings(signal_deadband=0.07, signal_smoothing_span=5)
    monkeypatch.setattr(registry, "get_settings", lambda: custom)

    async def _fake_configs():
        return {}

    monkeypatch.setattr(registry, "load_strategy_configs", _fake_configs)
    ens = asyncio.run(registry.build_ensemble_for_symbol("BTCUSD"))
    assert ens.deadband == 0.07
    assert ens.smoothing_span == 5
    # exit line stays looser than entry, derived when not set explicitly
    assert ens.exit_deadband == pytest.approx(0.07 * 0.4)
    get_settings.cache_clear()
