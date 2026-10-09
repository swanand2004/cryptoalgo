"""Tests for the live trend_ladder strategy and its book-level sizing.

The strategy itself is validated by backtest; what these lock down is the
handful of properties a backtest CANNOT catch, because a backtest sees the
whole frame at once and live never does -- causality, the fail-safe when data
is missing or stale, and the arithmetic that turns a book-level target into
the contract count the existing sizer understands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.risk.book_vol_target import (
    book_return_series,
    compute_book_scalar,
    ladder_size_multiplier,
)
from app.risk.position_sizing import select_leverage
from app.strategy.trend_ladder import TrendLadderStrategy, _z as _z_of


def _frame(n: int = 9000, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.004, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=np.int64) * 3600,
            "open": close, "high": high, "low": low, "close": close,
            "volume": np.ones(n),
        }
    )
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    # A realistic 8-hourly settlement pattern: nonzero on one bar in eight.
    df["funding_rate"] = np.where(df.index % 8 == 0, 0.0001, 0.0)
    return df


def test_signal_stays_inside_the_contract():
    sig = TrendLadderStrategy().generate_signal(_frame())
    assert sig.notna().all()
    assert sig.between(-1.0, 1.0).all()


def test_signal_is_causal():
    """The whole backtest rests on this: a bar's signal must not move when
    later bars arrive. If it does, every reported number is lookahead."""
    df = _frame()
    strategy = TrendLadderStrategy()
    full = strategy.generate_signal(df)
    truncated = strategy.generate_signal(df.iloc[:-500].copy())

    cut = len(df) - 500
    np.testing.assert_allclose(
        truncated.iloc[-200:].to_numpy(), full.iloc[cut - 200:cut].to_numpy(), atol=1e-12
    )


def test_an_uptrend_is_long_and_a_downtrend_is_short():
    strategy = TrendLadderStrategy()
    up = strategy.generate_signal(_frame(drift=0.0006)).iloc[-1]
    down = strategy.generate_signal(_frame(drift=-0.0006)).iloc[-1]
    assert up > 0.2, f"a sustained uptrend should read clearly long, got {up}"
    assert down < -0.2, f"a sustained downtrend should read clearly short, got {down}"


def test_conviction_increases_monotonically_with_trend_strength():
    """The continuous form is where the profit factor comes from: a marginal
    crossover must not get the same size as a wide, established trend.

    Tested as MONOTONICITY in drift rather than as |weak| < |strong|. Every
    frame here shares one seed, so the paths differ only by deterministic
    drift -- but a near-zero-drift random walk still trends somewhere by
    chance, and its |signal| can legitimately be large. What must hold is
    that raising the drift raises the signal, every step of the way.
    """
    strategy = TrendLadderStrategy()
    drifts = [-0.0008, -0.0004, -0.0001, 0.0001, 0.0004, 0.0008]
    signals = [float(strategy.generate_signal(_frame(drift=d)).iloc[-1]) for d in drifts]
    assert signals == sorted(signals), f"signal not monotonic in trend strength: {signals}"
    assert signals[0] < -0.2 and signals[-1] > 0.2


def test_sparse_funding_is_forward_filled_not_averaged_as_zero():
    """Funding settles 8-hourly on an hourly grid, so seven bars in eight are
    zero. Letting those into the z-score collapses it toward the mean and
    silently disables the carry tilt."""
    strategy = TrendLadderStrategy()
    df = _frame()
    rng = np.random.default_rng(11)

    # Realistic settlements: an 8-hourly rate that actually varies (a constant
    # rate has zero variance, so its z-score is undefined and the tilt is
    # correctly zero -- which is a different thing from being disabled).
    rates = np.abs(rng.normal(0.0001, 0.00005, len(df)))
    settles = df.index % 8 == 0
    baseline = df.copy()
    baseline["funding_rate"] = np.where(settles, rates, 0.0)

    # Same series, but the RECENT stretch is crowded long. Deliberately short
    # relative to the 2160-bar z-window: a z-score measures deviation from the
    # trailing norm, so funding elevated for 90 solid days is no longer
    # crowded, it is the new normal, and the tilt correctly fades.
    crowded = baseline.copy()
    tail = crowded.index >= len(crowded) - 300
    crowded.loc[tail & settles, "funding_rate"] = rates[tail & settles] * 8

    assert strategy._carry(crowded).iloc[-1] < strategy._carry(baseline).iloc[-1], (
        "crowded long funding should push the carry tilt short"
    )
    assert strategy._carry(crowded).iloc[-1] < 0

    # The mechanism this guards. Settlements really are sparse on an hourly
    # grid, and the un-filled series is mostly zeros that are not rates at all
    # -- they are "no settlement this bar". Feeding those to the z-score
    # measures the settlement SCHEDULE rather than the crowding.
    naive = crowded["funding_rate"]
    filled = naive.replace(0.0, np.nan).ffill()
    assert (naive == 0).mean() > 0.8, "sanity: settlements should be sparse here"
    assert (filled.dropna() == 0).sum() == 0, "forward-fill must leave no gaps"

    naive_tilt = float((-_z_of(naive)).clip(-1, 1).fillna(0.0).iloc[-1])
    assert naive_tilt != pytest.approx(float(strategy._carry(crowded).iloc[-1])), (
        "the zero-laden series must not produce the same tilt as the filled one"
    )


def test_missing_columns_degrade_to_flat_rather_than_raising():
    strategy = TrendLadderStrategy()
    assert (strategy.generate_signal(pd.DataFrame({"close": [1.0, 2.0]})) == 0).all()
    # No funding column at all: trend still works, carry contributes nothing.
    df = _frame().drop(columns=["funding_rate"])
    assert strategy.generate_signal(df).notna().all()


# ---------------------------------------------------------------- book sizing
def _two_symbol_book(n: int = 2000):
    frames = {"A": _frame(n, seed=1), "B": _frame(n, seed=2)}
    signals = {s: pd.Series(0.5, index=f.index) for s, f in frames.items()}
    return frames, signals


def test_book_scalar_is_none_when_history_is_too_short():
    """Fail-safe: the caller must be able to tell 'cannot size' apart from
    'size at 1.0'. An unscaled book is an arbitrary bet, not a cautious one."""
    frames, signals = _two_symbol_book(n=100)
    k, vol = compute_book_scalar(frames, signals, target_vol=0.3, max_book_leverage=8.0)
    assert k is None and vol is None


def test_book_scalar_hits_the_volatility_target():
    frames, signals = _two_symbol_book()
    k, vol = compute_book_scalar(frames, signals, target_vol=0.30, max_book_leverage=8.0)
    assert k is not None and vol > 0
    # Scaling the unit book by k should land on the target.
    assert k * vol == pytest.approx(0.30, rel=1e-9)


def test_book_scalar_respects_the_leverage_ceiling():
    """A collapse in the volatility estimate must not translate into unbounded
    leverage."""
    frames, signals = _two_symbol_book()
    k, _ = compute_book_scalar(frames, signals, target_vol=50.0, max_book_leverage=8.0)
    assert k == pytest.approx(8.0)


def test_no_forward_return_is_fabricated_for_the_newest_bar():
    """The newest closed bar has no NEXT bar yet, so its forward return does
    not exist. Inventing one -- which a naive pct_change would -- is the same
    class of lookahead as the trailing-stop bug fixed on 2026-08-31."""
    frames, signals = _two_symbol_book()
    shocked = {s: f.copy() for s, f in frames.items()}
    for f in shocked.values():
        f.loc[f.index[-1], "close"] = float(f["close"].iloc[-1]) * 1.35

    book = book_return_series(shocked, signals)
    assert book.iloc[-1] == 0.0, "the newest bar must contribute no forward return"


def test_a_completed_shock_immediately_reduces_size():
    """The flip side, and the behaviour we WANT: once a large move has
    actually completed, it belongs in the volatility estimate and must shrink
    the book on the very next decision. Recorded as a test so nobody later
    'fixes' it into ignoring fresh information."""
    frames, signals = _two_symbol_book()
    baseline, _ = compute_book_scalar(frames, signals, 0.30, 8.0)

    shocked = {s: f.copy() for s, f in frames.items()}
    for f in shocked.values():
        # A completed 35% bar: it has a NEXT bar, so its return is real.
        f.loc[f.index[-2], "close"] = float(f["close"].iloc[-2]) * 1.35
    after, _ = compute_book_scalar(shocked, signals, 0.30, 8.0)
    assert after < baseline * 0.75, (
        f"a completed shock should cut size, got {after:.3f} vs {baseline:.3f}"
    )


def test_book_return_series_aligns_on_time_not_row_position():
    """SOL's history starts three years after BTC's, so the legs are different
    lengths. Summing them positionally would pair 2020 SOL with 2017 BTC."""
    long_frame = _frame(3000, seed=1)
    short_frame = _frame(3000, seed=2)
    short_frame = short_frame.iloc[1000:].reset_index(drop=True)
    short_frame["time"] = long_frame["time"].iloc[1000:].to_numpy()

    frames = {"LONG": long_frame, "SHORT": short_frame}
    signals = {
        "LONG": pd.Series(1.0, index=long_frame.index),
        "SHORT": pd.Series(1.0, index=short_frame.index),
    }
    book = book_return_series(frames, signals)
    assert book.index.is_monotonic_increasing
    assert len(book) == len(long_frame)
    # Before the short leg exists, the book is the long leg alone (scaled by N).
    assert book.loc[long_frame["time"].iloc[5]] != 0


def test_size_multiplier_reproduces_the_intended_notional():
    """`ladder_size_multiplier` exists purely so the book target rides on top
    of fixed_margin_contracts instead of forking it. This is that contract."""
    settings = Settings()
    signal, scalar, weight = 0.6, 1.25, 0.30
    mult = ladder_size_multiplier(
        signal, scalar, weight,
        settings.max_margin_per_trade_pct, settings.min_leverage, settings.max_leverage,
    )
    leverage = select_leverage(abs(signal), settings.min_leverage, settings.max_leverage)
    # What fixed_margin_contracts will actually compute, as a fraction of equity.
    notional = settings.max_margin_per_trade_pct * mult * leverage
    assert notional == pytest.approx(abs(signal) * weight * scalar)


def test_size_multiplier_is_zero_for_a_flat_signal():
    settings = Settings()
    assert ladder_size_multiplier(
        0.0, 1.0, 0.25, settings.max_margin_per_trade_pct,
        settings.min_leverage, settings.max_leverage,
    ) == 0.0


def test_symbol_weights_are_inverse_volatility_and_sum_to_one():
    """Equal weighting is only equal in DOLLARS. The high-vol leg then supplies
    most of the book's actual risk, which is not what equal weight was for."""
    from app.risk.book_vol_target import symbol_weights

    calm = _frame(3000, seed=1)
    wild = _frame(3000, seed=2)
    wild["close"] = 100.0 * np.exp(np.cumsum(
        np.random.default_rng(2).normal(0, 0.016, 3000)))  # ~4x the vol

    w = symbol_weights({"CALM": calm, "WILD": wild})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["CALM"] > w["WILD"], f"calmer leg should get more dollars: {w}"


def test_symbol_weights_fall_back_to_equal_when_vol_is_unmeasurable():
    from app.risk.book_vol_target import symbol_weights

    short = _frame(50, seed=3)
    w = symbol_weights({"A": short, "B": _frame(50, seed=4)})
    assert sum(w.values()) == pytest.approx(1.0)
    # Neither leg is measurable, so neither may be silently dropped to zero.
    assert all(v > 0 for v in w.values())
