"""Settings.book_vol_fast_up (trend-doctor F2 brake) in compute_book_scalar.

Off (the default) must be byte-identical to the validated behaviour; on, the scalar is sized off
max(168-bar, 720-bar) realised vol, so it can only ever LOWER leverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.risk.book_vol_target import compute_book_scalar


def _book(seed: int, n: int = 3000, spike_last: int = 0, spike_mult: float = 4.0):
    rng = np.random.default_rng(seed)
    frames, signals = {}, {}
    for j, s in enumerate(("BTCUSD", "ETHUSD")):
        r = rng.normal(0, 0.006 * (1 + 0.3 * j), n)
        if spike_last:
            r[-spike_last:] *= spike_mult
        close = 100 * np.exp(np.cumsum(r))
        f = pd.DataFrame({"time": np.arange(n) * 3600, "close": close})
        frames[s] = f
        signals[s] = pd.Series(np.sign(np.sin(np.arange(n) / 200.0 + j)), index=f.index)
    return frames, signals


@pytest.mark.parametrize("seed", range(5))
def test_off_is_byte_identical_to_default(seed):
    frames, signals = _book(seed, spike_last=100 * (seed % 2))
    a = compute_book_scalar(frames, signals, target_vol=0.5, max_book_leverage=8.0)
    b = compute_book_scalar(frames, signals, target_vol=0.5, max_book_leverage=8.0, fast_up=False)
    assert a == b  # exact tuple equality, not approx


@pytest.mark.parametrize("seed", range(10))
def test_fast_up_never_raises_leverage(seed):
    frames, signals = _book(seed, spike_last=(seed % 3) * 80, spike_mult=0.3 if seed % 2 else 4.0)
    k0, v0 = compute_book_scalar(frames, signals, target_vol=0.5, max_book_leverage=8.0)
    k1, v1 = compute_book_scalar(frames, signals, target_vol=0.5, max_book_leverage=8.0, fast_up=True)
    assert k1 <= k0 and v1 >= v0


def test_fast_up_cuts_leverage_after_a_vol_spike():
    frames, signals = _book(1, spike_last=120, spike_mult=4.0)
    k0, v0 = compute_book_scalar(frames, signals, target_vol=0.5, max_book_leverage=8.0)
    k1, v1 = compute_book_scalar(frames, signals, target_vol=0.5, max_book_leverage=8.0, fast_up=True)
    assert k1 < 0.8 * k0
    # and it is the 168-bar estimate (as of the previous bar) that sized it
    from app.risk.book_vol_target import book_return_series

    book = book_return_series(frames, signals)
    fast = float(book.rolling(168, min_periods=168).std().iloc[-2] * np.sqrt(24 * 365))
    assert v1 == pytest.approx(fast)


def test_calm_recent_week_keeps_the_30_day_estimate():
    frames, signals = _book(2, spike_last=150, spike_mult=0.2)
    assert compute_book_scalar(frames, signals, 0.5, 8.0, fast_up=True) == compute_book_scalar(frames, signals, 0.5, 8.0)


def test_trading_loop_passes_the_setting():
    import inspect

    import app.trading_loop as tl

    src = inspect.getsource(tl)
    assert 'fast_up=bool(getattr(runtime.settings, "book_vol_fast_up", False))' in src
