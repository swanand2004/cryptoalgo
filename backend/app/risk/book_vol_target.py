from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.risk.position_sizing import select_leverage

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 24 * 365

# 30 days of hourly bars. Long enough that the estimate is stable, short
# enough that it actually reacts when the regime changes.
VOL_LOOKBACK_BARS = 720
# 7 days: the fast leg of the optional vol brake (Settings.book_vol_fast_up).
FAST_VOL_LOOKBACK_BARS = 168


def symbol_weights(frames: dict[str, pd.DataFrame], lookback: int = VOL_LOOKBACK_BARS) -> dict[str, float]:
    """Risk-parity weights across the legs: each symbol's share is proportional
    to the INVERSE of its own trailing volatility, normalised to sum to 1.

    Equal weighting is only equal in dollars, not in risk -- SOL runs roughly
    1.6x BTC's volatility, so an equal-dollar book is really a SOL book with
    some BTC attached. Sizing by inverse vol makes each leg contribute about
    the same amount of risk, which is what "equal weight" was trying to
    achieve in the first place.

    Measured against equal weight on 2017-2026, 4 symbols, at matched 15% vol:
    TRAIN Sharpe 1.81 -> 1.87, VALIDATION 1.02 -> 0.98, TEST 0.89 -> 0.99.
    Better on the holdout, and -- unlike the funding tilt removed on the same
    day -- it introduces no fitted parameter that could be tuned to make it
    look good.

    Uses only trailing, shifted data. Falls back to equal weight for any leg
    without enough history to estimate volatility.
    """
    inv: dict[str, float] = {}
    for symbol, frame in frames.items():
        if frame.empty or len(frame) < lookback + 2:
            continue
        r = np.log(frame["close"].astype(float)).diff()
        # -2, not -1: the newest bar's own volatility contribution is only
        # complete once it has a successor, same convention as the scalar.
        vol = float(r.rolling(lookback).std().iloc[-2])
        if np.isfinite(vol) and vol > 0:
            inv[symbol] = 1.0 / vol

    if not inv:
        n = max(len(frames), 1)
        return {s: 1.0 / n for s in frames}

    total = sum(inv.values())
    weights = {s: inv.get(s, 0.0) / total for s in frames}
    missing = [s for s in frames if s not in inv]
    if missing:
        # A leg we cannot measure gets the average of those we can, rather
        # than zero -- zero would silently stop trading it.
        share = 1.0 / max(len(frames), 1)
        for s in missing:
            weights[s] = share
        total = sum(weights.values())
        weights = {s: w / total for s, w in weights.items()}
    return weights


def book_return_series(frames: dict[str, pd.DataFrame], signals: dict[str, pd.Series]) -> pd.Series:
    """The book's own historical return had it been held at unit scale.

    Legs set RELATIVE weights (signal / N); one scalar then sizes the whole
    book. Doing it the other way round -- vol-targeting each leg and then
    equal-weighting the legs -- is a real trap: three legs each at 15% combine
    to a book at roughly 5% because they are only ~0.5 correlated, so the book
    silently runs at a third of its intended risk and strategies that trade
    rarely look artificially calm.
    """
    n = len(frames)
    if n == 0:
        return pd.Series(dtype=float)

    # Must be the SAME weights the live sizing uses, or the scalar is derived
    # from a book that is not the one being traded.
    weights = symbol_weights(frames)

    legs: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        signal = signals.get(symbol)
        if signal is None or frame.empty:
            continue
        close = frame["close"].astype(float)
        # Return earned over the bar a position decided on the PREVIOUS bar is
        # held for, so the series cannot contain a return the signal had not
        # been set before.
        ret = close.pct_change().shift(-1).fillna(0.0)
        leg = (signal.reindex(frame.index).fillna(0.0) * weights.get(symbol, 1.0 / n)) * ret
        legs[symbol] = pd.Series(leg.to_numpy(), index=frame["time"].to_numpy())

    if not legs:
        return pd.Series(dtype=float)

    # Align on time, not on row position: SOL's history starts in 2020 and the
    # frames are therefore different lengths.
    return pd.DataFrame(legs).fillna(0.0).sum(axis=1).sort_index()


def compute_book_scalar(
    frames: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    target_vol: float,
    max_book_leverage: float,
    lookback_bars: int = VOL_LOOKBACK_BARS,
    fast_up: bool = False,
) -> tuple[float | None, float | None]:
    """Returns (scalar k, the realised annualised vol it was derived from).

    k is the single multiplier that takes the book from unit scale to
    `target_vol`. It reads the PREVIOUS bar's volatility estimate -- reading
    the current bar's is the same class of lookahead as the trailing-stop bug
    fixed on 2026-08-31, and it would size the book using the very move it is
    supposed to be sizing for.

    Returns (None, None) when there is not enough history to estimate
    volatility. The caller must treat that as "do not trade", never as
    "assume 1.0": an unscaled book at these signal magnitudes is not a
    conservative default, it is an arbitrary one.
    """
    book = book_return_series(frames, signals)
    if len(book) < lookback_bars + 2:
        return None, None

    realised = book.rolling(lookback_bars, min_periods=lookback_bars // 2).std() * np.sqrt(HOURS_PER_YEAR)
    # -2 rather than -1: the last row's return is unknowable while the bar is
    # still the newest one we have, so the last fully-formed estimate is the
    # one before it. This is the .shift(1) from the study, done positionally.
    if len(realised) < 2 or not np.isfinite(realised.iloc[-2]) or realised.iloc[-2] <= 0:
        return None, None

    vol = float(realised.iloc[-2])
    if fast_up:
        # Settings.book_vol_fast_up (trend-doctor F2 brake, default off): a 7-day
        # estimate alongside the 30-day one, and the HIGHER of the two sizes the
        # book, so a volatility spike cuts leverage within days instead of weeks.
        # max() means it can only ever lower the scalar; a NaN 168-bar value
        # (too little history) falls back to the 720-bar estimate alone.
        fast = book.rolling(FAST_VOL_LOOKBACK_BARS, min_periods=FAST_VOL_LOOKBACK_BARS).std() * np.sqrt(HOURS_PER_YEAR)
        fast_vol = float(fast.iloc[-2]) if len(fast) >= 2 else float("nan")
        if np.isfinite(fast_vol) and fast_vol > vol:
            vol = fast_vol
    return float(min(target_vol / vol, max_book_leverage)), vol


def ladder_size_multiplier(
    signal: float,
    book_scalar: float,
    weight: float,
    margin_per_trade_pct: float,
    min_leverage: float,
    max_leverage: float,
) -> float:
    """Translates a book-level target notional into the `size_multiplier` that
    `fixed_margin_contracts` already understands.

    That function sizes as:

        notional = equity * margin_per_trade_pct * size_multiplier * leverage(|signal|)

    and the book wants:

        notional = equity * |signal| * weight * book_scalar

    where `weight` is this symbol's risk-parity share (see symbol_weights);
    it was a flat 1/N until 2026-09-09. The multiplier is just the ratio of
    the two. Expressing it this way
    means the vol target rides on top of the EXISTING sizing path rather than
    forking it -- `fixed_margin_contracts`, `should_rebalance`, the margin
    ceilings and every risk check downstream keep working untouched, and there
    is only ever one place that turns a signal into contracts.
    """
    if book_scalar <= 0 or weight <= 0 or margin_per_trade_pct <= 0:
        return 0.0
    leverage = select_leverage(min(1.0, abs(signal)), min_leverage, max_leverage)
    if leverage <= 0:
        return 0.0
    target_fraction = abs(signal) * weight * book_scalar
    return float(max(0.0, target_fraction / (margin_per_trade_pct * leverage)))
