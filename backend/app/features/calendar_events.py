from __future__ import annotations

import datetime as dt

import numpy as np

_UNIX_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
WEEKLY_EXPIRY_HOUR_UTC = 8  # Deribit's (and most crypto options venues') standard expiry time
FRIDAY_WEEKDAY = 4


def hours_until_weekly_options_expiry(times_utc: np.ndarray) -> np.ndarray:
    """Hours remaining until the next Friday 08:00 UTC -- the standard
    weekly crypto options expiry time (Deribit and most other venues).
    A real, documented event ("max pain" gravitational pull toward a
    settlement strike, volatility compression approaching expiry) with
    genuinely MIXED recent evidence on reliability -- current research
    shows episodes where it dampens volatility as expected and recent
    ones where price sat far from the "predicted" magnet, so this is
    implemented to be TESTED, not assumed to help.

    A pure calendar calculation, not a live API call -- deterministic, no
    external dependency, no missing-data case, and no need to look this up
    per-symbol (SOLUSD has no Deribit options market of its own, but the
    weekly Friday cycle is a broad crypto-market-wide volatility pattern
    driven by BTC/ETH options, not something specific to having its own
    listed contracts).
    """
    times_arr = np.asarray(times_utc, dtype=float)
    out = np.empty(len(times_arr), dtype=float)
    for i, t in enumerate(times_arr):
        current = _UNIX_EPOCH + dt.timedelta(seconds=float(t))
        days_ahead = (FRIDAY_WEEKDAY - current.weekday()) % 7
        expiry = (current + dt.timedelta(days=days_ahead)).replace(
            hour=WEEKLY_EXPIRY_HOUR_UTC, minute=0, second=0, microsecond=0
        )
        if expiry <= current:
            expiry += dt.timedelta(days=7)
        out[i] = (expiry - current).total_seconds() / 3600.0
    return out


def weekly_expiry_proximity(times_utc: np.ndarray) -> np.ndarray:
    """hours_until_weekly_options_expiry rescaled to [0, 1], where 1.0 =
    right at expiry and 0.0 = right after the previous one (maximally far
    from the next) -- a more ML-friendly "how close are we" framing of
    the same underlying calendar fact.
    """
    hours = hours_until_weekly_options_expiry(times_utc)
    return 1.0 - (hours / 168.0)
