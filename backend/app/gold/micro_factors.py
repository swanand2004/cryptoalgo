from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.gold.macro_calendar import next_macro_event

# Intraday / microstructure factors for gold that can be computed from the
# data this project actually has.
#
# WHAT IS DELIBERATELY ABSENT, and why -- stating it rather than quietly
# shipping a weaker proxy under the same name:
#
#   order-flow imbalance (signed aggressive volume), queue and depth
#   imbalance, microprice deviation from mid -- ALL require a live order
#   book. Gold here is PAXG/USDT spot OHLCV via ccxt. There is no book, so
#   these are not approximated, they are simply not present. A "volume
#   imbalance" built from candle volume is not order-flow imbalance and
#   calling it that would be dishonest.
#
#   COMEX vs London OTC lead-lag -- genuinely the closest analogue to the
#   Delta-vs-Binance work, and genuinely not buildable: neither CME tick
#   data nor LBMA intraday quotes are freely available.
#
# TWO KINDS OF FACTOR, kept separate on purpose:
#
#   DIRECTIONAL factors score in [-1, 1] and vote on which way gold goes.
#   REGIME factors do NOT vote. They scale conviction. Proximity to an FOMC
#       release tells you liquidity is thin and the next print is a coin
#       flip -- it says nothing about direction, and forcing it into a
#       directional score would be inventing a signal out of a risk warning.
#
# Collapsing those two into one number is how a risk gate silently becomes a
# fake alpha signal, so they are returned separately.
#
# MEASURED, then RE-measured after finding a bug in the first measurement.
#
# The first pass reported vol_term_structure at 44.5-48.6% and gold's hourly
# up-rate at 45.3%, and concluded "below 50% everywhere, therefore no edge".
# Those numbers were wrong. PAXG is thinly traded and prints a lot of exactly
# flat bars -- 10.7% of 15m bars have a zero 4-bar forward return, 7.2% of
# hourly bars have a zero 1-bar return. np.sign(0) is 0, so a flat bar counted
# as WRONG for a signal and for its exact opposite at the same time, which
# subtracted a near-constant offset from every figure and quietly moved the
# coin-flip line from 50% down to ~46%. Comparing to 50% under those
# conditions was meaningless. Corrected, gold's up-rate is 51-52%, not 45.3%,
# and vol_term_structure is ~50%: flat, not negative.
#
# RE-MEASURED 2026-09-04 excluding flat bars, on 128,213 15m bars and 52,587
# hourly bars, with a three-way chronological split -- sign fixed on TRAIN,
# factors selected on VALIDATION, TEST read exactly once. Thirteen factors,
# including the estimators built for exactly this situation (Roll 1984
# serial-covariance spread, Corwin-Schultz 2012 high-low spread, Amihud 2002
# illiquidity, a Kyle-lambda impact proxy) -- the first pass had skipped that
# literature on the grounds that there was no order book, which was the wrong
# reason to skip it.
#
# The strongest factor, close_location (where in its own range the bar shut),
# decays exactly the way a lucky factor does:
#     15m, 4 bars ahead:   TRAIN 59.7%  ->  VALID 55.8%  ->  TEST 50.8%
# Every hand-built combination behaves the same: 53-55% on validation, 50-51%
# on test.
#
# And the number that settles it -- TEST accuracy against simply being long:
#     15m/4    combo 50.5%   long-only 51.5%
#     15m/24   combo 50.8%   long-only 52.5%
#     1h/24    combo 51.3%   long-only 55.1%
#     1h/72    combo 50.3%   long-only 56.2%
# Gold drifts up. Every hand-weighted micro combination is WORSE than holding,
# and the gap widens with horizon. 50% was never the right bar.
#
# NET RESULT: still no directional vote from a hand-weighted micro combination
# -- but now for a measured reason rather than a mistaken one. The factors
# themselves are not worthless: they are wired into the MODEL's feature bank
# (see features.py, micro_* columns) where a tree can look for conditional
# structure, which is a different question from whether a fixed linear
# combination of them predicts direction. Here they scale conviction only.
MICRO_DIRECTIONAL_FACTORS: tuple[str, ...] = ()

# Gold's trading day, in UTC. Asia is the physical-demand session (China,
# India); London is where OTC price discovery concentrates; COMEX is where
# futures volume concentrates.
SESSION_BOUNDS_UTC = {
    "asia": (0, 7),      # Shanghai/Mumbai active
    "london": (7, 13),   # LBMA / London OTC
    "comex": (13, 21),   # NY futures open through settle
    "off": (21, 24),     # thin handover into Asia
}

# LBMA Gold Price auctions: 10:30 and 15:00 London time. These are real,
# scheduled, concentrated liquidity events -- the auction genuinely sets a
# benchmark that contracts settle against.
#
# The a-priori sign is DELIBERATELY none. There was a well-documented
# pre-fix drift before the 2015 reform (and litigation about it), but the
# LBMA Gold Price replaced the old telephone fix with an electronic auction
# specifically to kill that. Assuming the old pattern still holds a decade
# later would be trading a folk memory, so proximity is treated as a
# LIQUIDITY EVENT (regime), not a direction.
LONDON_FIX_HOURS_UTC = (10, 15)  # ~GMT; ignores the BST/GMT shift by design (see docstring)
FIX_WINDOW_MINUTES = 45

# Realized-vol term structure: short window over long window. Above 1 means
# vol is expanding relative to its own recent baseline.
RV_SHORT_BARS = 24
RV_LONG_BARS = 168
VOL_Z_WINDOW = 720
FACTOR_Z_CAP = 2.5

# Within this many hours of a scheduled US release, conviction is cut: the
# book thins out ahead of the print and the reaction to it is not something
# a trend model has any edge on.
EVENT_BLACKOUT_HOURS = 4.0
EVENT_CONVICTION_FLOOR = 0.35


@dataclass
class MicroFactor:
    name: str
    kind: str  # "directional" | "regime"
    value: float | None
    score: float          # directional: [-1,1] vote. regime: 0.0 (never votes)
    conviction: float     # regime: [0,1] multiplier. directional: 1.0
    direction: str
    detail: str


def _z(series: pd.Series, window: int = VOL_Z_WINDOW) -> float | None:
    s = series.dropna()
    if len(s) < max(30, window // 6):
        return None
    tail = s.tail(window)
    sd = tail.std()
    if not sd or not np.isfinite(sd) or sd == 0:
        return None
    return float((s.iloc[-1] - tail.mean()) / sd)


def session_for(ts: dt.datetime) -> str:
    h = ts.astimezone(dt.timezone.utc).hour
    for name, (lo, hi) in SESSION_BOUNDS_UTC.items():
        if lo <= h < hi:
            return name
    return "off"


def minutes_to_nearest_fix(ts: dt.datetime) -> float:
    """Absolute minutes to the closer of the two LBMA auctions.

    Uses fixed UTC hours rather than tracking London's BST/GMT switch. That
    introduces up to an hour of error for half the year, which is acceptable
    ONLY because this feeds a regime flag with a 45-minute window, never a
    directional signal -- if it were voting on direction the DST handling
    would have to be exact.
    """
    ts = ts.astimezone(dt.timezone.utc)
    minutes_now = ts.hour * 60 + ts.minute
    return float(min(abs(minutes_now - h * 60) for h in LONDON_FIX_HOURS_UTC))


def compute_micro_factors(frame: pd.DataFrame, now: dt.datetime | None = None) -> list[MicroFactor]:
    """Intraday factors from a gold OHLCV frame (hourly or finer).

    `frame` needs [time, close]. Missing or short data yields neutral,
    non-voting factors rather than an exception -- gold prediction must keep
    working when a series is thin.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    out: list[MicroFactor] = []

    # ---- DIRECTIONAL: realized-vol term structure ------------------------
    # Gold is a safe haven, so vol expansion tends to come WITH a bid rather
    # than against it. Sign committed (+1) before measuring.
    if frame is not None and not frame.empty and "close" in frame.columns and len(frame) > RV_LONG_BARS:
        ret = np.log(frame["close"].astype(float)).diff()
        rv_short = ret.rolling(RV_SHORT_BARS).std()
        rv_long = ret.rolling(RV_LONG_BARS).std()
        ratio = rv_short / rv_long.replace(0, np.nan)
        z = _z(ratio)
        latest = float(ratio.iloc[-1]) if pd.notna(ratio.iloc[-1]) else None
        if z is None:
            out.append(MicroFactor("vol_term_structure", "directional", latest, 0.0, 1.0,
                                   "neutral", "vol term structure: insufficient history"))
        else:
            score = float(np.clip(z, -FACTOR_Z_CAP, FACTOR_Z_CAP) / FACTOR_Z_CAP)
            out.append(MicroFactor(
                "vol_term_structure", "directional", latest, round(score, 4), 1.0,
                "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral",
                f"{RV_SHORT_BARS}h/{RV_LONG_BARS}h realized-vol ratio {latest:.2f} (z {z:+.2f}) -- "
                f"vol expansion tends to come with a safe-haven bid"))
    else:
        out.append(MicroFactor("vol_term_structure", "directional", None, 0.0, 1.0,
                               "neutral", "vol term structure: no data"))

    # ---- REGIME: which session ------------------------------------------
    session = session_for(now)
    out.append(MicroFactor(
        "session", "regime", None, 0.0,
        # The thin handover into Asia is where spreads are worst; conviction
        # is trimmed there rather than a direction being guessed.
        0.6 if session == "off" else 1.0,
        session,
        f"{session} session -- liquidity and spread regimes shift at each handoff"))

    # ---- REGIME: London fix proximity -----------------------------------
    mins = minutes_to_nearest_fix(now)
    near_fix = mins <= FIX_WINDOW_MINUTES
    out.append(MicroFactor(
        "london_fix", "regime", round(mins, 1), 0.0,
        0.7 if near_fix else 1.0,
        "near auction" if near_fix else "away from auction",
        f"{mins:.0f} min from the nearer LBMA auction (10:30 / 15:00 London) -- "
        f"scheduled liquidity event, direction NOT assumed (the pre-2015 pre-fix "
        f"drift was specifically what the electronic auction reformed)"))

    # ---- REGIME: scheduled US release proximity --------------------------
    try:
        nxt = next_macro_event(now)
        event, hours = nxt.get("event"), nxt.get("hours_away")
    except Exception:
        event, hours = None, None
    if event and hours is not None and abs(hours) <= EVENT_BLACKOUT_HOURS:
        out.append(MicroFactor(
            "macro_event", "regime", round(float(hours), 2), 0.0, EVENT_CONVICTION_FLOOR,
            f"{event} in {hours:.1f}h",
            f"{event} within {EVENT_BLACKOUT_HOURS:.0f}h -- the book thins out ahead of the "
            f"print and the reaction is not something a trend model has an edge on"))
    else:
        detail = f"next US release {event} in {hours:.0f}h" if event and hours is not None else "no release imminent"
        out.append(MicroFactor("macro_event", "regime", round(float(hours), 2) if hours is not None else None,
                               0.0, 1.0, "clear", detail))

    return out


def directional_score(factors: list[MicroFactor]) -> float:
    """Mean vote of the directional factors that PASSED measurement.

    Currently empty (see MICRO_DIRECTIONAL_FACTORS): vol term structure was
    measured below 50% at every horizon with its committed sign, so it is
    computed and shown but does not vote. This returns 0.0 by design, not by
    oversight -- the micro side scales conviction, it does not pick sides.
    """
    votes = [f.score for f in factors
             if f.kind == "directional" and f.name in MICRO_DIRECTIONAL_FACTORS and f.score != 0.0]
    return float(np.mean(votes)) if votes else 0.0


def conviction_multiplier(factors: list[MicroFactor]) -> float:
    """Product of the regime conviction multipliers, in [0, 1].

    Multiplicative rather than averaged: two independent reasons to be
    cautious (thin session AND an imminent FOMC) should compound, not cancel
    to something middling.
    """
    mult = 1.0
    for f in factors:
        if f.kind == "regime":
            mult *= max(0.0, min(1.0, f.conviction))
    return float(mult)
