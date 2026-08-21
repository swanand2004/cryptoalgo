from __future__ import annotations

import numpy as np

from app.astro.natal import GENESIS_UTC, genesis_natal_chart
from app.astro.panchang import BENEFIC_PLANETS, MALEFIC_PLANETS, NAKSHATRA_LORDS

# Vimshottari Dasha: the CENTRAL predictive technique of Vedic (Jyotish)
# astrology -- arguably more load-bearing in real predictive practice than
# the transit/aspect mechanisms elsewhere in this codebase. Where Gochara
# (transit, see natal.py's natal_transit_score and the aspect functions in
# panchang.py) asks "where are the planets RIGHT NOW relative to the natal
# chart," Dasha asks a structurally different question: "which planet's
# significations are IN FORCE over this extended period of the chart's
# life," with transits then read AS UNFOLDING WITHIN that period's theme.
# Verified against multiple independent sources (see the session's research):
# a fixed 120-year cycle, fixed order, fixed per-planet lengths.
DASHA_YEARS: dict[str, int] = {
    "ketu": 7, "venus": 20, "sun": 6, "moon": 10, "mars": 7,
    "rahu": 18, "jupiter": 16, "saturn": 19, "mercury": 17,
}
DASHA_ORDER = ["ketu", "venus", "sun", "moon", "mars", "rahu", "jupiter", "saturn", "mercury"]
DASHA_TOTAL_YEARS = sum(DASHA_YEARS.values())  # 120, verified
assert DASHA_TOTAL_YEARS == 120

# Standard Vimshottari software convention: a "year" is a fixed 365.25-day
# solar year, not a sidereal/tropical/lunar year -- this is what makes Dasha
# lengths reproducible across sources rather than a moving target.
YEAR_SECONDS = 365.25 * 86400.0

_mahadasha_schedule_cache: dict[str, list[tuple[str, float, float]]] = {}


def _nakshatra_position(moon_sidereal_longitude: float) -> tuple[int, float]:
    """(nakshatra_index [0,26], fraction already traversed [0,1)) for a
    sidereal Moon longitude -- the fraction determines how much of the natal
    Dasha lord's period was already "used up" before genesis."""
    span = 360.0 / 27.0
    idx = int(moon_sidereal_longitude // span) % 27
    frac = (moon_sidereal_longitude % span) / span
    return idx, frac


def mahadasha_schedule(symbol: str, cycles: int = 3) -> list[tuple[str, float, float]]:
    """[(lord, start_unix, end_unix), ...] covering `cycles` full 120-year
    Vimshottari cycles from `symbol`'s genesis moment. Computed once and
    cached (genesis never changes); 3 cycles = 360 years is comfortably
    enough to cover any symbol's entire practical lifetime from either
    side of "now."

    The FIRST Mahadasha is a PARTIAL balance, not a full period: classical
    Vimshottari rule is that the natal Moon's nakshatra determines the
    starting lord, and only the REMAINING fraction of that lord's period is
    left to run, proportional to how far the Moon had already progressed
    through that nakshatra span at the moment of birth (here, genesis).
    """
    cached = _mahadasha_schedule_cache.get(symbol)
    if cached is not None:
        return cached
    if symbol not in GENESIS_UTC:
        return []
    chart = genesis_natal_chart(symbol)
    if not chart:
        return []

    genesis_unix = GENESIS_UTC[symbol].timestamp()
    idx, frac = _nakshatra_position(chart["moon"])
    start_lord = NAKSHATRA_LORDS[idx]
    start_pos = DASHA_ORDER.index(start_lord)

    schedule: list[tuple[str, float, float]] = []
    cursor = genesis_unix

    first_years = DASHA_YEARS[start_lord] * (1.0 - frac)
    end = cursor + first_years * YEAR_SECONDS
    schedule.append((start_lord, cursor, end))
    cursor = end

    total_lords = len(DASHA_ORDER) * cycles
    for i in range(1, total_lords):
        lord = DASHA_ORDER[(start_pos + i) % len(DASHA_ORDER)]
        end = cursor + DASHA_YEARS[lord] * YEAR_SECONDS
        schedule.append((lord, cursor, end))
        cursor = end

    _mahadasha_schedule_cache[symbol] = schedule
    return schedule


def antardasha_schedule(mahadasha_lord: str, maha_start: float, maha_end: float) -> list[tuple[str, float, float]]:
    """9 sub-periods (Antardasha/Bhukti) within one Mahadasha, in the SAME
    cyclic DASHA_ORDER but STARTING FROM the Mahadasha lord itself, each
    sized proportional to (mahadasha_years * antardasha_years / 120) --
    the standard Vimshottari proportional-division rule, which is exactly
    why every Antardasha set sums back to its parent Mahadasha's full span.
    """
    maha_years = DASHA_YEARS[mahadasha_lord]
    start_pos = DASHA_ORDER.index(mahadasha_lord)
    cursor = maha_start
    out: list[tuple[str, float, float]] = []
    for i in range(len(DASHA_ORDER)):
        sub_lord = DASHA_ORDER[(start_pos + i) % len(DASHA_ORDER)]
        sub_years = maha_years * DASHA_YEARS[sub_lord] / DASHA_TOTAL_YEARS
        end = cursor + sub_years * YEAR_SECONDS
        out.append((sub_lord, cursor, end))
        cursor = end
    # Floating-point drift over up to 20 years of accumulated sub-periods is
    # negligible for this purpose but the last boundary is snapped to the
    # parent's exact end anyway, so a query for the last instant of the
    # Mahadasha never falls just short of every Antardasha segment.
    if out:
        lord, start, _ = out[-1]
        out[-1] = (lord, start, maha_end)
    return out


def _lord_at(schedule: list[tuple[str, float, float]], t: float) -> str:
    if not schedule:
        return ""
    starts = np.array([s for _, s, _ in schedule])
    pos = int(np.searchsorted(starts, t, side="right")) - 1
    if pos < 0 or pos >= len(schedule):
        return ""
    lord, start, end = schedule[pos]
    return lord if start <= t < end else ""


def dasha_lords_at(symbol: str, times_utc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mahadasha_lords, antardasha_lords) at each timestamp, as arrays of
    lowercase planet-name strings ("" where unresolvable -- unknown symbol,
    or outside the precomputed schedule window)."""
    times_arr = np.asarray(times_utc, dtype=float)
    maha_schedule = mahadasha_schedule(symbol)
    if not maha_schedule:
        empty = np.array([""] * len(times_arr), dtype=object)
        return empty, empty

    maha_starts = np.array([s for _, s, _ in maha_schedule])
    maha_out = np.empty(len(times_arr), dtype=object)
    antar_out = np.empty(len(times_arr), dtype=object)
    # Only ~27 Mahadasha segments exist across 3 cycles no matter how many
    # bars ask -- cache each one's Antardasha breakdown the first time any
    # bar falls in it, rather than recomputing per bar.
    antardasha_cache: dict[int, list[tuple[str, float, float]]] = {}

    for i, t in enumerate(times_arr):
        pos = int(np.searchsorted(maha_starts, t, side="right")) - 1
        if pos < 0 or pos >= len(maha_schedule):
            maha_out[i] = ""
            antar_out[i] = ""
            continue
        lord, start, end = maha_schedule[pos]
        if not (start <= t < end):
            maha_out[i] = ""
            antar_out[i] = ""
            continue
        maha_out[i] = lord
        if pos not in antardasha_cache:
            antardasha_cache[pos] = antardasha_schedule(lord, start, end)
        antar_out[i] = _lord_at(antardasha_cache[pos], t)

    return maha_out, antar_out


def _benefic_malefic_bias(lords: np.ndarray) -> np.ndarray:
    out = np.zeros(len(lords), dtype=float)
    out[np.isin(lords, list(BENEFIC_PLANETS))] = 1.0
    out[np.isin(lords, list(MALEFIC_PLANETS))] = -1.0
    return out


def dasha_bias(symbol: str, times_utc: np.ndarray) -> np.ndarray:
    """Net Dasha-lord benefic/malefic tilt, in roughly [-1.5, 1.5]: the
    ANTARDASHA (sub-period) lord's classical nature at full weight, plus
    the MAHADASHA (major-period) lord's nature at half weight as slower
    background context.

    Antardasha is weighted primary deliberately: Mahadashas run 6-20 years,
    so for BTC/ETH/SOL's genesis dates a Mahadasha is essentially CONSTANT
    across any realistic backtest window (there's nothing for an ablation
    test to even measure in that component within one run) -- Antardasha
    periods run months to a few years and are the ones that actually
    transition within a normal backtest slice.

    Fails closed to all-zero for an unrecognized symbol, exactly like
    natal_transit_score elsewhere in this package.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    maha, antar = dasha_lords_at(symbol, times_arr)
    return _benefic_malefic_bias(antar) + 0.5 * _benefic_malefic_bias(maha)
