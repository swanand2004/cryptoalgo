from __future__ import annotations

import datetime as dt

import numpy as np

from app.astro.ephemeris import sidereal_longitude
from app.astro.panchang import _angular_separation, ASPECT_ORB_DEGREES, CHALLENGING_ASPECT_ANGLES, HARMONIOUS_ASPECT_ANGLES

# Each asset's real, documented "birth moment" -- the genesis/first block --
# the same reference point actual financial astrologers use for mundane
# ("first trade") charts of an instrument. Verified via research, not
# assumed: BTC's Genesis Block timestamp is embedded in the block itself;
# ETH's is Unix 1438269973 (block #0, per Etherscan); SOL's is its
# documented Mainnet Beta genesis.
GENESIS_UTC = {
    "BTCUSD": dt.datetime(2009, 1, 3, 18, 15, 5, tzinfo=dt.timezone.utc),
    "ETHUSD": dt.datetime(2015, 7, 30, 15, 26, 13, tzinfo=dt.timezone.utc),
    "SOLUSD": dt.datetime(2020, 3, 16, 14, 29, 0, tzinfo=dt.timezone.utc),
}

NATAL_BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "rahu", "ketu"]

# Transiting bodies used for transit-to-natal analysis: Jupiter and Saturn
# are the two slow ("generational-phase") planets financial-astrology
# sources specifically call out for marking multi-month/year turning
# points, plus Rahu (the lunar node -- eclipse-adjacent, also specifically
# named in the same sources) alongside them.
TRANSIT_BODIES = ["jupiter", "saturn", "rahu"]

_natal_chart_cache: dict[str, dict[str, float]] = {}


def genesis_natal_chart(symbol: str) -> dict[str, float]:
    """Sidereal longitude of all 9 Navagraha at `symbol`'s genesis moment --
    computed once and cached, since it's a fixed historical instant that
    never changes no matter how many times/bars ask for it.
    """
    cached = _natal_chart_cache.get(symbol)
    if cached is not None:
        return cached
    if symbol not in GENESIS_UTC:
        return {}

    t = [GENESIS_UTC[symbol].timestamp()]
    chart = {body: float(sidereal_longitude(body, t)[0]) for body in NATAL_BODIES}
    _natal_chart_cache[symbol] = chart
    return chart


def natal_transit_score(symbol: str, times_utc: np.ndarray) -> np.ndarray:
    """Net transit-to-natal aspect tilt: the slow-moving signal real
    financial astrology uses for multi-month/year phase calls (as opposed
    to the fast, daily Panchang gates elsewhere in graha.py) -- +1 for each
    of transiting Jupiter/Rahu in a tight harmonious aspect (trine/sextile)
    to the asset's OWN natal Sun or Moon, -1 for transiting Saturn in a
    tight challenging aspect (square/opposition) to the same. Returns all
    zeros for an unrecognized symbol (fails closed, not open) or when a
    natal chart can't be computed.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    chart = genesis_natal_chart(symbol)
    if not chart:
        return np.zeros(len(times_arr), dtype=float)

    score = np.zeros(len(times_arr), dtype=float)
    for natal_point in ("sun", "moon"):
        natal_lon = chart[natal_point]

        for planet in ("jupiter", "rahu"):
            transit_lon = sidereal_longitude(planet, times_arr)
            sep = _angular_separation(transit_lon, np.full_like(transit_lon, natal_lon))
            in_aspect = np.any([np.abs(sep - a) <= ASPECT_ORB_DEGREES for a in HARMONIOUS_ASPECT_ANGLES], axis=0)
            score += in_aspect.astype(float)

        transit_saturn = sidereal_longitude("saturn", times_arr)
        sep = _angular_separation(transit_saturn, np.full_like(transit_saturn, natal_lon))
        in_aspect = np.any([np.abs(sep - a) <= ASPECT_ORB_DEGREES for a in CHALLENGING_ASPECT_ANGLES], axis=0)
        score -= in_aspect.astype(float)

    return score
