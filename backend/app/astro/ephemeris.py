from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from skyfield.api import load

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EPHEMERIS_FILE = DATA_DIR / "de421.bsp"

# JPL DE421 -- real planetary/lunar ephemeris data (the same underlying
# astronomy used by professional astrology software and Swiss Ephemeris),
# not a stub. skyfield downloads this (~17MB) automatically to
# EPHEMERIS_FILE on first use if it isn't already present.
BODY_NAMES = {
    "sun": "sun",
    "moon": "moon",
    "mercury": "mercury barycenter",
    "venus": "venus barycenter",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
}

_ts = None
_eph = None
_earth = None

# Unix epoch (1970-01-01T00:00:00Z) expressed as a Julian date.
JD_UNIX_EPOCH = 2440587.5


def _load():
    global _ts, _eph, _earth
    if _eph is None:
        _ts = load.timescale()
        _eph = load(str(EPHEMERIS_FILE))
        _earth = _eph["earth"]
    return _ts, _eph, _earth


def _mean_lunar_node_longitude(times_utc: np.ndarray) -> np.ndarray:
    """Rahu -- the Moon's MEAN ascending node -- via the standard polynomial
    (Meeus, "Astronomical Algorithms", ch. 47). This is an orbital element
    (where the Moon's orbital plane currently crosses the ecliptic), not a
    physical body, so it doesn't go through the JPL ephemeris body lookup
    like every other planet here -- it's computed directly from Julian
    centuries since J2000.0.
    """
    jd = JD_UNIX_EPOCH + np.asarray(times_utc, dtype=float) / 86400.0
    T = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000.0
    omega = 125.04452 - 1934.136261 * T + 0.0020708 * T**2 + T**3 / 450000.0
    return omega % 360.0


def tropical_longitudes(body_key: str, times_utc: np.ndarray) -> np.ndarray:
    """Vectorized geocentric ecliptic (tropical) longitude in degrees, for
    an array of UTC unix-second timestamps. One call per body covers every
    row at once -- not looped per-row -- so this stays fast even across a
    400-bar backtest frame.

    Timestamps are converted straight to a Julian date via arithmetic (NOT
    through Python's `datetime.fromtimestamp`, which raises OSError on
    Windows for timestamps close to the 1970 epoch -- exactly what
    synthetic/backtest fixtures tend to use). Treating the JD as Terrestrial
    Time rather than correcting through UT1/UTC skips a ~69s offset, which
    shifts a fast-moving body like the Moon by roughly 0.0009 degrees --
    negligible next to the 12-degree-wide Tithi/Nakshatra buckets this feeds
    into.

    "rahu"/"ketu" (the lunar nodes -- two of the Navagraha, not physical
    bodies) are handled separately via the mean-node formula above; Ketu is
    always exactly opposite Rahu, by definition (the descending node).
    """
    if body_key == "rahu":
        return _mean_lunar_node_longitude(times_utc)
    if body_key == "ketu":
        return (_mean_lunar_node_longitude(times_utc) + 180.0) % 360.0

    ts, eph, earth = _load()
    body = eph[BODY_NAMES[body_key]]
    jd = JD_UNIX_EPOCH + np.asarray(times_utc, dtype=float) / 86400.0
    t = ts.tt_jd(jd)
    _, lon, _ = earth.at(t).observe(body).ecliptic_latlon()
    return lon.degrees


def heliocentric_longitude(body_key: str, times_utc: np.ndarray) -> np.ndarray:
    """Ecliptic longitude as seen FROM THE SUN, not from Earth -- the
    coordinate frame financial/mundane astrology sources specifically
    prefer for cycle work, because it reflects the planet's TRUE orbital
    motion rather than the geocentric view. Apparent retrograde motion is
    entirely a geocentric illusion (caused by Earth overtaking, or being
    overtaken by, another planet in its own orbit) -- a planet's
    heliocentric longitude increases (near-)monotonically, with no
    retrograde loops at all, which is exactly why this frame is described
    as giving "cleaner" cycles. Not meaningful for the Moon (Earth's own
    satellite, not a Sun-orbiting body in its own right) or Rahu/Ketu (not
    physical bodies) -- callers should only pass Mercury/Venus/Mars/
    Jupiter/Saturn (or the Sun, which reduces to Earth's own heliocentric
    longitude + 180 degrees, i.e. Earth's position in its orbit).
    """
    ts, eph, _ = _load()
    sun = eph["sun"]
    body = eph[BODY_NAMES[body_key]]
    jd = JD_UNIX_EPOCH + np.asarray(times_utc, dtype=float) / 86400.0
    t = ts.tt_jd(jd)
    _, lon, _ = sun.at(t).observe(body).ecliptic_latlon()
    return lon.degrees


def declination(body_key: str, times_utc: np.ndarray) -> np.ndarray:
    """Geocentric declination in degrees [-90, 90] -- angular distance
    north(+)/south(-) of the celestial equator. A genuinely different
    coordinate AXIS from ecliptic longitude (which is what every other
    function in this module uses): two planets can be far apart in
    longitude yet share the same declination ("parallel," treated like a
    conjunction in this second system) or sit at opposite declinations
    ("contra-parallel," treated like an opposition) -- a real, distinct
    aspect system in both Western and Vedic astrology, computed here via
    the equatorial (RA/Dec) frame rather than the ecliptic one.
    """
    if body_key == "rahu":
        return np.zeros(len(np.atleast_1d(times_utc)), dtype=float)
    if body_key == "ketu":
        return np.zeros(len(np.atleast_1d(times_utc)), dtype=float)
    ts, eph, earth = _load()
    body = eph[BODY_NAMES[body_key]]
    jd = JD_UNIX_EPOCH + np.asarray(times_utc, dtype=float) / 86400.0
    t = ts.tt_jd(jd)
    _, dec, _ = earth.at(t).observe(body).radec()
    return dec.degrees


def angular_velocity(body_key: str, times_utc: np.ndarray, lookback_seconds: float = 86400.0) -> np.ndarray:
    """Continuous signed geocentric longitude velocity, in degrees/day --
    the same underlying quantity is_retrograde already thresholds to a
    boolean (negative = retrograde), but the CONTINUOUS magnitude is real
    information a binary flag throws away: a planet just barely stationing
    (velocity near 0, about to turn retrograde or direct) is a
    meaningfully different state from one moving at its normal full speed,
    and a binary retrograde flag can't distinguish them.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    now_lon = tropical_longitudes(body_key, times_arr)
    past_lon = tropical_longitudes(body_key, times_arr - lookback_seconds)
    delta = ((now_lon - past_lon + 180.0) % 360.0) - 180.0
    days = lookback_seconds / 86400.0
    return delta / days


def lahiri_ayanamsa(times_utc: np.ndarray) -> np.ndarray:
    """Lahiri ayanamsa: the precession offset between the tropical
    (Western) and sidereal (Vedic) zodiacs, in degrees. Standard linear
    approximation -- ~23.85 deg at J2000.0 (2000-01-01 12:00 UTC),
    increasing by Earth's axial precession rate of ~50.29 arcsec/year. This
    is the conventional ayanamsa used by most Vedic Panchang calculations.
    """
    j2000 = dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc).timestamp()
    years_since_j2000 = (np.asarray(times_utc, dtype=float) - j2000) / (365.25 * 86400.0)
    return 23.85 + years_since_j2000 * (50.29 / 3600.0)


def sidereal_longitude(body_key: str, times_utc: np.ndarray) -> np.ndarray:
    """Sidereal (Vedic zodiac) longitude in degrees, [0, 360)."""
    tropical = tropical_longitudes(body_key, times_utc)
    ayanamsa = lahiri_ayanamsa(times_utc)
    return (tropical - ayanamsa) % 360.0


def is_retrograde(body_key: str, times_utc: np.ndarray, lookback_seconds: float = 86400.0) -> np.ndarray:
    """True where the body's tropical longitude has DECREASED over the
    lookback window (accounting for the 360->0 wraparound) -- apparent
    retrograde motion. The Sun and Moon never retrograde in this sense;
    calling this for them is meaningless (always False in practice) but not
    an error.
    """
    now_lon = tropical_longitudes(body_key, times_utc)
    past_lon = tropical_longitudes(body_key, np.asarray(times_utc, dtype=float) - lookback_seconds)
    delta = ((now_lon - past_lon + 180.0) % 360.0) - 180.0  # shortest signed angular delta
    return delta < 0
