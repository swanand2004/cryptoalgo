from __future__ import annotations

import datetime as dt

import numpy as np
from skyfield import almanac
from skyfield.api import wgs84

from app.astro.ephemeris import (
    _load,
    angular_velocity,
    declination,
    heliocentric_longitude,
    is_retrograde,
    sidereal_longitude,
    tropical_longitudes,
)

IST_OFFSET_SECONDS = 5.5 * 3600
SECONDS_PER_DAY = 86400

# Vedic Panchang is computed for a specific place -- New Delhi is the
# conventional reference point for "Indian Standard Time" astronomy.
NEW_DELHI = wgs84.latlon(28.6139, 77.2090)

NAKSHATRA_NAMES = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]

YOGA_NAMES = [
    "Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda",
    "Sukarma", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva",
    "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyana",
    "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla",
    "Brahma", "Indra", "Vaidhriti",
]

TITHI_NAMES = [
    "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi",
    "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi",
    "Trayodashi", "Chaturdashi", "Purnima/Amavasya",
]

# The 5-fold Tithi auspiciousness cycle -- verified, standard classification:
# each lunar day (1-15 within EITHER paksha) falls into one of 5 repeating
# categories based on (tithi_number_within_paksha - 1) % 5. Nanda ("joy") and
# Purna ("fulfillment") are the most auspicious for new beginnings; Bhadra
# and Jaya are good but more suited to routine/strategic activity; Rikta
# ("empty") is the one traditionally avoided for starting anything new --
# distinct from Vishti Karana (a half-Tithi construct) and from paksha
# (waxing/waning), a third, independent lunar-calendar signal.
TITHI_CLASS_NAMES = ["Nanda", "Bhadra", "Jaya", "Rikta", "Purna"]
TITHI_CLASS_BIAS = {"Nanda": 1.0, "Bhadra": 0.5, "Jaya": 0.5, "Rikta": -1.0, "Purna": 1.0}

# Traditional weekday -> Rahu Kaal segment table (1-indexed, of 8 equal
# day-length divisions between sunrise and sunset). Weekday convention here:
# Sunday=0 ... Saturday=6 (the standard Hindu calendar week start).
RAHU_KAAL_SEGMENT = {0: 8, 1: 2, 2: 7, 3: 5, 4: 6, 5: 4, 6: 3}

# Nakshatra ruling-planet (Vimshottari dasha lord) sequence: 9 planets x 3
# repeats = 27 nakshatras. Standard, verified ordering -- Ketu rules
# Ashwini/Magha/Mula, Venus rules Bharani/Purva Phalguni/Purva Ashadha, etc.
NAKSHATRA_LORDS = ["ketu", "venus", "sun", "moon", "mars", "rahu", "jupiter", "saturn", "mercury"] * 3

# Classical Vedic benefic/malefic (shubha/papa graha) classification -- used
# by financial-astrology sources the same way as classical Jyotish texts:
# Jupiter/Venus/Moon/(well-placed)Mercury are the natural benefics; Saturn/
# Mars/Rahu/Ketu/Sun are malefic (Sun only mildly so; treated the same here
# for simplicity). Moon is included here as a NAKSHATRA-LORD classification
# (it's one of the 3 natural benefics in classical Jyotish) -- separate from
# and in addition to the paksha (waxing/waning) tilt elsewhere, which judges
# the Moon's PHASE rather than its lordship.
BENEFIC_PLANETS = {"jupiter", "venus", "mercury", "moon"}
MALEFIC_PLANETS = {"saturn", "mars", "rahu", "ketu", "sun"}

# The 11 Karanas (half-Tithi periods, ~60 per lunar month): 7 "chara"
# (movable) repeat 8 times covering karana-index 1-56; the remaining 3
# "sthira" (fixed) each occur once, at the very end of the month. Karana 0
# (Kimstughna) is a special fixed karana at the very start of the month.
CHARA_KARANAS = ["Bava", "Balava", "Kaulava", "Taitila", "Garaja", "Vanija", "Vishti"]
STHIRA_KARANAS = ["Shakuni", "Chatushpada", "Naga"]
KIMSTUGHNA = "Kimstughna"

# The two most severe of the 9 traditionally inauspicious Nitya Yogas (out
# of 27) -- "calamity" and "lack of support," respectively; the other 7
# (Vishkambha, Atiganda, Shula, Ganda, Vyaghata, Vajra, Parigha) are
# moderate-weight avoidances, not full gates.
SEVERE_INAUSPICIOUS_YOGAS = {16, 26}  # Vyatipata, Vaidhriti
MODERATE_INAUSPICIOUS_YOGAS = {0, 5, 8, 9, 12, 14, 18}  # Vishkambha, Atiganda, Shula, Ganda, Vyaghata, Vajra, Parigha

# Combustion (Surya Asta) orb, in degrees of separation from the Sun, per
# classical source (Brihat Parasara Hora Shastra). Only these 6 planets can
# be combust -- Rahu/Ketu are shadow points, not physical bodies, and are
# exempt. Direct-motion orbs used uniformly here (Mercury/Venus have a
# tighter orb specifically when retrograde in BPHS, a refinement not applied
# here to keep the rule a single number per planet).
COMBUSTION_ORB_DEGREES = {
    "moon": 12.0,
    "mars": 17.0,
    "mercury": 14.0,
    "jupiter": 11.0,
    "venus": 10.0,
    "saturn": 15.0,
}

# Aspect angles (degrees) treated as "in aspect" within ASPECT_ORB of exact,
# and their classical harmonious/challenging classification.
ASPECT_ORB_DEGREES = 6.0
HARMONIOUS_ASPECT_ANGLES = (60.0, 120.0)  # sextile, trine
CHALLENGING_ASPECT_ANGLES = (90.0, 180.0)  # square, opposition

_sunrise_sunset_cache: dict[str, tuple[float, float]] = {}


def tithi_index(times_utc: np.ndarray) -> np.ndarray:
    """Tithi index [0, 29]. Each tithi spans 12 deg of Moon-minus-Sun
    sidereal angular separation (30 tithis per synodic month)."""
    moon = sidereal_longitude("moon", times_utc)
    sun = sidereal_longitude("sun", times_utc)
    diff = (moon - sun) % 360.0
    return np.floor(diff / 12.0).astype(int)


def is_waxing_moon(times_utc: np.ndarray) -> np.ndarray:
    """Shukla paksha (waxing, tithi 0-14) vs Krishna paksha (waning, 15-29)."""
    return tithi_index(times_utc) < 15


def tithi_class_bias(times_utc: np.ndarray) -> np.ndarray:
    """+1.0 (Nanda/Purna) / +0.5 (Bhadra/Jaya) / -1.0 (Rikta) -- the 5-fold
    Tithi auspiciousness cycle, independent of paksha (which judges
    waxing/waning) and Vishti Karana (a half-Tithi construct): the
    classification repeats identically within each paksha, so
    tithi_index % 5 gives the class directly (tithi_index 0-29 -> tithi
    NUMBER-within-paksha 1-15 -> class (number-1) % 5, and since 15 is a
    multiple of 5 this collapses to tithi_index % 5 exactly).
    """
    idx = tithi_index(times_utc) % 5
    names = np.array(TITHI_CLASS_NAMES)[idx]
    return np.array([TITHI_CLASS_BIAS[n] for n in names], dtype=float)


def nakshatra_index(times_utc: np.ndarray) -> np.ndarray:
    """Nakshatra index [0, 26], the Moon's sidereal longitude divided into
    27 equal spans of 13deg20'."""
    moon = sidereal_longitude("moon", times_utc)
    return np.floor(moon / (360.0 / 27.0)).astype(int)


def yoga_index(times_utc: np.ndarray) -> np.ndarray:
    """Yoga index [0, 26]: (Sun + Moon sidereal longitude) divided into 27
    equal spans."""
    moon = sidereal_longitude("moon", times_utc)
    sun = sidereal_longitude("sun", times_utc)
    total = (moon + sun) % 360.0
    return np.floor(total / (360.0 / 27.0)).astype(int)


def nakshatra_lord(times_utc: np.ndarray) -> np.ndarray:
    """Which of the 9 Navagraha rules the Moon's CURRENT nakshatra (its
    Vimshottari dasha lord), as an array of lowercase planet-name strings."""
    idx = nakshatra_index(times_utc)
    return np.array(NAKSHATRA_LORDS)[idx]


def nakshatra_lord_bias(times_utc: np.ndarray) -> np.ndarray:
    """+1.0 where the Moon's current nakshatra is ruled by a classical
    benefic, -1.0 where ruled by a malefic. Every one of the 9 Navagraha
    classifies as one or the other here, so this never returns 0 in
    practice -- guarded anyway for safety."""
    lords = nakshatra_lord(times_utc)
    bias = np.zeros(len(lords), dtype=float)
    bias[np.isin(lords, list(BENEFIC_PLANETS))] = 1.0
    bias[np.isin(lords, list(MALEFIC_PLANETS))] = -1.0
    return bias


def karana_index(times_utc: np.ndarray) -> np.ndarray:
    """Karana index [0, 59]: half-Tithi, i.e. Moon-Sun sidereal angular
    separation divided into 60 equal 6-degree spans."""
    moon = sidereal_longitude("moon", times_utc)
    sun = sidereal_longitude("sun", times_utc)
    diff = (moon - sun) % 360.0
    return np.floor(diff / 6.0).astype(int)


def karana_name(idx: int) -> str:
    if idx == 0:
        return KIMSTUGHNA
    if 1 <= idx <= 56:
        return CHARA_KARANAS[(idx - 1) % 7]
    return STHIRA_KARANAS[idx - 57]  # 57/58/59 -> Shakuni/Chatushpada/Naga


def is_vishti_karana(times_utc: np.ndarray) -> np.ndarray:
    """Vishti (a.k.a. Bhadra) -- the most feared of the 11 Karanas,
    traditionally avoided for auspicious beginnings. It's the 7th of the 7
    repeating "chara" karanas, so it recurs roughly every ~3-4 days for
    ~11-12 hours (unlike Rahu Kaal, which is a fixed ~90min solar-time
    window every single day) -- a genuinely different mechanism from Rahu
    Kaal (lunar-solar angular Panchang vs. solar clock time), so the two do
    NOT always coincide."""
    idx = karana_index(times_utc)
    is_chara_range = (idx >= 1) & (idx <= 56)
    return is_chara_range & (((idx - 1) % 7) == 6)


def yoga_severity(times_utc: np.ndarray) -> np.ndarray:
    """0 = normal, 1 = moderate inauspicious Yoga, 2 = severe (Vyatipata/
    Vaidhriti) -- see SEVERE_INAUSPICIOUS_YOGAS / MODERATE_INAUSPICIOUS_YOGAS."""
    idx = yoga_index(times_utc)
    severity = np.zeros(len(np.atleast_1d(idx)), dtype=int)
    severity[np.isin(idx, list(MODERATE_INAUSPICIOUS_YOGAS))] = 1
    severity[np.isin(idx, list(SEVERE_INAUSPICIOUS_YOGAS))] = 2
    return severity


def _angular_separation(lon_a: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Shortest angular separation between two longitudes, in [0, 180]."""
    diff = np.abs(lon_a - lon_b) % 360.0
    return np.minimum(diff, 360.0 - diff)


def benefic_malefic_aspect_score(times_utc: np.ndarray) -> np.ndarray:
    """Net planetary-aspect (drishti-adjacent) tilt relative to the Moon --
    the fastest-moving, most heavily emphasized body in Panchang-based
    timing: +1 for each of Jupiter/Venus currently in a tight harmonious
    aspect (trine/sextile, within ASPECT_ORB_DEGREES of exact), -1 for each
    of Saturn/Mars in a tight challenging aspect (square/opposition).
    Aspects are angular DIFFERENCES between two bodies at the same instant,
    so the tropical/sidereal ayanamsa offset cancels out identically --
    tropical longitudes are used directly, no ayanamsa correction needed.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    moon = tropical_longitudes("moon", times_arr)
    score = np.zeros(len(times_arr), dtype=float)

    for planet in ("jupiter", "venus"):
        sep = _angular_separation(moon, tropical_longitudes(planet, times_arr))
        in_aspect = np.any([np.abs(sep - a) <= ASPECT_ORB_DEGREES for a in HARMONIOUS_ASPECT_ANGLES], axis=0)
        score += in_aspect.astype(float)

    for planet in ("saturn", "mars"):
        sep = _angular_separation(moon, tropical_longitudes(planet, times_arr))
        in_aspect = np.any([np.abs(sep - a) <= ASPECT_ORB_DEGREES for a in CHALLENGING_ASPECT_ANGLES], axis=0)
        score -= in_aspect.astype(float)

    return score


def malefic_retrograde_count(times_utc: np.ndarray) -> np.ndarray:
    """How many of {Mars, Saturn} are currently apparent-retrograde. (Mercury
    retrograde is handled separately, as its own well-known effect; Rahu/Ketu
    are excluded since the mean lunar node always regresses through the
    zodiac by definition -- "retrograde" isn't a meaningful on/off signal for
    it. Jupiter/Venus retrograde are left out too: classical interpretation
    of their retrograde is far more mixed/contested than Mars/Saturn's, so
    including them would be reaching past what's well-established.)
    """
    times_arr = np.asarray(times_utc, dtype=float)
    count = np.zeros(len(times_arr), dtype=int)
    for planet in ("mars", "saturn"):
        count += is_retrograde(planet, times_arr).astype(int)
    return count


def combust_planet_count(times_utc: np.ndarray) -> np.ndarray:
    """How many of {Mercury, Venus, Mars, Jupiter, Saturn} are currently
    "combust" -- within their classical orb (BPHS) of angular separation
    from the Sun, where a planet's own significations are considered dimmed/
    weakened. Moon is excluded here even though it has a listed orb (12deg):
    Moon-Sun proximity is already exactly what Tithi/paksha measures (a New
    Moon IS Moon-Sun conjunction), so counting it again here would double up
    an effect the paksha tilt already captures.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    sun = tropical_longitudes("sun", times_arr)
    count = np.zeros(len(times_arr), dtype=int)
    for planet in ("mercury", "venus", "mars", "jupiter", "saturn"):
        sep = _angular_separation(sun, tropical_longitudes(planet, times_arr))
        count += (sep <= COMBUSTION_ORB_DEGREES[planet]).astype(int)
    return count


def vara_index(times_utc: np.ndarray) -> np.ndarray:
    """Weekday in IST, Sunday=0 ... Saturday=6."""
    ist_seconds = np.asarray(times_utc, dtype=float) + IST_OFFSET_SECONDS
    days_since_epoch = np.floor(ist_seconds / SECONDS_PER_DAY).astype(int)
    # 1970-01-01 was a Thursday (Python weekday()==3, Mon=0). Hindu
    # convention wants Sunday=0, so Thursday=4 under that convention.
    return (days_since_epoch + 4) % 7


_UNIX_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _ist_date_for(time_utc: float) -> dt.date:
    # Pure timedelta arithmetic, NOT datetime.fromtimestamp -- the latter
    # raises OSError on Windows for timestamps close to the 1970 epoch
    # (exactly what synthetic/backtest fixtures tend to use).
    ist_dt = _UNIX_EPOCH + dt.timedelta(seconds=time_utc) + dt.timedelta(seconds=IST_OFFSET_SECONDS)
    return ist_dt.date()


def _sunrise_sunset_utc(date_ist: dt.date) -> tuple[float, float]:
    """Real astronomical sunrise/sunset (unix seconds, UTC) for New Delhi on
    the given IST calendar date -- via skyfield's discrete-event almanac
    search, not a fixed clock-time guess. Cached per calendar date since the
    search itself is comparatively expensive and a date's sunrise/sunset
    only needs computing once no matter how many intraday bars ask for it.
    """
    key = date_ist.isoformat()
    cached = _sunrise_sunset_cache.get(key)
    if cached is not None:
        return cached

    ts, eph, _ = _load()
    midnight_ist_utc = dt.datetime(date_ist.year, date_ist.month, date_ist.day, tzinfo=dt.timezone.utc) - dt.timedelta(
        seconds=IST_OFFSET_SECONDS
    )
    t0 = ts.from_datetime(midnight_ist_utc)
    t1 = ts.from_datetime(midnight_ist_utc + dt.timedelta(days=1))

    f = almanac.sunrise_sunset(eph, NEW_DELHI)
    times, events = almanac.find_discrete(t0, t1, f)

    sunrise = None
    sunset = None
    for ti, ei in zip(times, events):
        ts_unix = ti.utc_datetime().timestamp()
        if ei == 1 and sunrise is None:
            sunrise = ts_unix
        elif ei == 0 and sunset is None:
            sunset = ts_unix

    if sunrise is None or sunset is None:
        # Polar-night-style edge case shouldn't occur at Delhi's latitude,
        # but fail to a safe fixed approximation rather than raising.
        sunrise = midnight_ist_utc.timestamp() + 6 * 3600
        sunset = midnight_ist_utc.timestamp() + 18 * 3600

    result = (sunrise, sunset)
    _sunrise_sunset_cache[key] = result
    return result


def rahu_kaal_window_utc(date_ist: dt.date) -> tuple[float, float]:
    """(start, end) unix seconds (UTC) of Rahu Kaal on the given IST date:
    one of 8 equal divisions of that day's real sunrise-to-sunset span, the
    specific division picked by the traditional weekday table."""
    sunrise, sunset = _sunrise_sunset_utc(date_ist)
    hindu_weekday = (date_ist.weekday() + 1) % 7  # Python Mon=0 -> Hindu Sun=0
    segment = RAHU_KAAL_SEGMENT[hindu_weekday]
    part = (sunset - sunrise) / 8.0
    start = sunrise + (segment - 1) * part
    end = sunrise + segment * part
    return start, end


def is_rahu_kaal(times_utc: np.ndarray) -> np.ndarray:
    """Vectorized (via per-unique-date caching, not a per-bar astronomy
    query) Rahu Kaal membership mask."""
    times_arr = np.asarray(times_utc, dtype=float)
    result = np.zeros(len(times_arr), dtype=bool)
    for idx, t in enumerate(times_arr):
        date_ist = _ist_date_for(t)
        start, end = rahu_kaal_window_utc(date_ist)
        result[idx] = start <= t < end
    return result


# Hora (planetary hour): the 7 classical "physical" grahas (Rahu/Ketu are
# shadow points -- nodes, not bodies -- and traditionally have no Hora of
# their own) cycling in the fixed "Chaldean order," named for the
# descending-apparent-orbital-period sequence Hellenistic and Vedic
# planetary-hour astrology alike are built on: Saturn (slowest) down to the
# Moon (fastest). Verified, standard, uncontested convention -- every
# tradition's weekday-planet table (Sunday/Sun, Monday/Moon, ...) is a
# DERIVED consequence of this fixed 7-hour cycle landing back on the same
# planet exactly 24 hours (24 mod 7 == 3 steps) after the previous day's
# first hour, not an independent rule.
HORA_CHALDEAN_ORDER = ["saturn", "jupiter", "mars", "sun", "venus", "mercury", "moon"]

# Hindu weekday (Sunday=0) -> that day's own ruling planet, i.e. the lord of
# its very first daytime Hora (the one starting exactly at sunrise).
WEEKDAY_LORD = {0: "sun", 1: "moon", 2: "mars", 3: "mercury", 4: "jupiter", 5: "venus", 6: "saturn"}


def hora_lord_at(time_utc: float) -> str:
    """Which of the 7 classical grahas rules the Hora (planetary hour)
    containing this instant. Uses New Delhi's REAL sunrise/sunset (same
    reference as Rahu Kaal) split into 12 unequal daytime Horas and 12
    unequal nighttime Horas -- the genuine classical construction, not the
    common simplified "60-minute equal hour" shortcut.

    The sunrise-to-sunrise cycle containing `time_utc` (not necessarily the
    same calendar date -- a pre-dawn instant belongs to YESTERDAY's cycle,
    still running through its night half) determines both the day-lord
    anchor and the Hora count; getting this anchor wrong for pre-dawn
    instants would jump the sequence to the wrong weekday a few hours early.
    """
    date_ist = _ist_date_for(time_utc)
    sunrise, _ = _sunrise_sunset_utc(date_ist)

    if time_utc < sunrise:
        anchor_date = date_ist - dt.timedelta(days=1)
    else:
        anchor_date = date_ist
    anchor_sunrise, anchor_sunset = _sunrise_sunset_utc(anchor_date)

    if anchor_sunrise <= time_utc < anchor_sunset:
        hora_length = (anchor_sunset - anchor_sunrise) / 12.0
        hora_number = int((time_utc - anchor_sunrise) / hora_length)
    else:
        next_sunrise, _ = _sunrise_sunset_utc(anchor_date + dt.timedelta(days=1))
        hora_length = (next_sunrise - anchor_sunset) / 12.0
        hora_number = 12 + int((time_utc - anchor_sunset) / hora_length)
    hora_number = min(max(hora_number, 0), 23)

    hindu_weekday = (anchor_date.weekday() + 1) % 7
    day_lord = WEEKDAY_LORD[hindu_weekday]
    start_pos = HORA_CHALDEAN_ORDER.index(day_lord)
    return HORA_CHALDEAN_ORDER[(start_pos + hora_number) % 7]


def hora_bias(times_utc: np.ndarray) -> np.ndarray:
    """+1 where the current Hora's lord is a classical benefic, -1 where
    malefic -- same BENEFIC_PLANETS/MALEFIC_PLANETS classification used
    throughout this module, applied to a genuinely different clock: hourly-
    resolution planetary rulership, vs. Rahu Kaal's single ~90min daily
    gate. Deliberately added because it's matched to this system's actual
    1h/4h trading resolutions, where a day-level-only Panchang signal
    cannot move within a single bar.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    lords = np.array([hora_lord_at(t) for t in times_arr])
    bias = np.zeros(len(lords), dtype=float)
    bias[np.isin(lords, list(BENEFIC_PLANETS))] = 1.0
    bias[np.isin(lords, list(MALEFIC_PLANETS))] = -1.0
    return bias


def vara_lord_bias(times_utc: np.ndarray) -> np.ndarray:
    """+1/-1 benefic/malefic tilt from the WEEKDAY's own traditional ruler
    (same weekday-planet table as Hora's day-lord anchor, WEEKDAY_LORD) --
    a whole-day-level signal, genuinely coarser than Hora's hourly
    resolution and independent of it (a benefic day can still have
    malefic hours within it, and vice versa).
    """
    idx = vara_index(times_utc)
    lords = np.array([WEEKDAY_LORD[i] for i in idx])
    bias = np.zeros(len(lords), dtype=float)
    bias[np.isin(lords, list(BENEFIC_PLANETS))] = 1.0
    bias[np.isin(lords, list(MALEFIC_PLANETS))] = -1.0
    return bias


# Only these 5 "true planets" (not the luminaries Sun/Moon, not the shadow
# points Rahu/Ketu) can enter Graha Yuddha (planetary war) per BPHS.
GRAHA_YUDDHA_PLANETS = ("mercury", "venus", "mars", "jupiter", "saturn")
GRAHA_YUDDHA_ORB_DEGREES = 1.0


# The 7 classical "visible" bodies -- excludes Rahu/Ketu (shadow points
# with no physical position to aspect against in these comprehensive
# multi-pair systems, unlike the Moon-relative aspect score above which
# only ever looks at 4 specific planets, not Rahu/Ketu either).
CLASSICAL_SEVEN = ("sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn")
DECLINATION_PARALLEL_ORB_DEGREES = 1.0
# Mean daily motion (degrees/day) = 360 / sidereal orbital period in days --
# real orbital mechanics constants, used to normalize each planet's
# continuous angular velocity onto a comparable scale.
MEAN_DAILY_MOTION_DEGREES = {"mars": 360.0 / 687.0, "jupiter": 360.0 / 4331.0, "saturn": 360.0 / 10747.0}


def _pairwise_aspect_score(longitudes: dict[str, np.ndarray], bodies: tuple[str, ...]) -> np.ndarray:
    n = len(next(iter(longitudes.values())))
    score = np.zeros(n, dtype=float)
    for i, p1 in enumerate(bodies):
        for p2 in bodies[i + 1 :]:
            sep = _angular_separation(longitudes[p1], longitudes[p2])
            harmonious = np.any([np.abs(sep - a) <= ASPECT_ORB_DEGREES for a in HARMONIOUS_ASPECT_ANGLES], axis=0)
            challenging = np.any([np.abs(sep - a) <= ASPECT_ORB_DEGREES for a in CHALLENGING_ASPECT_ANGLES], axis=0)
            pair_has_benefic = (p1 in BENEFIC_PLANETS) or (p2 in BENEFIC_PLANETS)
            pair_has_malefic = (p1 in MALEFIC_PLANETS) or (p2 in MALEFIC_PLANETS)
            score += (harmonious & pair_has_benefic).astype(float)
            score -= (challenging & pair_has_malefic).astype(float)
    return score


def all_pairs_aspect_score(times_utc: np.ndarray) -> np.ndarray:
    """A Bradley-Barometer-style comprehensive aggregate (Donald Bradley,
    1940s -- a real, historically documented financial-astrology oscillator
    built from a weighted SUM of many planetary aspects at once, rather
    than reading one pair in isolation): every one of the 21 pairs among
    the 7 classical bodies (not just Moon-to-{Jupiter,Venus,Saturn,Mars},
    which is all benefic_malefic_aspect_score above reads), scored +1 per
    pair if EITHER member is a classical benefic and the pair is in a
    tight harmonious aspect, -1 if EITHER member is malefic and the pair
    is in a tight challenging aspect. Deliberately does not use per-pair
    "planetary friendship" weights (Naisargika Maitri) the way Bradley's
    original scheme reportedly did -- that's a separate, more elaborate
    system this codebase doesn't implement, and guessing at undocumented
    historical weights would not be honest sourcing.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    longitudes = {p: tropical_longitudes(p, times_arr) for p in CLASSICAL_SEVEN}
    return _pairwise_aspect_score(longitudes, CLASSICAL_SEVEN)


def helio_aspect_score(times_utc: np.ndarray) -> np.ndarray:
    """Same comprehensive benefic/malefic aspect scoring as
    all_pairs_aspect_score, but among Mercury/Venus/Mars/Jupiter/Saturn's
    HELIOCENTRIC longitudes -- the frame financial-astrology sources favor
    for cleaner cycles, since apparent retrograde (a purely geocentric
    viewing illusion) never appears here. Sun/Moon excluded: the Sun has
    no meaningful heliocentric position of its own (it's this frame's
    origin) and the Moon is Earth's satellite, not a Sun-orbiting body.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    bodies = ("mercury", "venus", "mars", "jupiter", "saturn")
    longitudes = {p: heliocentric_longitude(p, times_arr) for p in bodies}
    return _pairwise_aspect_score(longitudes, bodies)


def declination_aspect_score(times_utc: np.ndarray) -> np.ndarray:
    """Parallel/contra-parallel declination aspects to the Moon -- a
    genuinely different coordinate AXIS (declination, not ecliptic
    longitude) from every other aspect function in this module, scored the
    same Moon-relative, benefic/malefic-asymmetric way as
    benefic_malefic_aspect_score: +1 for each of Jupiter/Venus PARALLEL
    (same declination within orb -- the equatorial-coordinate analogue of
    a conjunction) to the Moon, -1 for each of Saturn/Mars CONTRA-PARALLEL
    (equal-and-opposite declination -- the analogue of an opposition).
    """
    times_arr = np.asarray(times_utc, dtype=float)
    moon_dec = declination("moon", times_arr)
    score = np.zeros(len(times_arr), dtype=float)
    for planet in ("jupiter", "venus"):
        parallel = np.abs(declination(planet, times_arr) - moon_dec) <= DECLINATION_PARALLEL_ORB_DEGREES
        score += parallel.astype(float)
    for planet in ("saturn", "mars"):
        contra_parallel = np.abs(declination(planet, times_arr) + moon_dec) <= DECLINATION_PARALLEL_ORB_DEGREES
        score -= contra_parallel.astype(float)
    return score


def planet_velocity_normalized(body_key: str, times_utc: np.ndarray) -> np.ndarray:
    """Continuous angular velocity, normalized by the body's own real mean
    daily motion (MEAN_DAILY_MOTION_DEGREES, an orbital-mechanics constant)
    so it's comparable across planets that move at very different natural
    speeds. This is GEOCENTRIC apparent velocity divided by a HELIOCENTRIC
    mean-motion constant, so it swings considerably wider than a naive
    "+-1 is normal" intuition suggests -- confirmed empirically up to
    roughly +-4x for Mars, since apparent motion as seen from Earth mixes
    in Earth's own orbital speed (fastest well after opposition, negative
    during retrograde), not just the target planet's own orbital speed.
    A continuous replacement for the binary is_retrograde flag: a planet
    just barely stationing (velocity near 0, about to turn) is a
    meaningfully different state from one moving at full normal speed,
    which a boolean can't distinguish.
    """
    velocity = angular_velocity(body_key, times_utc)
    return velocity / MEAN_DAILY_MOTION_DEGREES[body_key]


def graha_yuddha_intensity(times_utc: np.ndarray) -> np.ndarray:
    """How tight the TIGHTEST current "planetary war" is, in [0, 1]: two of
    {Mercury, Venus, Mars, Jupiter, Saturn} within GRAHA_YUDDHA_ORB_DEGREES
    of each other in tropical longitude (the aspect angle, so the sidereal/
    tropical ayanamsa offset cancels out identically -- same reasoning as
    benefic_malefic_aspect_score). 1.0 = exact conjunction, 0.0 = no pair
    anywhere near the 1-degree orb right now.

    Deliberately reports intensity, not a winner/loser call: classical
    sources disagree on the exact victory criterion (BPHS invokes Shadbala,
    brightness, AND latitude together, a much larger system this codebase
    doesn't implement) -- the one point of universal agreement is which
    pair is closest and how close, which is what's captured here.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    longitudes = {p: tropical_longitudes(p, times_arr) for p in GRAHA_YUDDHA_PLANETS}
    min_sep = np.full(len(times_arr), 180.0)
    for i, p1 in enumerate(GRAHA_YUDDHA_PLANETS):
        for p2 in GRAHA_YUDDHA_PLANETS[i + 1 :]:
            sep = _angular_separation(longitudes[p1], longitudes[p2])
            min_sep = np.minimum(min_sep, sep)
    return np.clip(1.0 - min_sep / GRAHA_YUDDHA_ORB_DEGREES, 0.0, 1.0)
