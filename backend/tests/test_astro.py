import datetime as dt

import numpy as np
import pytest

from app.astro import ephemeris, panchang

skyfield = pytest.importorskip("skyfield")


def _unix(y, m, d, h=0, mi=0, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp()


def test_tropical_longitude_is_vectorized_and_in_range():
    times = [_unix(2024, 1, 1), _unix(2024, 6, 1), _unix(2024, 12, 1)]
    lons = ephemeris.tropical_longitudes("sun", times)
    assert len(lons) == 3
    assert np.all((lons >= 0) & (lons < 360))


def test_sun_moves_roughly_one_degree_per_day():
    lon0 = ephemeris.tropical_longitudes("sun", [_unix(2024, 3, 1)])[0]
    lon1 = ephemeris.tropical_longitudes("sun", [_unix(2024, 3, 2)])[0]
    delta = (lon1 - lon0 + 360) % 360
    assert 0.8 < delta < 1.2


def test_lahiri_ayanamsa_increases_over_time_and_matches_j2000_anchor():
    at_j2000 = panchang  # noop import check
    a2000 = ephemeris.lahiri_ayanamsa([_unix(2000, 1, 1, 12)])[0]
    a2024 = ephemeris.lahiri_ayanamsa([_unix(2024, 1, 1, 12)])[0]
    assert abs(a2000 - 23.85) < 0.01
    assert a2024 > a2000


def test_sidereal_longitude_is_tropical_minus_ayanamsa_mod_360():
    t = [_unix(2024, 6, 15)]
    tropical = ephemeris.tropical_longitudes("mars", t)
    sidereal = ephemeris.sidereal_longitude("mars", t)
    ayanamsa = ephemeris.lahiri_ayanamsa(t)
    expected = (tropical - ayanamsa) % 360.0
    assert np.allclose(sidereal, expected)


def test_is_retrograde_returns_boolean_array_matching_input_length():
    times = [_unix(2024, 1, 1), _unix(2024, 6, 1)]
    result = ephemeris.is_retrograde("mercury", times)
    assert result.dtype == bool
    assert len(result) == 2


def test_tithi_index_in_valid_range():
    times = [_unix(2024, m, 15) for m in range(1, 13)]
    idx = panchang.tithi_index(times)
    assert np.all((idx >= 0) & (idx <= 29))


def test_nakshatra_index_in_valid_range():
    times = [_unix(2024, m, 15) for m in range(1, 13)]
    idx = panchang.nakshatra_index(times)
    assert np.all((idx >= 0) & (idx <= 26))


def test_yoga_index_in_valid_range():
    times = [_unix(2024, m, 15) for m in range(1, 13)]
    idx = panchang.yoga_index(times)
    assert np.all((idx >= 0) & (idx <= 26))


def test_is_waxing_moon_consistent_with_tithi_index():
    times = [_unix(2024, m, d) for m in range(1, 4) for d in (1, 8, 15, 22)]
    tithi = panchang.tithi_index(times)
    waxing = panchang.is_waxing_moon(times)
    assert np.array_equal(waxing, tithi < 15)


def test_vara_index_cycles_correctly_across_consecutive_days():
    # 2024-01-01 00:00 UTC + IST offset (5:30) falls just after IST midnight
    # on 2024-01-01, a Monday -> Hindu weekday index 1 (Sun=0..Sat=6).
    base = _unix(2024, 1, 1, 1)  # 1 AM UTC = 6:30 AM IST, safely mid-day IST
    varas = [panchang.vara_index([base + i * 86400])[0] for i in range(7)]
    assert varas[0] == 1  # Monday
    assert varas == [(1 + i) % 7 for i in range(7)]


def test_rahu_kaal_window_is_one_eighth_of_daytime_and_within_bounds():
    date = dt.date(2024, 6, 15)
    sunrise, sunset = panchang._sunrise_sunset_utc(date)
    start, end = panchang.rahu_kaal_window_utc(date)
    expected_duration = (sunset - sunrise) / 8.0
    assert abs((end - start) - expected_duration) < 1.0
    assert sunrise <= start < end <= sunset


def test_rahu_kaal_segment_matches_weekday_table():
    # 2024-06-17 is a Monday -> Hindu weekday 1 -> segment 2 (table value).
    monday = dt.date(2024, 6, 17)
    sunrise, sunset = panchang._sunrise_sunset_utc(monday)
    start, _ = panchang.rahu_kaal_window_utc(monday)
    part = (sunset - sunrise) / 8.0
    expected_start = sunrise + (2 - 1) * part
    assert abs(start - expected_start) < 1.0


def test_is_rahu_kaal_agrees_with_rahu_kaal_window_utc():
    date = dt.date(2024, 6, 15)
    start, end = panchang.rahu_kaal_window_utc(date)
    mid = (start + end) / 2.0
    before = start - 3600
    after = end + 3600
    result = panchang.is_rahu_kaal([before, mid, after])
    assert list(result) == [False, True, False]


def test_sunrise_sunset_cache_returns_same_value_on_repeat_call():
    date = dt.date(2024, 6, 15)
    first = panchang._sunrise_sunset_utc(date)
    second = panchang._sunrise_sunset_utc(date)
    assert first == second


def test_rahu_ketu_are_always_exactly_opposite():
    times = [_unix(2024, m, 15) for m in range(1, 13)]
    rahu = ephemeris.tropical_longitudes("rahu", times)
    ketu = ephemeris.tropical_longitudes("ketu", times)
    assert np.allclose((rahu - ketu) % 360.0, 180.0)


def test_mean_lunar_node_regresses_through_the_zodiac():
    """The Moon's mean node always moves BACKWARD (regresses) through the
    zodiac -- opposite every other body here -- roughly -19deg/year."""
    lon0 = ephemeris.tropical_longitudes("rahu", [_unix(2024, 1, 1)])[0]
    lon1 = ephemeris.tropical_longitudes("rahu", [_unix(2025, 1, 1)])[0]
    delta = ((lon1 - lon0 + 180) % 360) - 180
    assert -25 < delta < -13


def test_nakshatra_lords_match_the_documented_vimshottari_sequence():
    # Verified groupings: Ketu rules Ashwini(0)/Magha(9)/Mula(18); Venus
    # rules Bharani(1)/Purva Phalguni(10)/Purva Ashadha(19); Moon rules
    # Rohini(3)/Hasta(12)/Shravana(21).
    assert panchang.NAKSHATRA_LORDS[0] == panchang.NAKSHATRA_LORDS[9] == panchang.NAKSHATRA_LORDS[18] == "ketu"
    assert panchang.NAKSHATRA_LORDS[1] == panchang.NAKSHATRA_LORDS[10] == panchang.NAKSHATRA_LORDS[19] == "venus"
    assert panchang.NAKSHATRA_LORDS[3] == panchang.NAKSHATRA_LORDS[12] == panchang.NAKSHATRA_LORDS[21] == "moon"
    assert len(panchang.NAKSHATRA_LORDS) == 27
    assert len(set(panchang.NAKSHATRA_LORDS)) == 9  # all 9 Navagraha represented


def test_nakshatra_lord_bias_classifies_every_lord():
    times = [_unix(2024, m, d) for m in range(1, 13) for d in (1, 10, 20)]
    bias = panchang.nakshatra_lord_bias(times)
    assert np.all(np.isin(bias, [-1.0, 1.0]))  # never 0 -- every Navagraha classifies


def test_karana_index_in_valid_range():
    times = [_unix(2024, m, 15) for m in range(1, 13)]
    idx = panchang.karana_index(times)
    assert np.all((idx >= 0) & (idx <= 59))


def test_karana_index_is_twice_tithi_index_ish():
    """Each Tithi (12deg) contains exactly 2 Karanas (6deg each) -- so
    karana_index // 2 tracks tithi_index (with a possible +-1 at exact
    boundaries, which a real timestamp essentially never lands on)."""
    times = [_unix(2024, m, 15, 10) for m in range(1, 13)]
    tithi = panchang.tithi_index(times)
    karana = panchang.karana_index(times)
    assert np.array_equal(karana // 2, tithi)


def test_karana_name_covers_all_11_names_without_error():
    seen = {panchang.karana_name(i) for i in range(60)}
    expected = set(panchang.CHARA_KARANAS) | set(panchang.STHIRA_KARANAS) | {panchang.KIMSTUGHNA}
    assert seen == expected
    assert len(expected) == 11


def test_vishti_karana_is_the_7th_chara_karana():
    # karana_index 7 -> (7-1)%7==6 -> the 7th chara karana -> Vishti.
    assert panchang.karana_name(7) == "Vishti"
    result = panchang.is_vishti_karana(np.array([0.0]))
    assert result.dtype == bool


def test_is_vishti_karana_agrees_with_karana_name():
    times = [_unix(2024, m, d) for m in range(1, 13) for d in (1, 5, 10, 15, 20, 25)]
    idx = panchang.karana_index(times)
    names = [panchang.karana_name(int(i)) for i in idx]
    vishti_mask = panchang.is_vishti_karana(times)
    for name, is_vishti in zip(names, vishti_mask):
        assert is_vishti == (name == "Vishti")


def test_yoga_severity_matches_documented_inauspicious_indices():
    times = [_unix(2024, m, d) for m in range(1, 13) for d in (1, 8, 15, 22)]
    idx = panchang.yoga_index(times)
    severity = panchang.yoga_severity(times)
    for i, s in zip(idx, severity):
        if i in panchang.SEVERE_INAUSPICIOUS_YOGAS:
            assert s == 2
        elif i in panchang.MODERATE_INAUSPICIOUS_YOGAS:
            assert s == 1
        else:
            assert s == 0


def test_benefic_malefic_aspect_score_is_bounded():
    times = [_unix(2024, m, d) for m in range(1, 13) for d in (1, 15)]
    score = panchang.benefic_malefic_aspect_score(times)
    assert np.all(score >= -2.0) and np.all(score <= 2.0)


def test_malefic_retrograde_count_is_between_0_and_2():
    times = [_unix(2024, m, 15) for m in range(1, 13)]
    count = panchang.malefic_retrograde_count(times)
    assert np.all((count >= 0) & (count <= 2))


def test_combust_planet_count_is_between_0_and_5():
    times = [_unix(2024, m, d) for m in range(1, 13) for d in (1, 15)]
    count = panchang.combust_planet_count(times)
    assert np.all((count >= 0) & (count <= 5))


def test_mercury_is_combust_when_conjunct_the_sun():
    """Mercury never strays more than ~28deg from the Sun, so a real
    Mercury-Sun conjunction (well within its 14deg combustion orb) is easy
    to find -- search forward for one rather than assuming a date."""
    for offset_days in range(400):
        t = _unix(2024, 1, 1) + offset_days * 86400
        sun = ephemeris.tropical_longitudes("sun", [t])[0]
        mercury = ephemeris.tropical_longitudes("mercury", [t])[0]
        sep = min(abs(sun - mercury) % 360.0, 360.0 - abs(sun - mercury) % 360.0)
        if sep <= 5.0:  # comfortably inside Mercury's 14deg orb
            count = panchang.combust_planet_count([t])[0]
            assert count >= 1
            return
    raise AssertionError("no Mercury-Sun conjunction found in range -- widen the search")


def test_combustion_orb_table_excludes_shadow_planets_and_the_sun_itself():
    # Rahu/Ketu are shadow points (no physical body to be "near the Sun"),
    # and the Sun can't be combust relative to itself -- only the 5 classical
    # planets that combust_planet_count actually checks have listed orbs.
    for excluded in ("sun", "rahu", "ketu"):
        assert excluded not in panchang.COMBUSTION_ORB_DEGREES
    for included in ("mercury", "venus", "mars", "jupiter", "saturn"):
        assert included in panchang.COMBUSTION_ORB_DEGREES
