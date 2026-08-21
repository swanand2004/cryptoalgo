import datetime as dt

import numpy as np
import pytest

from app.astro import panchang

skyfield = pytest.importorskip("skyfield")


def _unix(y, m, d, h=0):
    return dt.datetime(y, m, d, h, tzinfo=dt.timezone.utc).timestamp()


def test_all_pairs_aspect_score_bounded_and_finite_on_real_dates():
    times = np.array([_unix(2026, m, 1, 12) for m in range(1, 13)])
    score = panchang.all_pairs_aspect_score(times)
    assert np.isfinite(score).all()
    # 21 pairs total, each contributing at most +-1 -- a hard ceiling.
    assert (np.abs(score) <= 21).all()


def test_all_pairs_aspect_score_math_matches_known_synthetic_longitudes(monkeypatch):
    # Every one of the 21 pairs checked by hand: only Mercury-Jupiter sits
    # at an exact aspect angle (120deg, trine) -- all other 20 pairs are
    # deliberately placed far from every aspect angle (60/90/120/180
    # +-6deg orb), so the score must land on exactly +1 (both benefic).
    positions = {
        "sun": 0.0, "moon": 1.0, "mercury": 50.0, "venus": 5.0,
        "mars": 8.0, "jupiter": 170.0, "saturn": 15.0,
    }

    def fake_tropical(body, times):
        return np.full(len(times), positions[body])

    monkeypatch.setattr(panchang, "tropical_longitudes", fake_tropical)
    score = panchang.all_pairs_aspect_score(np.array([0.0]))
    assert score[0] == pytest.approx(1.0)


def test_helio_aspect_score_bounded_and_finite_on_real_dates():
    times = np.array([_unix(2026, m, 1, 12) for m in range(1, 13)])
    score = panchang.helio_aspect_score(times)
    assert np.isfinite(score).all()
    assert (np.abs(score) <= 10).all()  # C(5,2) = 10 pairs among the 5 heliocentric bodies


def test_declination_aspect_score_bounded_and_finite_on_real_dates():
    times = np.array([_unix(2026, m, 1, 12) for m in range(1, 13)])
    score = panchang.declination_aspect_score(times)
    assert np.isfinite(score).all()
    assert (np.abs(score) <= 4).all()  # 2 benefics (+) + 2 malefics (-) max


def test_declination_aspect_score_detects_parallel_and_contra_parallel(monkeypatch):
    def fake_declination(body, times):
        values = {"moon": 10.0, "jupiter": 10.3, "saturn": -10.1, "venus": 80.0, "mars": 80.0}
        return np.full(len(times), values[body])

    monkeypatch.setattr(panchang, "declination", fake_declination)
    score = panchang.declination_aspect_score(np.array([0.0]))
    # jupiter parallel moon (+1), saturn contra-parallel moon (-1),
    # venus/mars far from either parallel or contra-parallel to moon (0 each).
    assert score[0] == pytest.approx(0.0)


def test_planet_velocity_normalized_matches_mean_daily_motion_math():
    times = np.array([_unix(2026, 6, 1, 12)])
    for body in ("mars", "jupiter", "saturn"):
        raw = panchang.angular_velocity(body, times)
        normalized = panchang.planet_velocity_normalized(body, times)
        assert normalized[0] == pytest.approx(raw[0] / panchang.MEAN_DAILY_MOTION_DEGREES[body])


def test_planet_velocity_normalized_bounded_reasonably_on_real_dates():
    # This is GEOCENTRIC apparent velocity (angular_velocity), normalized
    # by the body's HELIOCENTRIC mean motion -- the two don't move in
    # lockstep, since geocentric motion mixes in Earth's own orbital speed
    # (fastest well after opposition, when Earth is pulling away fastest;
    # negative during retrograde). Confirmed empirically to swing up to
    # ~+-4x for Mars, not the tighter +-1-ish range orbital period alone
    # would suggest -- this bound reflects that measured reality, not a
    # guess.
    times = np.array([_unix(2026, m, 1, 12) for m in range(1, 13)])
    for body in ("mars", "jupiter", "saturn"):
        normalized = panchang.planet_velocity_normalized(body, times)
        assert np.isfinite(normalized).all()
        assert (np.abs(normalized) < 6.0).all()
