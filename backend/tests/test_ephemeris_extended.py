import datetime as dt

import numpy as np
import pytest

skyfield = pytest.importorskip("skyfield")

from app.astro.ephemeris import (
    angular_velocity,
    declination,
    heliocentric_longitude,
    is_retrograde,
    tropical_longitudes,
)


def _unix(y, m, d, h=0):
    return dt.datetime(y, m, d, h, tzinfo=dt.timezone.utc).timestamp()


def test_heliocentric_longitude_bounded_and_finite():
    times = np.array([_unix(2026, i, 15, 12) for i in range(1, 13)])
    for body in ("mercury", "venus", "mars", "jupiter", "saturn"):
        lon = heliocentric_longitude(body, times)
        assert np.isfinite(lon).all()
        assert (lon >= 0).all() and (lon < 360).all()


def test_declination_bounded_and_finite():
    times = np.array([_unix(2026, i, 15, 12) for i in range(1, 13)])
    for body in ("sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"):
        dec = declination(body, times)
        assert np.isfinite(dec).all()
        assert (dec >= -90).all() and (dec <= 90).all()


def test_sun_declination_oscillates_within_the_obliquity_of_the_ecliptic():
    # A real, independently-known sanity check: the Sun's declination
    # swings within +-~23.5 degrees (Earth's axial tilt) over a year, never
    # beyond it.
    times = np.array([_unix(2026, m, 15, 12) for m in range(1, 13)])
    dec = declination("sun", times)
    assert dec.max() < 24.0
    assert dec.min() > -24.0
    assert dec.max() - dec.min() > 30.0  # genuinely swings, not flat


def test_rahu_ketu_declination_is_a_defined_neutral_not_a_crash():
    times = np.array([_unix(2026, 6, 1, 12)])
    assert declination("rahu", times)[0] == 0.0
    assert declination("ketu", times)[0] == 0.0


def test_angular_velocity_sign_matches_is_retrograde():
    # The continuous velocity and the existing boolean must never disagree
    # -- they're built from the exact same underlying delta.
    times = np.array([_unix(2026, m, d, 12) for m in range(1, 13) for d in (1, 15)])
    for body in ("mercury", "venus", "mars", "jupiter", "saturn"):
        velocity = angular_velocity(body, times)
        retro = is_retrograde(body, times)
        assert np.array_equal(velocity < 0, retro)


def test_heliocentric_longitude_has_no_retrograde_even_when_geocentric_does():
    """The whole point of the heliocentric frame: apparent retrograde
    motion is a purely geocentric viewing-angle illusion. Scans real dates
    for a genuine geocentric-retrograde window (rather than assuming one),
    then confirms the SAME body's heliocentric longitude keeps advancing
    (mod wraparound) across that exact window instead of reversing.
    """
    candidate_days = [_unix(2026, m, d, 12) for m in range(1, 13) for d in range(1, 29, 2)]
    times = np.array(candidate_days)

    def _longest_contiguous_run(mask: np.ndarray) -> tuple[int, int] | None:
        best = None
        run_start = None
        for i, flag in enumerate(mask):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                if best is None or (i - 1 - run_start) > (best[1] - best[0]):
                    best = (run_start, i - 1)
                run_start = None
        if run_start is not None and (best is None or (len(mask) - 1 - run_start) > (best[1] - best[0])):
            best = (run_start, len(mask) - 1)
        return best

    body = None
    run = None
    # Mercury retrogrades 3-4 times/year (~3 weeks each) -- almost certain
    # to be caught by this scan; Mars as a fallback (rarer, ~2.5 months
    # every ~26 months) in case Mercury's window is missed by the stride.
    for candidate_body in ("mercury", "mars"):
        retro = is_retrograde(candidate_body, times)
        candidate_run = _longest_contiguous_run(retro)
        if candidate_run is not None and candidate_run[1] > candidate_run[0]:
            body, run = candidate_body, candidate_run
            break
    if body is None:
        pytest.skip("no geocentric retrograde window found for mercury or mars in this scan of 2026 -- widen the scan if this ever fires")

    start, end = times[run[0]], times[run[1]]
    helio_start = heliocentric_longitude(body, np.array([start]))[0]
    helio_end = heliocentric_longitude(body, np.array([end]))[0]
    # Forward progress (mod 360) over the same window geocentric longitude
    # was moving backward in -- allow wraparound near 0/360.
    forward_progress = (helio_end - helio_start) % 360.0
    assert forward_progress < 180.0  # genuinely moved forward, not backward
