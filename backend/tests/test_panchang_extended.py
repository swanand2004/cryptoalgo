import datetime as dt

import numpy as np
import pytest

from app.astro import panchang

skyfield = pytest.importorskip("skyfield")


def _unix(y, m, d, h=0):
    return dt.datetime(y, m, d, h, tzinfo=dt.timezone.utc).timestamp()


@pytest.mark.parametrize(
    "desired_index,expected_bias",
    [(0, 1.0), (1, 0.5), (2, 0.5), (3, -1.0), (4, 1.0), (15, 1.0), (18, -1.0)],
)
def test_tithi_class_bias_matches_known_indices(monkeypatch, desired_index, expected_bias):
    # tithi_index -> class is (index % 5): 0=Nanda(+1), 1=Bhadra(+0.5),
    # 2=Jaya(+0.5), 3=Rikta(-1), 4=Purna(+1) -- verified directly against a
    # controlled tithi_index via monkeypatched sidereal longitudes
    # (moon - sun = 12deg * desired_index, +1deg to stay mid-tithi away
    # from any boundary), since the class is a pure function of the index.
    moon_lon = 12.0 * desired_index + 1.0

    def fake_sidereal(body, times):
        return np.full(len(times), moon_lon if body == "moon" else 0.0)

    monkeypatch.setattr(panchang, "sidereal_longitude", fake_sidereal)
    bias = panchang.tithi_class_bias(np.array([0.0]))
    assert bias[0] == pytest.approx(expected_bias)


def test_vara_lord_bias_matches_weekday_lord_classification():
    # 2026-08-16 is a Sunday -> "sun" (malefic); 2026-08-17 Monday -> "moon" (benefic).
    sunday = panchang.vara_lord_bias(np.array([_unix(2026, 8, 16, 10)]))[0]
    monday = panchang.vara_lord_bias(np.array([_unix(2026, 8, 17, 10)]))[0]
    assert sunday == -1.0  # Sun is malefic
    assert monday == 1.0  # Moon is benefic


def test_vara_lord_bias_covers_all_seven_weekdays_with_no_neutral():
    times = [_unix(2026, 8, 16 + i, 10) for i in range(7)]
    biases = panchang.vara_lord_bias(np.array(times))
    assert set(biases.tolist()) <= {1.0, -1.0}
    assert len(biases) == 7


def test_graha_yuddha_intensity_math_is_correct_given_known_longitudes(monkeypatch):
    def fake_tropical(body, times):
        # Mars and Saturn placed 0.4 degrees apart (within the 1-degree
        # orb); everything else far away -- the tightest pair must win.
        positions = {"mercury": 10.0, "venus": 50.0, "mars": 100.0, "jupiter": 200.0, "saturn": 100.4}
        return np.full(len(times), positions[body])

    monkeypatch.setattr(panchang, "tropical_longitudes", fake_tropical)
    intensity = panchang.graha_yuddha_intensity(np.array([0.0]))
    assert intensity[0] == pytest.approx(1.0 - 0.4 / 1.0, abs=1e-9)


def test_graha_yuddha_intensity_is_zero_when_nothing_is_close(monkeypatch):
    def fake_tropical(body, times):
        positions = {"mercury": 0.0, "venus": 60.0, "mars": 120.0, "jupiter": 180.0, "saturn": 240.0}
        return np.full(len(times), positions[body])

    monkeypatch.setattr(panchang, "tropical_longitudes", fake_tropical)
    intensity = panchang.graha_yuddha_intensity(np.array([0.0]))
    assert intensity[0] == 0.0


def test_graha_yuddha_intensity_bounded_and_finite_on_real_dates():
    times = np.array([_unix(2026, 1, 1, 12), _unix(2026, 6, 1, 12), _unix(2026, 12, 1, 12)])
    intensity = panchang.graha_yuddha_intensity(times)
    assert np.isfinite(intensity).all()
    assert (intensity >= 0).all() and (intensity <= 1).all()
