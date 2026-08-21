import datetime as dt

import numpy as np
import pytest

from app.features.calendar_events import (
    hours_until_weekly_options_expiry,
    weekly_expiry_proximity,
)


def _unix(y, m, d, h=0, mi=0):
    return dt.datetime(y, m, d, h, mi, tzinfo=dt.timezone.utc).timestamp()


# 2026-08-21 is a real Friday (verified against this session's other date
# checks) -- used throughout as the reference expiry instant.


def test_one_hour_before_friday_expiry():
    t = _unix(2026, 8, 21, 7)  # 07:00 UTC, expiry is 08:00 UTC
    hours = hours_until_weekly_options_expiry(np.array([t]))[0]
    assert hours == pytest.approx(1.0, abs=1e-6)


def test_one_hour_after_friday_expiry_rolls_to_next_week():
    t = _unix(2026, 8, 21, 9)  # 09:00 UTC, just past this week's expiry
    hours = hours_until_weekly_options_expiry(np.array([t]))[0]
    assert hours == pytest.approx(167.0, abs=1e-6)


def test_exact_expiry_instant_reports_the_next_ones_full_week():
    t = _unix(2026, 8, 21, 8)  # exactly at expiry
    hours = hours_until_weekly_options_expiry(np.array([t]))[0]
    assert hours == pytest.approx(168.0, abs=1e-6)


def test_previous_fridays_expiry_instant_is_exactly_one_week_out():
    t = _unix(2026, 8, 14, 8)  # exactly at the PRIOR week's expiry
    hours = hours_until_weekly_options_expiry(np.array([t]))[0]
    assert hours == pytest.approx(168.0, abs=1e-6)


def test_proximity_is_bounded_0_to_1_and_matches_the_hours_calculation():
    times = np.array([_unix(2026, 8, d, h) for d in range(15, 22) for h in (0, 8, 16)])
    hours = hours_until_weekly_options_expiry(times)
    proximity = weekly_expiry_proximity(times)
    assert np.allclose(proximity, 1.0 - hours / 168.0)
    assert (proximity >= 0).all() and (proximity <= 1.0 + 1e-9).all()


def test_proximity_is_near_one_right_before_expiry_and_near_zero_right_after():
    just_before = weekly_expiry_proximity(np.array([_unix(2026, 8, 21, 7, 59)]))[0]
    just_after = weekly_expiry_proximity(np.array([_unix(2026, 8, 21, 8, 1)]))[0]
    assert just_before > 0.99
    assert just_after < 0.01
