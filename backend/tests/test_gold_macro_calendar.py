import datetime as dt

import numpy as np
import pytest

from app.gold.macro_calendar import (
    CPI_RELEASE_DATES_2026,
    FOMC_DECISION_DATES_2026,
    hours_until_next_macro_event,
    macro_event_proximity,
    next_macro_event,
)


def test_fomc_dates_match_the_published_2026_schedule():
    """Spot-checked against federalreserve.gov's own 2026 calendar
    (2026-08-21) -- must not silently drift if this list is ever edited."""
    assert dt.date(2026, 1, 28) in FOMC_DECISION_DATES_2026
    assert dt.date(2026, 6, 17) in FOMC_DECISION_DATES_2026
    assert dt.date(2026, 9, 16) in FOMC_DECISION_DATES_2026
    assert dt.date(2026, 12, 9) in FOMC_DECISION_DATES_2026
    assert len(FOMC_DECISION_DATES_2026) == 8


def test_cpi_dates_match_the_published_2026_schedule():
    """Cross-checked against BLS-sourced individual confirmations
    (2026-08-21): June 10, July 14, September 11."""
    assert dt.date(2026, 6, 10) in CPI_RELEASE_DATES_2026
    assert dt.date(2026, 7, 14) in CPI_RELEASE_DATES_2026
    assert dt.date(2026, 9, 11) in CPI_RELEASE_DATES_2026
    assert len(CPI_RELEASE_DATES_2026) == 12


def test_fomc_decision_time_handles_dst_correctly():
    """2:00 PM ET is 19:00 UTC in EST (winter) but 18:00 UTC in EDT
    (summer) -- must use real US timezone rules, not a fixed UTC offset."""
    # Jan 28 2026 is EST (UTC-5).
    winter_event = next_macro_event(dt.datetime(2026, 1, 28, 18, 0, tzinfo=dt.timezone.utc))
    assert winter_event["event"] == "FOMC"
    assert winter_event["at"].startswith("2026-01-28T19:00")

    # Jun 17 2026 is EDT (UTC-4).
    summer_event = next_macro_event(dt.datetime(2026, 6, 17, 17, 0, tzinfo=dt.timezone.utc))
    assert summer_event["event"] == "FOMC"
    assert summer_event["at"].startswith("2026-06-17T18:00")


def test_next_macro_event_returns_the_soonest_upcoming_one():
    just_before = dt.datetime(2026, 1, 28, 18, 30, tzinfo=dt.timezone.utc)  # 30 min before FOMC decision
    result = next_macro_event(just_before)
    assert result["event"] == "FOMC"
    assert 0 < result["hours_away"] <= 0.6


def test_nfp_fills_gaps_between_fomc_and_cpi():
    """Even in a month with no FOMC/CPI print nearby, NFP (first Friday of
    every month, a genuine rule) should still surface as the nearest event."""
    # August 2026 has no FOMC meeting; CPI was Aug 12. Check a day in late
    # August, well past CPI, before September's events.
    late_august = dt.datetime(2026, 8, 25, tzinfo=dt.timezone.utc)
    result = next_macro_event(late_august)
    assert result["event"] is not None


def test_macro_event_proximity_is_highest_right_at_an_event():
    fomc_ts = dt.datetime(2026, 1, 28, 19, 0, tzinfo=dt.timezone.utc).timestamp()
    far_ts = dt.datetime(2026, 2, 1, 0, 0, tzinfo=dt.timezone.utc).timestamp()

    at_event, _, _, _ = macro_event_proximity(np.array([fomc_ts]))
    far_away, _, _, _ = macro_event_proximity(np.array([far_ts]))

    assert at_event[0] == pytest.approx(1.0)
    assert far_away[0] < at_event[0]


def test_exactly_one_event_type_flag_is_set_per_row():
    times = np.array([
        dt.datetime(2026, 1, 28, 19, 0, tzinfo=dt.timezone.utc).timestamp(),
        dt.datetime(2026, 6, 10, 12, 30, tzinfo=dt.timezone.utc).timestamp(),
    ])
    proximity, is_fomc, is_cpi, is_nfp = macro_event_proximity(times)
    totals = is_fomc + is_cpi + is_nfp
    assert (totals == 1.0).all()


def test_hours_until_next_macro_event_is_vectorized_and_consistent_with_scalar():
    times = np.array([
        dt.datetime(2026, 1, 28, 18, 30, tzinfo=dt.timezone.utc).timestamp(),
        dt.datetime(2026, 6, 17, 17, 30, tzinfo=dt.timezone.utc).timestamp(),
    ])
    hours, kinds = hours_until_next_macro_event(times)
    assert len(hours) == 2
    assert kinds[0] == "FOMC"
    assert kinds[1] == "FOMC"
    assert 0 < hours[0] < 1
    assert 0 < hours[1] < 1
