from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

_ET = ZoneInfo("America/New_York")
_UNIX_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

# Real, published 2026 macro-event dates -- the specific US economic events
# most documented to move gold (via real-yield/USD-strength expectations)
# and forex broadly. Verified via web research against the Federal
# Reserve's and BLS's own schedules (2026-08-21), not assumed or invented.
# Needs a manual update once the Fed/BLS publish their 2027 calendars --
# these are administratively SET dates, not something derivable by a rule
# the way NFP's "first Friday of the month" is (see _nfp_dates).

# FOMC: 8 meetings/year, each 2 days; the rate decision is announced at
# 2:00 PM ET on the SECOND day. Source: federalreserve.gov/monetarypolicy/
# fomccalendars.htm.
FOMC_DECISION_DATES_2026 = [
    dt.date(2026, 1, 28), dt.date(2026, 3, 18), dt.date(2026, 4, 29), dt.date(2026, 6, 17),
    dt.date(2026, 7, 29), dt.date(2026, 9, 16), dt.date(2026, 10, 28), dt.date(2026, 12, 9),
]
FOMC_DECISION_TIME_ET = dt.time(14, 0)

# CPI: released 8:30 AM ET, ~2nd week of the month following the reference
# month. Source: BLS release calendar (via usinflationcalculator.com,
# cross-checked against confirmed individual BLS-sourced dates).
CPI_RELEASE_DATES_2026 = [
    dt.date(2026, 1, 13), dt.date(2026, 2, 13), dt.date(2026, 3, 11), dt.date(2026, 4, 10),
    dt.date(2026, 5, 12), dt.date(2026, 6, 10), dt.date(2026, 7, 14), dt.date(2026, 8, 12),
    dt.date(2026, 9, 11), dt.date(2026, 10, 14), dt.date(2026, 11, 10), dt.date(2026, 12, 10),
]
CPI_RELEASE_TIME_ET = dt.time(8, 30)

# NFP (the monthly BLS "Employment Situation" jobs report): a genuine
# deterministic RULE, not a published list -- first Friday of every month,
# 8:30 AM ET, unchanged for decades, so this is computed for a wide date
# range rather than looked up.
NFP_RELEASE_TIME_ET = dt.time(8, 30)
_NFP_YEAR_RANGE = (2024, 2029)


def _combine_et_utc(date: dt.date, time_et: dt.time) -> dt.datetime:
    return dt.datetime.combine(date, time_et, tzinfo=_ET).astimezone(dt.timezone.utc)


def _nfp_dates() -> list[dt.date]:
    dates = []
    for year in range(*_NFP_YEAR_RANGE):
        for month in range(1, 13):
            first_of_month = dt.date(year, month, 1)
            days_to_friday = (4 - first_of_month.weekday()) % 7  # Monday=0 ... Friday=4
            dates.append(first_of_month + dt.timedelta(days=days_to_friday))
    return dates


def _build_event_table() -> tuple[np.ndarray, np.ndarray]:
    """(sorted unix timestamps, matching event-kind labels) -- built ONCE at
    import time, not per row: the whole point of a fixed schedule is that
    it doesn't need recomputing.
    """
    events: list[tuple[float, str]] = []
    for d in FOMC_DECISION_DATES_2026:
        events.append((_combine_et_utc(d, FOMC_DECISION_TIME_ET).timestamp(), "FOMC"))
    for d in CPI_RELEASE_DATES_2026:
        events.append((_combine_et_utc(d, CPI_RELEASE_TIME_ET).timestamp(), "CPI"))
    for d in _nfp_dates():
        events.append((_combine_et_utc(d, NFP_RELEASE_TIME_ET).timestamp(), "NFP"))
    events.sort(key=lambda x: x[0])
    times = np.array([e[0] for e in events], dtype=float)
    kinds = np.array([e[1] for e in events], dtype=object)
    return times, kinds


_EVENT_TIMES, _EVENT_KINDS = _build_event_table()


def _nearest_event_index(times_utc: np.ndarray) -> np.ndarray:
    """For each timestamp, the index into _EVENT_TIMES of whichever event
    (past or future) is closest in absolute time -- a past event still
    counts as "just happened" for a few hours, matching next_macro_event's
    "-1 hour" grace window, generalized continuously.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    right_idx = np.clip(np.searchsorted(_EVENT_TIMES, times_arr, side="left"), 0, len(_EVENT_TIMES) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(_EVENT_TIMES) - 1)
    left_dist = np.abs(times_arr - _EVENT_TIMES[left_idx])
    right_dist = np.abs(_EVENT_TIMES[right_idx] - times_arr)
    return np.where(left_dist <= right_dist, left_idx, right_idx)


def hours_until_next_macro_event(times_utc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: (signed hours to the nearest event -- negative if it just
    passed, positive if upcoming, kind label) for every row at once.
    """
    times_arr = np.asarray(times_utc, dtype=float)
    idx = _nearest_event_index(times_arr)
    hours = (_EVENT_TIMES[idx] - times_arr) / 3600.0
    return hours, _EVENT_KINDS[idx]


# Proximity is rescaled against a 10-day (240h) window -- CPI/NFP are at
# most ~35 days apart and FOMC ~7 weeks, so within any 10-day window at
# least one of the three is normally either just-passed or approaching;
# rescaling against that keeps the feature meaningfully non-zero most of
# the time rather than saturating at 0 for weeks between rarer FOMC prints.
_PROXIMITY_WINDOW_HOURS = 240.0


def macro_event_proximity(times_utc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (proximity, is_fomc, is_cpi, is_nfp) arrays: proximity in
    [0, 1] (1 = right at the nearest event, 0 = at/beyond the rescaling
    window), plus which TYPE of event is nearest -- CPI/FOMC/NFP move gold
    through different channels (rate-path vs inflation-print vs labor-
    market), so which one is closest matters, not just "how close."
    """
    hours, kinds = hours_until_next_macro_event(times_utc)
    proximity = np.clip(1.0 - np.abs(hours) / _PROXIMITY_WINDOW_HOURS, 0.0, 1.0)
    is_fomc = (kinds == "FOMC").astype(float)
    is_cpi = (kinds == "CPI").astype(float)
    is_nfp = (kinds == "NFP").astype(float)
    return proximity, is_fomc, is_cpi, is_nfp


def next_macro_event(now_utc: dt.datetime | None = None) -> dict:
    """The single nearest UPCOMING (or just-passed within the last hour)
    macro event, for display on the Events panel.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    now_ts = now.timestamp()
    future_mask = _EVENT_TIMES >= now_ts - 3600
    if not future_mask.any():
        return {"event": None, "hours_away": None, "at": None}
    future_times = _EVENT_TIMES[future_mask]
    future_kinds = _EVENT_KINDS[future_mask]
    i = int(np.argmin(future_times))
    event_time = future_times[i]
    hours_away = (event_time - now_ts) / 3600.0
    at = (_UNIX_EPOCH + dt.timedelta(seconds=float(event_time))).isoformat()
    return {"event": str(future_kinds[i]), "hours_away": round(float(hours_away), 1), "at": at}
