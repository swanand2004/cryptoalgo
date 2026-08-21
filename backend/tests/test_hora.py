import datetime as dt

import numpy as np

from app.astro.panchang import (
    BENEFIC_PLANETS,
    HORA_CHALDEAN_ORDER,
    MALEFIC_PLANETS,
    WEEKDAY_LORD,
    _sunrise_sunset_utc,
    hora_bias,
    hora_lord_at,
)


def test_every_weekday_first_hora_matches_its_traditional_ruler():
    # 2026-08-16 is a Sunday.
    for i in range(7):
        d = dt.date(2026, 8, 16 + i)
        sunrise, _ = _sunrise_sunset_utc(d)
        lord = hora_lord_at(sunrise + 1.0)  # 1s into the day's first Hora
        hindu_weekday = (d.weekday() + 1) % 7
        assert lord == WEEKDAY_LORD[hindu_weekday]


def test_24_hora_sequence_is_contiguous_across_the_day_night_boundary():
    d = dt.date(2026, 8, 16)
    sunrise, sunset = _sunrise_sunset_utc(d)
    next_sunrise, _ = _sunrise_sunset_utc(d + dt.timedelta(days=1))
    day_len = (sunset - sunrise) / 12.0
    night_len = (next_sunrise - sunset) / 12.0

    sequence = [hora_lord_at(sunrise + h * day_len + 1.0) for h in range(12)]
    sequence += [hora_lord_at(sunset + h * night_len + 1.0) for h in range(12)]

    start_idx = HORA_CHALDEAN_ORDER.index(sequence[0])
    expected = [HORA_CHALDEAN_ORDER[(start_idx + i) % 7] for i in range(24)]
    assert sequence == expected


def test_pre_dawn_instant_continues_yesterdays_cycle_not_todays():
    # 2026-08-17 is a Monday -- an instant just before ITS sunrise should
    # still belong to Sunday's night Hora sequence (hora 23, continuing from
    # Sunday's own day-lord), not jump early to Monday's hora 0.
    monday = dt.date(2026, 8, 17)
    sunrise, _ = _sunrise_sunset_utc(monday)
    lord = hora_lord_at(sunrise - 60.0)

    sun_idx = HORA_CHALDEAN_ORDER.index("sun")  # Sunday's own ruler
    expected_hora_23 = HORA_CHALDEAN_ORDER[(sun_idx + 23) % 7]
    assert lord == expected_hora_23
    assert lord != WEEKDAY_LORD[(monday.weekday() + 1) % 7]  # NOT Monday's hora-0 lord


def test_hora_bias_matches_benefic_malefic_classification():
    d = dt.date(2026, 8, 16)  # Sunday -- hora 0 lord is "sun" (malefic)
    sunrise, sunset = _sunrise_sunset_utc(d)
    day_len = (sunset - sunrise) / 12.0

    sun_hora_time = sunrise + 1.0
    assert hora_lord_at(sun_hora_time) == "sun"
    assert hora_bias(np.array([sun_hora_time]))[0] == -1.0  # sun is malefic

    venus_hora_time = sunrise + day_len + 1.0  # hora 1: next in Chaldean order after sun
    assert hora_lord_at(venus_hora_time) == "venus"
    assert hora_bias(np.array([venus_hora_time]))[0] == 1.0  # venus is benefic


def test_every_hora_lord_is_classified_as_exactly_one_of_benefic_or_malefic():
    for planet in HORA_CHALDEAN_ORDER:
        assert (planet in BENEFIC_PLANETS) != (planet in MALEFIC_PLANETS)
