import datetime as dt

from app.core.time_utils import to_iso_utc


def test_none_returns_none():
    assert to_iso_utc(None) is None


def test_naive_datetime_gets_explicit_utc_offset():
    """This is the exact shape SQLite hands back after a round-trip: correct
    UTC digits, but no tzinfo -- the browser must see an explicit offset or it
    silently interprets the string as local time instead of UTC.
    """
    naive = dt.datetime(2026, 8, 12, 18, 58, 21, 894673)
    result = to_iso_utc(naive)
    assert result == "2026-08-12T18:58:21.894673+00:00"


def test_aware_datetime_passes_through_unchanged():
    aware = dt.datetime(2026, 8, 12, 18, 58, 21, tzinfo=dt.timezone.utc)
    assert to_iso_utc(aware) == aware.isoformat()


def test_round_trip_parses_back_to_the_same_instant():
    """Confirms a JS-style parse of the output string recovers the same UTC
    instant the naive value represented -- the actual bug being fixed.
    """
    naive = dt.datetime(2026, 8, 12, 18, 58, 21)
    iso = to_iso_utc(naive)
    parsed = dt.datetime.fromisoformat(iso)
    assert parsed == naive.replace(tzinfo=dt.timezone.utc)
