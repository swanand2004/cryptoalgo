import datetime as dt

from app.gold.events import (
    build_gold_events,
    documented_seasonality_note,
    upcoming_gold_buying_tithis,
)


def test_documented_seasonality_note_covers_every_month():
    for month in range(1, 13):
        now = dt.datetime(2026, month, 15, tzinfo=dt.timezone.utc)
        note = documented_seasonality_note(now)
        assert note["month"]
        assert note["effect"] in ("positive", "negative", "neutral")
        assert note["note"]
        assert "Baur" in note["source"]


def test_september_and_november_are_the_documented_autumn_effect_months():
    """The one specific, statistically-significant published finding (Baur,
    2013) -- must not silently drift if the lookup table is ever edited."""
    sept = documented_seasonality_note(dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc))
    nov = documented_seasonality_note(dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc))
    assert sept["effect"] == "positive"
    assert nov["effect"] == "positive"
    assert "autumn effect" in sept["note"].lower()


def test_upcoming_gold_buying_tithis_returns_both_and_within_one_lunar_month():
    now = dt.datetime(2026, 8, 21, tzinfo=dt.timezone.utc)
    result = upcoming_gold_buying_tithis(now)

    assert set(result.keys()) == {"shukla_tritiya", "krishna_trayodashi"}
    for key, entry in result.items():
        assert entry["days_away"] is not None
        # A tithi recurs every ~29.5 days -- "next occurrence" must always
        # be within one lunar month, never wildly far off.
        assert 0 <= entry["days_away"] <= 32
        assert entry["date"] is not None
        assert "label" in entry


def test_build_gold_events_bundles_all_sections():
    events = build_gold_events(dt.datetime(2026, 8, 21, tzinfo=dt.timezone.utc))
    assert "seasonality" in events
    assert "gold_buying_tithis" in events
    assert "next_macro_event" in events
    assert events["next_macro_event"]["event"] in ("FOMC", "CPI", "NFP")
