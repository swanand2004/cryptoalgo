from __future__ import annotations

import datetime as dt
from typing import Optional


def to_iso_utc(value: Optional[dt.datetime]) -> Optional[str]:
    """Serializes a datetime to an ISO string with an explicit UTC offset.

    Every datetime in this app is created as UTC (`datetime.now(timezone.utc)`
    or the ORM's `_utcnow`), but SQLite/SQLAlchemy's plain DateTime column
    strips the tzinfo on round-trip -- by the time a value comes back from the
    DB it's a naive datetime whose *digits* are still correct UTC, just
    missing the marker. Calling bare `.isoformat()` on that naive value
    produces a string with no timezone suffix, which browsers then parse as
    LOCAL time instead of UTC, silently shifting every timestamp shown in the
    UI. This restamps the (already-correct) UTC digits with an explicit
    offset so `new Date(...)` on the frontend parses them correctly.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()
