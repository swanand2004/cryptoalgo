from __future__ import annotations

import datetime as dt

import numpy as np

from app.astro.panchang import tithi_index

_UNIX_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

# Real, published academic research on gold's monthly seasonality -- NOT
# this system's own model output (gold's own price history here is only
# ~4 months deep, nowhere near enough to have learned this itself yet).
# Surfaced as documented background context, sourced, rather than silently
# folded into a feature the model can't actually have derived on its own:
#   - Baur, D.G. (2013), "The autumn effect of gold", Journal of Banking &
#     Finance / SSRN: September and November are the only calendar months
#     with a POSITIVE AND STATISTICALLY SIGNIFICANT average gold return
#     (1980-2010 sample) -- attributed partly to Indian wedding-season/
#     festival (Dhanteras/Diwali) gold demand.
#   - Broader "winter effect" findings (multiple sources): Dec/Jan/Feb
#     average positive, May/Jun/Jul average negative; Jan the single
#     strongest month historically; Mar/Jun/Oct the weakest.
_SEASONALITY_BY_MONTH: dict[int, dict] = {
    1: {"effect": "positive", "note": "Historically gold's strongest month on average (\"winter effect\")."},
    2: {"effect": "positive", "note": "Part of the documented winter-effect window (Dec-Feb average positive)."},
    3: {"effect": "negative", "note": "One of the three historically weakest months for gold."},
    4: {"effect": "neutral", "note": "No strong documented seasonal tilt either way."},
    5: {"effect": "negative", "note": "Start of the historically weak May-July stretch."},
    6: {"effect": "negative", "note": "Historically weak month; also one of the three weakest overall."},
    7: {"effect": "negative", "note": "End of the historically weak May-July stretch."},
    8: {"effect": "neutral", "note": "No strong documented seasonal tilt; some sources note early strength."},
    9: {"effect": "positive", "note": "One of only two months with a statistically significant positive \"autumn effect\" (Baur, 2013)."},
    10: {"effect": "negative", "note": "One of the three historically weakest months, despite sitting between the two autumn-effect months."},
    11: {"effect": "positive", "note": "The other statistically significant positive \"autumn effect\" month (Baur, 2013) -- coincides with Diwali/Dhanteras gold-buying season in India."},
    12: {"effect": "positive", "note": "Part of the documented winter-effect window (Dec-Feb average positive)."},
}

SEASONALITY_SOURCE = "Baur, D.G. (2013) \"The Autumn Effect of Gold\"; broader winter-effect studies (see Sources)"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def documented_seasonality_note(now_utc: dt.datetime | None = None) -> dict:
    """This calendar month's documented historical gold-return tendency,
    from published research -- see _SEASONALITY_BY_MONTH's citations. An
    informational overlay, not a claim this system's own (currently ~4-
    month-deep) price history has verified the effect itself.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    entry = _SEASONALITY_BY_MONTH[now.month]
    return {
        "month": MONTH_NAMES[now.month - 1],
        "effect": entry["effect"],
        "note": entry["note"],
        "source": SEASONALITY_SOURCE,
    }


def _next_tithi_occurrence(target_tithi: int, now_utc: dt.datetime, horizon_days: int = 35) -> dict:
    """Forward-scans (6-hour steps, then refines to the hour) for the next
    moment `tithi_index` equals `target_tithi` -- a real Panchang calendar
    calculation (app/astro/panchang.tithi_index), not an approximation.
    """
    start = now_utc.timestamp()
    coarse_times = start + np.arange(0, horizon_days * 86400, 6 * 3600)
    coarse_idx = tithi_index(coarse_times)
    hits = np.where(coarse_idx == target_tithi)[0]
    if len(hits) == 0:
        return {"days_away": None, "date": None}

    # Refine within the 6h window right before the first coarse hit, hourly.
    window_start = coarse_times[max(0, hits[0] - 1)]
    fine_times = window_start + np.arange(0, 7 * 3600, 3600)
    fine_idx = tithi_index(fine_times)
    fine_hits = np.where(fine_idx == target_tithi)[0]
    hit_time = fine_times[fine_hits[0]] if len(fine_hits) else coarse_times[hits[0]]

    hit_dt = _UNIX_EPOCH + dt.timedelta(seconds=float(hit_time))
    days_away = (hit_time - start) / 86400.0
    return {"days_away": float(round(days_away, 1)), "date": hit_dt.date().isoformat()}


def upcoming_gold_buying_tithis(now_utc: dt.datetime | None = None) -> dict:
    """Two real, well-documented Hindu-calendar gold-buying occasions,
    Akshaya Tritiya and Dhanteras, are each tied to a specific TITHI
    (Shukla Tritiya and Krishna Trayodashi respectively -- verified via web
    research, see Sources) that recurs every lunar month, not just once a
    year. This surfaces the next recurrence of EACH tithi as a monthly
    proxy -- clearly not literally "Akshaya Tritiya" (which is specifically
    the Vaishakha-month occurrence, a once-a-year event this system doesn't
    attempt to identify), but the same underlying auspicious-for-gold lunar
    day, real and computable every month.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    return {
        "shukla_tritiya": {
            **_next_tithi_occurrence(2, now),  # Tritiya (idx 2) of Shukla Paksha (idx 0-14)
            "label": "Shukla Tritiya (Akshaya Tritiya's tithi)",
        },
        "krishna_trayodashi": {
            **_next_tithi_occurrence(27, now),  # Trayodashi (idx 12) of Krishna Paksha (15 + 12 = 27)
            "label": "Krishna Trayodashi (Dhanteras' tithi)",
        },
    }


def build_gold_events(now_utc: dt.datetime | None = None) -> dict:
    """Everything the Gold page's Events panel shows -- one call, one
    place, so engine.py's snapshot builder doesn't need to know the details
    of either sub-calculation.
    """
    from app.gold.macro_calendar import next_macro_event

    return {
        "seasonality": documented_seasonality_note(now_utc),
        "gold_buying_tithis": upcoming_gold_buying_tithis(now_utc),
        "next_macro_event": next_macro_event(now_utc),
    }
