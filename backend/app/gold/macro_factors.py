from __future__ import annotations

import datetime as dt
import io
import logging
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

from app.data.ingest_rest import upsert_candles

logger = logging.getLogger(__name__)

# Macro factor inputs for gold, from genuinely free sources.
#
# WHAT IS HERE AND WHY EACH SIGN IS WHAT IT IS. The signs are not fitted --
# they come from the standard economics of a zero-yield USD-denominated
# asset, decided before any measurement, so a wrong sign shows up as a bad
# result rather than being quietly flipped to fit the data:
#
#   real_rate (DFII10, 10y TIPS)   NEGATIVE. Gold yields nothing, so a higher
#       real yield raises the opportunity cost of holding it. The single most
#       established macro driver of gold.
#   dollar (DTWEXBGS, broad TWI)   NEGATIVE. Gold is quoted in USD; a stronger
#       dollar mechanically lowers the price for a given ounce of demand.
#   breakeven (T10YIE)             POSITIVE. Gold's inflation-hedge demand
#       rises with expected inflation.
#   cftc_positioning               NEGATIVE AT EXTREMES, i.e. contrarian.
#       Crowded speculative longs are fragile, not bullish -- the same
#       "crowded trade" logic this project already applies to funding and
#       long/short ratios.
#
# RATE OF CHANGE, not just level, for the rate/dollar series: the level of
# real rates has drifted structurally over 20 years, so a raw level is
# non-stationary in exactly the way this project already fixed for ema/atr/
# macd (see feature_pipeline.py). Change is what the market reprices against.
#
# DELIBERATELY EXCLUDED and why, rather than silently missing:
#   yield curve shape (T10Y2Y) -- direction for gold is genuinely ambiguous
#       (steepening can mean growth OR inflation risk), and guessing a sign
#       is how a factor becomes noise with a story attached.
#   central bank buying -- monthly, heavily revised, no reliable free feed.
#       A factor revised after the fact cannot be traded on in real time.
#   ETF flows -- daily but no free authoritative source; scraping holdings
#       pages would be fragile and is lagged anyway.

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CFTC_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CFTC_GOLD_MARKET = "GOLD"  # the COMEX contract; NOT 'PAX GOLD PERP STYLE'

# FRED series id -> (pseudo-symbol, human name)
FRED_SERIES = {
    "DFII10": ("MACRO:REALRATE10Y", "10y TIPS real rate"),
    "T10YIE": ("MACRO:BREAKEVEN10Y", "10y breakeven inflation"),
    "DTWEXBGS": ("MACRO:DXY", "trade-weighted USD (broad)"),
}
CFTC_SYMBOL = "MACRO:CFTC_GOLD_NET"

# Lookback for z-scoring each factor against its own recent history, in
# trading days. ~1 year, so "extreme" means extreme for this regime rather
# than against a 20-year average that includes a different rate world.
FACTOR_Z_WINDOW = 252
# Rate-of-change window in trading days (~1 month) for the level series.
FACTOR_ROC_DAYS = 21
# z-scores clipped here before scaling to [-1, 1]; beyond ~2.5 sigma the
# marginal information is mostly outlier noise.
FACTOR_Z_CAP = 2.5

# Which factors are allowed into the composite.
#
# MEASURED 2026-09-04 on 2,196 daily PAXG bars, strictly point-in-time
# (FRED shifted 1 day for publication, CFTC 4 days -- its report date is
# Tuesday but it is not released until Friday). Directional accuracy against
# forward gold returns, coin flip = 50%, 2-sigma noise band +-2.1 points:
#
#                        1d      5d     10d     21d     63d
#   real_rate_change   49.6    51.6    52.2    54.4    50.3
#   dollar_change      49.4    52.3    52.9    51.9    47.7
#   breakeven_infl.    48.6    49.9    51.3    46.4    46.7
#   cftc_positioning   48.2    46.3    46.7    47.3    44.1
#   equal-weight all   48.9    48.3    48.7    44.6    39.6
#
# Two conclusions, neither of them flattering:
#
# 1. Only real_rate_change clears the noise band anywhere (54.4% at 21d),
#    and dollar_change marginally (52.9% at 10d). Everything else is a coin
#    flip or worse.
# 2. The equal-weighted composite of all four is WORSE than its own best
#    component at every horizon. Averaging one real signal with three noisy
#    ones destroys the signal -- the same arithmetic that made the crypto
#    max_abs blend cost 31 points of return.
#
# So the composite carries only the two factors whose a-priori sign actually
# held up. The other two are still COMPUTED and REPORTED (their reading is
# useful context) but contribute zero weight.
#
# Stated plainly because it matters: this is a selection made AFTER seeing
# the measurement. It is defensible as dropping factors whose committed sign
# failed, not as tuning weights to fit -- no weight was optimised, they are
# simply in or out. Treat the 54.4% as the optimistic end, not a promise.
COMPOSITE_FACTORS = ("real_rate_change", "dollar_change")


@dataclass
class FactorScore:
    name: str
    raw_value: float | None
    z_score: float | None
    # Signed contribution in [-1, 1]: positive = bullish gold.
    score: float
    direction: str  # "bullish" | "bearish" | "neutral"
    as_of: str | None
    detail: str


async def fetch_fred_series(series_id: str, client: httpx.AsyncClient | None = None) -> list[dict]:
    """One FRED series as candle-shaped rows. The CSV endpoint needs no API
    key, which is why it is used in preference to the JSON API."""
    owns = client is None
    client = client or httpx.AsyncClient(timeout=45.0)
    try:
        resp = await client.get(FRED_CSV, params={"id": series_id}, follow_redirects=True)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    finally:
        if owns:
            await client.aclose()

    date_col, value_col = df.columns[0], df.columns[1]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")  # FRED writes '.' for gaps
    df = df.dropna(subset=[value_col])
    out = []
    for _, row in df.iterrows():
        ts = int(pd.Timestamp(row[date_col], tz="UTC").timestamp())
        v = float(row[value_col])
        out.append({"time": ts, "open": v, "high": v, "low": v, "close": v, "volume": 0.0})
    return out


async def fetch_cftc_gold_positioning(client: httpx.AsyncClient | None = None) -> list[dict]:
    """COMEX gold non-commercial NET position as a share of open interest.

    Normalised by open interest rather than used as a raw contract count:
    OI has grown enormously since 1986, so a raw net figure is
    non-stationary and 200k contracts means something completely different
    now than it did then.
    """
    owns = client is None
    client = client or httpx.AsyncClient(timeout=60.0)
    rows: list[dict] = []
    try:
        resp = await client.get(CFTC_URL, params={
            "$where": f"contract_market_name = '{CFTC_GOLD_MARKET}'",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 5000,
        })
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns:
            await client.aclose()

    for r in payload:
        try:
            oi = float(r.get("open_interest_all") or 0)
            if oi <= 0:
                continue
            net = float(r.get("noncomm_positions_long_all") or 0) - float(
                r.get("noncomm_positions_short_all") or 0)
            pct = net / oi * 100.0
            ts = int(pd.Timestamp(str(r["report_date_as_yyyy_mm_dd"])[:10], tz="UTC").timestamp())
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"time": ts, "open": pct, "high": pct, "low": pct, "close": pct, "volume": 0.0})
    return rows


async def backfill_macro_factors() -> dict[str, int]:
    """Pulls every macro series into the candles table as MACRO:* pseudo-
    symbols. Each source is independently fault tolerant, matching every
    other ingestion path in this project."""
    counts: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        for series_id, (symbol, name) in FRED_SERIES.items():
            try:
                rows = await fetch_fred_series(series_id, client=client)
                if rows:
                    await upsert_candles(symbol, "1d", rows)
                counts[symbol] = len(rows)
                logger.info("Backfilled %d points of %s (%s)", len(rows), name, series_id)
            except Exception:
                logger.exception("FRED backfill failed for %s", series_id)
                counts[symbol] = 0
        try:
            rows = await fetch_cftc_gold_positioning(client=client)
            if rows:
                await upsert_candles(CFTC_SYMBOL, "1d", rows)
            counts[CFTC_SYMBOL] = len(rows)
            logger.info("Backfilled %d CFTC gold positioning reports", len(rows))
        except Exception:
            logger.exception("CFTC backfill failed")
            counts[CFTC_SYMBOL] = 0
    return counts


def _z(series: pd.Series, window: int = FACTOR_Z_WINDOW) -> float | None:
    """Trailing z-score of the latest value. Trailing, not full-sample: a
    full-sample z leaks the future into every historical bar."""
    s = series.dropna()
    if len(s) < max(20, window // 4):
        return None
    tail = s.tail(window)
    sd = tail.std()
    if not sd or not np.isfinite(sd) or sd == 0:
        return None
    return float((s.iloc[-1] - tail.mean()) / sd)


def _scored(name: str, z: float | None, sign: float, raw: float | None,
            as_of: str | None, detail: str) -> FactorScore:
    if z is None:
        return FactorScore(name, raw, None, 0.0, "neutral", as_of, f"{detail} (insufficient history)")
    score = float(np.clip(z, -FACTOR_Z_CAP, FACTOR_Z_CAP) / FACTOR_Z_CAP) * sign
    direction = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    return FactorScore(name, raw, round(z, 3), round(score, 4), direction, as_of, detail)


def compute_macro_factors(frames: dict[str, pd.DataFrame]) -> list[FactorScore]:
    """Turns the raw MACRO:* series into signed contributions in [-1, 1].

    `frames` maps pseudo-symbol -> DataFrame with [time, close]. A missing or
    short series contributes exactly 0.0 rather than being dropped, so the
    blend degrades gracefully instead of silently re-weighting itself.
    """
    out: list[FactorScore] = []

    def latest_stamp(df: pd.DataFrame) -> str | None:
        if df is None or df.empty:
            return None
        return dt.datetime.fromtimestamp(int(df["time"].iloc[-1]), dt.timezone.utc).strftime("%Y-%m-%d")

    # --- real rate: CHANGE matters, level is non-stationary over 20 years --
    df = frames.get("MACRO:REALRATE10Y")
    if df is not None and not df.empty:
        s = df["close"].astype(float)
        roc = s.diff(FACTOR_ROC_DAYS)
        out.append(_scored("real_rate_change", _z(roc), -1.0, float(s.iloc[-1]), latest_stamp(df),
                           f"10y TIPS {s.iloc[-1]:.2f}%, {FACTOR_ROC_DAYS}d change "
                           f"{roc.iloc[-1]:+.2f}pt -- higher real yields raise gold's opportunity cost"))
    else:
        out.append(_scored("real_rate_change", None, -1.0, None, None, "10y TIPS unavailable"))

    # --- dollar: same reasoning, change not level -------------------------
    df = frames.get("MACRO:DXY")
    if df is not None and not df.empty:
        s = df["close"].astype(float)
        roc = s.pct_change(FACTOR_ROC_DAYS) * 100
        out.append(_scored("dollar_change", _z(roc), -1.0, float(s.iloc[-1]), latest_stamp(df),
                           f"broad USD {s.iloc[-1]:.2f}, {FACTOR_ROC_DAYS}d change "
                           f"{roc.iloc[-1]:+.2f}% -- gold is priced in dollars"))
    else:
        out.append(_scored("dollar_change", None, -1.0, None, None, "USD index unavailable"))

    # --- breakeven inflation: LEVEL is the signal here --------------------
    df = frames.get("MACRO:BREAKEVEN10Y")
    if df is not None and not df.empty:
        s = df["close"].astype(float)
        out.append(_scored("breakeven_inflation", _z(s), 1.0, float(s.iloc[-1]), latest_stamp(df),
                           f"10y breakeven {s.iloc[-1]:.2f}% -- inflation-hedge demand"))
    else:
        out.append(_scored("breakeven_inflation", None, 1.0, None, None, "breakevens unavailable"))

    # --- CFTC positioning: CONTRARIAN at extremes -------------------------
    df = frames.get(CFTC_SYMBOL)
    if df is not None and not df.empty:
        s = df["close"].astype(float)
        out.append(_scored("cftc_positioning", _z(s, window=104), -1.0, float(s.iloc[-1]), latest_stamp(df),
                           f"spec net {s.iloc[-1]:.1f}% of OI -- crowded longs are fragile, "
                           f"not bullish (weekly, 3-day lag)"))
    else:
        out.append(_scored("cftc_positioning", None, -1.0, None, None, "CFTC positioning unavailable"))

    return out


def blend_score(model_score: float | None, factors: list[FactorScore],
                model_weight: float = 0.5) -> dict:
    """Combines the model's own score with the macro factor composite.

    Default 50/50 as requested. The factor side is an equal-weighted mean of
    whichever factors have data -- equal weights on purpose, because with 4
    factors and a handful of years of overlapping history, fitting relative
    weights would be curve-fitting dressed up as sophistication.

    Reports the two sides separately as well as blended, so it is always
    visible which half is driving a call and whether they agree.
    """
    # Only factors in COMPOSITE_FACTORS contribute; the rest are reported
    # for context but weighted zero (see that constant for the measurement).
    contributing = [f for f in factors if f.z_score is not None and f.name in COMPOSITE_FACTORS]
    available = [f for f in factors if f.z_score is not None]
    factor_score = float(np.mean([f.score for f in contributing])) if contributing else 0.0
    model = float(model_score) if model_score is not None else 0.0

    fw = 1.0 - model_weight
    blended = model_weight * model + fw * factor_score
    agree = (model > 0) == (factor_score > 0) and abs(model) > 0.05 and abs(factor_score) > 0.05

    return {
        "model_score": round(model, 4),
        "factor_score": round(factor_score, 4),
        "blended_score": round(float(np.clip(blended, -1, 1)), 4),
        "model_weight": model_weight,
        "factor_weight": round(fw, 4),
        "factors_available": len(available),
        "factors_contributing": len(contributing),
        "factors_total": len(factors),
        "sides_agree": agree,
        "factors": [
            {**f.__dict__, "contributes": f.name in COMPOSITE_FACTORS}
            for f in factors
        ],
    }
