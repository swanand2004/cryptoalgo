"""Delta Exchange India option chain, from PUBLIC endpoints only (no auth).

`GET /v2/tickers?contract_types=call_options,put_options&underlying_asset_symbols=BTC`
returns one row per listed option. Prices are quoted per 1 unit of the
underlying (1 BTC / 1 ETH); one contract is `contract_value` of that (0.001 BTC,
0.01 ETH), so a contract's premium in USD is price x contract_value.

Expiry is encoded in the symbol (C-BTC-90000-161026 = 16 Oct 2026) and every
Delta option settles at 12:00 UTC on that day (confirmed from an expired
product's `settlement_time`).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

UNDERLYINGS = ("BTC", "ETH")
CACHE_TTL_SECONDS = 60.0
SETTLEMENT_HOUR_UTC = 12

_cache: dict[str, tuple[float, list[dict]]] = {}
_locks: dict[str, asyncio.Lock] = {}


def _base_url() -> str:
    # Always production: testnet option quotes are meaningless for a forward
    # record, and these are public market-data reads only.
    return get_settings().delta_rest_url_production.rstrip("/")


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_expiry(symbol: str) -> dt.datetime | None:
    """C-BTC-90000-161026 -> 2026-10-16 12:00 UTC. None if it does not parse."""
    try:
        code = symbol.rsplit("-", 1)[-1]
        if len(code) != 6:
            return None
        day, month, year = int(code[0:2]), int(code[2:4]), 2000 + int(code[4:6])
        return dt.datetime(year, month, day, SETTLEMENT_HOUR_UTC, tzinfo=dt.timezone.utc)
    except (ValueError, IndexError):
        return None


def parse_ticker(row: dict) -> dict | None:
    """One raw Delta option ticker -> a flat, typed row. None if unusable."""
    ctype = row.get("contract_type")
    if ctype not in ("call_options", "put_options"):
        return None
    symbol = str(row.get("symbol") or "")
    expiry = parse_expiry(symbol)
    strike = _f(row.get("strike_price"))
    if expiry is None or strike is None:
        return None
    quotes = row.get("quotes") or {}
    greeks = row.get("greeks") or {}
    bid, ask = _f(quotes.get("best_bid")), _f(quotes.get("best_ask"))
    return {
        "symbol": symbol,
        "type": "call" if ctype == "call_options" else "put",
        "strike": strike,
        "expiry": expiry,
        "mark": _f(row.get("mark_price")),
        "iv": _f(quotes.get("mark_iv")) if quotes.get("mark_iv") is not None else _f(row.get("mark_vol")),
        "bid": bid if bid and bid > 0 else None,
        "ask": ask if ask and ask > 0 else None,
        "bid_iv": _f(quotes.get("bid_iv")),
        "ask_iv": _f(quotes.get("ask_iv")),
        "delta": _f(greeks.get("delta")),
        "gamma": _f(greeks.get("gamma")),
        "theta": _f(greeks.get("theta")),
        "vega": _f(greeks.get("vega")),
        "oi_contracts": _f(row.get("oi_contracts")),
        "oi": _f(row.get("oi")),
        "volume": _f(row.get("volume")),
        "spot": _f(row.get("spot_price")) or _f(greeks.get("spot")),
        "contract_value": _f(row.get("contract_value")),
    }


async def _fetch_raw(underlying: str) -> list[dict]:
    url = f"{_base_url()}/v2/tickers"
    params = {"contract_types": "call_options,put_options", "underlying_asset_symbols": underlying}
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=20.0)) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
    if not payload.get("success", True):
        raise RuntimeError(f"Delta tickers failed: {payload}")
    return list(payload.get("result") or [])


async def get_chain(underlying: str, *, force: bool = False) -> tuple[list[dict], float]:
    """Parsed option rows for one underlying plus the fetch time (unix s).

    Cached ~60 s per underlying. If a refresh fails and an older copy exists,
    the older copy is returned (callers can show its age); with no copy at
    all, the error propagates.
    """
    underlying = underlying.upper()
    if underlying not in UNDERLYINGS:
        raise ValueError(f"unsupported underlying {underlying!r}")
    lock = _locks.setdefault(underlying, asyncio.Lock())
    async with lock:
        cached = _cache.get(underlying)
        if not force and cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1], cached[0]
        try:
            raw = await _fetch_raw(underlying)
        except Exception:
            if cached:
                logger.warning("Option chain refresh failed for %s -- serving the %0.fs old copy",
                               underlying, time.time() - cached[0], exc_info=True)
                return cached[1], cached[0]
            raise
        rows = [r for r in (parse_ticker(x) for x in raw) if r is not None]
        fetched = time.time()
        _cache[underlying] = (fetched, rows)
        return rows, fetched


def build_chain_view(rows: list[dict], *, expiry_code: str | None = None, width: int = 10,
                     now: dt.datetime | None = None) -> dict:
    """Pure: rows -> the page's view. Expiry list, then for one expiry the
    `width` strikes either side of spot with call and put side by side."""
    now = now or dt.datetime.now(dt.timezone.utc)
    live = [r for r in rows if r["expiry"] > now]
    expiries = sorted({r["expiry"] for r in live})
    spot = next((r["spot"] for r in live if r.get("spot")), None)

    def code_of(e: dt.datetime) -> str:
        return e.strftime("%d%m%y")

    exp_list = [
        {"code": code_of(e), "expiry": e.isoformat(), "days": round((e - now).total_seconds() / 86400, 2),
         "count": sum(1 for r in live if r["expiry"] == e)}
        for e in expiries
    ]
    chosen = next((e for e in expiries if code_of(e) == expiry_code), None) if expiry_code else None
    if chosen is None and expiries:
        # Default: the nearest expiry with at least a day left, else the nearest.
        chosen = next((e for e in expiries if (e - now).total_seconds() >= 86400), expiries[0])

    strikes_view: list[dict] = []
    if chosen is not None:
        by_strike: dict[float, dict] = {}
        for r in live:
            if r["expiry"] != chosen:
                continue
            slot = by_strike.setdefault(r["strike"], {"strike": r["strike"], "call": None, "put": None})
            slot[r["type"]] = _public_row(r)
        ordered = sorted(by_strike.values(), key=lambda s: s["strike"])
        if spot is not None and ordered:
            atm = min(range(len(ordered)), key=lambda i: abs(ordered[i]["strike"] - spot))
            lo, hi = max(0, atm - width), min(len(ordered), atm + width + 1)
            ordered = ordered[lo:hi]
        strikes_view = ordered

    return {
        "spot": spot,
        "expiries": exp_list,
        "expiry": code_of(chosen) if chosen else None,
        "expiry_time": chosen.isoformat() if chosen else None,
        "contract_value": next((r["contract_value"] for r in live if r.get("contract_value")), None),
        "strikes": strikes_view,
    }


def _public_row(r: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, dt.datetime) else v) for k, v in r.items()}


async def fetch_product(symbol: str) -> dict | None:
    """Public product record -- for an expired option it carries `state` and
    `settlement_price` (the exchange's own intrinsic per underlying unit)."""
    url = f"{_base_url()}/v2/products/{symbol}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=20.0)) as client:
        resp = await client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
    return payload.get("result") if isinstance(payload, dict) else None
