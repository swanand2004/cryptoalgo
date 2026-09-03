from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# Delta-neutral funding harvest, priced from the live options chain.
#
# The trade: SHORT the perpetual (which is paid funding) and hold a LONG
# SYNTHETIC FORWARD in options (long call + short put, same strike and
# expiry) to cancel the price exposure. Both legs are USD-quoted on Delta,
# so unlike spot-vs-perp cash-and-carry there is no INR/FX leg.
#
#   P&L  =  funding received  -  the forward premium paid  -  execution cost
#
# Measured inputs, 2026-09-02/03:
#   - Delta funding is clamp-pinned at the 0.01%/8h interest rate on 88.6% of
#     BTC bars, i.e. ~+10.95%/yr, contractual rather than sentiment-driven.
#   - Binance funding over 7 years was positive in 8/8 years for BTC (+11.6%
#     mean) and ETH (+13.9%). SOL was positive in only 4/7 years with a
#     7-year mean of +0.17%, so it is EXCLUDED here rather than assumed to
#     behave like the majors.
#   - The live option chain implies a forward premium of ~5.0-5.5%/yr,
#     consistent across every tenor from 8 to 85 days (that consistency is
#     what identifies it as carry rather than mispricing).
#
# This module PRICES the opportunity. It does not execute: Position/
# OrderRequest carry no strike/expiry/option-type, and symbols.py resolves
# only perpetual_futures, so the execution layer cannot represent an option
# leg yet. Deliberately kept as a measurement surface until that exists.

BASE_URL = "https://api.india.delta.exchange"

# BTC and ETH only -- see the SOL note above.
CARRY_SYMBOLS = ("BTCUSD", "ETHUSD")
UNDERLYING = {"BTCUSD": "BTC", "ETHUSD": "ETH"}

# Delta's published taker fees. Both legs of the option structure plus the
# perp leg are charged, and this deliberately assumes TAKER on everything:
# an edge that only survives as a resting maker is not one to rely on.
OPTION_TAKER_FEE = 0.0003
PERP_TAKER_FEE = 0.0005
ROUND_TRIP_COST_PCT = (OPTION_TAKER_FEE * 2 + PERP_TAKER_FEE) * 100

# Below this the structure is not worth carrying: it is inside the noise of
# the bid/ask on three legs.
MIN_NET_CARRY_PCT_PER_YEAR = 2.0
# Minimum executable size on the legs you must LIFT, in contracts.
#
# Gating on open interest instead (as this did initially) measures the wrong
# thing and rejects perfectly tradable strikes. Checked live 2026-09-03: the
# best structure showed open interest of just 1-5 contracts but 4,442
# contracts on the call ask and 6,663 on the put bid -- roughly $342k and
# $513k of notional. Low OI with deep quotes simply means market makers are
# posting size at a strike few people have traded, which is a liquidity
# signal, not a warning. Contract value is 0.001 BTC, so 1 contract is only
# ~$77 of notional and OI counts in the single digits are meaningless either
# way.
MIN_QUOTE_DEPTH_CONTRACTS = 100
# Short tenors are dominated by execution cost (the same spread amortised
# over fewer days), which is exactly what the 14-day case showed: -0.95%/yr
# net against +4.11% at 60 days.
MIN_DAYS_TO_EXPIRY = 25


@dataclass
class CarryQuote:
    symbol: str
    expiry: str
    days_to_expiry: float
    strike: float
    perp_mark: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    open_interest: float
    # Annualised, all in percent.
    implied_forward_rate: float
    funding_rate: float
    gross_carry: float
    cost_drag: float
    net_carry: float
    tradable: bool
    reasons: list[str] = field(default_factory=list)
    # Executable size on the legs being lifted -- the real liquidity test,
    # as opposed to open_interest which is only trade history.
    quote_depth: float = 0.0


def _parse_expiry(ddmmyy: str) -> dt.datetime:
    """Delta option symbols carry DDMMYY, settling 12:00 UTC."""
    return dt.datetime(2000 + int(ddmmyy[4:6]), int(ddmmyy[2:4]), int(ddmmyy[:2]),
                       12, 0, tzinfo=dt.timezone.utc)


def annualised_forward_rate(synthetic_forward: float, perp_mark: float, years: float) -> float:
    """The forward premium the options imply, as a continuous annual rate.

    A dated forward legitimately sits above the perp when carry is positive;
    that gap is NOT an arbitrage. Confirmed empirically: the implied rate came
    out at ~5.0-5.5%/yr across every tenor from 8 to 85 days, and a genuine
    mispricing would not be that consistent across the curve.
    """
    if perp_mark <= 0 or years <= 0:
        return 0.0
    return (synthetic_forward / perp_mark - 1.0) / years * 100.0


def evaluate_carry(
    symbol: str,
    expiry: str,
    strike: float,
    perp_mark: float,
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    funding_rate_pct_per_year: float,
    open_interest: float,
    call_ask_size: float = 0.0,
    put_bid_size: float = 0.0,
    now: dt.datetime | None = None,
) -> CarryQuote:
    """Prices one strike/expiry as a delta-neutral carry structure.

    Uses EXECUTABLE prices, not mids: to hold the long synthetic forward you
    lift the call ask and hit the put bid. Mid-price maths flatters this trade
    badly -- on the live chain the median combined call+put spread was 0.153%
    of notional, which is most of a year's carry at these rates.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    days = (_parse_expiry(expiry) - now).total_seconds() / 86400.0
    years = max(days / 365.25, 1e-9)

    # Cost to BUY the synthetic forward: pay the call ask, receive the put bid.
    synthetic_forward = strike + (call_ask - put_bid)
    implied = annualised_forward_rate(synthetic_forward, perp_mark, years)

    gross = funding_rate_pct_per_year - implied
    # Open and unwind: the round trip is paid once each way over the holding
    # period, then annualised over however long the structure is held.
    cost_drag = ROUND_TRIP_COST_PCT * 2 * 365.25 / max(days, 1e-9)
    net = gross - cost_drag

    reasons: list[str] = []
    if days < MIN_DAYS_TO_EXPIRY:
        reasons.append(f"only {days:.0f}d to expiry (< {MIN_DAYS_TO_EXPIRY}d): cost drag dominates")
    if net < MIN_NET_CARRY_PCT_PER_YEAR:
        reasons.append(f"net carry {net:.2f}%/yr < {MIN_NET_CARRY_PCT_PER_YEAR}%/yr floor")
    # Depth on the legs actually being LIFTED to build the long synthetic
    # forward: the call ask and the put bid.
    depth = min(call_ask_size, put_bid_size)
    if depth < MIN_QUOTE_DEPTH_CONTRACTS:
        reasons.append(
            f"only {depth:.0f} contracts quoted (< {MIN_QUOTE_DEPTH_CONTRACTS}): too thin to fill")
    if call_bid <= 0 or put_bid <= 0:
        reasons.append("missing a two-sided quote on one leg")

    return CarryQuote(
        symbol=symbol, expiry=expiry, days_to_expiry=days, strike=strike,
        perp_mark=perp_mark, call_bid=call_bid, call_ask=call_ask,
        put_bid=put_bid, put_ask=put_ask, open_interest=open_interest,
        quote_depth=depth,
        implied_forward_rate=implied, funding_rate=funding_rate_pct_per_year,
        gross_carry=gross, cost_drag=cost_drag, net_carry=net,
        tradable=not reasons, reasons=reasons,
    )


async def fetch_carry_quotes(symbols: tuple[str, ...] = CARRY_SYMBOLS) -> list[CarryQuote]:
    """Prices every strike/expiry on the live chain for the given symbols."""
    quotes: list[CarryQuote] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        perp_resp = await client.get(f"{BASE_URL}/v2/tickers",
                                     params={"contract_types": "perpetual_futures"})
        perp_resp.raise_for_status()
        perps = {t["symbol"]: t for t in perp_resp.json()["result"]}

        opt_resp = await client.get(f"{BASE_URL}/v2/tickers",
                                    params={"contract_types": "call_options,put_options"})
        opt_resp.raise_for_status()
        options = opt_resp.json()["result"]

    for symbol in symbols:
        perp = perps.get(symbol)
        if perp is None:
            continue
        perp_mark = float(perp["mark_price"])
        # Delta quotes funding as percent per 8h; 3 settlements a day.
        funding_annual = float(perp.get("funding_rate") or 0.0) * 3 * 365

        chain: dict[tuple[str, float], dict] = {}
        for t in options:
            if t.get("underlying_asset_symbol") != UNDERLYING.get(symbol):
                continue
            q = t.get("quotes") or {}
            if not (q.get("best_bid") and q.get("best_ask")):
                continue
            parts = t["symbol"].split("-")
            if len(parts) != 4:
                continue
            kind, _, strike, expiry = parts
            entry = chain.setdefault((expiry, float(strike)), {})
            entry[kind] = {"bid": float(q["best_bid"]), "ask": float(q["best_ask"]),
                           "oi": float(t.get("oi") or 0),
                           "bid_size": float(q.get("bid_size") or 0),
                           "ask_size": float(q.get("ask_size") or 0)}

        for (expiry, strike), legs in chain.items():
            if "C" not in legs or "P" not in legs:
                continue
            c, p = legs["C"], legs["P"]
            try:
                quotes.append(evaluate_carry(
                    symbol=symbol, expiry=expiry, strike=strike, perp_mark=perp_mark,
                    call_bid=c["bid"], call_ask=c["ask"], put_bid=p["bid"], put_ask=p["ask"],
                    funding_rate_pct_per_year=funding_annual,
                    open_interest=min(c["oi"], p["oi"]),
                    call_ask_size=c["ask_size"], put_bid_size=p["bid_size"],
                ))
            except (ValueError, KeyError):
                continue
    return quotes


def best_quote(quotes: list[CarryQuote]) -> CarryQuote | None:
    """Highest net carry among structures that actually clear every gate."""
    tradable = [q for q in quotes if q.tradable]
    return max(tradable, key=lambda q: q.net_carry) if tradable else None
