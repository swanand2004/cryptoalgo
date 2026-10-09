"""The PAPER "S7" options tracker. Never places an order.

Pre-registered rules (do not tune them after looking at results):
  * Every Friday 12:00 UTC, for BTC and ETH, read the live trend-book signal
    (runtime.live_signals; fallback: TrendLadderStrategy on stored 1h candles).
  * signal > +0.2  -> buy the ~7-day call nearest +0.40 delta;
    signal < -0.2  -> buy the ~7-day put nearest -0.40 delta; otherwise nothing.
  * 1 contract (0.001 BTC / 0.01 ETH). Premium = best ask, or mark if no ask.
  * Fee = min(0.01% x notional, 3.5% x premium) x 1.18 (GST), charged on entry
    and on settlement only when the option finishes in the money.
  * Hold to expiry; settle at intrinsic value from the underlying at expiry.

The 2026-10-05 study found no options strategy that beats costs, so this is a
forward record only -- it decides whether S7 ever deserves real money.
"""

from __future__ import annotations

import datetime as dt
import logging
import math

from sqlalchemy import select

from app.data.db import SessionLocal
from app.data.models import Candle, OptionsPaperTrade
from app.options.chain import UNDERLYINGS, fetch_product, get_chain

logger = logging.getLogger(__name__)

SIGNAL_THRESHOLD = 0.2
TARGET_DELTA = 0.40
TARGET_DAYS = 7.0
MIN_DAYS = 2.0  # never "buy" an option that is expiring today/tomorrow
FEE_NOTIONAL_RATE = 0.0001  # 0.01% of underlying notional
FEE_PREMIUM_CAP = 0.035  # capped at 3.5% of the premium
GST = 1.18
ENTRY_WEEKDAY = 4  # Friday
ENTRY_HOUR_UTC = 12
ENTRY_WINDOW = dt.timedelta(hours=1)  # retries inside the hour if the first try fails
SETTLE_GRACE = dt.timedelta(minutes=5)  # let Delta publish the settlement price
CANDLE_FALLBACK_AFTER = dt.timedelta(hours=2)  # then settle off the BINANCE 1h bar
DEFAULT_CONTRACT_VALUE = {"BTC": 0.001, "ETH": 0.01}

NOTE = "Paper only. Our backtest found no options strategy that beats costs; this tracks S7 forward before any real money."


# ---------------------------------------------------------------- pure rules

def decide(signal: float | None, threshold: float = SIGNAL_THRESHOLD) -> str:
    """"CALL" above +threshold, "PUT" below -threshold, else "NONE"."""
    if signal is None or not math.isfinite(signal):
        return "NONE"
    if signal > threshold:
        return "CALL"
    if signal < -threshold:
        return "PUT"
    return "NONE"


def entry_price(row: dict) -> tuple[float | None, str | None]:
    """Premium per underlying unit: the best ask, or the mark if there is no ask."""
    ask, mark = row.get("ask"), row.get("mark")
    if ask is not None and ask > 0:
        return float(ask), "ask"
    if mark is not None and mark > 0:
        return float(mark), "mark"
    return None, None


def select_contract(rows: list[dict], decision: str, now: dt.datetime, *,
                    target_days: float = TARGET_DAYS, target_delta: float = TARGET_DELTA,
                    min_days: float = MIN_DAYS) -> dict | None:
    """The option S7 buys: the expiry whose time left is closest to
    `target_days` (at least `min_days` away), then within it the call nearest
    +target_delta or the put nearest -target_delta that has a usable price.
    Ties go to the earlier expiry / lower strike, so the pick is deterministic."""
    if decision not in ("CALL", "PUT"):
        return None
    kind = "call" if decision == "CALL" else "put"
    want = target_delta if kind == "call" else -target_delta
    usable = [
        r for r in rows
        if r.get("type") == kind and r.get("delta") is not None and entry_price(r)[0] is not None
        and (r["expiry"] - now).total_seconds() / 86400 >= min_days
    ]
    if not usable:
        return None
    expiries = sorted({r["expiry"] for r in usable})
    best_exp = min(expiries, key=lambda e: (abs((e - now).total_seconds() / 86400 - target_days), e))
    same = [r for r in usable if r["expiry"] == best_exp]
    return min(same, key=lambda r: (abs(r["delta"] - want), r["strike"]))


def option_fee(notional_usd: float, premium_usd: float) -> float:
    """Delta India option fee incl. 18% GST: min(0.01% notional, 3.5% premium) x 1.18."""
    return min(FEE_NOTIONAL_RATE * max(notional_usd, 0.0), FEE_PREMIUM_CAP * max(premium_usd, 0.0)) * GST


def intrinsic(kind: str, strike: float, underlying: float) -> float:
    """Per underlying unit. kind is "call"/"CALL" or "put"/"PUT"."""
    if kind.lower() == "call":
        return max(0.0, underlying - strike)
    return max(0.0, strike - underlying)


def settle_pnl(*, contract_value: float, premium: float, entry_fee: float,
               settle_value: float, underlying: float | None) -> dict:
    """Net USD P&L of 1 long option contract held to expiry.

    `premium` and `settle_value` are per underlying unit. The settlement fee is
    charged only when the option finishes in the money, on the settlement
    value as the "premium" and the underlying notional at expiry."""
    paid = premium * contract_value
    received = max(settle_value, 0.0) * contract_value
    settle_fee = 0.0
    if received > 0:
        notional = (underlying or 0.0) * contract_value
        settle_fee = option_fee(notional, received)
    pnl = received - settle_fee - paid - entry_fee
    cost = paid + entry_fee
    return {
        "settle_fee": settle_fee,
        "pnl_usd": pnl,
        "pnl_pct": pnl / cost if cost > 0 else None,
    }


def in_entry_window(now: dt.datetime) -> bool:
    start = now.replace(hour=ENTRY_HOUR_UTC, minute=0, second=0, microsecond=0)
    return now.weekday() == ENTRY_WEEKDAY and start <= now < start + ENTRY_WINDOW


def next_entry_time(now: dt.datetime) -> dt.datetime:
    start = now.replace(hour=ENTRY_HOUR_UTC, minute=0, second=0, microsecond=0)
    days = (ENTRY_WEEKDAY - now.weekday()) % 7
    cand = start + dt.timedelta(days=days)
    if cand + ENTRY_WINDOW <= now:
        cand += dt.timedelta(days=7)
    return cand


def _utc(d: dt.datetime | None) -> dt.datetime | None:
    if d is None:
        return None
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d.astimezone(dt.timezone.utc)


# ------------------------------------------------------------------ signal

async def read_signal(runtime, underlying: str) -> tuple[float | None, str | None]:
    """The trend book's current signal for {underlying}USD: the value the live
    trading tick published, else TrendLadderStrategy on the stored BINANCE 1h
    frame (closed bars only, refused if the newest bar is over 3h old)."""
    symbol = f"{underlying}USD"
    live = getattr(runtime, "live_signals", None) or {}
    val = live.get(symbol) if isinstance(live, dict) else None
    if val is not None:
        try:
            f = float(val)
            if math.isfinite(f):
                return f, "live"
        except (TypeError, ValueError):
            pass
    try:
        from app.data.query import load_ladder_frame
        from app.strategy.trend_ladder import TrendLadderStrategy
        from app.trading_loop import drop_incomplete_last_bar

        frame = await load_ladder_frame(symbol)
        if frame.empty:
            return None, None
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
        frame = drop_incomplete_last_bar(frame, resolution_seconds=3600, now_ts=now_ts).reset_index(drop=True)
        strategy = TrendLadderStrategy()
        if len(frame) < strategy.required_warmup_bars() + 5:
            return None, None
        if now_ts - float(frame["time"].iloc[-1]) > 3 * 3600:
            return None, None
        sig = float(strategy.generate_signal(frame).iloc[-1])
        return (sig, "trend_ladder") if math.isfinite(sig) else (None, None)
    except Exception:
        logger.exception("S7: trend_ladder fallback signal failed for %s", symbol)
        return None, None


# ------------------------------------------------------------------ the job

async def run_s7_tick(runtime, now: dt.datetime | None = None) -> None:
    """Scheduler entry point (every 5 min): settle anything expired, and in the
    Friday 12:00-13:00 UTC window record this week's decision per underlying."""
    now = now or dt.datetime.now(dt.timezone.utc)
    await settle_due(now)
    if in_entry_window(now):
        for underlying in UNDERLYINGS:
            try:
                await enter_week(runtime, underlying, now)
            except Exception:
                logger.exception("S7: entry failed for %s -- will retry inside the window", underlying)


async def enter_week(runtime, underlying: str, now: dt.datetime) -> OptionsPaperTrade | None:
    week = now.date().isoformat()
    async with SessionLocal() as session:
        exists = await session.execute(
            select(OptionsPaperTrade.id).where(OptionsPaperTrade.week == week, OptionsPaperTrade.underlying == underlying)
        )
        if exists.scalar_one_or_none() is not None:
            return None

    signal, source = await read_signal(runtime, underlying)
    if signal is None:
        logger.warning("S7: no trend signal for %s yet -- retrying next tick", underlying)
        return None
    decision = decide(signal)
    row = OptionsPaperTrade(week=week, underlying=underlying, signal=signal, signal_source=source,
                            decision=decision, entry_time=now, is_open=False)
    if decision == "NONE":
        row.note = f"|signal| <= {SIGNAL_THRESHOLD}: no trade"
    else:
        chain, _ = await get_chain(underlying, force=True)  # raises -> retried next tick
        pick = select_contract(chain, decision, now)
        if pick is None:
            row.decision = "NONE"
            row.note = f"no {decision.lower()} met the rule (expiry >= {MIN_DAYS:g}d with a delta and a price)"
        else:
            price, price_src = entry_price(pick)
            cv = pick.get("contract_value") or DEFAULT_CONTRACT_VALUE[underlying]
            spot = pick.get("spot") or 0.0
            row.symbol = pick["symbol"]
            row.strike = pick["strike"]
            row.expiry = pick["expiry"]
            row.contract_value = cv
            row.spot_at_entry = spot or None
            row.delta_at_entry = pick["delta"]
            row.iv_at_entry = pick.get("iv")
            row.premium = price
            row.premium_source = price_src
            row.entry_fee = option_fee(spot * cv, price * cv)
            row.is_open = True
    async with SessionLocal() as session:
        session.add(row)
        await session.commit()
    logger.info("S7 paper %s %s: signal %.3f (%s) -> %s %s", week, underlying, signal, source, row.decision,
                row.symbol or "")
    return row


async def _binance_close_at(underlying: str, at: dt.datetime) -> float | None:
    """Close of the BINANCE:{U}USD 1h bar that ends exactly at `at`."""
    ts = int(at.timestamp()) - 3600
    async with SessionLocal() as session:
        res = await session.execute(
            select(Candle.close).where(Candle.symbol == f"BINANCE:{underlying}USD", Candle.resolution == "1h",
                                       Candle.time == ts)
        )
        val = res.scalars().first()
    return float(val) if val is not None else None


async def settle_due(now: dt.datetime) -> int:
    """Settles every open paper option whose expiry has passed. Prefers
    Delta's own published settlement price; falls back to the BINANCE 1h close
    at expiry once that is 2h overdue. Returns how many were settled."""
    async with SessionLocal() as session:
        res = await session.execute(select(OptionsPaperTrade).where(OptionsPaperTrade.is_open.is_(True)))
        open_rows = list(res.scalars().all())
    settled = 0
    for t in open_rows:
        expiry = _utc(t.expiry)
        if expiry is None or now < expiry + SETTLE_GRACE:
            continue
        kind = "call" if t.decision == "CALL" else "put"
        settle_value = underlying = None
        source = None
        try:
            product = await fetch_product(t.symbol)
        except Exception:
            logger.warning("S7: could not read product %s", t.symbol, exc_info=True)
            product = None
        sp = (product or {}).get("settlement_price")
        if product and product.get("state") == "expired" and sp not in (None, ""):
            settle_value = max(float(sp), 0.0)
            source = "delta"
            if settle_value > 0:
                underlying = t.strike + settle_value if kind == "call" else t.strike - settle_value
            else:
                underlying = await _binance_close_at(t.underlying, expiry)
        elif now >= expiry + CANDLE_FALLBACK_AFTER:
            underlying = await _binance_close_at(t.underlying, expiry)
            if underlying is not None:
                settle_value = intrinsic(kind, t.strike, underlying)
                source = "binance_1h"
        if settle_value is None:
            continue
        out = settle_pnl(contract_value=t.contract_value, premium=t.premium, entry_fee=t.entry_fee or 0.0,
                         settle_value=settle_value, underlying=underlying)
        async with SessionLocal() as session:
            row = await session.get(OptionsPaperTrade, t.id)
            if row is None or not row.is_open:
                continue
            row.is_open = False
            row.settled_at = now
            row.settle_value = settle_value
            row.underlying_at_expiry = underlying
            row.settle_source = source
            row.settle_fee = out["settle_fee"]
            row.pnl_usd = out["pnl_usd"]
            row.pnl_pct = out["pnl_pct"]
            await session.commit()
        settled += 1
        logger.info("S7 paper settled %s at %.2f (%s): P&L %.4f USD", t.symbol, settle_value, source, out["pnl_usd"])
    return settled


# ---------------------------------------------------------------- snapshot

def _row_json(t: OptionsPaperTrade) -> dict:
    def iso(d):
        d = _utc(d)
        return d.isoformat() if d else None

    cost = None
    if t.premium is not None and t.contract_value is not None:
        cost = t.premium * t.contract_value + (t.entry_fee or 0.0)
    return {
        "id": t.id, "week": t.week, "underlying": t.underlying, "signal": t.signal, "signal_source": t.signal_source,
        "decision": t.decision, "note": t.note, "entry_time": iso(t.entry_time), "symbol": t.symbol,
        "strike": t.strike, "expiry": iso(t.expiry), "contract_value": t.contract_value,
        "spot_at_entry": t.spot_at_entry, "delta_at_entry": t.delta_at_entry, "iv_at_entry": t.iv_at_entry,
        "premium": t.premium, "premium_source": t.premium_source, "entry_fee": t.entry_fee, "cost_usd": cost,
        "is_open": t.is_open, "settled_at": iso(t.settled_at), "underlying_at_expiry": t.underlying_at_expiry,
        "settle_value": t.settle_value, "settle_source": t.settle_source, "settle_fee": t.settle_fee,
        "pnl_usd": t.pnl_usd, "pnl_pct": t.pnl_pct,
    }


async def s7_snapshot(now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    async with SessionLocal() as session:
        res = await session.execute(
            select(OptionsPaperTrade).order_by(OptionsPaperTrade.week.desc(), OptionsPaperTrade.underlying)
        )
        rows = [_row_json(t) for t in res.scalars().all()]
    settled = [r for r in rows if r["decision"] != "NONE" and not r["is_open"] and r["pnl_usd"] is not None]
    total_pnl = sum(r["pnl_usd"] for r in settled)
    total_cost = sum(r["cost_usd"] or 0.0 for r in settled)
    return {
        "note": NOTE,
        "rules": {
            "signal_threshold": SIGNAL_THRESHOLD, "target_delta": TARGET_DELTA, "target_days": TARGET_DAYS,
            "entry": "Fridays 12:00 UTC", "contracts": "1 (0.001 BTC / 0.01 ETH)",
            "premium": "best ask, else mark",
            "fee": "min(0.01% notional, 3.5% premium) x 1.18, on entry and on settlement if ITM",
            "exit": "hold to expiry, settle at intrinsic",
        },
        "next_entry": next_entry_time(now).isoformat(),
        "stats": {
            "weeks": len({r["week"] for r in rows}),
            "trades": sum(1 for r in rows if r["decision"] != "NONE"),
            "open": sum(1 for r in rows if r["is_open"]),
            "settled": len(settled),
            "wins": sum(1 for r in settled if r["pnl_usd"] > 0),
            "total_pnl_usd": total_pnl,
            "total_cost_usd": total_cost,
            "return_on_premium": total_pnl / total_cost if total_cost > 0 else None,
        },
        "rows": rows,
    }
