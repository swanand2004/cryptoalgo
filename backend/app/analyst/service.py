"""Runs the Analyst on the server: gathers every free input from the database,
analyses each configured symbol, keeps the latest result for the page, and
manages the Analyst's PAPER trades (opened on BUY/SELL, closed only at stop or
target). Nothing here ever reaches an exchange."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time

import httpx
import numpy as np
import pandas as pd
from sqlalchemy import select

from app.analyst.factors import Inputs, analyse
from app.data.db import SessionLocal
from app.data.ingest_rest import upsert_candles
from app.data.models import AnalystTrade
from app.data.query import load_candles_df

logger = logging.getLogger(__name__)

COST_PER_SIDE = 0.0007  # Delta taker 0.05% + 18% GST + slippage, as in every study here
HOURLY_BARS = 24 * 330
STABLE_SYMBOL = "STABLE:TOTAL"
_latest: dict[str, dict] = {}
_last_run: dict = {"at": None}
# The gate that stopped (or OPENED) each coin on the latest run, for the page's
# "why no trade?" (read-only copy of what open_best_trade reports; never read back).
_latest_gates: dict[str, str] = {}
_latest_gold: dict[str, dict] = {}
_last_gold_run: dict = {"at": None}
# Every-minute runs: deep hourly history is re-read only when a new bar has
# closed, and the daily stablecoin series only every few hours.
_hourly_cache: dict[str, tuple[int, pd.DataFrame]] = {}
_stable_refreshed: dict = {"at": 0.0}
STABLE_REFRESH_S = 6 * 3600


def _series(df: pd.DataFrame, col: str = "close", scale: float = 1.0) -> pd.Series | None:
    if df is None or df.empty:
        return None
    s = pd.Series(df[col].astype(float).to_numpy() * scale, index=pd.to_datetime(df["time"], unit="s", utc=True))
    return s[~s.index.duplicated(keep="last")].sort_index()


async def refresh_stablecoin_supply() -> None:
    """Total stablecoin supply (DefiLlama, free) as a daily pseudo-series."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        rows = (await client.get("https://stablecoins.llama.fi/stablecoincharts/all")).json()
    candles = []
    for r in rows:
        v = float((r.get("totalCirculatingUSD") or {}).get("peggedUSD") or 0.0)
        if v > 0:
            t = int(r["date"])
            candles.append({"time": t, "open": v, "high": v, "low": v, "close": v, "volume": 0.0})
    if candles:
        await upsert_candles(STABLE_SYMBOL, "1d", candles)


async def _hourly_frame(symbol: str) -> pd.DataFrame:
    # Keyed on (time, close) of the newest TWO stored rows, not just the newest
    # time: the Binance refresh re-writes the forming hour and finalises the
    # previous one, and a time-only key kept a frame whose "closed" bar had been
    # stored a few minutes into its hour (fixed 2026-10-07).
    newest = await load_candles_df(f"BINANCE:{symbol}", "1h", limit=2)
    if newest.empty:
        return newest
    key = tuple((int(t), float(c)) for t, c in zip(newest["time"], newest["close"]))
    cached = _hourly_cache.get(symbol)
    if cached is None or cached[0] != key:
        _hourly_cache[symbol] = (key, await load_candles_df(f"BINANCE:{symbol}", "1h", limit=HOURLY_BARS))
    return _hourly_cache[symbol][1]


def final_bars(hourly: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
    """(the bars whose stored values are FINAL, whether the previous hour is
    still unconfirmed). Every Binance refresh also stores the forming hour, so a
    bar is final only once a newer row exists: the newest stored row is either
    the forming hour or a bar written before its hour ended (no refresh since).
    In the second case it is held back until the next refresh brings its final
    values -- the Analyst then repeats the previous hour's view, never a partial
    bar. In steady state this drops exactly the forming hour, as before."""
    if not len(hourly):
        return hourly, False
    pending = bool(hourly.index[-1] < now.floor("h"))
    return hourly.iloc[:-1], pending


def analyst_symbols(runtime) -> list[str]:
    wanted = getattr(runtime.settings, "analyst_symbol_list", None) or list(runtime.registry.symbols)
    return [s for s in wanted if s in runtime.registry.symbols]


async def gather_inputs(runtime, symbol: str) -> Inputs | None:
    hourly_df = await _hourly_frame(symbol)
    if hourly_df.empty:
        return None
    hourly = hourly_df.set_index(pd.to_datetime(hourly_df["time"], unit="s", utc=True))[["open", "high", "low", "close"]]
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
    now = pd.Timestamp.now(tz="UTC")
    hourly, pending = final_bars(hourly, now)  # never analyse an unfinished bar
    if live_bar_on(runtime) and not pending:
        # (pending: the previous hour is not final yet, so no forming bar is
        # appended after a gap -- closed bars only until the next refresh)
        hourly, _ = with_forming_bar(hourly, await load_candles_df(symbol, "1m", limit=LIVE_BAR_MINUTES), now)

    from app.gold.macro_calendar import next_macro_event

    nxt = next_macro_event()
    events = []
    if nxt.get("at") and nxt.get("hours_away") is not None and 0 <= nxt["hours_away"] <= 24:
        events.append((dt.datetime.fromisoformat(nxt["at"]), str(nxt["event"])))
    signal = getattr(runtime, "live_signals", {}).get(symbol)
    return Inputs(
        hourly=hourly,
        strategy_signal=float(signal) if signal is not None else None,
        # Binance perp funding (a real per-8h rate, as a fraction). Delta's own
        # FUNDING: series is the premium index -- sign-inverted, ~4x off.
        funding=_series(await load_candles_df(f"BFUND:{symbol}", "8h", limit=60)),
        open_interest=_series(await load_candles_df(f"BOI:{symbol}", "1h", limit=200)),
        long_short_ratio=_series(await load_candles_df(f"BLSR:{symbol}", "1h", limit=48)),
        fear_greed=_shift_day(_series(await load_candles_df("FNG:GLOBAL", "1d", limit=30))),
        stablecoin_supply=_shift_day(_series(await load_candles_df(STABLE_SYMBOL, "1d", limit=1200))),
        news=_series(await load_candles_df(f"NEWS:{symbol}", "1h", limit=72)),
        dollar=_shift_day(_series(await load_candles_df("MACRO:DXY", "1d", limit=120))),
        implied_vol=_series(await load_candles_df(f"DVOL:{symbol}", "1h", limit=24 * 95)),
        events=events,
    )


# ----------------------------------------------------------------- live-bar mode
# Settings.analyst_live_bar (default OFF = the closed-bar behaviour, untouched):
# the hourly frame also carries the FORMING hour, built from the stored 1-minute
# candles, so the score moves every minute instead of once an hour. A call then
# has to HOLD for analyst_live_bar_hold_minutes before it can trade or flip a
# trade (hold_call), so a score wobbling around the line cannot flip-flop.
# Pre-registered backtest: scratchpad team/realtime (PREREG.txt, results there).
LIVE_BAR_MINUTES = 120  # 1m rows read per run: the whole forming hour, with room
LIVE_BAR_MAX_AGE_S = 300  # newest 1m candle older than this = feed down: closed bars only
LIVE_CALL_GAP_S = 180  # a call not seen for this long (missed runs, restart) starts over
_live_calls: dict[str, tuple[str, float, float]] = {}  # symbol -> (side, first seen, last seen), unix s


def live_bar_on(runtime) -> bool:
    return bool(getattr(getattr(runtime, "settings", None), "analyst_live_bar", False))


def with_forming_bar(hourly: pd.DataFrame, minutes: pd.DataFrame | None,
                     now: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """(closed hourly bars + the forming hour, minutes in it). The forming bar:
    open = the hour's first 1m open, high/low = extremes so far, close = the
    latest 1m close. Any stored row for the current hour (the Binance refresh
    stores a partial one) is replaced. No fresh 1m data -> closed bars only."""
    start = now.floor("h")
    closed = hourly[hourly.index < start]
    if minutes is None or minutes.empty:
        return closed, 0
    t = minutes["time"].astype("int64").to_numpy()
    s0, s1 = int(start.timestamp()), int(now.timestamp())
    m = minutes[(t >= s0) & (t <= s1)].sort_values("time")
    if m.empty or s1 - int(m["time"].iloc[-1]) > LIVE_BAR_MAX_AGE_S:
        return closed, 0
    idx = pd.DatetimeIndex([start], name=closed.index.name)
    if isinstance(closed.index, pd.DatetimeIndex) and len(closed):
        idx = idx.as_unit(closed.index.unit)
    bar = pd.DataFrame({"open": [float(m["open"].iloc[0])], "high": [float(m["high"].astype(float).max())],
                        "low": [float(m["low"].astype(float).min())], "close": [float(m["close"].iloc[-1])]},
                       index=idx)
    return pd.concat([closed, bar]), len(m)


def hold_call(r: dict, symbol: str, now_s: float, settings) -> None:
    """Live-bar stability rule: a BUY/SELL (at or above analyst_min_score) only
    counts once it has held, on every run, for analyst_live_bar_hold_minutes;
    until then the result reads WAIT (so it neither opens a trade nor flips
    one) with a note saying how long it has held."""
    hold_s = max(0, int(getattr(settings, "analyst_live_bar_hold_minutes", 15))) * 60
    floor = float(getattr(settings, "analyst_min_score", 0.0))
    raw = r["decision"] if r["decision"] != "WAIT" and abs(r["score"]) >= floor else "WAIT"
    if raw == "WAIT":
        _live_calls.pop(symbol, None)
        return
    prev = _live_calls.get(symbol)
    first = prev[1] if prev is not None and prev[0] == raw and now_s - prev[2] <= LIVE_CALL_GAP_S else now_s
    _live_calls[symbol] = (raw, first, now_s)
    held = now_s - first
    r["live_bar"] = {"call": raw, "held_s": round(held), "needed_s": hold_s}
    if held < hold_s:
        r["decision"] = "WAIT"
        r["notes"] = list(r.get("notes") or []) + [
            f"Live-bar {raw} call forming: held {held / 60:.0f} of {hold_s / 60:.0f} min before it can trade"]


def _shift_day(s: pd.Series | None) -> pd.Series | None:
    """A daily value dated D is complete only at the end of D."""
    if s is None:
        return None
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=1)
    return s


async def run_analysis(runtime) -> dict[str, dict]:
    """Analyses every configured symbol, then opens paper trades for new calls."""
    started = time.monotonic()
    if time.monotonic() - _stable_refreshed["at"] > STABLE_REFRESH_S or _stable_refreshed["at"] == 0.0:
        try:
            await refresh_stablecoin_supply()
            _stable_refreshed["at"] = time.monotonic()
        except Exception:
            logger.exception("Analyst: stablecoin supply refresh failed (factor drops out)")
    results: dict[str, dict] = {}
    from app.analyst.execution import shadow_on

    shadow = shadow_on(getattr(runtime, "settings", None))
    frames: dict[str, object] = {}  # each coin's analysed frame, for the shadow variants' ATR rank
    for symbol in analyst_symbols(runtime):
        try:
            inp = await gather_inputs(runtime, symbol)
            if shadow and inp is not None:
                frames[symbol] = getattr(inp, "hourly", None)
            # Pure CPU work: off the event loop, so the trading tick and the
            # 15-second risk supervisors are never held up by an analysis run.
            a = await asyncio.to_thread(analyse, symbol, inp) if inp is not None else None
            if a is not None:
                r = a.to_dict()
                _to_live_price(r, _live_price(runtime, symbol))
                r["factors"] += live_flow_readings(runtime, symbol)
                if live_bar_on(runtime):
                    hold_call(r, symbol, time.time(), runtime.settings)
                results[symbol] = r
        except Exception:
            logger.exception("Analyst failed for %s", symbol)
    _latest.clear()
    _latest.update(results)
    _last_run["at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    # The best high-confidence call that agrees with the trend book, when nothing
    # is open -- a real order under trading_engine "analyst"/"combined", else paper.
    from app.analyst.execution import open_best_trade

    gates: dict[str, str] = {}
    await open_best_trade(runtime, results, gates=gates)
    _latest_gates.clear()
    _latest_gates.update(gates)
    # One INFO line per run, so "why no trade" can be read back from app.log:
    # each coin's call, conviction, and the gate that stopped it (or OPENED).
    logger.info(
        "Analyst: %d symbols in %.1fs | %s", len(results), time.monotonic() - started,
        ", ".join(f"{s} {r['decision']} {r['conviction']}/10 {gates.get(s, '?')}" for s, r in sorted(results.items())),
    )
    if shadow:  # AFTER the live decision, on copies: a failure here never touches it
        try:
            from app.analyst.execution import atr_rank, open_variant_trades

            ranks = {s: await asyncio.to_thread(atr_rank, h) for s, h in frames.items() if s in results}
            await open_variant_trades(runtime, {s: dict(r) for s, r in results.items()}, ranks)
        except Exception:
            logger.exception("Analyst shadow variants failed (live trade unaffected)")
    return results


def _live_price(runtime, symbol: str) -> float | None:
    cache = getattr(runtime, "cache", None)
    price = cache.get_last_price(symbol) if cache is not None else None
    return float(price) if price else None


def _to_live_price(r: dict, price: float | None) -> None:
    """Re-centres the plan on Delta's live price (the analysis itself runs on
    closed hourly bars): same stop and target DISTANCES, so the trade is the
    one a person could actually place right now."""
    if price is None or r.get("entry") is None:
        return
    stop_dist, tgt_dist = r["entry"] - r["stop_loss"], r["take_profit"] - r["entry"]
    r["price"] = r["entry"] = price
    r["stop_loss"], r["take_profit"] = price - stop_dist, price + tgt_dist


def live_flow_readings(runtime, symbol: str) -> list[dict]:
    """Tick-level order flow from the live websocket feed (order book + trade
    tape), shown beside the analysis. NOT scored: on 9 years of data order-flow
    signals were not profitable after fees (see the order-flow toolkit study),
    so letting them vote would make the decision less accurate, not more."""
    cache = getattr(runtime, "cache", None)
    if cache is None:
        return []
    from app.features.microstructure import compute_microstructure

    try:
        m = compute_microstructure(cache, symbol)
    except Exception:
        return []
    out = []
    if m.book_imbalance is not None:
        side = "bids heavier" if m.book_imbalance > 0.1 else "asks heavier" if m.book_imbalance < -0.1 else "balanced"
        out.append({"key": "book", "family": "live", "name": "Order book imbalance (live)",
                    "score": 0.0, "reading": f"{m.book_imbalance:+.2f} -- {side}"})
    if m.trade_flow_imbalance is not None:
        side = "buyers pressing" if m.trade_flow_imbalance > 0.1 else "sellers pressing" if m.trade_flow_imbalance < -0.1 else "two-way"
        out.append({"key": "flow", "family": "live", "name": "Trade flow, last 200 ticks (live)",
                    "score": 0.0, "reading": f"{m.trade_flow_imbalance:+.2f} -- {side}"})
    if m.spread_bps is not None:
        out.append({"key": "spread", "family": "live", "name": "Bid-ask spread (live)",
                    "score": 0.0, "reading": f"{m.spread_bps:.1f} bps"})
    return out


def best_opportunity(results: dict[str, dict]) -> dict | None:
    """The single strongest BUY/SELL across every symbol (largest |score|)."""
    calls = [r for r in results.values() if r["decision"] != "WAIT" and r.get("entry") is not None]
    return max(calls, key=lambda r: abs(r["score"])) if calls else None


async def run_gold_analysis(runtime) -> dict | None:
    """The same Analyst on gold (Delta XAUTUSD): price-based families plus the
    dollar, US real yields and the event calendar. Crypto-only inputs (Fear &
    Greed, stablecoins, crypto funding/positioning) are left out. Paper only."""
    from app.gold.ingest import GOLD_SPOT_SYMBOL

    hourly_df = await load_candles_df(GOLD_SPOT_SYMBOL, "1h", limit=HOURLY_BARS)
    if hourly_df.empty:
        return None
    hourly = hourly_df.set_index(pd.to_datetime(hourly_df["time"], unit="s", utc=True))[["open", "high", "low", "close"]]
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
    if len(hourly) and hourly.index[-1] + pd.Timedelta(hours=1) > pd.Timestamp.now(tz="UTC"):
        hourly = hourly.iloc[:-1]
    from app.gold.macro_calendar import next_macro_event

    nxt = next_macro_event()
    events = []
    if nxt.get("at") and nxt.get("hours_away") is not None and 0 <= nxt["hours_away"] <= 24:
        events.append((dt.datetime.fromisoformat(nxt["at"]), str(nxt["event"])))
    inp = Inputs(
        hourly=hourly,
        dollar=_shift_day(_series(await load_candles_df("MACRO:DXY", "1d", limit=120))),
        real_yield=_shift_day(_series(await load_candles_df("MACRO:REALRATE10Y", "1d", limit=120))),
        events=events,
    )
    a = await asyncio.to_thread(analyse, GOLD_SPOT_SYMBOL, inp)
    if a is None:
        return None
    r = a.to_dict()
    latest = await load_candles_df(GOLD_SPOT_SYMBOL, "15m", limit=1)
    _to_live_price(r, float(latest["close"].iloc[-1]) if len(latest) else None)
    _latest_gold.clear()
    _latest_gold[GOLD_SPOT_SYMBOL] = r
    _last_gold_run["at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    from app.analyst.execution import open_best_trade

    await open_best_trade(runtime, {GOLD_SPOT_SYMBOL: r}, paper_only=True, require_book=False)
    return r


async def gold_snapshot() -> dict:
    from app.gold.ingest import GOLD_SPOT_SYMBOL

    async with SessionLocal() as session:
        trades = (await session.execute(select(AnalystTrade).where(AnalystTrade.symbol == GOLD_SPOT_SYMBOL)
                                        .order_by(AnalystTrade.entry_time.desc()).limit(200))).scalars().all()
    latest = await load_candles_df(GOLD_SPOT_SYMBOL, "15m", limit=1)
    price = float(latest["close"].iloc[-1]) if len(latest) else None
    rows = [_trade_row(t, price) for t in trades]
    closed = [r for r in rows if not r["open"]]
    rs = [r["r_multiple"] for r in closed if r["r_multiple"] is not None]
    return {
        "updated": _last_gold_run["at"],
        "analysis": _latest_gold.get(GOLD_SPOT_SYMBOL),
        "open_trades": [r for r in rows if r["open"]],
        "closed_trades": closed[:60],
        "stats": {"closed": len(closed), "wins": sum(1 for r in rs if r > 0),
                  "win_rate": (sum(1 for r in rs if r > 0) / len(rs)) if rs else None,
                  "avg_r": float(np.mean(rs)) if rs else None, "total_r": float(np.sum(rs)) if rs else 0.0},
    }


async def update_paper_trades() -> None:
    """Closes open paper trades whose stop or target has been touched, walking
    the closed hourly bars since entry (a bar touching both = stopped)."""
    async with SessionLocal() as session:
        # Paper records only: a real position is managed tick by tick by the
        # executor, which also reconciles it with the exchange after downtime.
        q = select(AnalystTrade).where(AnalystTrade.is_open.is_(True), AnalystTrade.contracts == 0)
        from app.analyst.execution import shadow_on
        from app.config import get_settings

        if not shadow_on(get_settings()):  # setting off: no variant work at all
            q = q.where(AnalystTrade.variant.is_(None))
        trades = (await session.execute(q)).scalars().all()
        for t in trades:
            bars = await load_candles_df(f"BINANCE:{t.symbol}", "1h", limit=24 * 60)
            if bars.empty:
                continue
            entry_ts = t.entry_time.replace(tzinfo=dt.timezone.utc).timestamp() if t.entry_time.tzinfo is None else t.entry_time.timestamp()
            now_ts = time.time()
            bars = bars[(bars["time"] >= entry_ts) & (bars["time"] + 3600 <= now_ts)]
            d = 1 if t.side == "BUY" else -1
            # Each bar is judged against a stop that existed AT THAT BAR, not the
            # current one: manage_open_trades trails t.stop_loss tick by tick,
            # and re-walking bars from before a trail against it closed trades
            # retroactively at a stop that did not exist yet. The trail can
            # only have reached t.stop_loss once price reached the recorded
            # peak, so bars up to that point use the initial stop (the
            # conservative one) and bars after it the current stop.
            # Risk (for R) is measured from the initial stop, as everywhere else.
            initial = t.initial_stop or t.stop_loss
            stop = initial
            risk = abs(t.entry - initial)
            for _, b in bars.iterrows():
                hit_sl = b["low"] <= stop if d > 0 else b["high"] >= stop
                hit_tp = b["high"] >= t.take_profit if d > 0 else b["low"] <= t.take_profit
                if not (hit_sl or hit_tp):
                    if t.peak is not None and (b["high"] >= t.peak if d > 0 else b["low"] <= t.peak):
                        stop = t.stop_loss  # the trail exists from the next bar on
                    continue
                if hit_sl or hit_tp:
                    px = stop if hit_sl else t.take_profit
                    t.is_open = False
                    t.exit_price = float(px)
                    t.exit_time = dt.datetime.fromtimestamp(b["time"] + 3600, dt.timezone.utc)
                    t.result = "SL" if hit_sl else "TP"
                    t.r_multiple = float(d * (px - t.entry) / risk - 2 * COST_PER_SIDE * t.entry / risk) if risk > 0 else 0.0
                    logger.info("Analyst paper %s %s closed %s at %.6g (%+.2fR)", t.side, t.symbol, t.result, px, t.r_multiple)
                    break
        await session.commit()


def _trade_row(t: AnalystTrade, price: float | None) -> dict:
    d = 1 if t.side == "BUY" else -1
    risk = abs(t.entry - t.stop_loss) or np.nan
    mark = t.exit_price if not t.is_open else price
    r_now = t.r_multiple if not t.is_open else (d * (mark - t.entry) / risk if mark else None)
    return {
        "id": t.id, "symbol": t.symbol, "side": t.side, "conviction": t.conviction,
        "entry_time": t.entry_time.isoformat(), "entry": t.entry, "stop_loss": t.stop_loss,
        "take_profit": t.take_profit, "open": t.is_open,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None, "exit_price": t.exit_price,
        "result": t.result, "r_multiple": None if r_now is None or not np.isfinite(r_now) else float(r_now),
        "pnl_pct": float(d * ((mark or t.entry) / t.entry - 1) * 100), "reasons": t.reasons,
        "live": bool(t.live), "contracts": int(t.contracts or 0), "extended": bool(t.extended),
    }


async def analyst_snapshot(runtime) -> dict:
    async with SessionLocal() as session:
        trades = (await session.execute(select(AnalystTrade).where(AnalystTrade.symbol != "XAUTUSD",
                                                                   AnalystTrade.variant.is_(None))
                                        .order_by(AnalystTrade.entry_time.desc()).limit(300))).scalars().all()
    price = lambda s: runtime.cache.get_last_price(s) if getattr(runtime, "cache", None) else None  # noqa: E731
    rows = [_trade_row(t, price(t.symbol)) for t in trades]
    closed = [r for r in rows if not r["open"]]
    rs = [r["r_multiple"] for r in closed if r["r_multiple"] is not None]
    best = best_opportunity(_latest)
    signals = getattr(runtime, "live_signals", None) or {}
    book: dict[str, float | None] = {}
    for s in _latest:
        try:
            book[s] = float(signals[s]) if signals.get(s) is not None else None
        except (TypeError, ValueError):
            book[s] = None
    return {
        "updated": _last_run["at"],
        "best": best["symbol"] if best else None,
        "analyses": sorted(_latest.values(), key=lambda a: -abs(a["score"])),
        "gates": dict(_latest_gates),
        "book_signals": book,
        "risk_settings": _risk_settings(getattr(runtime, "settings", None)),
        "open_trades": [r for r in rows if r["open"]],
        "closed_trades": closed[:100],
        "stats": {
            "closed": len(closed),
            "wins": sum(1 for r in rs if r > 0),
            "win_rate": (sum(1 for r in rs if r > 0) / len(rs)) if rs else None,
            "avg_r": float(np.mean(rs)) if rs else None,
            "total_r": float(np.sum(rs)) if rs else 0.0,
        },
        "variants": await variant_race(getattr(runtime, "settings", None)),
    }


VARIANT_LABELS = {"live": "Live (since race start)", "base": "Base: no extra rule",
                  "C": "C: skip top-third ATR rank", "B": "B: SELL only at score <= -0.40", "CB": "C + B"}


async def variant_race(settings) -> dict:
    """The shadow variants' paper books next to the live record over the SAME
    span (live trades entered since the first variant trade), for the page's
    strategy race. Read-only; variant rows never reach the live stats."""
    from app.analyst.execution import SHADOW_VARIANTS, shadow_on

    if not shadow_on(settings):
        return {"enabled": False, "since": None, "rows": []}
    from sqlalchemy import func, or_

    shadow = AnalystTrade.variant.is_not(None)
    async with SessionLocal() as session:
        since = (await session.execute(select(func.min(AnalystTrade.entry_time)).where(shadow))).scalar()
        rows = [] if since is None else (await session.execute(
            select(AnalystTrade.variant, AnalystTrade.is_open, AnalystTrade.r_multiple).where(
                AnalystTrade.symbol != "XAUTUSD", or_(shadow, AnalystTrade.entry_time >= since)))).all()
    keys = ["live", *SHADOW_VARIANTS]
    rs: dict[str, list[float]] = {k: [] for k in keys}
    opened = dict.fromkeys(keys, 0)
    for v, is_open, r in rows:
        k = v or "live"
        if k not in rs:
            continue
        if is_open:
            opened[k] += 1
        elif r is not None:
            rs[k].append(float(r))
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=dt.timezone.utc)
    return {
        "enabled": True,
        "since": since.isoformat() if since is not None else None,
        "rows": [{"key": k, "label": VARIANT_LABELS[k], "closed": len(rs[k]), "open": opened[k],
                  "wins": sum(1 for x in rs[k] if x > 0),
                  "win_rate": (sum(1 for x in rs[k] if x > 0) / len(rs[k])) if rs[k] else None,
                  "avg_r": float(np.mean(rs[k])) if rs[k] else None,
                  "total_r": float(np.sum(rs[k])) if rs[k] else 0.0} for k in keys],
    }


def _risk_settings(settings) -> dict:
    """The Analyst's risk settings in force, for the page. Read-only."""
    if settings is None:
        return {}
    from app.analyst.execution import trade_risk_cap

    g = lambda k, d=None: getattr(settings, k, d)  # noqa: E731
    return {
        "trading_engine": g("trading_engine"),
        "executes": g("trading_engine") in ("analyst", "combined"),
        "risk_pct": g("analyst_risk_pct"),
        "risk_cap": trade_risk_cap(settings),
        "target_leverage": g("analyst_target_leverage", 0.0),
        "min_liq_to_stop": g("analyst_min_liq_to_stop", 0.0),
        "max_leverage": g("max_leverage"),
        "maker_execution": g("maker_execution"),
        "trail_mode": g("analyst_trail_mode"),
        "trail_atr": g("analyst_trail_atr"),
        "trail_atr_tight": g("analyst_trail_atr_tight"),
        "min_score": g("analyst_min_score"),
        "min_conviction": g("analyst_min_conviction"),
        "live_bar": bool(g("analyst_live_bar", False)),
    }
