"""The Analyst as the trader (Settings.trading_engine = "analyst").

One position at a time, on the single best opportunity among the major coins,
only when the call is high-probability (conviction >= analyst_min_conviction).
Orders go through the normal OrderManager, so every risk limit, the kill switch
and the paper/live switch apply exactly as they do for the trend book.

Exits are dynamic but one-directional (user, 2026-10-06: "SL/TP according to
the situation, but never pull the SL back after moving it forward"):
- the stop trails behind the best price since entry by analyst_trail_atr daily
  ATRs, tightening to analyst_trail_atr_tight once the trade is +1R, and from
  +1R it never sits below breakeven; it only ever moves in the trade's favour;
- at TP1 the trade closes -- unless the Analyst still makes the same call with
  high conviction, in which case it stays open, the stop is locked at +0.5R and
  the next target is set 1R further;
- a high-conviction call the OTHER way closes it at once.
Under analyst_trail_mode = "after_tp1" (the default since 2026-10-07) the ATR
trail and the breakeven floor are off: the stop stays at the level stop and
moves only by the TP1 lock. Live trades also keep a reduce-only stop order on
Delta at the stop (analyst_exchange_stops).
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import time
from dataclasses import dataclass

from sqlalchemy import select

from app.data.app_state import get_trading_mode, is_kill_switch_on
from app.data.db import SessionLocal
from app.data.models import AnalystTrade
from app.execution.broker_base import OrderRequest

logger = logging.getLogger(__name__)

COST_PER_SIDE = 0.0007
STRATEGY_TAG = "consolidated"  # the live broker keeps one position per product under this tag
MARGIN_USE = 0.85
# Isolated-margin liquidation estimate: Delta maintenance margin is 0.25% (BTC,
# ETH) to 0.5% (SOL, BNB) of notional; take the larger, plus ~0.12% for the
# close fee (0.05% taker + 18% GST, rounded up), so the estimate is never
# further away than the real liquidation.
LIQ_MAINTENANCE = 0.005
LIQ_FEES = 0.0012
# Exchange-side stops (Settings.analyst_exchange_stops). Their state lives in
# AnalystTrade.reasons["exchange_stop"] = {"id", "price", "at"} so it survives
# a restart; their client_order_id carries STOP_TAG so a close can sweep any
# duplicate (e.g. a placement whose reply was lost to a timeout).
STOP_TAG = "an-stop-"
STOP_SYNC_SECONDS = 60.0   # check/move the exchange stop at most once a minute
STOP_MIN_MOVE_ATR = 0.1    # ... and move it only when the software stop moved >= 0.1 ATR
_RESTING = ("open", "pending")  # "pending" = an untriggered stop order on Delta


def trade_risk_cap(settings) -> float:
    """The share of equity one contract may risk: the tighter of the user's
    analyst_max_trade_risk and the CRO's analyst_risk_cap (0 = no CRO cap)."""
    cap = settings.analyst_max_trade_risk
    cro = getattr(settings, "analyst_risk_cap", 0.0) or 0.0
    return min(cap, cro) if cro > 0 else cap


def liquidation_distance(leverage: float) -> float:
    """Estimated fraction of the entry price an isolated position at
    `leverage` can move against it before Delta liquidates it."""
    return 1.0 / leverage - LIQ_MAINTENANCE - LIQ_FEES


def plan_size(*, equity: float, price: float, stop: float, contract_value: float, settings) -> tuple[int, float] | None:
    """(contracts, leverage) risking analyst_risk_pct of equity to the stop.
    If that rounds to zero, ONE contract is allowed only while its risk stays
    within trade_risk_cap of equity; otherwise the coin is skipped.
    Leverage is the lowest that margins the position inside 85% of equity.
    The trade is also skipped when liquidation would not sit at least
    analyst_min_liq_to_stop times as far away as the stop."""
    risk_per_contract = abs(price - stop) * contract_value
    if equity <= 0 or price <= 0 or contract_value <= 0 or risk_per_contract <= 0:
        return None
    n = int(settings.analyst_risk_pct * equity // risk_per_contract)
    if n == 0 and risk_per_contract <= trade_risk_cap(settings) * equity:
        n = 1
    if n == 0:
        return None
    leverage = max(1.0, float(math.ceil(n * contract_value * price / (MARGIN_USE * equity))))
    if leverage > settings.max_leverage:
        return None
    ratio = getattr(settings, "analyst_min_liq_to_stop", 0.0) or 0.0
    stop_pct = abs(price - stop) / price
    if ratio > 0 and liquidation_distance(leverage) < ratio * stop_pct:
        return None
    return n, _target_leverage(leverage, stop_pct, ratio, settings)


def _target_leverage(leverage: float, stop_pct: float, ratio: float, settings) -> float:
    """analyst_target_leverage (0 = off): raise the leverage toward the target
    -- a smaller posted margin, so less to lose if the bot is down -- but never
    to one whose liquidation sits closer than `ratio` x the stop distance (at
    least beyond the stop when the ratio check is off); then the highest whole
    leverage that is still safe. Never lowers the computed leverage."""
    target = getattr(settings, "analyst_target_leverage", 0.0) or 0.0
    if target <= leverage:
        return leverage
    floor = max(ratio, 1.0)
    lev = float(math.floor(min(target, settings.max_leverage)))
    while lev > leverage and liquidation_distance(lev) < floor * stop_pct:
        lev -= 1.0
    return max(leverage, lev)


@dataclass
class ExitUpdate:
    stop: float
    take_profit: float
    peak: float
    extended: bool
    action: str | None  # None = hold; "TP" / "SL" / "TRAIL" = close now


def update_exit(*, side: str, entry: float, stop: float, initial_stop: float, atr: float, peak: float,
                price: float, take_profit: float, extended: bool, still_strong: bool, settings) -> ExitUpdate:
    d = 1 if side == "BUY" else -1
    risk = abs(entry - initial_stop) or 1e-12
    peak = max(peak, price) if d > 0 else min(peak, price)
    best_r = d * (peak - entry) / risk
    new_stop = stop
    # analyst_trail_mode "after_tp1" (loss doctor FIX A, owner-approved
    # 2026-10-07): no ATR trail and no breakeven floor -- the stop holds at the
    # level stop and only TP1's +0.5R lock below moves it. "live" = the trail.
    if getattr(settings, "analyst_trail_mode", "live") != "after_tp1":
        trail = (settings.analyst_trail_atr_tight if best_r >= 1 else settings.analyst_trail_atr) * atr
        candidate = peak - d * trail
        if best_r >= 1:  # past +1R the stop never sits below breakeven (plus a sliver for fees)
            floor = entry + d * 0.1 * risk
            candidate = max(candidate, floor) if d > 0 else min(candidate, floor)
        new_stop = max(stop, candidate) if d > 0 else min(stop, candidate)  # never moves back

    action = None
    if (price >= take_profit) if d > 0 else (price <= take_profit):
        if still_strong:
            extended = True
            lock = entry + d * 0.5 * risk
            new_stop = max(new_stop, lock) if d > 0 else min(new_stop, lock)
            take_profit = take_profit + d * risk
        else:
            action = "TP"
    if action is None and ((price <= new_stop) if d > 0 else (price >= new_stop)):
        moved = (new_stop > initial_stop) if d > 0 else (new_stop < initial_stop)
        action = "TRAIL" if moved else "SL"
    return ExitUpdate(new_stop, take_profit, peak, extended, action)


def _executes(settings) -> bool:
    return settings.trading_engine in ("analyst", "combined")


def agrees_with_book(side: str, book_signal: float | None, line: float = 0.15) -> bool:
    """The Analyst trades only in the direction the validated trend book holds
    for that coin (its entry line, 0.15): BUY needs the book long, SELL short."""
    if book_signal is None:
        return False
    return book_signal >= line if side == "BUY" else book_signal <= -line


def _live_price(runtime, symbol: str) -> float | None:
    cache = getattr(runtime, "cache", None)
    p = cache.get_last_price(symbol) if cache is not None else None
    return float(p) if p else None


async def _stored_price(symbol: str) -> float | None:
    """Latest stored close, for a symbol the live websocket does not stream
    (gold's XAUTUSD is refreshed into the candles table every few minutes)."""
    from app.data.query import load_candles_df

    df = await load_candles_df(symbol, "15m", limit=1)
    return float(df["close"].iloc[-1]) if len(df) else None


def _strong(call: dict | None, side: str, settings) -> bool:
    return bool(call) and call["decision"] == side and call["conviction"] >= settings.analyst_min_conviction


async def _ml_gate_multipliers(settings) -> dict[str, float] | None:
    """analyst_ml_gate "g2" (2026-10-09, pre-registered in cryptoalgo_team/mlgate/PREREG.txt):
    enter only on coins the frozen ML book filter rates 1.5x today. 10 unseen coins, R/trade
    +0.155/+0.274 vs +0.078/+0.139 and R/yr +28.7/+56.8 vs +18.8/+36.7 (2019-22/2023-26),
    8/10 and 9/10 coins up, above the random-gate p95 (+0.081/+0.180 R/trade), DSR 1.00;
    majors 2023-26 +0.617 vs +0.412 R/trade. Fewer trades (~25% fewer).
    Returns None = no gate: setting off, or no fresh record (fail OPEN onto the
    validated baseline rules rather than freezing the Analyst)."""
    if getattr(settings, "analyst_ml_gate", "off") != "g2":
        return None
    try:
        from app.ml.book_filter import current_multipliers

        mult = await current_multipliers()
    except Exception:
        logger.exception("Analyst ML gate: multipliers unavailable -- gate open")
        return None
    return mult or None


def _call_gate(r: dict, settings) -> str | None:
    """Why a call is not even a candidate, or None when it is."""
    if r["decision"] == "WAIT":
        return "wait"
    if r.get("entry") is None:
        return "no_entry"
    if abs(r["score"]) < settings.analyst_min_score:
        return f"score<{settings.analyst_min_score:g}"
    return None


# ------------------------------------------------ exchange-side stops (live only)
# The software stop above is checked every 5 s; these keep a reduce-only
# stop-market order resting on Delta at the same level, so a bot outage does
# not leave the position with only its liquidation price. None of these ever
# raises into the caller: a failure is logged and retried on a later pass.
def _stop_broker(runtime):
    """The live broker when exchange-side stops apply, else None (setting off =
    the old software-only behaviour, or a broker without the stop helpers)."""
    if not getattr(runtime.settings, "analyst_exchange_stops", False):
        return None
    broker = getattr(getattr(runtime, "order_manager", None), "live_broker", None)
    return broker if hasattr(broker, "place_stop_order") else None


def _stop_state(t: AnalystTrade) -> dict:
    return dict((t.reasons or {}).get("exchange_stop") or {})


def _set_stop_state(t: AnalystTrade, st: dict) -> None:
    # Reassign the whole dict: a JSON column does not see in-place edits.
    t.reasons = {**(t.reasons or {}), "exchange_stop": st}


async def _place_exchange_stop(broker, t: AnalystTrade, now: float) -> str | None:
    cid = f"{STOP_TAG}{t.symbol}-{int(now * 1000)}"[:32]
    oid = await broker.place_stop_order(t.symbol, "sell" if t.side == "BUY" else "buy", int(t.contracts),
                                        float(t.stop_loss), cid)
    if oid is None:
        logger.error("Analyst LIVE %s %s has NO exchange stop (software stop only) -- retrying in %ds",
                     t.side, t.symbol, STOP_SYNC_SECONDS)
    return oid


async def _sync_exchange_stop(broker, t: AnalystTrade, atr: float, now: float) -> None:
    """At most once a minute: re-place the stop if it is missing (the deadman
    switch, a manual cancel, or a failed first placement), else move it when the
    software stop has moved >= STOP_MIN_MOVE_ATR ATRs. A move edits in place, or
    places the new stop BEFORE cancelling the old one: never a gap."""
    try:
        st = _stop_state(t)
        if now - float(st.get("at") or 0.0) < STOP_SYNC_SECONDS:
            return
        oid = st.get("id")
        if oid:
            rows = await broker.resting_orders(t.symbol)
            if rows is None:  # unknown is not missing: look again next minute
                _set_stop_state(t, {**st, "at": now})
                return
            if any(str(r.get("id")) == str(oid) and r.get("state", "pending") in _RESTING for r in rows):
                if abs(float(t.stop_loss) - float(st.get("price") or t.stop_loss)) < STOP_MIN_MOVE_ATR * atr:
                    _set_stop_state(t, {**st, "at": now})
                    return
                new = await broker.edit_stop_order(t.symbol, oid, float(t.stop_loss))
                if new is None:
                    new = await _place_exchange_stop(broker, t, now)
                    if new is not None:
                        await broker.cancel_stop_order(t.symbol, oid)
                if new is None:  # the old stop still rests at its old level
                    _set_stop_state(t, {**st, "at": now})
                    return
                _set_stop_state(t, {"id": new, "price": float(t.stop_loss), "at": now})
                return
            # Not resting: cancelled (e.g. by the deadman switch) or triggered.
            # Re-place it; reduce-only, so on a position that is already flat it
            # can do nothing, and the SYNC close below sweeps it away.
            logger.warning("Analyst LIVE %s %s: exchange stop %s is gone -- re-placing at %.6g",
                           t.side, t.symbol, oid, t.stop_loss)
        new = await _place_exchange_stop(broker, t, now)
        _set_stop_state(t, {"id": new, "price": float(t.stop_loss) if new else None, "at": now})
    except Exception:
        logger.exception("Analyst exchange-stop upkeep failed for %s -- retrying next minute", t.symbol)


async def _cancel_exchange_stops(broker, t: AnalystTrade) -> None:
    """The trade is closed: cancel its stop and sweep any other Analyst-tagged
    stop on that product (a duplicate whose placement reply was lost), so none
    is left resting against a later position on the same coin."""
    try:
        st = _stop_state(t)
        ids = {str(st["id"])} if st.get("id") else set()
        for r in await broker.resting_orders(t.symbol) or []:
            if str(r.get("client_order_id") or "").startswith(STOP_TAG) and r.get("id") is not None:
                ids.add(str(r["id"]))
        failed = [oid for oid in sorted(ids) if not await broker.cancel_stop_order(t.symbol, oid)]
        if failed:
            logger.error("Analyst %s %s closed but exchange stop(s) %s could NOT be cancelled -- cancel by hand",
                         t.side, t.symbol, ", ".join(failed))
        _set_stop_state(t, {**st, "id": None, "at": time.time(), "cancelled": sorted(ids)})
    except Exception:
        logger.exception("Analyst %s closed but its exchange stop could not be cancelled -- check Delta", t.symbol)


async def open_best_trade(runtime, results: dict[str, dict], *, paper_only: bool = False,
                          require_book: bool = True, gates: dict[str, str] | None = None) -> AnalystTrade | None:
    """Opens the single best high-probability call when nothing is open IN THIS
    MARKET (the symbols in `results`: crypto majors, or gold on its own).

    `gates`, when given, is filled with the reason each symbol did or did not
    trade this run (for the per-run summary log). It never changes a decision."""
    settings = runtime.settings
    gates = {} if gates is None else gates
    for sym, r in results.items():
        gates[sym] = _call_gate(r, settings) or "candidate"

    def _block_all(reason: str) -> None:
        for sym, g in gates.items():
            if g == "candidate":
                gates[sym] = reason

    if await is_kill_switch_on() or not results:
        _block_all("kill_switch")
        return None
    from app.gold.ingest import GOLD_SPOT_SYMBOL

    # One open trade per MARKET, judged on the whole market -- not just the
    # coins analysed this minute: a coin whose analysis failed this run still
    # holds its open (possibly real) position. Gold is its own market.
    in_market = (AnalystTrade.symbol == GOLD_SPOT_SYMBOL if GOLD_SPOT_SYMBOL in results
                 else AnalystTrade.symbol != GOLD_SPOT_SYMBOL)
    async with SessionLocal() as session:
        if (await session.execute(select(AnalystTrade.id).where(
                AnalystTrade.is_open.is_(True), in_market, AnalystTrade.variant.is_(None)))).first():
            _block_all("trade_open")
            return None
    calls = sorted(
        (r for r in results.values()
         if r["decision"] != "WAIT" and r.get("entry") is not None and abs(r["score"]) >= settings.analyst_min_score),
        key=lambda r: -abs(r["score"]),
    )
    executes = _executes(settings) and not paper_only
    mode = await get_trading_mode()
    equity, held = None, set()
    if executes:
        try:
            equity = await runtime.order_manager.get_equity()
        except Exception:
            logger.warning("Analyst: equity unavailable -- not opening a trade this minute")
            _block_all("no_equity")
            return None
        held = {k.split(":")[0] for k, p in (await runtime.order_manager.get_positions()).items() if p.size != 0}
    signals = getattr(runtime, "live_signals", {}) or {}
    ml_mult = await _ml_gate_multipliers(settings) if require_book else None
    for r in calls:
        symbol, side = r["symbol"], r["decision"]
        if require_book and not agrees_with_book(side, signals.get(symbol)):
            gates[symbol] = "book_disagrees"
            continue  # synergy rule: only in the trend book's own direction for this coin
        if ml_mult and float(ml_mult.get(symbol, 1.0)) < 1.5:
            gates[symbol] = "ml_gate"
            continue  # analyst_ml_gate "g2": the book filter must rate this coin 1.5x today
        if symbol in held:
            gates[symbol] = "book_holds_coin"
            continue  # the trend book already holds this coin -- one position per coin
        price = _live_price(runtime, symbol) or r["entry"]
        stop_dist, tgt_dist = r["entry"] - r["stop_loss"], r["take_profit"] - r["entry"]
        stop, target = price - stop_dist, price + tgt_dist
        atr = r.get("atr") or abs(stop_dist) / 1.5
        contracts, leverage, live = 0, None, False
        if executes:
            product = runtime.registry.get(symbol)
            sized = plan_size(equity=equity, price=price, stop=stop, contract_value=product.contract_value, settings=settings)
            if sized is None:
                logger.info("Analyst: %s %s skipped -- one contract would risk more than %.0f%% of $%.2f, "
                            "or liquidation would sit too close to the stop",
                            side, symbol, trade_risk_cap(settings) * 100, equity)
                gates[symbol] = "too_risky_to_size"
                continue
            contracts, leverage = sized
            result = await runtime.order_manager.submit(OrderRequest(
                symbol=symbol, side="buy" if side == "BUY" else "sell", size=contracts, order_type="market",
                strategy=STRATEGY_TAG, quality=min(1.0, abs(r["score"])), leverage=leverage,
            ))
            if result.state == "rejected" or result.filled_size == 0:
                logger.warning("Analyst %s %s order not filled: %s", side, symbol, result.reason)
                gates[symbol] = "not_filled"
                continue
            if result.avg_fill_price:
                shift = result.avg_fill_price - price
                price, stop, target = price + shift, stop + shift, target + shift
            contracts, live = result.filled_size, mode == "live"
        trade = AnalystTrade(
            symbol=symbol, side=side, conviction=r["conviction"], score=r["score"], size_factor=r["size_factor"],
            entry=price, stop_loss=stop, take_profit=target, initial_stop=stop, atr=atr, peak=price,
            contracts=contracts, live=live, entry_time=dt.datetime.now(dt.timezone.utc),
            reasons={"families": r["families"], "notes": r["notes"],
                     "factors": [{k: f[k] for k in ("name", "score", "reading")} for f in r["factors"]]},
        )
        if live and (stop_broker := _stop_broker(runtime)) is not None:
            # Exchange-side stop right after the fill (a failure is retried by
            # manage_open_trades within a minute; the software stop runs anyway).
            now = time.time()
            try:
                oid = await _place_exchange_stop(stop_broker, trade, now)
            except Exception:
                logger.exception("Analyst exchange stop placement failed for %s", symbol)
                oid = None
            _set_stop_state(trade, {"id": oid, "price": float(stop) if oid else None, "at": now})
        async with SessionLocal() as session:
            session.add(trade)
            await session.commit()
        logger.info("Analyst %s %s %s x%d @ %.6g (SL %.6g / TP1 %.6g, conviction %d, %s)",
                    "LIVE" if live else "paper", side, symbol, contracts, price, stop, target, r["conviction"],
                    f"{leverage:.0f}x" if leverage else "no order")
        gates[symbol] = "OPENED"
        _block_all("outranked")
        return trade
    return None


async def manage_open_trades(runtime, latest: dict[str, dict]) -> None:
    """Every few seconds: trail the stop, decide TP1, and close on a live tick.
    The real records first; then (Settings.analyst_shadow_variants) the paper
    shadow-variant books, which can never hold up or alter the real pass."""
    await _manage_live_trades(runtime, latest)
    if shadow_on(getattr(runtime, "settings", None)):
        try:
            await manage_variant_trades(runtime, latest)
        except Exception:
            logger.exception("Analyst shadow-variant exits failed (live trades unaffected)")


async def _manage_live_trades(runtime, latest: dict[str, dict]) -> None:
    settings = runtime.settings
    async with SessionLocal() as session:
        trades = (await session.execute(select(AnalystTrade).where(
            AnalystTrade.is_open.is_(True), AnalystTrade.variant.is_(None)))).scalars().all()
        if not trades:
            return
        positions = None
        mode = await get_trading_mode()
        for t in trades:
            price = _live_price(runtime, t.symbol) or await _stored_price(t.symbol)
            if price is None:
                continue
            # Live vs paper is a property of the RECORD (did it send an order?),
            # not of today's engine setting: a real position opened under
            # "combined" must still close on the exchange after a switch to "book".
            # The order manager routes by the CURRENT mode, so a record only goes
            # through it while that mode matches the one it was opened in --
            # otherwise a paper record could close a real (book) position on the
            # same coin after a paper->live switch.
            on_broker = bool(t.contracts) and bool(t.live) == (mode == "live")
            # Exchange-side stop: only for a real position this pass manages.
            stop_broker = _stop_broker(runtime) if on_broker and t.live else None
            if on_broker:
                if positions is None:
                    positions = await runtime.order_manager.get_positions()
                pos = positions.get(f"{t.symbol}:{STRATEGY_TAG}")
                if pos is None or pos.size == 0:  # closed elsewhere: kill switch, liquidation, by hand, exchange stop
                    if stop_broker is not None:
                        await _cancel_exchange_stops(stop_broker, t)
                    _close_record(t, price, "SYNC")
                    continue
            atr = t.atr or abs(t.entry - (t.initial_stop or t.stop_loss)) / 1.5
            call = latest.get(t.symbol)
            u = update_exit(side=t.side, entry=t.entry, stop=t.stop_loss, initial_stop=t.initial_stop or t.stop_loss,
                            atr=atr, peak=t.peak or t.entry, price=price, take_profit=t.take_profit,
                            extended=bool(t.extended), still_strong=_strong(call, t.side, settings), settings=settings)
            t.stop_loss, t.take_profit, t.peak, t.extended = u.stop, u.take_profit, u.peak, u.extended
            action = u.action
            opposite = "SELL" if t.side == "BUY" else "BUY"
            if action is None and _strong(call, opposite, settings):
                action = "FLIP"
            if action is None:
                if stop_broker is not None:
                    await _sync_exchange_stop(stop_broker, t, atr, time.time())
                continue
            if on_broker:
                result = await runtime.order_manager.close_position(t.symbol, STRATEGY_TAG)
                if result is not None and (result.state == "rejected" or not result.filled_size):
                    # The exchange did not close it: keep the record open (and
                    # its stop watched) and try again on the next pass, rather
                    # than orphaning a real position the book would then adopt.
                    # The exchange stop stays resting meanwhile.
                    logger.error("Analyst %s %s close (%s) failed on the exchange: %s -- retrying next pass",
                                 t.side, t.symbol, action, result.reason)
                    continue
                if stop_broker is not None:  # flat now (or already): its exchange stop must go
                    await _cancel_exchange_stops(stop_broker, t)
                px = result.avg_fill_price if result is not None and result.avg_fill_price else price
            else:  # paper record: a target fills at its level, a stop where the tick crossed it
                if t.contracts:
                    logger.warning("Analyst %s %s was opened in %s mode but the bot is now %s -- closing the "
                                   "record without an order", t.side, t.symbol, "live" if t.live else "paper", mode)
                px = t.take_profit if action == "TP" else price
            _close_record(t, px, action)
        await session.commit()


def _close_record(t: AnalystTrade, px: float, result: str) -> None:
    d = 1 if t.side == "BUY" else -1
    risk = abs(t.entry - (t.initial_stop or t.stop_loss))
    t.is_open, t.exit_price, t.result = False, float(px), result
    t.exit_time = dt.datetime.now(dt.timezone.utc)
    t.r_multiple = float(d * (px - t.entry) / risk - 2 * COST_PER_SIDE * t.entry / risk) if risk > 0 else 0.0
    logger.info("Analyst %s %s %s closed %s at %.6g (%+.2fR)",
                "LIVE" if t.live else f"shadow-{t.variant}" if t.variant else "paper",
                t.side, t.symbol, result, px, t.r_multiple)


# ------------------------------------------------ shadow variants (paper only)
# A forward test of the loss autopsy's near-miss rules (Settings.
# analyst_shadow_variants). Each variant keeps its own PAPER book -- one open
# trade per coin per variant -- with the live entry rules (call gate, score
# line, trend-book agreement, kill switch) plus its extra rule, and the live
# exits (update_exit with the live settings, FLIP on a strong opposite call).
# "base" is the control: the same per-coin paper book with NO extra rule, so C
# / B / C+B are compared with like (the live record is one best trade at a
# time, sized, and skips coins the book holds). Nothing here touches the
# order manager, the exchange or a live record.
SHADOW_VARIANTS: dict[str, tuple[bool, bool]] = {  # name -> (vol gate C, strict shorts B)
    "base": (False, False), "C": (True, False), "B": (False, True), "CB": (True, True),
}
VOL_GATE_RANK = 2 / 3      # C: skip an entry when atr_rank > 2/3 (top third)
STRICT_SHORT_SCORE = -0.40  # B: SELL only when score <= -0.40


def shadow_on(settings) -> bool:
    return bool(getattr(settings, "analyst_shadow_variants", False))


def atr_rank(hourly) -> float | None:
    """Where today's daily ATR14 (Wilder) / close sits among the last 365 daily
    values (share below it, 0..1), from the bars passed in -- the Analyst's own
    frame, closed bars only, so past data only. None with < 120 days (no gate),
    as in the autopsy (scratchpad team/doctor_analyst/compute_unseen.py)."""
    try:
        import pandas as pd

        from app.analyst.factors import _resample

        daily = _resample(hourly[["open", "high", "low", "close"]], "1D")
        hh, ll, cc = daily["high"], daily["low"], daily["close"]
        tr = pd.concat([hh - ll, (hh - cc.shift()).abs(), (ll - cc.shift()).abs()], axis=1).max(axis=1)
        natr = (tr.ewm(alpha=1 / 14, adjust=False).mean() / cc).iloc[-365:]
        if len(natr) < 120 or not math.isfinite(float(natr.iloc[-1])):
            return None
        return float((natr < natr.iloc[-1]).mean())
    except Exception:
        return None


def variant_gate(variant: str, r: dict, rank: float | None) -> str | None:
    """Why `variant` skips a call the live rules would take, or None."""
    vol, strict = SHADOW_VARIANTS[variant]
    if vol and rank is not None and rank > VOL_GATE_RANK:
        return "vol_top_third"
    if strict and r["decision"] == "SELL" and r["score"] > STRICT_SHORT_SCORE:
        return "weak_short"
    return None


async def open_variant_trades(runtime, results: dict[str, dict], ranks: dict[str, float | None]) -> list[AnalystTrade]:
    """Opens a paper trade in each variant's book for every call the live entry
    rules accept and the variant's extra rule allows, on coins where that
    variant has nothing open. Never sends an order."""
    settings = getattr(runtime, "settings", None)
    if not shadow_on(settings) or not results or await is_kill_switch_on():
        return []
    from app.gold.ingest import GOLD_SPOT_SYMBOL

    async with SessionLocal() as session:
        busy = {(v, s) for v, s in (await session.execute(select(AnalystTrade.variant, AnalystTrade.symbol).where(
            AnalystTrade.is_open.is_(True), AnalystTrade.variant.is_not(None)))).all()}
    signals = getattr(runtime, "live_signals", {}) or {}
    now = dt.datetime.now(dt.timezone.utc)
    new: list[AnalystTrade] = []
    for symbol, r in results.items():
        if symbol == GOLD_SPOT_SYMBOL or _call_gate(r, settings) is not None:
            continue
        if not agrees_with_book(r["decision"], signals.get(symbol)):
            continue
        price = _live_price(runtime, symbol) or r["entry"]
        stop_dist, tgt_dist = r["entry"] - r["stop_loss"], r["take_profit"] - r["entry"]
        stop, target = price - stop_dist, price + tgt_dist
        atr = r.get("atr") or abs(stop_dist) / 1.5
        rank = ranks.get(symbol)
        for v in SHADOW_VARIANTS:
            if (v, symbol) in busy or variant_gate(v, r, rank) is not None:
                continue
            new.append(AnalystTrade(
                symbol=symbol, side=r["decision"], conviction=r["conviction"], score=r["score"],
                size_factor=r.get("size_factor", 1.0), entry=price, stop_loss=stop, take_profit=target,
                initial_stop=stop, atr=atr, peak=price, contracts=0, live=False, variant=v, entry_time=now,
                reasons={"variant": v, "atr_rank": rank, "families": r.get("families"), "notes": r.get("notes")},
            ))
    if new:
        async with SessionLocal() as session:
            session.add_all(new)
            await session.commit()
        logger.info("Analyst shadow: opened %s", ", ".join(f"{t.variant} {t.side} {t.symbol}" for t in new))
    return new


async def manage_variant_trades(runtime, latest: dict[str, dict]) -> None:
    """The live exit rules on the shadow books: update_exit with the live
    settings, FLIP on a strong opposite call, a target filled at its level and
    a stop where the tick crossed it (as a live paper record). Paper only."""
    settings = runtime.settings
    async with SessionLocal() as session:
        trades = (await session.execute(select(AnalystTrade).where(
            AnalystTrade.is_open.is_(True), AnalystTrade.variant.is_not(None)))).scalars().all()
        if not trades:
            return
        for t in trades:
            price = _live_price(runtime, t.symbol) or await _stored_price(t.symbol)
            if price is None:
                continue
            atr = t.atr or abs(t.entry - (t.initial_stop or t.stop_loss)) / 1.5
            call = latest.get(t.symbol)
            u = update_exit(side=t.side, entry=t.entry, stop=t.stop_loss, initial_stop=t.initial_stop or t.stop_loss,
                            atr=atr, peak=t.peak or t.entry, price=price, take_profit=t.take_profit,
                            extended=bool(t.extended), still_strong=_strong(call, t.side, settings), settings=settings)
            t.stop_loss, t.take_profit, t.peak, t.extended = u.stop, u.take_profit, u.peak, u.extended
            action = u.action
            if action is None and _strong(call, "SELL" if t.side == "BUY" else "BUY", settings):
                action = "FLIP"
            if action is not None:
                _close_record(t, t.take_profit if action == "TP" else price, action)
        await session.commit()
