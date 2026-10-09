"""Owner alerts on Telegram: trades, halts, equity outages and a daily summary.

OFF unless BOTH `telegram_bot_token` and `telegram_chat_id` are set (backend/.env
or real env vars). Off means every function here returns without doing anything:
no network, no task, no log line.

Safety contract -- an alert must never hurt trading:
  * `notify()` is synchronous and fire-and-forget: it only schedules a background
    task on the running loop and returns at once. The caller never awaits Telegram.
  * The send has a hard timeout (SEND_TIMEOUT_S), and EVERY failure (no loop, DNS,
    HTTP error, bad token, timeout, a bug in here) is swallowed with a warning.
  * Rate limit: at most one message per event key per RATE_LIMIT_S. Trade events are
    keyed per symbol ("trade_close:ETHUSD"), so a kill switch that flattens four coins
    still reports all four; a burst on ONE key is dropped and counted, and the count is
    added to that key's next message.
  * The bot token sits in the request URL, so neither our warnings nor httpx's own
    INFO request log may ever print the URL (see _HideTelegramUrls below).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"
SEND_TIMEOUT_S = 5.0  # whole send, connect + response
RATE_LIMIT_S = 60.0
EQUITY_ALERT_AFTER_S = 300.0  # "equity unavailable for more than 5 min"
MAX_TEXT = 3900  # Telegram's limit is 4096 characters

_last_sent: dict[str, float] = {}  # event key -> monotonic time of the last send
_suppressed: dict[str, int] = {}  # event key -> messages dropped by the rate limit since
_tasks: set[asyncio.Task] = set()  # strong refs, so a pending send is not garbage-collected
_equity = {"down_since": None, "alerted": False}


def _now() -> float:
    return time.monotonic()  # one seam, so tests move this clock and not asyncio's


class _HideTelegramUrls(logging.Filter):
    """httpx logs every request URL at INFO ("HTTP Request: POST https://...") and the
    app logs INFO to app.log -- for Telegram that URL contains the bot token. Drop
    those records (Delta request lines are untouched)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "api.telegram.org" not in record.getMessage()
        except Exception:
            return True


_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(f, _HideTelegramUrls) for f in _httpx_logger.filters):
    _httpx_logger.addFilter(_HideTelegramUrls())


def _credentials(settings: Settings | None) -> tuple[str, str] | None:
    s = settings or get_settings()
    token = (getattr(s, "telegram_bot_token", "") or "").strip()
    chat_id = (getattr(s, "telegram_chat_id", "") or "").strip()
    return (token, chat_id) if token and chat_id else None


def is_enabled(settings: Settings | None = None) -> bool:
    try:
        return _credentials(settings) is not None
    except Exception:
        return False


def notify(event: str, text: str, settings: Settings | None = None) -> bool:
    """Queues one Telegram message and returns immediately. True = a send was scheduled
    (not that it arrived); False = off, rate-limited, or failed to schedule. Never raises."""
    try:
        creds = _credentials(settings)
        if creds is None:
            return False
        now = _now()
        last = _last_sent.get(event)
        if last is not None and now - last < RATE_LIMIT_S:
            _suppressed[event] = _suppressed.get(event, 0) + 1
            return False
        loop = asyncio.get_running_loop()  # RuntimeError outside a loop -> swallowed below
        dropped = _suppressed.pop(event, 0)
        if dropped:
            text += f"\n(+{dropped} more '{event}' alert(s) were rate-limited before this one)"
        _last_sent[event] = now
        task = loop.create_task(_send(creds[0], creds[1], text))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        return True
    except Exception as exc:
        logger.warning("Telegram alert '%s' not queued (%s)", event, type(exc).__name__)
        return False


async def _send(token: str, chat_id: str, text: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_S) as client:
            resp = await asyncio.wait_for(
                client.post(
                    f"{TELEGRAM_API}/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text[:MAX_TEXT], "disable_web_page_preview": True},
                ),
                timeout=SEND_TIMEOUT_S,
            )
        if resp.status_code != 200:
            logger.warning("Telegram alert not delivered (HTTP %s)", resp.status_code)
    except Exception as exc:
        # Type name only: an httpx error's text can contain the URL, i.e. the token.
        logger.warning("Telegram alert failed (%s)", type(exc).__name__)


# --- Event helpers: the hooks call these, never notify() with hand-built text. ---

def _px(price) -> str:
    try:
        return f"{float(price):,.6g}"
    except (TypeError, ValueError):
        return "?"


def trade_opened(symbol: str, side: str, size: int, price: float, mode: str = "live") -> None:
    notify(f"trade_open:{symbol}", f"OPENED {mode.upper()} {side.upper()} {symbol}\n"
                                   f"size {abs(size)} contract(s) @ {_px(price)}")


def trade_closed(symbol: str, side: str, size: int, entry: float, exit_price: float,
                 pnl: float, fees: float, reason: str, mode: str = "live") -> None:
    net = pnl - fees
    notify(f"trade_close:{symbol}", f"CLOSED {mode.upper()} {side.upper()} {symbol} ({reason})\n"
                                    f"{abs(size)} contract(s): {_px(entry)} -> {_px(exit_price)}\n"
                                    f"P&L {net:+.4f} USD net (gross {pnl:+.4f}, fees {fees:.4f})")


def kill_switch_on(reason: str) -> None:
    notify("kill_switch", f"KILL SWITCH ON -- trading halted, positions being flattened.\nReason: {reason}")


def equity_read_failed(reason: str) -> None:
    """Called on EVERY failed Delta balance read. Alerts once, when failures have
    lasted EQUITY_ALERT_AFTER_S without a single good read in between."""
    try:
        if not is_enabled():
            return
        now = _now()
        if _equity["down_since"] is None:
            _equity["down_since"] = now
        if not _equity["alerted"] and now - _equity["down_since"] >= EQUITY_ALERT_AFTER_S:
            _equity["alerted"] = True
            minutes = (now - _equity["down_since"]) / 60
            notify("equity_unavailable", f"EQUITY UNAVAILABLE for {minutes:.0f} min -- Delta balance reads "
                                         f"keep failing ({reason}). Risk checks are skipping until it recovers.")
    except Exception as exc:
        logger.warning("Telegram equity-outage tracking failed (%s)", type(exc).__name__)


def equity_read_ok() -> None:
    """Called on every good balance read: clears the outage, and says so if we alerted."""
    if _equity["down_since"] is None:
        return
    alerted = _equity["alerted"]
    _equity["down_since"], _equity["alerted"] = None, False
    if alerted:
        notify("equity_restored", "Equity readable again -- Delta balance reads recovered.")


async def send_daily_summary(runtime) -> None:
    """Scheduler job, 03:35 UTC = 09:05 IST: YESTERDAY's result (owner 2026-10-09:
    "every next day yesterday's PnL"). Each section is best-effort: a failing
    read becomes "unavailable" in the message instead of cancelling it."""
    if not is_enabled(getattr(runtime, "settings", None)):
        return
    try:
        text = await _build_summary(runtime)
    except Exception as exc:
        logger.warning("Telegram daily summary not built (%s)", type(exc).__name__)
        return
    notify("daily_summary", text, settings=getattr(runtime, "settings", None))


async def _build_summary(runtime) -> str:
    from sqlalchemy import func, select

    from app.data.app_state import get_trading_mode, is_kill_switch_on
    from app.data.db import SessionLocal
    from app.data.models import AnalystTrade, ClosedTrade

    from app.data.models import EquitySnapshot

    now = dt.datetime.now(dt.timezone.utc)
    today = dt.datetime.combine(now.date(), dt.time())  # naive UTC, like the DB columns
    day_start = today - dt.timedelta(days=1)  # yesterday 00:00 UTC
    mode = await get_trading_mode()
    lines = [f"Yesterday {day_start:%Y-%m-%d} (UTC day) -- {mode.upper()} mode, "
             f"kill switch {'ON' if await is_kill_switch_on() else 'off'}"]

    try:  # equity change over yesterday from the minute equity snapshots
        async with SessionLocal() as session:
            first = (await session.execute(select(EquitySnapshot.equity).where(
                EquitySnapshot.mode == mode, EquitySnapshot.timestamp >= day_start, EquitySnapshot.timestamp < today)
                .order_by(EquitySnapshot.timestamp.asc()).limit(1))).scalar_one_or_none()
            last = (await session.execute(select(EquitySnapshot.equity).where(
                EquitySnapshot.mode == mode, EquitySnapshot.timestamp >= day_start, EquitySnapshot.timestamp < today)
                .order_by(EquitySnapshot.timestamp.desc()).limit(1))).scalar_one_or_none()
        if first and last:
            ch = last - first
            lines.append(f"Yesterday's P&L: {ch:+.2f} USD ({ch / first:+.2%}), equity {first:.2f} -> {last:.2f}"
                         " (deposits/withdrawals included)")
        else:
            lines.append("Yesterday's P&L: no equity records")
    except Exception:
        lines.append("Yesterday's P&L: unavailable")

    equity = None
    try:
        equity = await runtime.order_manager.get_equity()
        lines.append(f"Equity: {equity:.2f} USD")
    except Exception as exc:
        lines.append(f"Equity: unavailable ({type(exc).__name__})")

    try:
        tracker = runtime.risk_manager.daily_pnl
        if equity is not None and tracker.day == now.date() and tracker.starting_equity > 0:
            pnl = equity - tracker.starting_equity
            lines.append(f"Today so far (since 00:00 UTC): {pnl:+.2f} USD ({pnl / tracker.starting_equity:+.2%})")
        else:
            lines.append("Day P&L: no anchor for today yet")
    except Exception:
        lines.append("Day P&L: unavailable")

    try:
        positions = await runtime.order_manager.get_positions()
        held = [f"{p.symbol} {p.size:+d}" for p in positions.values() if p.size]
        lines.append("Open: " + (", ".join(held) if held else "flat"))
    except Exception:
        lines.append("Open: unavailable")

    try:
        async with SessionLocal() as session:
            n, net = (await session.execute(
                select(func.count(ClosedTrade.id), func.coalesce(func.sum(ClosedTrade.realized_pnl - ClosedTrade.fees), 0.0))
                .where(ClosedTrade.mode == mode, ClosedTrade.exit_time >= day_start, ClosedTrade.exit_time < today)
            )).one()
            opened, live = (await session.execute(
                select(func.count(AnalystTrade.id), func.coalesce(func.sum(AnalystTrade.live), 0))
                .where(AnalystTrade.entry_time >= day_start, AnalystTrade.entry_time < today,
                       AnalystTrade.variant.is_(None))
            )).one()
        lines.append(f"Trades closed yesterday: {n} (net {float(net):+.4f} USD)")
        lines.append(f"Analyst trades opened yesterday: {opened} ({int(live)} live)")
    except Exception:
        lines.append("Yesterday's trades: unavailable")

    try:
        from app.analyst import service

        calls = sorted(service._latest.items())
        if calls:
            lines.append("Analyst calls: " + ", ".join(
                f"{s} {r.get('decision', '?')} {r.get('conviction', '?')}/10" for s, r in calls))
    except Exception:
        pass
    return "\n".join(lines)
