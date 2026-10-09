from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.data.app_state import get_state, is_kill_switch_on, set_state
from app.data.market_cache import MarketDataCache
from app.risk.position_sizing import select_leverage

logger = logging.getLogger(__name__)


@dataclass
class ExternalFlow:
    """A deposit (+) or withdrawal (-) on the exchange wallet -- money that
    moved in or out, not money the strategy made or lost."""
    uid: str
    time: dt.datetime
    amount: float


@dataclass
class DailyPnLTracker:
    day: dt.date = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).date())
    realized_pnl: float = 0.0
    starting_equity: float = 0.0
    # When starting_equity was taken. A deposit/withdrawal AFTER it is not in
    # it and must be added; one before it is already included.
    anchored_at: dt.datetime | None = None

    def roll_if_new_day(self, equity: float) -> None:
        today = dt.datetime.now(dt.timezone.utc).date()
        if today != self.day:
            self.day = today
            self.realized_pnl = 0.0
            self.starting_equity = equity
            self.anchored_at = dt.datetime.now(dt.timezone.utc)

    def anchor(self, equity: float) -> None:
        self.starting_equity = equity
        self.anchored_at = dt.datetime.now(dt.timezone.utc)

    def add_realized(self, pnl: float) -> None:
        self.realized_pnl += pnl


class RiskManager:
    """Enforces the hard limits that separate a systematic trading system from a
    toy script: per-symbol and portfolio exposure caps, a max-leverage ceiling,
    a daily-loss kill-switch that flattens and halts trading, and a circuit
    breaker that respects Delta's own system_status / trading_status signals.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.daily_pnl = DailyPnLTracker()
        # Three supervisors apply flows concurrently; without this two of them
        # read the same "applied" list and add one deposit twice (seen live).
        self._flows_lock = asyncio.Lock()

    async def load_daily_pnl_state(self) -> None:
        """Hydrates the in-memory daily-PnL tracker from persisted state at
        startup. Without this, the tracker (and therefore the daily-loss
        kill switch) lives purely in memory -- any `--reload` restart, or a
        crash/recovery, silently re-arms it at zero realized losses, even
        mid-day after real losses already happened. Only restores state from
        the SAME calendar day; a stale prior day's numbers are discarded
        exactly like `roll_if_new_day` would discard them anyway.
        """
        day_str = await get_state("daily_pnl_day")
        if not day_str:
            return
        try:
            stored_day = dt.date.fromisoformat(day_str)
        except ValueError:
            return
        if stored_day != dt.datetime.now(dt.timezone.utc).date():
            return
        self.daily_pnl.day = stored_day
        self.daily_pnl.realized_pnl = float(await get_state("daily_pnl_realized", "0.0") or 0.0)
        self.daily_pnl.starting_equity = float(await get_state("daily_pnl_starting_equity", "0.0") or 0.0)
        raw_anchor = await get_state("daily_pnl_anchored_at")
        self.daily_pnl.anchored_at = dt.datetime.fromisoformat(raw_anchor) if raw_anchor else None

    async def rebase_for_mode_switch(self, equity: float) -> None:
        """Re-anchors the drawdown peak and today's starting equity to the
        account being SWITCHED TO. Paper and live share one set of risk state,
        so without this the first check after paper ($5,732 peak) -> live
        ($3.67 account) reads as a 99.9% drawdown and an equal daily loss,
        trips both breakers, flattens and forces paper within seconds. The
        limits themselves are unchanged -- they now measure the right account."""
        await self.set_high_water_mark(equity)
        self.daily_pnl.day = dt.datetime.now(dt.timezone.utc).date()
        self.daily_pnl.realized_pnl = 0.0
        self.daily_pnl.anchor(max(equity, 0.0))
        await self._persist_daily_pnl_state()

    async def set_high_water_mark(self, equity: float) -> None:
        """Sets the drawdown peak and stamps WHEN, so a later deposit or
        withdrawal can tell whether the peak already contains it."""
        await set_state("equity_high_water_mark", str(max(equity, 0.0)))
        await set_state("equity_high_water_mark_at", dt.datetime.now(dt.timezone.utc).isoformat())

    async def apply_external_flows(self, flows: list[ExternalFlow]) -> list[ExternalFlow]:
        """Keeps deposits and withdrawals out of P&L. Without this a deposit
        reads as profit (2026-10-05: a Rs 1,000 deposit, +$11.76 on a $3.24
        day, tripped the +10% profit target, flattened and halted) and a
        withdrawal reads as a loss that trips the daily-loss and drawdown
        halts. Each flow is applied exactly once (by id):

        - today's starting equity moves by the amount if the flow came AFTER
          it was taken (one before it is already inside it);
        - the drawdown peak moves by the amount if the flow came after the
          peak was last set (a peak set later already contains it). With no
          stamp yet -- the first run after this shipped -- the peak is taken
          to contain every flow so far, and stamped now.

        Returns the flows newly applied."""
        async with self._flows_lock:
            return await self._apply_external_flows(flows)

    async def _apply_external_flows(self, flows: list[ExternalFlow]) -> list[ExternalFlow]:
        raw = await get_state("external_flows_applied")
        applied: list[str] = json.loads(raw) if raw else []
        seen = set(applied)
        now = dt.datetime.now(dt.timezone.utc)
        raw_hwm_at = await get_state("equity_high_water_mark_at")
        if not raw_hwm_at:
            raw_hwm_at = now.isoformat()
            await set_state("equity_high_water_mark_at", raw_hwm_at)
        hwm_at = dt.datetime.fromisoformat(raw_hwm_at)
        day_start = dt.datetime.combine(self.daily_pnl.day, dt.time(), tzinfo=dt.timezone.utc)
        anchored_at = self.daily_pnl.anchored_at or day_start

        new: list[ExternalFlow] = []
        for flow in sorted(flows, key=lambda f: f.time):
            if flow.uid in seen:
                continue
            seen.add(flow.uid)
            applied.append(flow.uid)
            new.append(flow)
            if flow.time.date() == self.daily_pnl.day and flow.time >= anchored_at and self.daily_pnl.starting_equity > 0:
                self.daily_pnl.starting_equity = max(self.daily_pnl.starting_equity + flow.amount, 0.0)
            if flow.time >= hwm_at:
                raw_hwm = await get_state("equity_high_water_mark")
                if raw_hwm:
                    await set_state("equity_high_water_mark", str(max(float(raw_hwm) + flow.amount, 0.0)))
            logger.info(
                "External %s of %.2f USD at %s kept out of P&L (day start now %.2f)",
                "deposit" if flow.amount >= 0 else "withdrawal", abs(flow.amount), flow.time.isoformat(),
                self.daily_pnl.starting_equity,
            )
        if new:
            await set_state("external_flows_applied", json.dumps(applied[-500:]))
            await self._persist_daily_pnl_state()
        return new

    async def _persist_daily_pnl_state(self) -> None:
        await set_state("daily_pnl_day", self.daily_pnl.day.isoformat())
        await set_state("daily_pnl_realized", str(self.daily_pnl.realized_pnl))
        await set_state("daily_pnl_starting_equity", str(self.daily_pnl.starting_equity))
        if self.daily_pnl.anchored_at is not None:
            await set_state("daily_pnl_anchored_at", self.daily_pnl.anchored_at.isoformat())

    async def check_daily_loss(self, equity: float) -> tuple[bool, str]:
        """Mark-to-market (realized + unrealized), not realized-only: a
        position sitting deep underwater but not yet closed is real risk
        already taken, not risk that only counts once a stop eventually
        fires. Realized-only would miss exactly the case a daily-loss
        circuit breaker exists for -- a bad day that's still open, not
        just one already booked.

        Equity-relative (daily_loss_limit_pct of the day's STARTING equity),
        symmetric with check_daily_profit_target below -- not a fixed dollar
        figure, which would silently become a tighter and tighter fraction of
        the account as it grows.
        """
        self.daily_pnl.roll_if_new_day(equity)
        if self.daily_pnl.starting_equity == 0.0:
            self.daily_pnl.anchor(equity)
        await self._persist_daily_pnl_state()

        if self.daily_pnl.starting_equity <= 0:
            return True, ""  # nothing to take a percentage of yet

        total_pnl_today = equity - self.daily_pnl.starting_equity
        limit_usd = self.daily_pnl.starting_equity * self.settings.daily_loss_limit_pct
        if total_pnl_today <= -limit_usd:
            return False, (
                f"Daily loss limit breached: equity {total_pnl_today:.2f} USD today "
                f"(started {self.daily_pnl.starting_equity:.2f}, now {equity:.2f}) "
                f"<= -{limit_usd:.2f} USD ({self.settings.daily_loss_limit_pct:.0%})"
            )
        return True, ""

    async def check_daily_profit_target(self, equity: float) -> tuple[bool, str]:
        """Mirror of check_daily_loss for the upside: once today's mark-to-
        market gain reaches `daily_profit_target_pct` of the day's starting
        equity, stop -- lock the day in rather than risk giving it back.
        Shares the exact same DailyPnLTracker/starting_equity anchor as
        check_daily_loss (one "today" concept, two symmetric limits on it).
        """
        self.daily_pnl.roll_if_new_day(equity)
        if self.daily_pnl.starting_equity == 0.0:
            self.daily_pnl.anchor(equity)
        await self._persist_daily_pnl_state()

        if self.settings.daily_profit_target_pct <= 0:
            # 0 = off. Without this, a 0 target is "any non-negative day" and
            # halts trading on every flat or green day. (The day anchor above
            # is still kept -- the status and exit-level displays read it.)
            return True, ""

        if self.daily_pnl.starting_equity <= 0:
            return True, ""  # nothing to take a percentage of yet

        total_pnl_today = equity - self.daily_pnl.starting_equity
        target_usd = self.daily_pnl.starting_equity * self.settings.daily_profit_target_pct
        if total_pnl_today >= target_usd:
            return False, (
                f"Daily profit target reached: equity +{total_pnl_today:.2f} USD today "
                f"(started {self.daily_pnl.starting_equity:.2f}, now {equity:.2f}) "
                f">= +{target_usd:.2f} USD ({self.settings.daily_profit_target_pct:.0%})"
            )
        return True, ""

    async def check_max_drawdown(self, equity: float) -> tuple[bool, str]:
        """Peak-to-trough drawdown against a PERSISTENT high-water mark --
        the multi-day risk the daily breakers are blind to by construction
        (three consecutive 9% down days each pass check_daily_loss while the
        account is down 27% from its peak).

        The high-water mark ratchets UP only: it's raised whenever equity
        makes a new high and otherwise left alone, so a drawdown is always
        measured from the real peak, not from wherever the last check
        happened to run. Re-based to current equity only on a deliberate
        manual acknowledgment (see routes_trading.update_kill_switch), never
        automatically -- otherwise a halt would clear itself on any bounce,
        which is exactly how a drawdown policy stops being a policy.
        """
        if equity <= 0:
            return True, ""

        raw_hwm = await get_state("equity_high_water_mark")
        high_water_mark = float(raw_hwm) if raw_hwm else 0.0

        if equity > high_water_mark:
            await self.set_high_water_mark(equity)
            return True, ""

        if high_water_mark <= 0:
            return True, ""

        drawdown_pct = (high_water_mark - equity) / high_water_mark
        if drawdown_pct >= self.settings.max_drawdown_pct:
            return False, (
                f"Max drawdown breached: equity {equity:.2f} is {drawdown_pct:.1%} below the "
                f"high-water mark of {high_water_mark:.2f} "
                f"(limit {self.settings.max_drawdown_pct:.0%})"
            )
        return True, ""

    async def kill_switch_status(self, equity: float) -> dict | None:
        """Human-facing explanation of WHY the kill switch is currently on --
        the dashboard shows this as a banner so an engaged switch is never a
        mystery. Distinguishes the three ways it can end up on, in priority
        order: today's own auto-triggered-day flags (set by
        _daily_profit_target_supervisor / _daily_loss_supervisor in
        scheduler.py) if either matches today, otherwise a manual toggle
        (dashboard button, or any other engagement not caused by one of those
        two breach checks today).
        """
        if not await is_kill_switch_on():
            return None

        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        starting = self.daily_pnl.starting_equity
        pnl_usd = equity - starting if starting else 0.0
        pnl_pct = (pnl_usd / starting) if starting else 0.0

        # Checked FIRST and deliberately not day-scoped: a drawdown halt is
        # the most serious state the switch can be in, and unlike the daily
        # flags it persists across day boundaries until acknowledged.
        if (await get_state("drawdown_halt_active")) == "on":
            raw_hwm = await get_state("equity_high_water_mark")
            hwm = float(raw_hwm) if raw_hwm else 0.0
            dd = (hwm - equity) / hwm if hwm > 0 else 0.0
            return {
                "type": "max_drawdown",
                "message": (
                    f"Max drawdown halt: {dd:.1%} below the high-water mark of ${hwm:,.2f} "
                    f"(limit {self.settings.max_drawdown_pct:.0%}). This does NOT clear at the "
                    "next day -- turning the kill switch off acknowledges it and re-bases the peak."
                ),
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "starting_equity": starting,
                "equity": equity,
            }

        if await get_state("daily_profit_target_auto_triggered_day") == today:
            kind = "profit_target"
            message = (
                f"Daily profit target reached (+{pnl_pct:.1%}) -- trading halted "
                "for the rest of today to lock in the gain."
            )
        elif await get_state("daily_kill_switch_auto_triggered_day") == today:
            kind = "daily_loss"
            message = f"Daily loss limit reached ({pnl_pct:.1%}) -- trading halted for the rest of today."
        else:
            kind = "manual"
            message = "Kill switch manually engaged."

        return {
            "type": kind,
            "message": message,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "starting_equity": starting,
            "equity": equity,
        }

    async def record_realized_pnl(self, pnl: float) -> None:
        self.daily_pnl.add_realized(pnl)
        await self._persist_daily_pnl_state()

    def system_status_ok(self, cache: MarketDataCache) -> tuple[bool, str]:
        status = (cache.system_status or {}).get("status", "live")
        if status in ("maintenance",):
            return False, f"Delta system status is '{status}' -- trading paused"
        return True, ""

    def cap_position_size(
        self,
        proposed_contracts: int,
        symbol: str,
        price: float,
        contract_value: float,
        equity: float,
        current_portfolio_notional_usd: float,
        current_portfolio_margin_usd: float = 0.0,
        quality: float = 1.0,
        leverage: float | None = None,
    ) -> int:
        """`leverage` is the leverage the order will actually be placed at,
        when the caller chose one (the book's margin leverage, see
        Settings.book_margin_use). Margin is then measured at THAT leverage,
        and the per-trade margin cap -- a leftover of the fixed-margin-per-trade
        sizing, where margin was the risk unit -- no longer applies: the book
        already sets each symbol's notional, and at 1-2x a single position's
        margin legitimately exceeds 10% of equity. Total margin is capped at
        book_margin_use, the same share the leverage was chosen to fit in."""
        if proposed_contracts == 0 or price <= 0:
            return 0

        notional_per_contract = contract_value * price
        proposed_notional = abs(proposed_contracts) * notional_per_contract

        max_symbol_notional = equity * self.settings.max_leverage
        proposed_notional = min(proposed_notional, max_symbol_notional)

        max_portfolio_notional = equity * self.settings.max_portfolio_leverage
        remaining_portfolio_budget = max(0.0, max_portfolio_notional - current_portfolio_notional_usd)
        proposed_notional = min(proposed_notional, remaining_portfolio_budget)

        # Notional/leverage caps alone don't stop total MARGIN committed
        # across every simultaneous position from exceeding the account --
        # with each strategy now trading its own independent position, that's
        # a real scenario, not a theoretical one. A real exchange would
        # reject this order outright as "insufficient margin"; this is that
        # same check, scaling the order down to what actually still fits.
        # Margin uses the SAME leverage tier this trade will actually be
        # stamped with (see PaperBroker), so this cap and the resulting
        # position's displayed margin always agree.
        if proposed_notional > 0:
            book_leverage = leverage is not None and leverage > 0
            if not book_leverage:
                leverage = select_leverage(quality, self.settings.min_leverage, self.settings.max_leverage)
            proposed_margin = proposed_notional / leverage

            # Per-trade cap FIRST: without this, one trade alone can request
            # margin up to the whole per-trade budget, which -- being the
            # first order in -- would pass the aggregate check below by
            # itself and leave nothing for the other strategies/symbols
            # queued behind it. Capping each trade's own footprint
            # guarantees several can coexist instead of the first one
            # winning the whole budget.
            max_single_trade_margin = equity * self.settings.max_margin_per_trade_pct
            if not book_leverage and proposed_margin > max_single_trade_margin:
                proposed_notional = max_single_trade_margin * leverage
                proposed_margin = max_single_trade_margin

            max_total_margin = equity * (
                max(self.settings.max_margin_utilization, self.settings.book_margin_use)
                if book_leverage else self.settings.max_margin_utilization
            )
            remaining_margin_budget = max(0.0, max_total_margin - current_portfolio_margin_usd)
            if proposed_margin > remaining_margin_budget:
                scale = remaining_margin_budget / proposed_margin if proposed_margin > 0 else 0.0
                proposed_notional *= scale

        capped_contracts = int(proposed_notional / notional_per_contract) if notional_per_contract else 0
        return capped_contracts if proposed_contracts >= 0 else -capped_contracts

    async def evaluate_kill_switch(self, equity: float) -> bool:
        """Returns True only if the kill-switch has been MANUALLY engaged (via
        the Kill Switch button / API) -- a daily-loss breach no longer
        auto-engages it. It only logs a warning so the breach stays visible;
        deciding whether to actually halt trading is left entirely to a
        manual toggle. (Previously this auto-engaged and re-engaged on every
        tick for as long as the breach persisted, which fought a manual
        "turn it back off" with no way to keep trading for the rest of the
        day even deliberately.)
        """
        if await is_kill_switch_on():
            return True
        ok, reason = await self.check_daily_loss(equity)
        if not ok:
            logger.warning("Daily loss limit breached (kill switch NOT auto-engaged): %s", reason)
        return False

    async def trading_allowed(self, cache: MarketDataCache, equity: float) -> tuple[bool, str]:
        if await self.evaluate_kill_switch(equity):
            return False, "Kill switch is active"
        ok, reason = self.system_status_ok(cache)
        if not ok:
            return False, reason
        return True, ""
