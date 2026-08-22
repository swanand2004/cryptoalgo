from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.data.app_state import get_state, is_kill_switch_on, set_state
from app.data.market_cache import MarketDataCache
from app.risk.position_sizing import select_leverage

logger = logging.getLogger(__name__)


@dataclass
class DailyPnLTracker:
    day: dt.date = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).date())
    realized_pnl: float = 0.0
    starting_equity: float = 0.0

    def roll_if_new_day(self, equity: float) -> None:
        today = dt.datetime.now(dt.timezone.utc).date()
        if today != self.day:
            self.day = today
            self.realized_pnl = 0.0
            self.starting_equity = equity

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

    async def _persist_daily_pnl_state(self) -> None:
        await set_state("daily_pnl_day", self.daily_pnl.day.isoformat())
        await set_state("daily_pnl_realized", str(self.daily_pnl.realized_pnl))
        await set_state("daily_pnl_starting_equity", str(self.daily_pnl.starting_equity))

    async def check_daily_loss(self, equity: float) -> tuple[bool, str]:
        """Mark-to-market (realized + unrealized), not realized-only: a
        position sitting deep underwater but not yet closed is real risk
        already taken, not risk that only counts once a stop eventually
        fires. Realized-only would miss exactly the case a daily-loss
        circuit breaker exists for -- a bad day that's still open, not
        just one already booked.
        """
        self.daily_pnl.roll_if_new_day(equity)
        if self.daily_pnl.starting_equity == 0.0:
            self.daily_pnl.starting_equity = equity
        await self._persist_daily_pnl_state()
        total_pnl_today = equity - self.daily_pnl.starting_equity
        if total_pnl_today <= -self.settings.max_daily_loss_usd:
            return False, (
                f"Daily loss limit breached: equity {total_pnl_today:.2f} USD today "
                f"(started {self.daily_pnl.starting_equity:.2f}, now {equity:.2f}) "
                f"<= -{self.settings.max_daily_loss_usd:.2f} USD"
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
            self.daily_pnl.starting_equity = equity
        await self._persist_daily_pnl_state()

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
    ) -> int:
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
            if proposed_margin > max_single_trade_margin:
                proposed_notional = max_single_trade_margin * leverage
                proposed_margin = max_single_trade_margin

            max_total_margin = equity * self.settings.max_margin_utilization
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
