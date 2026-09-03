from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class DeltaEnv(str, Enum):
    TESTNET = "testnet"
    PRODUCTION = "production"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


# Shared by main.py's CORS middleware (HTTP) and routes_ws.py's manual origin
# check (WebSocket -- CORSMiddleware only inspects HTTP scope, so it does NOT
# protect the /ws/dashboard route at all; both must agree on one allowlist).
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    delta_api_key: str = ""
    delta_api_secret: str = ""
    delta_env: DeltaEnv = DeltaEnv.TESTNET

    delta_rest_url_production: str = "https://api.india.delta.exchange"
    delta_rest_url_testnet: str = "https://cdn-ind.testnet.deltaex.org"

    delta_ws_public_url_production: str = "wss://public-socket.india.delta.exchange"
    delta_ws_private_url_production: str = "wss://socket.india.delta.exchange"
    delta_ws_public_url_testnet: str = "wss://socket-ind-pub.testnet.deltaex.org"
    delta_ws_private_url_testnet: str = "wss://socket-ind.testnet.deltaex.org"

    symbols: str = "BTCUSD,ETHUSD,SOLUSD"

    # Safety rail: the app always boots in PAPER mode regardless of this value.
    # This setting only controls what the dashboard defaults its toggle display to.
    default_trading_mode: TradingMode = TradingMode.PAPER

    database_url: str = "sqlite+aiosqlite:///./cryptoalgo.db"

    # Leverage is a genuine, deliberate per-trade choice (see
    # position_sizing.select_leverage), not a rounded-off display value: each
    # trade picks a whole-number tier between min_leverage and max_leverage
    # based on its conviction (the ensemble signal's magnitude), weak
    # conviction using the floor and full conviction using the ceiling.
    min_leverage: float = 10.0
    max_leverage: float = 50.0
    # Equity-relative, not a fixed dollar figure -- so it scales automatically
    # as the account grows/shrinks instead of needing manual re-tuning.
    # Sized to match max_margin_utilization's break-even point (see below):
    # at max_leverage=50 and max_margin_per_trade_pct=0.10, the number of
    # full-sized concurrent trades the margin cap allows is
    # max_margin_utilization / max_margin_per_trade_pct = 4, so 4 * (equity *
    # 0.10 * 50) = 20x equity in aggregate notional is the break-even --
    # raised just enough that MARGIN decides how many trades can be open at
    # once, not this notional cap silently binding first.
    max_portfolio_leverage: float = 20
    # Fraction of equity allowed to be committed as MARGIN across every open
    # position combined (not notional -- actual margin, i.e. notional/leverage).
    # A real exchange rejects an order once required margin exceeds your
    # balance; this is that same check, with a buffer left for adverse moves
    # before liquidation rather than cutting it exactly at 100%.
    #
    # Was 0.85, sized purely off "how many trades do we want open at once,"
    # with no account taken of how CORRELATED those trades' returns actually
    # are -- an assumption never measured until now. Measured directly
    # (2026-08-31, event-driven backtest of each of the 3 live strategies in
    # isolation, per symbol, over the same real history): mean pairwise
    # correlation across all 9 strategy/symbol combinations is 0.733 (BTC
    # trend_momentum vs. graha alone: 0.955 -- they're barely different bets
    # at all). Plugged into the standard equal-weight portfolio-volatility
    # formula (sqrt(1/N + (N-1)/N * rho), N = max_concurrent_positions = 4)
    # against the independence this cap implicitly assumed (rho = 0), the
    # REALIZED portfolio volatility for the same aggregate margin
    # utilization is ~1.98x what was intended -- this book was never as
    # diversified as "N positions" made it look, so it was running roughly
    # twice the risk 0.85 was actually meant to represent. 0.40 (0.85/1.98)
    # restores the ORIGINALLY INTENDED portfolio-level risk, given what the
    # correlation actually is rather than what it was assumed to be. See
    # [[project-risk-controls]] for the full derivation and correlation
    # matrix. Revisit if a genuinely uncorrelated strategy is ever added --
    # this number is only valid for the current 3-strategy lineup's measured
    # correlation, not a universal constant.
    max_margin_utilization: float = 0.40
    # Fraction of equity committed as margin, EVERY trade -- not just a
    # ceiling anymore (per explicit request), the actual sizing target:
    # position_sizing.fixed_margin_contracts sizes every fresh entry to
    # exactly this fraction of equity as margin, at whatever leverage tier
    # the signal's conviction selects (see select_leverage). risk_manager's
    # cap_position_size still separately enforces this as a ceiling too
    # (defense in depth), which on a fresh entry is a no-op since both derive
    # from the same number and the same selected leverage. With 3 strategies
    # x 3 symbols trading independently but measured at 0.733 mean pairwise
    # correlation (see max_margin_utilization above) -- not the near-
    # independence originally assumed -- 1/10th of equity per trade now
    # allows 4 to run concurrently (0.40 / 0.10 = 4), matching
    # max_concurrent_positions below exactly rather than the previous 8-9.
    max_margin_per_trade_pct: float = 0.10
    # Hard cap on how many positions may be open SIMULTANEOUSLY across the
    # whole account (every symbol, every strategy combined) -- a position
    # count limit, separate from and in addition to the margin/leverage caps
    # above. Was 5 (from back when the margin caps alone still allowed
    # ~8-9 concurrent trades and this trimmed that down); now set to match
    # the margin cap's own natural limit of 4 exactly (see
    # max_margin_utilization's correlation-derivation comment above), so
    # this is a consistent backstop rather than a looser, redundant one.
    # Extending or flipping an EXISTING position never counts as a new one
    # against this cap -- only genuinely opening from flat does, so this
    # never blocks managing risk on positions already open.
    max_concurrent_positions: int = 4
    # Equity-relative, not a fixed dollar figure -- was a flat MAX_DAILY_LOSS_USD
    # (200) until 2026-08-23, which meant the actual loss tolerance shrank as a
    # fraction of equity every time the account grew (on a ~$6,700 account, $200
    # is only ~3%, tripping the breaker at a third of the symmetric 10% profit
    # target below). Switched to a percentage of the day's STARTING equity so
    # the two circuit breakers stay genuinely symmetric at any account size.
    daily_loss_limit_pct: float = 0.10
    # Mirror of daily_loss_limit_pct for the upside: once today's mark-to-market
    # gain reaches this fraction of the day's STARTING equity, trading halts
    # and flattens for the rest of the day (see risk_manager.check_daily_
    # profit_target / scheduler._daily_profit_target_supervisor) -- locking in
    # a good day instead of risking giving it back.
    daily_profit_target_pct: float = 0.10
    # Peak-to-trough drawdown limit, measured against a PERSISTENT equity
    # high-water mark rather than any single day's starting equity -- the one
    # universal fund risk control this system lacked until 2026-08-31.
    #
    # The gap it closes, concretely: losing 9% a day for three consecutive
    # days passes every daily check (each day individually under
    # daily_loss_limit_pct) while the account is actually down 27% from its
    # peak. Daily limits are blind to multi-day bleed by construction; this
    # is the limit that sees it.
    #
    # Deliberately a MORE serious state than the daily breakers: it does NOT
    # auto-clear at the next day boundary the way they do (see
    # scheduler._daily_kill_switch_reset, which explicitly skips while a
    # drawdown halt is active). It clears only on a deliberate manual
    # kill-switch-off, which also re-bases the high-water mark to that
    # moment's equity so an acknowledged drawdown doesn't instantly re-fire
    # off the old peak.
    max_drawdown_pct: float = 0.20
    paper_starting_balance_usd: float = 10000
    atr_stop_multiple: float = 2.5  # stop-loss distance, in ATRs, from entry
    atr_take_profit_multiple: float = 3.0  # take-profit distance, in ATRs (1.5:1 reward:risk)
    # Once a position has moved this many ATRs in its favor, the trailing
    # stop stops using the wide entry stop_distance (2 ATR) and switches to
    # BOTH a floor at breakeven AND a tighter trail distance. Without this,
    # a position that spikes to within a whisker of take-profit and then
    # reverses can round-trip all the way back to breakeven or even a small
    # loss, because the stop only trails at a constant 2 ATRs behind the
    # peak regardless of how close to target price got.
    # Was 1.0 -- lowered per explicit "make the trailing more aggressive"
    # request so the breakeven floor + tightened trail engage at HALF the
    # favorable move they used to (a smaller move now counts as "committed
    # enough" to start protecting), not just a tighter trail once triggered.
    # RAISED 0.5 -> 3.0 on 2026-09-03 after the exit policy was measured over
    # 9 years / 9,411 trades. At 0.5 the tightened trail engaged after only half
    # an ATR of favourable move while losers still ran the full stop, giving a
    # structural ~1:4 reward:risk that no achievable win rate can beat: the
    # strategy won 58-60% of trades and still went to -99%, needing 61-66% just
    # to break even. At 3.0 (>= the take-profit distance) the trail effectively
    # never engages before the target, so winners are no longer cut short.
    # Measured effect, selected on 2017-2023 and reported on untouched
    # 2023-2026: payoff ratio 0.59 -> 1.53, out-of-sample return -75.2% -> +1.4%.
    # That is ruin converted to roughly breakeven -- NOT a profitable strategy.
    # Deflated Sharpe remains 0.000 and only 2 of 36 exit configs made money
    # out-of-sample, so this is a risk fix, not a validated edge.
    atr_breakeven_trigger_multiple: float = 3.0
    # Was 1.0, then 0.8 -- tightened again (same request as above) so the
    # trailing stop hugs the peak noticeably closer once triggered, locking
    # in more of a favorable move sooner at the cost of getting stopped out
    # by ordinary pullbacks more often. Doesn't change the wide pre-trigger
    # stop or when tightening starts (see atr_breakeven_trigger_multiple).
    atr_tightened_trail_multiple: float = 2.0  # trail distance, in ATRs, once triggered

    # -- Signal thresholds (app/strategy/ensemble.py) -----------------------
    # Was hardcoded as SignalEnsemble constructor defaults, so the single most
    # important number in the system -- the line that decides whether anything
    # trades at all -- could not be changed without editing code.
    #
    # Why it needed lowering (measured 2026-09-03): since consolidating to one
    # blended position on 2026-09-01, ZERO trades had fired in 47+ hours. Not a
    # bug -- the live signal path was verified to match the backtest exactly
    # (fires on 15-17% of 4h bars) and the last 12 bars genuinely peaked at
    # 0.125 / 0.115 / 0.071 against the 0.15 line. The blend is a weighted
    # AVERAGE, so it can never exceed its largest member and collapses toward
    # zero whenever members disagree -- and trend_momentum vs tsmom, which
    # normally correlate +0.73, were pointing in opposite directions.
    #
    # Consolidation therefore raised the effective bar without anyone choosing
    # to: the same 0.15 that a single member cleared on ~36% of bars is only
    # cleared by the blend on ~19%.
    signal_deadband: float = 0.15
    # Exit line, deliberately looser than entry so a position is not flattened
    # by the same noise that would not have been enough to open it. None means
    # "derive as deadband * 0.4", preserving the original relationship.
    signal_exit_deadband: float | None = None
    signal_smoothing_span: int = 3

    # -- Regime-aware sizing (app/strategy/regime.py) -----------------------
    # DEFAULT OFF -- measured, and it failed its own test.
    #
    # The idea: holding contract count constant while the asset's ATR doubles
    # is silently a doubled risk position, so cut size in high-vol regimes.
    # Thresholds were fixed from convention (Wilder's ADX 25, symmetric 25/75
    # percentiles, 0.5x in high vol) BEFORE measuring, and deliberately not
    # tuned afterwards.
    #
    # Measured 2026-09-02, one pre-committed config, baseline vs regime across
    # BTC/ETH/SOL on 1h and 4h:
    #     shallower drawdown : 6/6   (mean -29.90% -> -26.62%)
    #     higher return      : 0/6   (mean  -6.74% -> -15.19%)
    #     higher Sharpe      : 0/6   (mean  -0.38  ->  -1.91)
    #
    # The risk claim held perfectly. It is still turned off, because Sharpe
    # degrading on 6 of 6 with no exceptions means the trade is inefficient:
    # a size cut that merely scaled risk down would leave Sharpe roughly flat.
    # Falling everywhere means it is cutting size on the WRONG bars -- the
    # high-vol stretches that were, in this sample, the profitable ones.
    # BTC 4h is the clearest: +26.09% -> +5.84%, surrendering 20 points of
    # return to buy 2.4 points of drawdown.
    #
    # Kept wired and configurable rather than deleted: the machinery is
    # correct and tested, the DEFAULT PARAMETERISATION is what failed. A
    # different multiplier or a vol measure that separates "volatile and
    # trending" from "volatile and chopping" could plausibly work -- but that
    # is a new hypothesis needing its own out-of-sample test, not a knob to
    # twist until this backtest looks better.
    regime_sizing_enabled: bool = False
    regime_high_vol_size_multiplier: float = 0.5
    regime_low_vol_size_multiplier: float = 1.0
    regime_adx_trend_threshold: float = 25.0  # Wilder's standard trend/no-trend line
    regime_vol_lookback_bars: int = 200
    regime_low_vol_percentile: float = 0.25
    regime_high_vol_percentile: float = 0.75

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    @property
    def underlying_assets(self) -> list[str]:
        # BTCUSD -> BTC, ETHUSD -> ETH, SOLUSD -> SOL
        return [s.replace("USD", "") for s in self.symbol_list]

    @property
    def rest_url(self) -> str:
        return (
            self.delta_rest_url_production
            if self.delta_env == DeltaEnv.PRODUCTION
            else self.delta_rest_url_testnet
        )

    @property
    def ws_public_url(self) -> str:
        return (
            self.delta_ws_public_url_production
            if self.delta_env == DeltaEnv.PRODUCTION
            else self.delta_ws_public_url_testnet
        )

    @property
    def ws_private_url(self) -> str:
        return (
            self.delta_ws_private_url_production
            if self.delta_env == DeltaEnv.PRODUCTION
            else self.delta_ws_private_url_testnet
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.delta_api_key and self.delta_api_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
