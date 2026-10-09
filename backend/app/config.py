from __future__ import annotations

import os
import re

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
# Vite dev server port, moved 5173 -> 5180 on 2026-09-08. The regex below
# already admits any loopback port, so this list is the documented default
# rather than the thing that makes it work.
ALLOWED_ORIGINS = ["http://localhost:5180", "http://127.0.0.1:5180"]

# The public URL when deployed (e.g. "https://dash.example.com"), comma-
# separated if several. Only ever set alongside DASHBOARD_PASSWORD (see
# app/api/auth.py): the origin check stops OTHER sites' pages from reading the
# socket, it does not stop a person typing the URL into a browser.
ALLOWED_ORIGINS += [o.strip().rstrip("/") for o in os.environ.get("PUBLIC_ORIGINS", "").split(",") if o.strip()]

# Any private/loopback address on the dev port is also acceptable.
#
# Why this exists: the Vite dev server was changed on 2026-09-03 to bind all
# interfaces (host: true), because binding only "localhost" resolved to the
# IPv6 loopback and left the dashboard unreachable over IPv4. That fix made
# Vite advertise a LAN URL (e.g. http://192.168.0.101:5173), and opening it
# sent an Origin the static allowlist above rejected -- every dashboard
# WebSocket got a 403 with no explanation of why.
#
# The protection that actually matters is unchanged: a page on the public
# internet still cannot open this socket and read live equity/positions. What
# is now allowed is only loopback and RFC1918 private ranges -- i.e. this
# machine, or a device on the same home network, which is exactly who is
# meant to reach a locally-hosted dashboard.
_PRIVATE_ORIGIN_RE = re.compile(
    r"^https?://("
    r"localhost"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|\[::1\]"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?$"
)


def is_allowed_origin(origin: str | None, host: str | None = None) -> bool:
    """True if `origin` may open the dashboard socket.

    A missing Origin is allowed: non-browser clients (the project's own
    tests, a CLI websocket client) do not send one, and the header is only a
    meaningful signal when a browser sets it.

    Same-origin is allowed: when the backend serves the dashboard itself (the
    deployed setup), the page's Origin host equals the request's Host, which
    is what a cross-site hijack can never forge. That is what lets a public
    tunnel URL work without being listed anywhere in advance.
    """
    if origin is None:
        return True
    if host and origin.split("://", 1)[-1].rstrip("/") == host:
        return True
    if origin in ALLOWED_ORIGINS:
        return True
    extra = [o.strip().rstrip("/") for o in get_settings().public_origins.split(",") if o.strip()]
    if origin.rstrip("/") in extra:
        return True
    return bool(_PRIVATE_ORIGIN_RE.match(origin))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    delta_api_key: str = ""
    delta_api_secret: str = ""
    # Password gate for a public URL (app/api/auth.py). Read through Settings,
    # so backend/.env works as well as a real environment variable -- reading
    # os.environ alone missed .env and left the site open with no error.
    # Which local address family signed Delta requests leave from: "auto",
    # "ipv6" or "ipv4". Delta's IP whitelist matches the SOURCE address, and a
    # dual-stack server alternates between its two, so a whitelist holding one
    # of them fails about half the time. Pin the family whose address is
    # whitelisted. The VPS uses "ipv4": its IPv6 went "dadfailed" (duplicate
    # address on the provider's network) after a reboot on 2026-10-02 and has
    # carried no traffic since.
    delta_ip_family: str = "auto"
    dashboard_user: str = "admin"
    dashboard_password: str = ""
    public_origins: str = ""  # extra allowed Origins, comma-separated
    # Owner alerts on Telegram (app/notify/telegram.py): trades, kill switch,
    # equity outages, 18:00 IST summary. BOTH empty by default = OFF (no calls).
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    delta_env: DeltaEnv = DeltaEnv.TESTNET

    delta_rest_url_production: str = "https://api.india.delta.exchange"
    delta_rest_url_testnet: str = "https://cdn-ind.testnet.deltaex.org"

    delta_ws_public_url_production: str = "wss://public-socket.india.delta.exchange"
    delta_ws_private_url_production: str = "wss://socket.india.delta.exchange"
    delta_ws_public_url_testnet: str = "wss://socket-ind-pub.testnet.deltaex.org"
    delta_ws_private_url_testnet: str = "wss://socket-ind.testnet.deltaex.org"

    # BNBUSD added 2026-09-07 -- the one universe change out of 30 candidates
    # that improved train, validation AND test. See BINANCE_SPOT_SYMBOL in
    # app/data/ingest_binance_history.py for why nothing wider was taken.
    symbols: str = "BTCUSD,ETHUSD,SOLUSD,BNBUSD"
    # Size-aware universe (2026-10-04). Empty = off: every symbol above trades,
    # exactly as before. When set, these are the CORE symbols and every OTHER
    # symbol above is a small-account filler. A core symbol trades only while
    # one contract is at most `affordable_contract_fraction` x equity; the
    # fillers trade only while the balance cannot hold every core symbol.
    #
    # Why: whole contracts. One ETH contract is $27, BTC $85, SOL $122. On a
    # small balance the core book cannot be sized -- either nothing trades or
    # the min-lot fallback rounds UP to several times the intended risk.
    # Simulated with the bot's own sizing/rebalance/limit rules, 89 one-year
    # runs (2018-05..2025-09), today's contract sizes, median 1y [runs halted
    # at -60%]: $2.92: today's 1 ETH @9x -29% [48%] vs size-aware +28% [16%];
    # $30: +15% [30%] vs +42% [0%]; $100: +7% [55%] vs +39% [2%]; $300: +52%
    # [24%] vs +65% [1%]; $1000: +59% [9%] vs +65% [0%]. The fillers (HBAR,
    # MANA, XLM, ADA, TRX, STX -- picked by contract size alone, $0.10-0.39)
    # run the same strategy weaker but positive in all three windows (Sharpe
    # 0.95/0.85/0.67 at 50% vol vs core 1.65/1.10/1.14), so they are a
    # stand-in for a book that cannot be held yet, never an addition to one.
    # Scratchpad: account_sim.py, alt_quality.py (session 61348aee).
    core_symbols: str = ""
    # The Analyst (app/analyst) analyses and paper-trades ONLY these -- the four
    # major coins (user, 2026-10-06), independent of what the live book holds.
    analyst_symbols: str = "BTCUSD,ETHUSD,SOLUSD,BNBUSD"
    # Who places real orders: "book" = the trend book (trend_ladder + gann);
    # "analyst" = the Analyst's single best opportunity at a time (user,
    # 2026-10-06). With "analyst" the book still computes its signals -- the
    # Analyst reads them -- but sends no orders.
    # "combined" = both: the book trades whatever it can size, and the Analyst
    # trades its best opportunity on a coin the book is NOT holding, only in
    # the book's own direction for that coin.
    trading_engine: str = "book"
    analyst_min_conviction: int = 4  # confidence >= 40% (|score| >= 0.35) -- user lowered it, 2026-10-06
    # ENTRY floor on |score| (2026-10-07, owner: "it did not give a single trade
    # all day"). Pre-registered test (scratchpad team/factors/PREREG_threshold.txt,
    # 2019-26, book gate, 1.5/1.5 ATR, 7 bps + funding): on ETH -- the only major
    # tradable at ~$14 under the 15% risk cap -- |score| >= 0.20 gave +0.16/+0.05/
    # +0.21R per trade TRAIN/VAL/TEST at ~4.6 trades/month vs +0.10/+0.06/+0.27R
    # at ~3/month for 0.35; total R/yr 9.0/3.0/11.1 vs 3.5/2.3/9.6. Inside the
    # noise, single coin. Across all 4 coins 0.35 stays slightly better in VAL:
    # revisit when equity makes BTC/BNB/SOL tradable. 0.35 = the old behaviour.
    # analyst_min_conviction still gates FLIP exits and the TP1 extension.
    analyst_min_score: float = 0.20
    analyst_risk_pct: float = 0.02  # equity risked per trade (entry to stop)
    # A small account can only buy whole contracts: one contract may risk up to
    # this share of equity; a coin whose single contract risks more is skipped.
    # 0.70 = every major coin is allowed at a ~$14 balance (user, 2026-10-06): one
    # BTC contract then risks ~22% of equity at the stop, BNB ~27%, SOL up to ~64%.
    analyst_max_trade_risk: float = 0.70
    # CRO hard ceiling on the same thing (2026-10-07); the tighter of the two
    # applies, so this can only ever skip a trade. 0.15 at ~$14 = ETH only
    # (BTC ~21%, BNB ~24%, SOL ~49% per contract are skipped until equity
    # grows). Monte Carlo over the Analyst's backtest R (+0.06..+0.17R/trade,
    # whole contracts at today's prices): chance of a 60%-drawdown halt within
    # a year 63-87% at 0.70, 39-61% at 0.25, 4-6% at 0.15. 0 = off.
    analyst_risk_cap: float = 0.15
    # Skip an Analyst trade unless the estimated isolated-margin liquidation is
    # at least this many times as far from entry as the stop (1.0 = merely
    # beyond it). Only ever skips. 0 = off.
    analyst_min_liq_to_stop: float = 1.5
    # Owner asked for 15-20x (2026-10-07). 0 = off: the lowest leverage that
    # margins the contract (today ~3x on ETH). >0: use max(computed, target),
    # but never a leverage whose liquidation sits closer than
    # analyst_min_liq_to_stop x the stop (then the highest safe whole number,
    # ~6-10x on ETH's ~6-10% stops). Pre-registered backtest (scratchpad
    # team/leverage, ETH 2019-26, 1 contract at $14): with that clamp $/yr and
    # drawdown equal the 3x results exactly (liquidation never ends a trade)
    # while the margin at risk if the bot is down falls from $9.0 to ~$2.7-4.5.
    # A FIXED 15x/20x (no clamp) puts liquidation inside the stop on 80%/97% of
    # trades: $/yr TRAIN/VAL/TEST 16.6/2.2/20.9 at 3x vs 3.2/-8.0/6.9 at 15x and
    # -3.0/-16.4/-1.0 at 20x, P(account -50% in a year) up 0.27-0.49 -> 0.50-0.90.
    # 2026-10-07 owner asked for 15-20x. A FIXED 15-20x lost money (liquidation inside the stop on
    # 80-97% of trades: scratchpad team/leverage). Target 20 WITH the liquidation clamp
    # (analyst_min_liq_to_stop) lands ~6-10x on ETH: same $/yr and drawdown as ~3x, but margin at
    # risk $9.0 -> ~$2.7-4.5 (isolated margin caps an outage/gap loss). 0 = lowest leverage that fits.
    analyst_target_leverage: float = 20.0
    analyst_trail_atr: float = 1.5  # trailing stop distance (daily ATRs) ...
    analyst_trail_atr_tight: float = 1.0  # ... tightened once the trade is +1R
    # How the Analyst's stop moves (execution.update_exit). "after_tp1" = the
    # loss doctor's FIX A, exactly as pre-registered and tested: no ATR trail and
    # no breakeven floor; the stop holds at the initial level stop until TP1, and
    # the only move is TP1's +0.5R lock on a still-strong call (never backwards).
    # "live" = the old trail from bar 1 (the rollback value). Evidence:
    # scratchpad/team/doctor_analyst/PREREG.txt + out_eval_unseen.txt -- on 10
    # unseen coins, R/yr 18.8 (to 2022) / 36.7 (2023-26) vs 13.0 / 11.0 for
    # "live"; better on 6/10 coins (a near-miss on the 7/10 rule). The tested
    # "trail only from +1R" variant (A2) was weaker, so nothing trails after TP1.
    # Owner-approved 2026-10-07 ("live with fix A").
    analyst_trail_mode: str = "after_tp1"
    # Exchange-side stop for LIVE Analyst trades: a reduce-only stop-market
    # order on Delta at the trade's stop, moved (edit, else place-then-cancel)
    # when the software stop moves >= 0.1 ATR, at most once a minute, re-placed
    # if it went missing, cancelled when the trade closes. Protects the trade
    # when the bot is down. False = software stop only (the old behaviour).
    analyst_exchange_stops: bool = True
    # Nightly re-fit of the Analyst's family weights (app/analyst/training.py,
    # 02:40 UTC). Off = factors.FAMILY_WEIGHTS exactly as before. On, new
    # weights are used only after they beat the current ones on the most recent
    # 90 days (never used in the fit) by margins fixed in code.
    analyst_training_enabled: bool = False
    # Live-bar mode (owner, 2026-10-07: "updates every minute, not every hour").
    # On: the Analyst also scores the FORMING hourly bar (built from the stored
    # 1m candles) so its call can change every minute; a BUY/SELL must then hold
    # for analyst_live_bar_hold_minutes before it can open (or flip) a trade.
    # Off = the closed-hourly-bar behaviour, byte for byte. Pre-registered
    # 15-minute-step backtest on Binance 1m data 2022-26 (scratchpad
    # team/realtime): see the realtime report for the numbers and verdict.
    analyst_live_bar: bool = False
    analyst_live_bar_hold_minutes: int = 15
    affordable_contract_fraction: float = 0.5
    # Leverage stamped on every book-sized order: the LOWEST whole number that
    # margins the whole book's target notional inside this share of equity.
    # Leverage does not change exposure here (the book sets notional); it only
    # sets margin and therefore how far away Delta's isolated liquidation
    # sits. The old 10-50x conviction tiers put it 2-10% away -- inside an
    # ordinary day for an alt -- while every backtest assumes positions
    # survive normal swings and leaves exits to the signal and the daily
    # limits. At a 1-2x book this puts liquidation ~50% away.
    book_margin_use: float = 0.85

    # Mode for a NEW database only. After that the mode is whatever was last
    # chosen and it survives restarts -- risk limits halt with the kill switch
    # rather than switching to paper. Paper by default; the server sets live.
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
    #
    # RAISED 0.10 -> 0.20 on 2026-09-07, as a direct consequence of
    # book_target_vol going to 0.50 -- not an independent loosening of risk.
    # At that dial the worst single day in nine years of backtest is -17.5% to
    # -21%, so a 10% limit would have flattened the book on ordinary bad days
    # and turned the circuit breaker into the strategy. 0.20 keeps it as what
    # it is meant to be: a breaker for the genuinely abnormal day, expected to
    # fire roughly once in the whole sample. Lower this back to 0.10 the
    # moment book_target_vol goes back to 0.30 or below.
    daily_loss_limit_pct: float = 0.20
    # Mirror of daily_loss_limit_pct for the upside: once today's mark-to-market
    # gain reaches this fraction of the day's STARTING equity, trading halts
    # and flattens for the rest of the day (see risk_manager.check_daily_
    # profit_target / scheduler._daily_profit_target_supervisor) -- locking in
    # a good day instead of risking giving it back.
    #
    # 0 = OFF. Kept at 10% because the user asked for halts on hitting limits,
    # but measured as a cost: capping the day's gain removes exactly the trend
    # days the strategy's returns come from while every losing day stays
    # uncapped. Measured on the real small account in 5
    # settings, median one-year result with vs without the cap: 1 ETH @9x
    # -30% vs -22%; @3x -36% vs +18%; @1x +36% vs +44%; 6-coin small book +19%
    # vs +28% (scratchpad small_account_sim.py / account_sim.py). Same finding
    # as the 2026-10-02 "+1%/day take-profit" study, which lost money.
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
    #
    # RAISED 0.20 -> 0.60 on 2026-09-07, again as a consequence of
    # book_target_vol = 0.50 rather than a separate decision. The backtested
    # maximum drawdown at that dial is -50%; a 20% halt would fire in the
    # first ordinary drawdown and, because this state does NOT auto-clear,
    # would stop the system permanently until manually reset. 0.60 sits above
    # the historical worst and below the ~65% a 1.3x live-vs-backtest buffer
    # implies, so it still catches a genuine regime break.
    #
    # This is the number to look at hardest before going live: it means
    # accepting that more than half the account can be down at once.
    max_drawdown_pct: float = 0.60
    paper_starting_balance_usd: float = 10000
    # -- ATR brackets: OFF for trend_ladder, and this is a measured call ------
    # Run through this repo's own backtest engine on 2017-2026 hourly history,
    # BTC/ETH/SOL, the difference is not marginal:
    #
    #   live defaults (TP 3xATR + 2-stage trail)   -100.0%   3389 trades
    #   no take-profit, trail kept                 -100.0%   1516
    #   no TP, no trail tightening                 -100.0%   1527
    #   no TP/trail, wide 6xATR stop                -99.9%   1745
    #   signal exits only                          +967.6%   1030
    #
    # The mechanism is re-entry churn, not the stop level. trend_ladder holds
    # ~214 hours and keeps its sign across the whole trend; a stop closes the
    # position, the unchanged signal re-opens it on the next bar, and the pair
    # repeats -- three times the trades, each one crystallising a loss and
    # paying two more fees. A wider stop only slows the bleed. Take-profit is
    # separately fatal for the same reason the study found: profit factor 2.31
    # lives in the right tail, and capping it removes the winners while
    # keeping every loser.
    #
    # Risk is therefore carried by position SIZE (book_target_vol, which
    # shrinks the whole book as volatility rises) plus the daily-loss kill
    # switch and drawdown supervisor -- not by per-trade brackets.
    #
    # Turning this back on requires a broker that can stay flat until the
    # signal actually flips. Neither PaperBroker nor LiveBroker does today, so
    # re-enabling it without that change reproduces the -100% row exactly.
    atr_brackets_enabled: bool = False
    # Rebalances as passive limit orders at the touch (maker fee 0.02% vs taker
    # 0.05%), falling back to market after maker_timeout_seconds. "paper" until
    # paper fills show the fee saving is real and no trades are missed; then
    # "all". See app/execution/maker.py.
    # 2026-10-07: "all" -- the execution study measured +2.37 bps/side vs taker
    # (98% filled within 5 min; scratchpad team/execution), the live-path bugs
    # (cancel of a filled order, flatten_all, post_only) were fixed, and the
    # owner approved limit orders.
    maker_execution: str = "all"  # "off" | "paper" | "all"
    maker_timeout_seconds: int = 300  # 2026-10-07 execution study: 98% fill, +2.37 bps/side; 900 adds only +0.05
    # Raw websocket ticks nothing reads (app/data/retention.py) are kept this
    # long, then pruned daily, so a 24/7 server's disk does not fill up.
    raw_tick_retention_days: float = 2.0
    # Nightly crypto-ML + gold model retrains inside the trading process (the
    # default, fine on a laptop) or NOT, when a separate low-priority job runs
    # them instead (`python -m app.jobs.retrain`, see deploy/vps/). Out of
    # process, their ~1.1 GB peak and 15-25 CPU-minutes never touch trading,
    # and all of that memory is returned to the OS when the job exits.
    retrain_in_process: bool = True
    # Small-account mode (LIVE only; 0 = off). When normal sizing rounds a
    # position to 0 contracts on a small balance, take ONE contract instead --
    # but only while total open notional stays <= this many x equity, and at
    # the LOWEST leverage the free balance can margin, so liquidation sits as
    # far away as the money allows. With $3.67 and 10x, only ETH ($27.5 a
    # contract, 7.5x) fits; BTC/BNB/SOL (21-33x) stay flat.
    live_min_lot_max_leverage: float = 0.0
    live_min_lot_margin_use: float = 0.85  # share of equity one min-lot may post as margin
    atr_stop_multiple: float = 2.5  # stop-loss distance, in ATRs, from entry
    atr_take_profit_multiple: float = 3.0  # take-profit distance, in ATRs (1.5:1 reward:risk)
    # Once a position has moved this many ATRs in its favor, the trailing
    # stop stops using the wide entry stop_distance (atr_stop_multiple ATRs)
    # and switches to
    # BOTH a floor at breakeven AND a tighter trail distance. Without this,
    # a position that spikes to within a whisker of take-profit and then
    # reverses can round-trip all the way back to breakeven or even a small
    # loss, because the stop only trails at a constant atr_stop_multiple ATRs
    # behind the peak regardless of how close to target price got.
    #
    # AS CONFIGURED THIS STAGE NEVER ENGAGES -- see the 3.0 below. The
    # mechanism is kept working and tested (tests/test_trailing_policy.py)
    # because disabling it is a measured decision that could be revisited,
    # not because it is in use.
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
    # 0.05 exactly, matching the hysteresis pair the trend_ladder study
    # validated. None would derive 0.15 * 0.4 = 0.06 -- close, but not the
    # pair that was actually measured.
    signal_exit_deadband: float | None = 0.05
    # 1 = no extra EMA smoothing. Smoothing exists to damp bar-to-bar noise in
    # signals that flip hard; trend_ladder is already a mean of twelve EMA
    # pairs squashed through tanh, so it arrives smooth. Layering another EMA
    # on top would lag it ~3 further bars and would no longer be the signal
    # the study measured. Raise it if a jumpier strategy is wired back in.
    signal_smoothing_span: int = 1

    # -- Book-level volatility targeting (app/risk/book_vol_target.py) ------
    # The ONE dial that sets how much money this system risks. Everything
    # else about the strategy is fixed by measurement; this is the choice.
    #
    # Sharpe is invariant to it, so the row you pick trades return against
    # drawdown one-for-one. Measured on the nine-year study, net of costs:
    #
    #   target   CAGR    maxDD   worst day   p99 leverage   9yr multiple
    #    0.10    15.6%  -11.0%       -3.5%          0.41x           3.7x
    #    0.15    23.6%  -16.3%       -5.3%          0.62x           6.8x
    #    0.20    31.8%  -21.5%       -7.0%          0.83x          12.1x
    #    0.30    48.4%  -31.5%      -10.5%          1.24x          35.5x
    #    0.40    65.0%  -40.9%      -14.0%          1.65x          92.2x
    #    0.50    81.0%  -49.5%      -17.5%          2.07x         213.4x
    #
    # Two things to hold onto before raising it. Backtested drawdowns
    # understate live ones -- plan for roughly 1.3x the historical figure. And
    # the rolling walk-forward (re-picking the strategy each year on prior
    # data only) returned 38.7% at 0.30, not 48.4%; the recent 2024-2026 span
    # returned 15.3%. Size for the walk-forward number.
    # SET TO 0.50 ON 2026-09-07 at explicit, repeated user direction ("very
    # aggressive"), after the trade-off was stated and reaffirmed. This is the
    # bottom row of the table above: ~81-91% CAGR on the full sample against a
    # -50% maximum drawdown, a -17% to -21% worst day, and p99 notional near
    # 2.1x equity. Liquidation is still not the binding constraint at that
    # leverage (it would take a ~48% adverse move); drawdown tolerance is.
    #
    # Two things that do not change by turning this up: the walk-forward
    # estimate is ~65% at this setting rather than the full-sample 91%, and
    # 2024-2026 alone ran at roughly a quarter of the full-sample rate. Drop
    # this to 0.30 for the previous, better-tested operating point.
    book_target_vol: float = 0.50
    # Ceiling on total book notional as a multiple of equity. Never binds at
    # or below a 0.50 target (p99 is ~2.07x there) -- it exists so a collapse
    # in the volatility estimate cannot translate into unbounded leverage.
    book_max_leverage: float = 8.0
    # Trend-doctor F2 brake: size the book off max(7-day, 30-day) realised vol
    # instead of the 30-day alone, so leverage drops within days of a vol spike.
    # Can only LOWER leverage. Off = the validated behaviour, unchanged.
    book_vol_fast_up: bool = False
    # ML book filter v1 (app/ml/book_filter.py): a frozen LightGBM daily size
    # multiplier (0.5/1.0/1.5) per major on the trend book. "off" = nothing
    # runs; "shadow" = computed at 00:05 UTC and logged, ZERO effect on sizing;
    # "live" = applied to the book weights (clipped to book_max_leverage, 1.0
    # when stale). Validated cycle 5 with thin margins -> shadow first.
    ml_book_filter: str = "shadow"
    # The ladder's signal comes from the BINANCE:{symbol} hourly series (see
    # data/query.load_ladder_frame). If that series stops updating, the signal
    # silently freezes at its last value while the market moves. Past this
    # age, the loop refuses to trade rather than acting on a stale view.
    ladder_max_bar_age_seconds: int = 3 * 3600

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

    # Crypto blended prediction: how much of the call comes from the ML model,
    # the rest from measured factors (app/strategy/crypto_blend.py).
    #
    # 0.40 as requested. Worth knowing what it costs: out-of-fold the model
    # scores BELOW a coin flip on all three symbols (BTC 49.01%, ETH 47.98%,
    # SOL 48.81%), so this weight subtracts roughly 0.15 points from the
    # factor half's accuracy -- inside the noise band, but the sign is not in
    # doubt. Configurable so the trade-off stays a decision rather than a
    # constant buried in a module.
    crypto_blend_model_weight: float = 0.40

    # DB growth (maint, 2026-10-07; app/data/retention.py). orderbook_snapshots
    # is read by nothing and was the biggest raw table (~68 MB/day), so it is
    # kept 1 day instead of raw_tick_retention_days. 1m/5m candles older than
    # these are deleted; the longest readers are the Analyst live bar (120 1m
    # rows) and the Markets chart (300 bars by default). 0 disables either.
    orderbook_retention_days: float = 1.0
    candle_1m_retention_days: float = 4.0
    candle_5m_retention_days: float = 14.0

    # Analyst shadow variants (cycle 8): on every Analyst run, also record what
    # the near-miss rules from the loss autopsy would do -- C (no entry when the
    # coin's daily ATR/close rank is in its top third), B (SELL only at score
    # <= -0.40), C+B, and "base" (no extra rule, the control) -- each as its own
    # PAPER book in analyst_trades.variant. Never sends an order, never gates or
    # touches the live trade, never counts in the live stats. False = no variant
    # work at all.
    analyst_shadow_variants: bool = True

    # ML gate on Analyst entries (2026-10-09): "g2" = enter only on coins the
    # frozen ML book filter (app/ml/book_filter.py) rates 1.5x for today. PASSED
    # its pre-registered test (cryptoalgo_team/mlgate): 10 unseen coins R/yr
    # +28.7/+56.8 vs +18.8/+36.7, beats the random-gate p95, DSR 1.00. Crypto
    # Analyst only (gold is ungated). No fresh record -> gate open. "off" = old rule.
    analyst_ml_gate: str = "g2"

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]

    @property
    def analyst_symbol_list(self) -> list[str]:
        return [s.strip().upper() for s in self.analyst_symbols.split(",") if s.strip()]

    @property
    def core_symbol_list(self) -> list[str]:
        """Core symbols of the size-aware universe; [] when the feature is off."""
        return [s.strip().upper() for s in self.core_symbols.split(",") if s.strip()]

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
