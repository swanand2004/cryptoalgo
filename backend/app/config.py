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
    # Raised alongside max_margin_per_trade_pct so MARGIN is what decides
    # how many trades can be open at once, not this cap silently binding
    # first: at max_leverage=50 and max_margin_per_trade_pct=0.10, 9 fully
    # margin-capped trades need up to 9 * (equity*0.10*50) = 45x equity in
    # aggregate notional even though their aggregate MARGIN is only the
    # intended 90% of equity. 45x is the exact break-even for that; this is
    # a real, substantial increase in the aggregate notional-exposure
    # ceiling (not just margin) as a direct consequence of prioritizing the
    # 8-9-concurrent-trades target.
    max_portfolio_leverage: float = 45
    # Fraction of equity allowed to be committed as MARGIN across every open
    # position combined (not notional -- actual margin, i.e. notional/leverage).
    # A real exchange rejects an order once required margin exceeds your
    # balance; this is that same check, with a buffer left for adverse moves
    # before liquidation rather than cutting it exactly at 100%.
    max_margin_utilization: float = 0.85
    # Fraction of equity committed as margin, EVERY trade -- not just a
    # ceiling anymore (per explicit request), the actual sizing target:
    # position_sizing.fixed_margin_contracts sizes every fresh entry to
    # exactly this fraction of equity as margin, at whatever leverage tier
    # the signal's conviction selects (see select_leverage). risk_manager's
    # cap_position_size still separately enforces this as a ceiling too
    # (defense in depth), which on a fresh entry is a no-op since both derive
    # from the same number and the same selected leverage. With 5 strategies
    # x 3 symbols trading independently, 1/9th of equity per trade keeps room
    # for 8-9 to run concurrently (0.85 / 0.10 = 8.5) instead of one eating
    # the whole account's margin.
    max_margin_per_trade_pct: float = 0.10
    # Hard cap on how many positions may be open SIMULTANEOUSLY across the
    # whole account (every symbol, every strategy combined) -- a position
    # count limit, separate from and in addition to the margin/leverage caps
    # above. With 7 strategies x 3 symbols each trading its own independent
    # position, the margin caps alone still allow up to ~8-9 concurrent
    # trades; this trims that down to a smaller, easier-to-reason-about
    # number. Extending or flipping an EXISTING position never counts as a
    # new one against this cap -- only genuinely opening from flat does, so
    # this never blocks managing risk on positions already open.
    max_concurrent_positions: int = 5
    max_daily_loss_usd: float = 200
    # Mirror of max_daily_loss_usd for the upside: once today's mark-to-market
    # gain reaches this fraction of the day's STARTING equity, trading halts
    # and flattens for the rest of the day (see risk_manager.check_daily_
    # profit_target / scheduler._daily_profit_target_supervisor) -- locking in
    # a good day instead of risking giving it back. A percentage rather than
    # a fixed dollar figure (unlike the loss limit) because a profit TARGET is
    # naturally expressed relative to what you started the day with.
    daily_profit_target_pct: float = 0.10
    paper_starting_balance_usd: float = 10000
    atr_stop_multiple: float = 2.0  # stop-loss distance, in ATRs, from entry
    atr_take_profit_multiple: float = 3.0  # take-profit distance, in ATRs (1.5:1 reward:risk)
    # Once a position has moved this many ATRs in its favor, the trailing
    # stop stops using the wide entry stop_distance (2 ATR) and switches to
    # BOTH a floor at breakeven AND a tighter trail distance. Without this,
    # a position that spikes to within a whisker of take-profit and then
    # reverses can round-trip all the way back to breakeven or even a small
    # loss, because the stop only trails at a constant 2 ATRs behind the
    # peak regardless of how close to target price got.
    atr_breakeven_trigger_multiple: float = 1.0
    # Was 1.0 -- tightened so the trailing stop follows the peak a bit more
    # closely (locks in more of a favorable move, a bit sooner) once
    # triggered, without changing anything about the wide pre-trigger stop
    # or when tightening starts.
    atr_tightened_trail_multiple: float = 0.8  # trail distance, in ATRs, once triggered

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
