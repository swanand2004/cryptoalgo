from __future__ import annotations

import datetime as dt
import logging
import math

import pandas as pd

from app.data.ingest_rest import _RESOLUTION_SECONDS
from app.data.query import load_candles_df, load_ladder_frame, load_pseudo_series_df
from app.data.app_state import get_trading_mode, is_kill_switch_on
from app.execution.broker_base import OrderRequest
from app.execution.maker import cancel_resting, execute_as_maker, maker_enabled
from app.features.feature_pipeline import build_feature_frame
from app.features.microstructure import compute_microstructure
from app.risk.position_sizing import (
    SizingInputs,
    apply_signal_hysteresis,
    fixed_margin_contracts,
    should_rebalance,
)
from app.risk.book_vol_target import compute_book_scalar, ladder_size_multiplier, symbol_weights
from app.ml.book_filter import book_filter_weights
from app.strategy.ensemble import SignalEnsemble, WeightedStrategy
from app.strategy.gann_angle import GannAngleStrategy
from app.strategy.trend_ladder import TrendLadderStrategy

# Strategies whose signal is computed on the deep BINANCE: hourly frame rather
# than one of Delta's short per-resolution frames.
LADDER_FRAME_STRATEGIES = {TrendLadderStrategy.name, GannAngleStrategy.name}

logger = logging.getLogger(__name__)

LADDER_RESOLUTION_SECONDS = 3600

# Delta's own frames are used only for the stop-placing ATR and a fallback
# price, so they need to satisfy ATR(14)'s warm-up, not the strategy's.
ATR_WARMUP_BARS = 30

SIGNAL_RESOLUTION = "15m"
SIGNAL_RESOLUTION_SECONDS = 900

# The single strategy tag every live position now carries. As of 2026-08-31
# the loop trades ONE weighted blend of trend_momentum + graha + tsmom per
# symbol rather than one position per member -- see _tick_symbol for the
# measured justification (0.733 mean pairwise correlation between the three,
# so separate positions were paying triple fees for ~1.3 effective bets).
CONSOLIDATED_STRATEGY = "consolidated"

# Best-performing timeframe per (symbol, strategy), from isolated backtests
# across 5m/15m/1h/4h history: every real strategy showed a strong, monotonic
# relationship between longer timeframe and better risk-adjusted performance
# (funding_carry and mean_reversion especially -- 15m is far too fast for a
# multi-bar mean-reversion or a multi-day carry thesis to play out before its
# ATR-based stop, sized off that same short timeframe, cuts it off).
# Anything not listed here falls back to SIGNAL_RESOLUTION: "ensemble" (kept
# on the baseline cadence rather than mixing resolutions inside one blended
# signal), ml_signal (only ever trained at 15m), and orderbook_imbalance (a
# genuinely fast microstructure signal -- 15m is its natural cadence, not a
# leftover default).
FAVORABLE_RESOLUTION: dict[str, dict[str, str]] = {
    "BTCUSD": {
        "trend_momentum": "4h",
        "graha": "4h",
        "tsmom": "4h",
    },
    # ETH's trend_momentum/tsmom moved 1h -> 4h on 2026-08-31, after the
    # backtest engine's intrabar-lookahead and free-stop-fill bugs were fixed
    # (see [[project-backtest-engine-bugs]] -- every resolution choice in this
    # table was originally made on the broken engine, so they were re-derived
    # from scratch). 4h beats 1h essentially everywhere on the corrected
    # engine: in-sample on 6 of 6 BTC/SOL combos, often hugely (BTC graha
    # +9.5% at 4h vs -21.8% at 1h), and out-of-sample on these two ETH combos
    # by a wide margin -- trend_momentum -5.13% -> +11.21% (Sharpe -0.80 ->
    # 3.02) and tsmom -6.58% -> +14.34% (Sharpe -0.96 -> 2.89), each with
    # roughly HALF the max drawdown. The mechanism is visible in the live
    # trade data rather than assumed: 4h takes ~1/4 the trades (19 vs 72, 39
    # vs 136), and real closed trades showed fees consuming 36-89% of gross
    # profit, so quartering trade count is a direct, large cost saving.
    "ETHUSD": {
        "trend_momentum": "4h",
        "graha": "4h",  # after the deep Panchang rebuild, 4h beats 1h here too
        "tsmom": "4h",
    },
    "SOLUSD": {
        "trend_momentum": "4h",
        "graha": "4h",
        "tsmom": "4h",
    },
}

# The consolidated blend trades at 4h on every symbol -- the members all
# resolved to 4h independently (above), and 4h beat 1h and 15m across the
# board on the corrected engine, so there is no mixed-timeframe blending to
# reconcile: signal, ATR-based stop/target, and sizing all come from the
# same 4h bars.
for _sym in FAVORABLE_RESOLUTION:
    FAVORABLE_RESOLUTION[_sym][CONSOLIDATED_STRATEGY] = "4h"


def _resolution_for(symbol: str, strategy_name: str) -> str:
    return FAVORABLE_RESOLUTION.get(symbol, {}).get(strategy_name, SIGNAL_RESOLUTION)


async def run_trading_tick(runtime) -> None:
    """One pass over every configured symbol: rebuild features from the
    latest stored candles once, then let each individually enabled strategy
    trade its own independent position (tagged with that strategy's name),
    on its own favorable timeframe -- so every strategy builds a real, live
    paper track record on every symbol, not just a hypothetical backtest
    one. Runs identically regardless of paper/live mode -- OrderManager
    decides where the order goes.

    The blended ensemble ("ensemble") no longer trades its own position, per
    explicit request: it was a second book layered on top of the same 3
    member signals it's built from, running alongside them rather than
    replacing them. `runtime.ensembles` is still built and still used below
    purely as a container for `.members` (weights/enabled flags) -- see
    build_ensemble_for_symbol in strategy/registry.py -- and the ensemble's
    own backtest/report use elsewhere is untouched.

    As of 2026-09-07 sizing is decided at the BOOK level, not per symbol: one
    volatility scalar is computed across all three symbols before any of them
    executes, then handed to each. That ordering is the whole point -- the
    scalar depends on how correlated the legs currently are, which is not
    knowable from inside a single symbol's tick.
    """
    if await is_kill_switch_on():
        # Halted: the limits flatten on the way in; until the switch is turned
        # off (manually, or by the next-day reset for daily limits) nothing
        # trades -- in either mode.
        logger.debug("Kill switch is on -- trading halted, skipping tick")
        return

    frames, signals = await _build_ladder_frames(runtime)
    # Every coin's current trend direction, traded or not: the Analyst only
    # trades WITH the book (app/analyst/execution.agrees_with_book).
    live_signals = getattr(runtime, "live_signals", None)
    if isinstance(live_signals, dict):
        fresh = {s: float(sig.iloc[-1]) for s, sig in signals.items() if len(sig)}
        # A coin whose frame was dropped this tick (stale bars, load error) must
        # not keep its last signal: the Analyst's agrees_with_book gate would
        # otherwise trade "with the book" on a frozen value.
        for s in [s for s in live_signals if s not in fresh]:
            del live_signals[s]
        live_signals.update(fresh)
    if getattr(runtime.settings, "trading_engine", "book") == "analyst":
        # config.trading_engine: under "analyst" the book only computes the
        # signals above (the Analyst reads them) and sends no orders.
        return
    analyst_held = await _analyst_held_symbols(runtime)
    active = await _active_universe(runtime, frames)
    if active is None:
        logger.warning("Equity or prices unavailable -- cannot size the universe, holding every position")
        return
    if not active:
        # Nothing the balance can size today (and no fillers configured): the
        # book holds nothing; its signals above still feed the Analyst.
        logger.debug("Book: no symbol is tradeable at this balance today")
        return
    # The book (weights, vol scalar) is built from the ACTIVE symbols only: a
    # symbol the balance cannot size must not take a share of the risk budget.
    frames = {s: f for s, f in frames.items() if s in active}
    signals = {s: signals[s] for s in frames}
    book_scalar, realised_vol = compute_book_scalar(
        frames,
        signals,
        target_vol=runtime.settings.book_target_vol,
        max_book_leverage=runtime.settings.book_max_leverage,
        fast_up=bool(getattr(runtime.settings, "book_vol_fast_up", False)),
    )
    if book_scalar is None:
        logger.warning(
            "Book volatility scalar unavailable (%d/%d symbols have a usable ladder frame) "
            "-- holding every position rather than sizing off a guess",
            len(frames), len(runtime.registry.symbols),
        )
        return

    # Risk-parity shares, not 1/N: a leg's weight is inversely proportional to
    # its own volatility, so SOL does not quietly dominate a book it shares
    # with BTC.
    weights = symbol_weights(frames)
    # ML book filter: returns `weights` itself unless Settings.ml_book_filter == "live".
    weights = await book_filter_weights(runtime.settings, weights, signals, book_scalar)
    book_leverage = book_margin_leverage(runtime.settings, signals, weights, book_scalar)
    logger.info(
        "Book: realised vol %.1f%% -> scalar %.3f (target %.0f%%), %d symbols live, %.0fx margin leverage | weights %s",
        realised_vol * 100, book_scalar, runtime.settings.book_target_vol * 100, len(frames), book_leverage,
        ", ".join("%s %.2f" % (s, w) for s, w in sorted(weights.items())),
    )

    for symbol in runtime.registry.symbols:
        try:
            if symbol in analyst_held:
                continue  # the Analyst's position: the book does not resize or close it
            if symbol not in active:
                await _close_inactive(runtime, symbol)
                continue
            await _tick_symbol(
                runtime, symbol,
                ladder_frame=frames.get(symbol),
                book_scalar=book_scalar,
                weight=weights.get(symbol, 1.0 / max(len(runtime.registry.symbols), 1)),
                book_leverage=book_leverage,
            )
        except Exception:
            logger.exception("Trading tick failed for %s", symbol)


async def _analyst_held_symbols(runtime) -> set[str]:
    """Coins where the Analyst holds a REAL position (trading_engine "analyst"/"combined")."""
    if getattr(runtime.settings, "trading_engine", "book") not in ("analyst", "combined"):
        return set()
    from sqlalchemy import select

    from app.data.db import SessionLocal
    from app.data.models import AnalystTrade

    async with SessionLocal() as session:
        rows = await session.execute(select(AnalystTrade.symbol).where(
            AnalystTrade.is_open.is_(True), AnalystTrade.contracts > 0))
        return set(rows.scalars().all())


def book_margin_leverage(settings, signals: dict, weights: dict, book_scalar: float) -> float:
    """ONE exchange leverage for every book-sized order: the lowest whole
    number that margins the whole book's target notional inside
    `book_margin_use` of equity. Notional is set by the book (|signal| x
    weight x scalar x equity), so leverage only sets margin -- and therefore
    how far away Delta's isolated liquidation is. The same number on every
    position spreads the balance evenly instead of letting whichever symbol
    opens first lock most of it up (2026-10-05: BNB held $9.91 of $14.49,
    leaving ETH ~20x with liquidation INSIDE its daily-loss level)."""
    gross = sum(abs(float(sig.iloc[-1])) * weights.get(s, 0.0) for s, sig in signals.items() if len(sig))
    lev = math.ceil(max(gross * book_scalar, 1e-9) / max(settings.book_margin_use, 1e-9))
    return float(min(max(lev, 1), settings.max_leverage))


async def _active_universe(runtime, frames: dict) -> set[str] | None:
    """The symbols that may trade now (see Settings.core_symbols).

    Off (no core_symbols): every symbol, as before. On: core symbols whose ONE
    contract is at most affordable_contract_fraction x equity, plus the
    small-account fillers while not every core symbol is affordable. Decided
    once per UTC day and mode -- the simulation that validated it re-chose
    daily, and re-choosing every minute would churn positions whenever
    equity hovers at a threshold. None = cannot decide (equity or a core
    price unknown): the caller holds rather than guessing."""
    settings = runtime.settings
    core = [s for s in settings.core_symbol_list if s in runtime.registry.symbols]
    if not core:
        return set(runtime.registry.symbols)
    mode = await get_trading_mode()
    key = (mode, dt.datetime.now(dt.timezone.utc).date().isoformat())
    cached = getattr(runtime, "_universe_cache", None)
    if cached and cached[0] == key:
        return cached[1]
    try:
        equity = await runtime.order_manager.get_equity()
    except Exception:
        return None
    if equity <= 0:
        return None
    affordable = []
    for symbol in core:
        price = runtime.cache.get_last_price(symbol)
        if not price and symbol in frames:
            price = float(frames[symbol]["close"].iloc[-1])
        if not price:
            return None  # a core price is unknown: deciding now could stick for a day
        if runtime.registry.get(symbol).contract_value * price <= settings.affordable_contract_fraction * equity:
            affordable.append(symbol)
    active = set(affordable)
    if len(affordable) < len(core):
        active |= {s for s in runtime.registry.symbols if s not in core}
    runtime._universe_cache = (key, active)
    logger.info(
        "Universe for %s (%s, equity $%.2f): %s", key[1], mode, equity, ", ".join(sorted(active)),
    )
    return active


async def _close_inactive(runtime, symbol: str) -> None:
    """A symbol outside today's universe holds nothing: close what is left."""
    positions = await runtime.order_manager.get_positions()
    pos = positions.get(f"{symbol}:{CONSOLIDATED_STRATEGY}")
    if pos is not None and pos.size != 0:
        logger.info("Closing %s (%+d) -- outside today's size-aware universe", symbol, pos.size)
        await runtime.order_manager.close_position(symbol, CONSOLIDATED_STRATEGY)


async def _build_ladder_frames(runtime) -> tuple[dict, dict]:
    """Deep hourly signal frames plus each symbol's full signal series.

    A symbol is dropped -- and therefore not traded this tick -- if its frame
    is too short to converge the ladder, or if its newest closed bar is older
    than `ladder_max_bar_age_seconds`. A stale BINANCE: series would otherwise
    freeze the signal at its last value while the market kept moving, which
    looks exactly like a confident hold and is the most dangerous possible
    failure mode for a trend follower.
    """
    strategy = TrendLadderStrategy()
    warmup = strategy.required_warmup_bars()
    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
    max_age = runtime.settings.ladder_max_bar_age_seconds

    frames: dict = {}
    signals: dict = {}
    for symbol in runtime.registry.symbols:
        try:
            frame = await load_ladder_frame(symbol)
        except Exception:
            logger.exception("Could not load ladder frame for %s", symbol)
            continue
        if frame.empty:
            logger.warning("No BINANCE:%s hourly history -- %s will not trade", symbol, symbol)
            continue

        frame = drop_incomplete_last_bar(
            frame, resolution_seconds=LADDER_RESOLUTION_SECONDS, now_ts=now_ts
        ).reset_index(drop=True)
        if len(frame) < warmup + 5:
            logger.warning(
                "BINANCE:%s has %d hourly bars, ladder needs %d -- %s will not trade",
                symbol, len(frame), warmup + 5, symbol,
            )
            continue

        age = now_ts - float(frame["time"].iloc[-1])
        if age > max_age:
            logger.warning(
                "BINANCE:%s newest closed bar is %.1fh old (limit %.1fh) -- %s will not trade "
                "on a stale signal",
                symbol, age / 3600, max_age / 3600, symbol,
            )
            continue

        frames[symbol] = frame
        # The book's volatility must be measured on the signal that is
        # actually TRADED -- the full ensemble blend (trend_ladder +
        # gann_angle) -- not on one member of it. Sizing off the ladder alone
        # would estimate the risk of a book the system no longer holds.
        ensemble = getattr(runtime, "ensembles", {}).get(symbol)
        signals[symbol] = ensemble.smoothed_signal(frame) if ensemble is not None else strategy.generate_signal(frame)
    return frames, signals


def drop_incomplete_last_bar(
    candles: pd.DataFrame, resolution_seconds: int = SIGNAL_RESOLUTION_SECONDS, now_ts: float | None = None
) -> pd.DataFrame:
    """Drops the last row if that bar hasn't actually closed yet.

    Delta's candle endpoint (like most exchanges') includes the current,
    still-forming bar with whatever partial OHLC exists so far -- every
    indicator computed from it (EMA, MACD, RSI, ADX, Bollinger...) "repaints"
    as the bar keeps filling in, since its close is a moving target for the
    next `resolution_seconds`. Confirmed live: the newest bar was only ~2
    minutes old when checked. That let entries and exits fire within minutes
    of each other on nothing but the bar settling, not a genuine reversal --
    70% of signal-exits were closing within one bar's width, median hold
    time 4 minutes. Trading off only CLOSED bars is the standard fix for
    this class of repainting bug.
    """
    if len(candles) == 0:
        return candles
    if now_ts is None:
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
    if candles["time"].iloc[-1] + resolution_seconds > now_ts:
        return candles.iloc[:-1]
    return candles


async def _build_features_for_resolution(
    runtime, symbol: str, product, resolution: str, smoothed_book: float | None, smoothed_flow: float | None
) -> pd.DataFrame | None:
    candles = await load_candles_df(symbol, resolution, limit=400)
    candles = drop_incomplete_last_bar(candles, resolution_seconds=_RESOLUTION_SECONDS[resolution])
    if len(candles) < 30:
        return None

    funding = await load_pseudo_series_df(f"FUNDING:{symbol}", "1h")
    mark = await load_pseudo_series_df(f"MARK:{symbol}", resolution)
    spot = await load_pseudo_series_df(product.spot_index_symbol, resolution) if product.spot_index_symbol else None
    fear_greed = await load_pseudo_series_df("FNG:GLOBAL", "1d")
    binance_funding = await load_pseudo_series_df(f"BFUND:{symbol}", "8h")
    binance_oi = await load_pseudo_series_df(f"BOI:{symbol}", "1h")
    binance_lsr = await load_pseudo_series_df(f"BLSR:{symbol}", "1h")
    dvol = await load_pseudo_series_df(f"DVOL:{symbol}", "1h")
    news = await load_pseudo_series_df(f"NEWS:{symbol}", "1h")
    features = build_feature_frame(
        candles,
        funding=funding,
        mark=mark,
        spot=spot,
        fear_greed=fear_greed,
        binance_funding=binance_funding,
        binance_oi=binance_oi,
        binance_lsr=binance_lsr,
        dvol=dvol,
        news=news,
    )

    # Historical bars have no stored order-book state, so book_imbalance/
    # trade_flow_imbalance only exist for the *current* row -- filled from the
    # live market cache (computed ONCE per symbol per tick by the caller, not
    # per resolution, since it mutates a persistent EMA in the cache).
    # OrderbookImbalanceStrategy treats missing values as 0, so this only ever
    # affects the live signal, never the backtest.
    last_idx = features.index[-1]
    features.loc[last_idx, "book_imbalance"] = smoothed_book if smoothed_book is not None else 0.0
    features.loc[last_idx, "trade_flow_imbalance"] = smoothed_flow if smoothed_flow is not None else 0.0
    return features


async def _tick_symbol(
    runtime,
    symbol: str,
    *,
    ladder_frame=None,
    book_scalar: float | None = None,
    weight: float = 1.0,
    book_leverage: float | None = None,
) -> None:
    ensemble = runtime.ensembles.get(symbol)
    if ensemble is None:
        return
    if ladder_frame is None or book_scalar is None:
        # _build_ladder_frames already logged why. Holding is the correct
        # response to a missing signal -- not flattening, which would be a
        # position decision taken on the absence of information.
        return
    product = runtime.registry.get(symbol)

    # Smoothed via a persistent EMA (market_cache.smooth_imbalance), not the
    # raw instantaneous reading -- since every OTHER row in a feature frame is
    # 0, the ensemble's own EMA smoothing has no real history to average over
    # here and reduces to ~half the instant reading, with no memory across
    # ticks. That let this signal flip sign almost every tick (order books
    # genuinely do reverse minute to minute), causing rapid open/close/flip
    # churn. The EMA here persists across ticks in the cache instead. Computed
    # exactly once per symbol per tick (not per resolution below) since
    # calling it again would incorrectly advance the EMA a second time.
    micro = compute_microstructure(runtime.cache, symbol)
    smoothed_book = runtime.cache.smooth_imbalance(symbol, "book", micro.book_imbalance)
    smoothed_flow = runtime.cache.smooth_imbalance(symbol, "flow", micro.trade_flow_imbalance)

    # One feature dataframe per DISTINCT resolution actually needed this tick
    # -- not per strategy -- so this stays cheap even with several strategies
    # pointed at the same favorable timeframe (almost always just 2-3 distinct
    # resolutions: the 15m baseline plus whichever favorable timeframes this
    # symbol's strategies use).
    needed_resolutions = {SIGNAL_RESOLUTION, _resolution_for(symbol, CONSOLIDATED_STRATEGY)} | {
        _resolution_for(symbol, m.strategy.name) for m in ensemble.members if m.enabled and m.weight > 0
    }
    features_by_resolution: dict[str, pd.DataFrame] = {}
    for resolution in needed_resolutions:
        feats = await _build_features_for_resolution(runtime, symbol, product, resolution, smoothed_book, smoothed_flow)
        if feats is not None:
            features_by_resolution[resolution] = feats

    # The SIGNAL's warm-up is checked against the ladder frame in
    # _build_ladder_frames, not here. Delta's own frames now serve only two
    # purposes -- a fallback price and the ATR that places stops -- so gating
    # them on the strategy's warm-up would be checking the wrong series
    # against the wrong number: the ladder needs 6,400 hourly bars and this
    # frame holds 400 fifteen-minute ones, so the comparison is never
    # satisfiable and the loop silently never trades.
    baseline_features = features_by_resolution.get(SIGNAL_RESOLUTION)
    if baseline_features is None or len(baseline_features) < ATR_WARMUP_BARS:
        return

    price = runtime.cache.get_last_price(symbol) or float(baseline_features["close"].iloc[-1])
    if price <= 0:
        return

    positions = await runtime.order_manager.get_positions()
    equity = await runtime.order_manager.get_equity()

    # ONE blended position per symbol (2026-08-31), replacing the previous
    # "every member trades its own position" design.
    #
    # The case for it is measured, not tidiness: the three strategies' actual
    # return correlation is 0.733 mean pairwise across all 9 strategy/symbol
    # combos -- about 1.3 effective independent bets out of 9 (see
    # [[project-risk-controls]]). Running them as separate positions therefore
    # paid three sets of fees and committed three lots of margin for what is
    # very nearly a single directional bet, while real closed trades showed
    # fees eating 36-89% of gross profit. An earlier partial fix here only
    # merged FRESH same-direction entries; blending at the signal level
    # instead fixes it for the whole life of the position.
    #
    # Validated on the corrected backtest engine (in-sample AND held-out, so
    # this is not test-window fishing): consolidation beat separate positions
    # -15.89% vs -22.70% in-sample and +44.49% vs +42.55% out-of-sample,
    # while cutting trade count ~70% and fees ~70% in both windows.
    #
    # 2026-09-07: the SIGNAL now comes from `ladder_frame` -- deep BINANCE:
    # hourly history -- while the ATR that places stops still comes from
    # Delta's own candles. That split is deliberate. The ladder's slowest EMA
    # needs ~267 days to converge and Delta only has ~113, so the signal has
    # to be computed on the deep series; but a stop is an order resting at a
    # Delta price, so its distance must be measured in Delta's own bars.
    resolution = _resolution_for(symbol, CONSOLIDATED_STRATEGY)
    blended_features = features_by_resolution.get(resolution)
    if blended_features is None:
        blended_features = baseline_features
    atr = float(blended_features["atr_14"].iloc[-1]) if pd.notna(blended_features["atr_14"].iloc[-1]) else 0.0
    if atr <= 0:
        return

    # Any position still tagged with an individual member's name predates the
    # consolidation and is no longer managed by the loop below -- close it out
    # rather than orphaning it to drift until its stop or target happens to
    # fire.
    for key, pos in positions.items():
        pos_symbol, _, pos_strategy = key.partition(":")
        if pos_symbol != symbol or pos_strategy == CONSOLIDATED_STRATEGY or pos.size == 0:
            continue
        logger.info(
            "Closing legacy %s position on %s (%d contracts) -- superseded by the consolidated strategy",
            pos_strategy, symbol, pos.size,
        )
        await runtime.order_manager.close_position(symbol, pos_strategy)

    current, target_contracts, latest_signal = _compute_member_target(
        runtime, product, ladder_frame, price, positions, equity,
        strategy_name=CONSOLIDATED_STRATEGY, signal_ensemble=ensemble,
        book_scalar=book_scalar, weight=weight,
    )
    target_contracts, leverage, min_lot = await _min_lot_target(
        runtime, product, price, positions, equity,
        symbol=symbol, target_contracts=target_contracts, latest_signal=latest_signal,
    )
    await _execute_target(
        runtime, symbol, product, price, atr,
        strategy_name=CONSOLIDATED_STRATEGY, current=current,
        target_contracts=target_contracts, latest_signal=latest_signal,
        leverage=leverage if min_lot else book_leverage, min_lot=min_lot,
    )


async def _min_lot_target(
    runtime, product, price: float, positions: dict, equity: float,
    *, symbol: str, target_contracts: int, latest_signal: float,
) -> tuple[int, float | None, bool]:
    """Small-account mode (LIVE only, off unless live_min_lot_max_leverage > 0).

    On a small balance the strategy's sizing rounds every position to 0
    contracts -- one ETH contract is $27.5, a whole $3.67 account several
    times over. Rather than never trading, hold ONE contract in the signal's
    direction, but only while the account's TOTAL open notional stays within
    `live_min_lot_max_leverage` x equity, and at the LOWEST exchange leverage
    the free balance can margin -- more of the account backs the position, so
    liquidation sits as far away as the money allows. Coins whose single
    contract would break the cap stay flat and join on their own as equity
    grows. Returns (target, leverage, is_min_lot).

    Leverage is the same for every lot: ceil(cap / live_min_lot_margin_use),
    so a book filled up to the cap always fits the margin -- not the lowest the
    LEFTOVER margin allows, which let the first lot lock up most of the
    balance and squeezed (or shut out) the next ones.
    """
    settings = runtime.settings
    cap = getattr(settings, "live_min_lot_max_leverage", 0.0)
    if cap <= 0 or target_contracts != 0 or latest_signal == 0 or equity <= 0 or price <= 0:
        return target_contracts, None, False
    if await get_trading_mode() != "live":
        return target_contracts, None, False

    one_contract = product.contract_value * price
    others = [p for key, p in positions.items() if not key.startswith(f"{symbol}:") and p.size != 0]
    other_notional = sum(p.notional_usd for p in others)
    if other_notional + one_contract > cap * equity:
        return 0, None, False
    free_margin = settings.live_min_lot_margin_use * equity - sum(p.margin_usd for p in others)
    if free_margin <= 0:
        return 0, None, False
    # ONE leverage for every lot, fixed by the cap: the whole capped book
    # (cap x equity of notional) margins inside margin_use x equity, so lots
    # opened later are never squeezed by earlier ones (2026-10-06: at cap 25
    # the old incremental rule let BNB take the margin and SOL/BTC could not
    # open; before that, ETH sat at 34-36x behind BNB at 8x).
    leverage = max(1.0, float(math.ceil(cap / settings.live_min_lot_margin_use)))
    if one_contract / leverage > free_margin:
        return 0, None, False
    if leverage > settings.max_leverage:
        return 0, None, False
    logger.info(
        "Small-account lot %s: 1 contract ($%.2f) on $%.2f equity at %.0fx leverage (book now %.1fx of a %.0fx cap)",
        symbol, one_contract, equity, leverage, (other_notional + one_contract) / equity, cap,
    )
    return (1 if latest_signal > 0 else -1), leverage, True


async def get_live_signals(runtime, symbol: str) -> list[dict]:
    """Read-only peek at every strategy's CURRENT computed signal for
    `symbol` -- one entry per individual member (enabled or not, so a
    disabled strategy's signal is still visible), each on its own
    favorable timeframe, via the exact same feature-building path
    `_tick_symbol` uses. The blended ensemble itself no longer appears here
    -- it doesn't trade its own position anymore (see run_trading_tick's
    docstring), so its signal has nothing live to explain. Never submits
    an order and never advances the persistent orderbook-imbalance
    smoothing EMA (passes raw=None to `smooth_imbalance`, which reads the
    cached value without updating it), since this can be polled anytime
    from the dashboard, independent of and far more often than the real
    60s trading tick.
    """
    ensemble = runtime.ensembles.get(symbol)
    if ensemble is None:
        return []
    product = runtime.registry.get(symbol)

    smoothed_book = runtime.cache.smooth_imbalance(symbol, "book", None)
    smoothed_flow = runtime.cache.smooth_imbalance(symbol, "flow", None)

    needed_resolutions = {SIGNAL_RESOLUTION} | {_resolution_for(symbol, m.strategy.name) for m in ensemble.members}
    features_by_resolution: dict[str, pd.DataFrame] = {}
    for resolution in needed_resolutions:
        feats = await _build_features_for_resolution(runtime, symbol, product, resolution, smoothed_book, smoothed_flow)
        if feats is not None:
            features_by_resolution[resolution] = feats

    results: list[dict] = []

    for member in ensemble.members:
        # trend_ladder and gann_angle read the deep BINANCE: hourly series, not one of the
        # per-resolution Delta frames -- so the dashboard shows the signal the
        # loop is actually trading rather than one computed off 400 bars of
        # warm-up garbage.
        if member.strategy.name in LADDER_FRAME_STRATEGIES:
            frame = await load_ladder_frame(symbol)
            frame = drop_incomplete_last_bar(frame, resolution_seconds=LADDER_RESOLUTION_SECONDS)
            value = None
            if len(frame) >= member.strategy.required_warmup_bars() + 5:
                value = float(member.strategy.generate_signal(frame).iloc[-1])
            results.append({
                "strategy": member.strategy.name,
                "signal": value,
                "resolution": "1h",
                "enabled": member.enabled and member.weight > 0,
                "weight": member.weight,
            })
            continue

        resolution = _resolution_for(symbol, member.strategy.name)
        member_features = features_by_resolution.get(resolution)
        signal_value = None
        if member_features is not None:
            solo_ensemble = SignalEnsemble(
                [WeightedStrategy(member.strategy, weight=1.0)],
                deadband=ensemble.deadband,
                smoothing_span=ensemble.smoothing_span,
            )
            if len(member_features) >= solo_ensemble.warmup_bars() + 5:
                signal_value = float(solo_ensemble.smoothed_signal(member_features).iloc[-1])
        results.append(
            {
                "strategy": member.strategy.name,
                "signal": signal_value,
                "resolution": resolution,
                "enabled": member.enabled and member.weight > 0,
                "weight": member.weight,
            }
        )
    return results


async def get_live_ml_predictions(runtime) -> dict[str, dict[str, dict]]:
    """Read-only peek at the STANDALONE ML prediction model's current
    confidence-gated score, per SYMBOL and per trained RESOLUTION (see
    ML_RESOLUTIONS in app/ml/train.py -- 15m/1h/4h, each trained toward
    roughly the same ~4-hour-ahead question asked from a different
    granularity of data, not different horizons). Deliberately not
    organized by strategy: this isn't one of the 3 live-trading strategies
    at all (trend_momentum/graha/tsmom; see registry.py) -- it was cut
    from the ensemble on its own honest walk-forward numbers and still
    retrains daily purely for monitoring (see the Walk-Forward Ledger).
    Uses the exact same feature-building path the live trading tick
    itself uses, via `_build_features_for_resolution`, so this reflects
    what the model would say right now, not a stale training-time snapshot.

    `score` is the deployed, confidence-gated value (0.0 when the model is
    abstaining, per CONFIDENCE_THRESHOLD_MULTIPLE -- see app/ml/train.py);
    `decidable` reports whether this instant actually cleared that bar, so
    a genuine "no edge either way" reading isn't confused with "abstained."
    """
    from app.ml.predict import MLPredictor
    from app.ml.train import ML_RESOLUTIONS

    out: dict[str, dict[str, dict]] = {}
    for symbol in runtime.registry.symbols:
        product = runtime.registry.get(symbol)
        out[symbol] = {}
        for resolution in ML_RESOLUTIONS:
            features = await _build_features_for_resolution(runtime, symbol, product, resolution, None, None)
            predictor = MLPredictor(symbol, resolution=resolution)

            if features is None or not predictor.is_ready:
                out[symbol][resolution] = {"score": None, "decidable": False, "resolution": resolution}
                continue

            gated = predictor.predict_score_batch(features, require_confidence=True)
            ungated = predictor.predict_score_batch(features, require_confidence=False)
            gated_last = float(gated.iloc[-1])
            ungated_last = float(ungated.iloc[-1])
            decidable = gated_last != 0.0 or ungated_last == 0.0

            out[symbol][resolution] = {"score": gated_last, "decidable": decidable, "resolution": resolution}
    return out


def _regime_size_multiplier(runtime, features) -> tuple[float, str]:
    """How much of normal size this bar's volatility regime allows.

    Same app/strategy/regime.py classifier the backtest engine calls, so live
    and backtest sizing cannot silently diverge. Fails OPEN (1.0, normal size)
    on any error rather than blocking trading -- a regime classifier is a risk
    refinement, not a safety-critical gate, and a crash here must not take the
    trading loop down with it.
    """
    if not getattr(runtime.settings, "regime_sizing_enabled", False):
        return 1.0, "disabled"
    try:
        from app.strategy.regime import VolRegime, classify

        _, vol_regime = classify(
            features,
            adx_trend_threshold=runtime.settings.regime_adx_trend_threshold,
            vol_lookback_bars=runtime.settings.regime_vol_lookback_bars,
            low_vol_percentile=runtime.settings.regime_low_vol_percentile,
            high_vol_percentile=runtime.settings.regime_high_vol_percentile,
        )
        if len(vol_regime) == 0:
            return 1.0, "unknown"
        label = str(vol_regime.iloc[-1])
        return {
            VolRegime.LOW.value: runtime.settings.regime_low_vol_size_multiplier,
            VolRegime.NORMAL.value: 1.0,
            VolRegime.HIGH.value: runtime.settings.regime_high_vol_size_multiplier,
        }.get(label, 1.0), label
    except Exception:
        logger.exception("Regime classification failed -- sizing at normal size")
        return 1.0, "error"


def _compute_member_target(
    runtime, product, features, price: float, positions: dict, equity: float,
    *, strategy_name: str, signal_ensemble,
    book_scalar: float | None = None, weight: float = 1.0,
) -> tuple[int, int, float]:
    """Computes (current_contracts, target_contracts, latest_signal) for one
    strategy -- pure calculation, never submits anything. Split out from the
    old _tick_one so _tick_symbol can look at every member's target BEFORE
    executing any of them, to catch coinciding fresh entries (see its
    docstring) and consolidate them into a single order.
    """
    key = f"{product.symbol}:{strategy_name}"
    current = positions[key].size if key in positions else 0
    current_sign = 1 if current > 0 else (-1 if current < 0 else 0)

    signal_series = signal_ensemble.smoothed_signal(features)
    raw_signal = float(signal_series.iloc[-1])
    live_signals = getattr(runtime, "live_signals", None)
    if isinstance(live_signals, dict):
        live_signals[product.symbol] = raw_signal
    latest_signal = apply_signal_hysteresis(
        raw_signal, current_sign, signal_ensemble.deadband, signal_ensemble.exit_deadband
    )

    size_multiplier, vol_label = _regime_size_multiplier(runtime, features)

    if book_scalar is not None:
        # Book-level volatility targeting replaces the fixed-margin default
        # rather than stacking on top of it: `ladder_size_multiplier` returns
        # exactly the multiplier that makes fixed_margin_contracts produce the
        # book's own target notional, so the two cannot both apply.
        size_multiplier = ladder_size_multiplier(
            signal=latest_signal,
            book_scalar=book_scalar,
            weight=weight,
            margin_per_trade_pct=runtime.settings.max_margin_per_trade_pct,
            min_leverage=runtime.settings.min_leverage,
            max_leverage=runtime.settings.max_leverage,
        )
        vol_label = "book-vol-target"

    target_contracts = fixed_margin_contracts(
        SizingInputs(
            equity_usd=equity,
            signal=latest_signal,
            price=price,
            contract_value=product.contract_value,
            margin_per_trade_pct=runtime.settings.max_margin_per_trade_pct,
            min_leverage=runtime.settings.min_leverage,
            max_leverage=runtime.settings.max_leverage,
            size_multiplier=size_multiplier,
        )
    )
    if latest_signal != 0 and book_scalar is not None:
        # Not a reduction from "normal" -- a different sizing basis entirely,
        # so it is reported as the notional it targets rather than as a
        # percentage of a fixed-margin size that is no longer being used.
        logger.info(
            "%s %s: signal %.3f, book k %.3f -> %.3fx equity notional, target %d contracts",
            product.symbol, strategy_name, latest_signal, book_scalar,
            abs(latest_signal) * weight * book_scalar, target_contracts,
        )
    elif size_multiplier != 1.0 and latest_signal != 0:
        logger.info(
            "%s %s: %s regime -> sizing at %.0f%% of normal (signal %.3f, target %d contracts)",
            product.symbol, strategy_name, vol_label, size_multiplier * 100, latest_signal, target_contracts,
        )
    return current, target_contracts, latest_signal


async def _execute_target(
    runtime, symbol: str, product, price: float, atr: float,
    *, strategy_name: str, current: int, target_contracts: int, latest_signal: float,
    leverage: float | None = None, min_lot: bool = False,
) -> None:
    """Submits whatever order is needed to move `strategy_name`'s position
    from `current` to `target_contracts`. Given a precomputed target rather
    than deriving it from the strategy's own signal, so a coinciding-entry
    consolidation pass (see _tick_symbol) can override the size actually
    executed -- e.g. one strategy executing at the SUM of what two
    same-direction fresh entries independently wanted.
    """
    delta = target_contracts - current
    mode = await get_trading_mode()
    use_maker = maker_enabled(runtime.settings, mode)
    if not should_rebalance(current, target_contracts):
        if use_maker and delta == 0:
            # Target reached (or the signal moved back): nothing left to
            # work, so no resting order may linger and fill later.
            await cancel_resting(runtime, mode, symbol, strategy_name)
        return

    # A move that only shrinks the position toward flat (same direction,
    # magnitude decreasing, or exactly to flat) is risk-REDUCING and must
    # never be capped by the notional/margin risk limits -- those exist to
    # stop exposure from growing, not to trap a strategy holding more risk
    # than it wants because the portfolio's margin is busy elsewhere.
    # Flipping through flat to the opposite side still opens new exposure
    # for the flipped portion, so it is NOT marked reduce_only.
    is_reduction = current != 0 and (
        target_contracts == 0
        or (current > 0 and 0 <= target_contracts < current)
        or (current < 0 and current < target_contracts <= 0)
    )

    side = "buy" if delta > 0 else "sell"
    req = OrderRequest(
        symbol=symbol,
        side=side,
        size=abs(delta),
        order_type="market",
        strategy=strategy_name,
        # None disables the ATR stop/take-profit bracket entirely (see
        # Settings.atr_brackets_enabled -- measured at -100% vs +968%).
        atr=atr if runtime.settings.atr_brackets_enabled else None,
        quality=min(1.0, abs(latest_signal)),
        reduce_only=is_reduction,
        leverage=leverage,
        min_lot=min_lot and not is_reduction,
    )
    if use_maker:
        result = await execute_as_maker(runtime, mode, req)
        if result is None:
            return  # a resting order is already working this rebalance
    else:
        result = await runtime.order_manager.submit(req)
    if result.state == "rejected":
        logger.info("Order rejected for %s/%s: %s", symbol, strategy_name, result.reason)
    else:
        logger.info(
            "Tick %s/%s: signal=%.3f target=%d current=%d -> %s %d @ ~%.2f (%s)",
            symbol, strategy_name, latest_signal, target_contracts, current, side, abs(delta), price, result.state,
        )
