from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from app.data.ingest_rest import _RESOLUTION_SECONDS
from app.data.query import load_candles_df, load_pseudo_series_df
from app.execution.broker_base import OrderRequest
from app.features.feature_pipeline import build_feature_frame
from app.features.microstructure import compute_microstructure
from app.risk.position_sizing import (
    SizingInputs,
    apply_signal_hysteresis,
    fixed_margin_contracts,
    should_rebalance,
)
from app.strategy.ensemble import SignalEnsemble, WeightedStrategy

logger = logging.getLogger(__name__)

SIGNAL_RESOLUTION = "15m"
SIGNAL_RESOLUTION_SECONDS = 900

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
    "ETHUSD": {
        "trend_momentum": "1h",
        "graha": "4h",  # after the deep Panchang rebuild, 4h beats 1h here too (Sharpe 2.64 vs 1.44)
        "tsmom": "1h",
    },
    "SOLUSD": {
        "trend_momentum": "4h",
        "graha": "4h",
        "tsmom": "4h",
    },
}


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
    """
    for symbol in runtime.registry.symbols:
        try:
            await _tick_symbol(runtime, symbol)
        except Exception:
            logger.exception("Trading tick failed for %s", symbol)


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


async def _tick_symbol(runtime, symbol: str) -> None:
    ensemble = runtime.ensembles.get(symbol)
    if ensemble is None:
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
    needed_resolutions = {SIGNAL_RESOLUTION} | {
        _resolution_for(symbol, m.strategy.name) for m in ensemble.members if m.enabled and m.weight > 0
    }
    features_by_resolution: dict[str, pd.DataFrame] = {}
    for resolution in needed_resolutions:
        feats = await _build_features_for_resolution(runtime, symbol, product, resolution, smoothed_book, smoothed_flow)
        if feats is not None:
            features_by_resolution[resolution] = feats

    baseline_features = features_by_resolution.get(SIGNAL_RESOLUTION)
    if baseline_features is None or len(baseline_features) < ensemble.warmup_bars() + 5:
        return

    price = runtime.cache.get_last_price(symbol) or float(baseline_features["close"].iloc[-1])
    if price <= 0:
        return

    positions = await runtime.order_manager.get_positions()
    equity = await runtime.order_manager.get_equity()

    # Each individually enabled strategy trades its own position, on
    # ITS OWN favorable timeframe -- both the signal AND the ATR-based stop/
    # target/sizing come from that same resolution, so a strategy tuned for
    # 4h isn't undermined by a stop distance still sized off 15m noise.
    #
    # First pass: compute (never submit) every member's target. Several
    # strategies sharing the same underlying trend construction (e.g. graha's
    # base signal is literally trend_momentum's, only tilted by astrology) can
    # independently reach full conviction on the same clean move and open a
    # fresh, same-direction position at the same tick -- two positions for
    # what is, right now, one bet, paying double margin and fees for it. Only
    # a coinciding FRESH entry (from flat) is consolidated below; an existing
    # position being managed is untouched.
    computed: list[dict] = []
    for member in ensemble.members:
        if not member.enabled or member.weight <= 0:
            continue
        resolution = _resolution_for(symbol, member.strategy.name)
        member_features = features_by_resolution.get(resolution)
        if member_features is None:
            continue
        member_atr = (
            float(member_features["atr_14"].iloc[-1]) if pd.notna(member_features["atr_14"].iloc[-1]) else 0.0
        )
        if member_atr <= 0:
            continue
        solo_ensemble = SignalEnsemble(
            [WeightedStrategy(member.strategy, weight=1.0)],
            deadband=ensemble.deadband,
            smoothing_span=ensemble.smoothing_span,
        )
        if len(member_features) < solo_ensemble.warmup_bars() + 5:
            continue
        current, target_contracts, latest_signal = _compute_member_target(
            runtime, product, member_features, price, positions, equity,
            strategy_name=member.strategy.name, signal_ensemble=solo_ensemble,
        )
        computed.append(
            {
                "strategy_name": member.strategy.name,
                "atr": member_atr,
                "current": current,
                "target": target_contracts,
                "signal": latest_signal,
            }
        )

    # Group members opening FRESH (from flat) this tick by direction.
    fresh_by_direction: dict[int, list[dict]] = {}
    for entry in computed:
        if entry["current"] == 0 and entry["target"] != 0:
            fresh_by_direction.setdefault(1 if entry["target"] > 0 else -1, []).append(entry)

    # Names already handled by the consolidation pass below (both the
    # primary, executed at the combined size, and the others, skipped
    # entirely) -- tracked by name so the final loop can't double-execute
    # or accidentally skip the primary.
    already_handled: set[str] = set()
    for direction, entries in fresh_by_direction.items():
        if len(entries) < 2:
            continue
        primary = max(entries, key=lambda e: abs(e["signal"]))
        combined_target = direction * sum(abs(e["target"]) for e in entries)
        logger.info(
            "Coinciding fresh entries on %s: %s all open %s -- consolidating into %s at %d (was %s each)",
            symbol, [e["strategy_name"] for e in entries], "long" if direction > 0 else "short",
            primary["strategy_name"], combined_target, [e["target"] for e in entries],
        )
        await _execute_target(
            runtime, symbol, product, price, primary["atr"],
            strategy_name=primary["strategy_name"], current=primary["current"],
            target_contracts=combined_target, latest_signal=primary["signal"],
        )
        already_handled.update(e["strategy_name"] for e in entries)

    for entry in computed:
        if entry["strategy_name"] in already_handled:
            continue
        await _execute_target(
            runtime, symbol, product, price, entry["atr"],
            strategy_name=entry["strategy_name"], current=entry["current"],
            target_contracts=entry["target"], latest_signal=entry["signal"],
        )


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


def _compute_member_target(
    runtime, product, features, price: float, positions: dict, equity: float,
    *, strategy_name: str, signal_ensemble,
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
    latest_signal = apply_signal_hysteresis(
        raw_signal, current_sign, signal_ensemble.deadband, signal_ensemble.exit_deadband
    )

    target_contracts = fixed_margin_contracts(
        SizingInputs(
            equity_usd=equity,
            signal=latest_signal,
            price=price,
            contract_value=product.contract_value,
            margin_per_trade_pct=runtime.settings.max_margin_per_trade_pct,
            min_leverage=runtime.settings.min_leverage,
            max_leverage=runtime.settings.max_leverage,
        )
    )
    return current, target_contracts, latest_signal


async def _execute_target(
    runtime, symbol: str, product, price: float, atr: float,
    *, strategy_name: str, current: int, target_contracts: int, latest_signal: float,
) -> None:
    """Submits whatever order is needed to move `strategy_name`'s position
    from `current` to `target_contracts`. Given a precomputed target rather
    than deriving it from the strategy's own signal, so a coinciding-entry
    consolidation pass (see _tick_symbol) can override the size actually
    executed -- e.g. one strategy executing at the SUM of what two
    same-direction fresh entries independently wanted.
    """
    delta = target_contracts - current
    if not should_rebalance(current, target_contracts):
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
    result = await runtime.order_manager.submit(
        OrderRequest(
            symbol=symbol,
            side=side,
            size=abs(delta),
            order_type="market",
            strategy=strategy_name,
            atr=atr,
            quality=min(1.0, abs(latest_signal)),
            reduce_only=is_reduction,
        )
    )
    if result.state == "rejected":
        logger.info("Order rejected for %s/%s: %s", symbol, strategy_name, result.reason)
    else:
        logger.info(
            "Tick %s/%s: signal=%.3f target=%d current=%d -> %s %d @ ~%.2f (%s)",
            symbol, strategy_name, latest_signal, target_contracts, current, side, abs(delta), price, result.state,
        )
