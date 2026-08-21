from __future__ import annotations

import datetime as dt
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.data.app_state import (
    get_state,
    get_trading_mode,
    is_kill_switch_on,
    set_kill_switch,
    set_state,
    set_trading_mode,
)
from app.data.flush import flush_market_cache_to_db
from app.data.ingest_rest import backfill_symbol_history
from app.ml.train import train_and_save
from app.trading_loop import run_trading_tick

logger = logging.getLogger(__name__)


def build_scheduler(runtime) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)

    scheduler.add_job(
        flush_market_cache_to_db, "interval", seconds=5, args=[runtime.cache], id="flush_market_cache", max_instances=1
    )
    scheduler.add_job(
        runtime.paper_broker.check_resting_orders, "interval", seconds=5, id="paper_resting_orders", max_instances=1
    )
    scheduler.add_job(
        runtime.paper_broker.check_stop_loss_take_profit,
        "interval",
        seconds=5,
        id="paper_stop_loss_take_profit",
        max_instances=1,
    )
    # next_run_time=now makes each job fire immediately on startup instead of
    # waiting a full interval first -- without this, a 60s-interval job could
    # never get a chance to run at all if the process restarts (e.g. --reload
    # during development) more often than every 60 seconds, since APScheduler
    # otherwise schedules the first run one full interval after registration.
    scheduler.add_job(
        _trading_tick, "interval", seconds=60, args=[runtime], id="trading_tick", max_instances=1, next_run_time=now
    )
    scheduler.add_job(
        _incremental_backfill, "interval", minutes=5, args=[runtime], id="incremental_backfill", max_instances=1, next_run_time=now
    )
    scheduler.add_job(
        _incremental_backfill_external,
        "interval",
        minutes=60,
        args=[runtime],
        id="incremental_backfill_external",
        max_instances=1,
        next_run_time=now,
    )
    scheduler.add_job(
        _deadman_switch_supervisor, "interval", seconds=10, args=[runtime], id="deadman_supervisor", max_instances=1, next_run_time=now
    )
    scheduler.add_job(
        _live_sync, "interval", seconds=15, args=[runtime], id="live_sync", max_instances=1, next_run_time=now
    )
    scheduler.add_job(
        _daily_loss_supervisor, "interval", seconds=15, args=[runtime], id="daily_loss_supervisor", max_instances=1, next_run_time=now
    )
    # Runs once daily at 03:00 UTC (off-peak). Models were manually retrained
    # 2026-08-19 after fixing the spot-index/funding-window feature bugs (see
    # scheduler._incremental_backfill and feature_pipeline.build_feature_frame)
    # so live trading wasn't stuck on models trained against broken features
    # until the next scheduled run.
    scheduler.add_job(_retrain_models, "cron", hour=3, minute=0, args=[runtime], id="ml_retrain", max_instances=1)
    scheduler.add_job(
        _record_equity, "interval", seconds=60, args=[runtime], id="record_equity", max_instances=1, next_run_time=now
    )

    return scheduler


async def _record_equity(runtime) -> None:
    from app.data.db import SessionLocal
    from app.data.models import EquitySnapshot

    async with SessionLocal() as session:
        paper_equity = await runtime.paper_broker.get_equity()
        session.add(EquitySnapshot(mode="paper", balance=paper_equity, unrealized_pnl=0.0, equity=paper_equity))
        if runtime.settings.has_credentials:
            try:
                live_equity = await runtime.live_broker.get_equity()
                session.add(EquitySnapshot(mode="live", balance=live_equity, unrealized_pnl=0.0, equity=live_equity))
            except Exception:
                logger.exception("Failed to record live equity snapshot")
        await session.commit()


async def _trading_tick(runtime) -> None:
    mode = await get_trading_mode()
    logger.debug("Trading tick (mode=%s)", mode)
    await run_trading_tick(runtime)


async def _incremental_backfill(runtime) -> None:
    for product in runtime.registry.all():
        try:
            await backfill_symbol_history(runtime.rest_client, product.symbol, "15m", lookback_seconds=3 * 3600)
            await backfill_symbol_history(runtime.rest_client, product.symbol, "1h", lookback_seconds=6 * 3600)
            # MARK:/FUNDING:/OI: pseudo-symbols were only ever refreshed by
            # the one-time startup backfill (backfill_all) -- WS ingestion
            # never covers FUNDING:/OI: at all, and only produces 1m MARK:
            # candles, not the 15m ones the feature pipeline actually reads
            # (SIGNAL_RESOLUTION in trading_loop.py). Without refreshing
            # these here too, funding_rate/funding_z/basis features silently
            # freeze at their startup values for the life of the process --
            # no error, no warning, just stale signals and stale ML training
            # data from then on.
            await backfill_symbol_history(
                runtime.rest_client, f"MARK:{product.symbol}", "15m", lookback_seconds=3 * 3600
            )
            await backfill_symbol_history(
                runtime.rest_client, f"FUNDING:{product.symbol}", "1h", lookback_seconds=6 * 3600
            )
            await backfill_symbol_history(
                runtime.rest_client, f"OI:{product.symbol}", "1h", lookback_seconds=6 * 3600
            )
            # Spot index was never backfilled here OR at startup -- confirmed
            # against the live DB (zero rows, ever) -- so the "basis"
            # feature (mark_price - spot_price) has been silently defaulted
            # to 0 (neutral) for the life of the system rather than reading
            # a real value.
            if product.spot_index_symbol:
                await backfill_symbol_history(
                    runtime.rest_client, product.spot_index_symbol, "15m", lookback_seconds=3 * 3600
                )
        except Exception:
            logger.exception("Incremental backfill failed for %s", product.symbol)


async def _incremental_backfill_external(runtime) -> None:
    """Keeps the app/data/ingest_external.py pseudo-series (Fear & Greed,
    cross-exchange funding/OI/long-short, Deribit DVOL) fresh. Deliberately
    separate from _incremental_backfill (which runs every 5 minutes for
    Delta's own candles): these come from external, independently-rate-
    -limited services, so an hourly cadence with small per-tick windows is
    plenty -- FNG updates once a day, Binance funding every 8h, and hourly
    is already the native granularity of the OI/long-short/DVOL series
    themselves. Every source is independently fault-tolerant, same as the
    one-time deep backfill in ingest_external.backfill_external_all.
    """
    from app.data.ingest_external import (
        FNG_SYMBOL,
        fetch_binance_funding_history,
        fetch_binance_long_short_ratio,
        fetch_binance_open_interest_hist,
        fetch_deribit_dvol,
        fetch_fear_greed_index,
    )
    from app.data.ingest_rest import upsert_candles

    try:
        fng = await fetch_fear_greed_index(limit=3)
        await upsert_candles(FNG_SYMBOL, "1d", fng)
    except Exception:
        logger.exception("Incremental Fear & Greed refresh failed")

    for product in runtime.registry.all():
        symbol = product.symbol
        try:
            funding = await fetch_binance_funding_history(symbol, limit=3)
            await upsert_candles(f"BFUND:{symbol}", "8h", funding)
        except Exception:
            logger.exception("Incremental Binance funding refresh failed for %s", symbol)
        try:
            oi = await fetch_binance_open_interest_hist(symbol, period="1h", limit=6)
            await upsert_candles(f"BOI:{symbol}", "1h", oi)
        except Exception:
            logger.exception("Incremental Binance OI refresh failed for %s", symbol)
        try:
            lsr = await fetch_binance_long_short_ratio(symbol, period="1h", limit=6)
            await upsert_candles(f"BLSR:{symbol}", "1h", lsr)
        except Exception:
            logger.exception("Incremental Binance long/short refresh failed for %s", symbol)
        try:
            dvol = await fetch_deribit_dvol(symbol, lookback_seconds=6 * 3600)
            if dvol:
                await upsert_candles(f"DVOL:{symbol}", "1h", dvol)
        except Exception:
            logger.exception("Incremental Deribit DVOL refresh failed for %s", symbol)


async def _deadman_switch_supervisor(runtime) -> None:
    mode = await get_trading_mode()
    if mode == "live" and runtime.settings.has_credentials:
        if not runtime.deadman_switch.is_running:
            runtime.deadman_switch.start()
    else:
        if runtime.deadman_switch.is_running:
            await runtime.deadman_switch.stop()


async def _live_sync(runtime) -> None:
    mode = await get_trading_mode()
    if mode != "live" or not runtime.settings.has_credentials:
        return
    try:
        await runtime.live_broker.sync_fills_and_positions()
    except Exception:
        logger.exception("Live sync failed")


async def _daily_loss_supervisor(runtime) -> None:
    """Proactively (not just reactively on the next order submission)
    auto-engages the kill switch the FIRST time today's mark-to-market
    equity breaches max_daily_loss_usd -- flattening every open position
    and forcing paper mode, exactly like the manual kill-switch toggle
    (routes_trading.py's update_kill_switch). This is what makes the daily
    loss limit an actual circuit breaker instead of just a log line:
    risk_manager.check_daily_loss on its own only reports whether a breach
    is happening, and only when something else happens to ask (previously
    only on the next order submission, which never comes if every strategy
    is already flat or holding through a losing position with no new
    signal) -- nothing without this job actually acts on that report.

    Fires at most ONCE per calendar day (a persisted flag keyed to the
    same day risk_manager.daily_pnl itself tracks): a deliberate manual
    re-enable later that same day is respected, not immediately re-fought
    on the very next tick just because the account is still down for the
    day. That "don't fight a manual override" behavior is intentionally
    preserved from the reactive-only version this replaces -- what's
    restored is the actual halt-and-flatten the FIRST breach triggers,
    which the reactive-only version had dropped entirely.
    """
    if await is_kill_switch_on():
        return  # already engaged (manually, or by this job earlier today) -- nothing to do

    equity = await runtime.order_manager.get_equity()
    ok, reason = await runtime.risk_manager.check_daily_loss(equity)
    if ok:
        return

    today = runtime.risk_manager.daily_pnl.day.isoformat()
    if await get_state("daily_kill_switch_auto_triggered_day") == today:
        return  # already auto-triggered (and possibly manually re-enabled since) today

    logger.warning("Daily loss limit breached -- auto-engaging kill switch: %s", reason)
    try:
        mode_to_flatten = await get_trading_mode()
        await runtime.order_manager.flatten_all(mode_override=mode_to_flatten)
        await set_kill_switch(True)
        await set_trading_mode("paper")
    finally:
        # Recorded even if flattening raised partway through, so a bug in
        # the flatten path can't turn into re-attempting it (and re-halting
        # trading) every 15 seconds for the rest of the day.
        await set_state("daily_kill_switch_auto_triggered_day", today)


async def _retrain_models(runtime) -> None:
    from app.data.query import load_candles_df, load_pseudo_series_df
    from app.features.feature_pipeline import build_feature_frame
    from app.ml.train import ML_RESOLUTIONS

    for symbol in runtime.registry.symbols:
        for resolution, cfg in ML_RESOLUTIONS.items():
            try:
                candles = await load_candles_df(symbol, resolution, limit=cfg["limit"])
                if len(candles) < 200:
                    logger.info("Skipping %s retrain for %s: only %d candles", resolution, symbol, len(candles))
                    continue
                funding = await load_pseudo_series_df(f"FUNDING:{symbol}", "1h")
                mark = await load_pseudo_series_df(f"MARK:{symbol}", resolution)
                product = runtime.registry.get(symbol)
                spot = (
                    await load_pseudo_series_df(product.spot_index_symbol, resolution)
                    if product.spot_index_symbol
                    else None
                )
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
                train_and_save(features, symbol=symbol, resolution=resolution, horizon_bars=cfg["horizon_bars"])
            except Exception:
                logger.exception("Retrain failed for %s %s", symbol, resolution)

    await runtime.refresh_ensembles()
