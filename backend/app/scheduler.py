from __future__ import annotations

import asyncio
import datetime as dt
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.data.app_state import (
    get_state,
    get_trading_mode,
    is_kill_switch_on,
    set_kill_switch,
    set_state,
)
from app.data.flush import flush_market_cache_to_db
from app.execution.live_broker import fresh_balance
from app.data.ingest_rest import backfill_symbol_history
from app.gold.engine import build_gold_snapshot, save_gold_snapshot
from app.gold.ingest import gold_incremental_backfill
from app.data.retention import run_retention
from app.core.memory import release_free_memory
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
        runtime.order_manager.process_resting_fills, "interval", seconds=5, id="paper_resting_orders", max_instances=1
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
    # Runs BEFORE the first trading tick can matter, and often: this is the
    # trading signal's data source, not a research feed.
    scheduler.add_job(
        _incremental_ladder_history,
        "interval",
        minutes=15,
        args=[runtime],
        id="incremental_ladder_history",
        max_instances=1,
        next_run_time=now,
    )
    # The 15-minute refresh above can leave the just-closed hour stored with only
    # its first minutes for up to 15 min; this one, 30 s past every hour, stores
    # its final values (and the new forming hour) so the Analyst's closed-bar
    # view is current within about a minute (service.final_bars, 2026-10-07).
    scheduler.add_job(
        _binance_hour_close_refresh, "cron", minute=0, second=30, args=[runtime],
        id="binance_hour_close_refresh", max_instances=1, coalesce=True, misfire_grace_time=600,
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
    scheduler.add_job(
        _daily_profit_target_supervisor, "interval", seconds=15, args=[runtime],
        id="daily_profit_target_supervisor", max_instances=1, next_run_time=now,
    )
    scheduler.add_job(
        _drawdown_supervisor, "interval", seconds=15, args=[runtime],
        id="drawdown_supervisor", max_instances=1, next_run_time=now,
    )
    # The rules-based Analyst (app/analyst): every minute on the four major coins; paper exits on every live tick (5 s) plus a 15-min hourly-bar sweep
    # that catches anything crossed while the app was down.
    scheduler.add_job(
        _analyst_run, "interval", seconds=60, args=[runtime], id="analyst_run", max_instances=1,
        next_run_time=now + dt.timedelta(minutes=2),
    )
    scheduler.add_job(
        _analyst_gold_run, "interval", minutes=5, args=[runtime], id="analyst_gold_run", max_instances=1,
        next_run_time=now + dt.timedelta(minutes=3),
    )
    # FRED dollar / real yields / breakevens + CFTC: read by the gold page and the
    # Analyst's gold macro readings. Had no schedule (data froze 2026-09-02).
    scheduler.add_job(
        _macro_refresh, "cron", hour=6, minute=20, id="macro_refresh", max_instances=1,
        next_run_time=now + dt.timedelta(minutes=6),
    )
    scheduler.add_job(
        _analyst_paper_exits, "interval", minutes=15, id="analyst_paper_exits", max_instances=1,
        next_run_time=now + dt.timedelta(minutes=5),
    )
    scheduler.add_job(
        _analyst_live_exits, "interval", seconds=5, args=[runtime], id="analyst_live_exits", max_instances=1,
        next_run_time=now + dt.timedelta(minutes=4),
    )
    scheduler.add_job(
        _halted_means_flat, "interval", seconds=15, args=[runtime],
        id="halted_means_flat", max_instances=1, next_run_time=now,
    )
    scheduler.add_job(
        _daily_kill_switch_reset, "interval", seconds=60,
        id="daily_kill_switch_reset", max_instances=1, next_run_time=now,
    )
    # Runs once daily at 03:00 UTC (off-peak). Models were manually retrained
    # 2026-08-19 after fixing the spot-index/funding-window feature bugs (see
    # scheduler._incremental_backfill and feature_pipeline.build_feature_frame)
    # so live trading wasn't stuck on models trained against broken features
    # until the next scheduled run.
    if runtime.settings.retrain_in_process:
        scheduler.add_job(_retrain_models, "cron", hour=3, minute=0, args=[runtime], id="ml_retrain", max_instances=1)
    scheduler.add_job(
        _record_equity, "interval", seconds=60, args=[runtime], id="record_equity", max_instances=1, next_run_time=now
    )

    # Gold research: NOT part of trading (no PaperBroker position, no PnL --
    # per explicit request, a pure prediction/research page) -- its own
    # continuous data refresh + full parameter-bank recompute, independent
    # of every crypto job above.
    scheduler.add_job(
        _gold_research_tick, "interval", minutes=3, args=[runtime], id="gold_research_tick", max_instances=1, next_run_time=now,
    )
    # Offset from ml_retrain (03:00) to avoid both heavy retrains competing
    # for CPU at the exact same moment.
    if runtime.settings.retrain_in_process:
        scheduler.add_job(_gold_retrain, "cron", hour=3, minute=30, args=[runtime], id="gold_retrain", max_instances=1)
    # Opt-in Analyst weight training (Settings.analyst_training_enabled): clear
    # of the 03:00 ML and 03:30 gold retrains; does nothing while it is off.
    scheduler.add_job(_analyst_training, "cron", hour=2, minute=40, args=[runtime], id="analyst_training", max_instances=1)
    # PAPER S7 options tracker (app/options/s7.py): never places an order. Every
    # 5 min, aligned to :00, so the Friday 12:00 UTC entry fires on time; the
    # same tick settles expired paper options.
    scheduler.add_job(
        _options_s7_tick, "cron", minute="*/5", args=[runtime], id="options_s7", max_instances=1,
        coalesce=True, misfire_grace_time=240,
    )
    # Every 6 h (was daily): a daily prune let each raw table reach retention +
    # 1 day before it ran. Orderbook snapshots 1 day, 1m/5m candles 4/14 days
    # (app/data/retention.py run_retention; settings at the end of config.py).
    scheduler.add_job(
        run_retention, "cron", hour="4,10,16,22", minute=15, args=[runtime.settings],
        id="prune_raw_ticks", max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
    # ML book filter v1 (app/ml/book_filter.py): daily multipliers from the bar
    # closing 00:00 UTC, applied from 01:00. Retried until the price refresh has
    # finalised that bar; the 00:50 attempt stores a stale (all 1.0) record.
    scheduler.add_job(
        _ml_book_filter, "cron", hour=0, minute="5,20,35,50", args=[runtime],
        id="ml_book_filter", max_instances=1, coalesce=True, misfire_grace_time=600,
    )
    # Owner's Telegram morning report at 09:05 IST (03:35 UTC): yesterday's P&L. Returns at once
    # while telegram_bot_token / telegram_chat_id are unset (the default).
    scheduler.add_job(
        _telegram_daily_summary, "cron", hour=3, minute=35, args=[runtime],  # 09:05 IST, yesterday's P&L
        id="telegram_daily_summary", max_instances=1, misfire_grace_time=1800,
    )

    return scheduler


async def _telegram_daily_summary(runtime) -> None:
    from app.notify.telegram import send_daily_summary

    try:
        await send_daily_summary(runtime)
    except Exception:
        logger.warning("Telegram daily summary failed", exc_info=True)


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


async def _incremental_ladder_history(runtime) -> None:
    """Keeps BINANCE:{symbol} hourly prices and BFUNDHIST:{symbol} funding
    current, because the live trading signal is computed from them.

    This is not a research-data refresh, it is a TRADING dependency. Before
    trend_ladder went live these two series were backfill-only -- populated
    once by a script and never touched again -- which was fine when they only
    fed backtests. It is not fine now: a frozen price series makes the ladder
    hold its last signal forever while the market moves, and the loop cannot
    tell that apart from genuine conviction.

    The loop does defend itself (`ladder_max_bar_age_seconds` stops trading on
    a stale frame), so the worst case here is that the system stops trading
    rather than trades wrongly. This job is what keeps that from happening.

    Runs every 15 minutes against a ~10-day window: bars are hourly, so this
    is deliberately far more often than strictly needed, to leave room for
    several consecutive failures before the staleness guard trips.
    """
    from app.data.ingest_binance_history import backfill_deep_history, backfill_deep_funding

    symbols = [p.symbol for p in runtime.registry.all()]
    if not symbols:
        return
    start_ms = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).timestamp() * 1000)
    try:
        await backfill_deep_history(symbols=symbols, resolutions=("1h", "4h"), start_ms=start_ms)
    except Exception:
        logger.exception("Incremental BINANCE price refresh failed -- ladder may go stale")
    try:
        await backfill_deep_funding(symbols=symbols, start_ms=start_ms)  # recent window, not 2019 onwards
    except Exception:
        logger.exception("Incremental BFUNDHIST funding refresh failed")


async def _binance_hour_close_refresh(runtime) -> None:
    """Light top-of-hour BINANCE 1h refresh (last 3 h only): finalises the hour
    that just closed. The 15-minute _incremental_ladder_history stays the main
    refresh; this only shortens the window in which the newest stored "closed"
    bar is still the partial one."""
    from app.data.ingest_binance_history import backfill_deep_history

    symbols = [p.symbol for p in runtime.registry.all()]
    if not symbols:
        return
    start_ms = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)).timestamp() * 1000)
    try:
        await backfill_deep_history(symbols=symbols, resolutions=("1h",), start_ms=start_ms)
    except Exception:
        logger.exception("Top-of-hour BINANCE 1h refresh failed (the 15-minute refresh still runs)")


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
        fetch_and_store_news_sentiment,
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

    try:
        await fetch_and_store_news_sentiment([p.symbol for p in runtime.registry.all()])
    except Exception:
        logger.exception("Incremental news sentiment refresh failed")


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


async def _sync_external_flows(runtime) -> None:
    """Applies new deposits/withdrawals to the risk anchors (live only) so the
    checks that follow measure trading P&L, not money moved in or out."""
    if await get_trading_mode() != "live":
        return
    broker = getattr(getattr(runtime, "order_manager", None), "live_broker", None)
    if broker is None or not hasattr(broker, "external_flows"):
        return
    try:
        await runtime.risk_manager.apply_external_flows(await broker.external_flows())
    except Exception:
        logger.exception("Could not read deposits/withdrawals from Delta -- checks run on raw equity")


async def _analyst_run(runtime) -> None:
    from app.analyst.service import run_analysis

    try:
        await run_analysis(runtime)
    except Exception:
        logger.exception("Analyst run failed")


async def _analyst_gold_run(runtime) -> None:
    from app.analyst.service import run_gold_analysis

    try:
        await run_gold_analysis(runtime)
    except Exception:
        logger.exception("Analyst gold run failed")


async def _macro_refresh() -> None:
    from app.gold.macro_factors import backfill_macro_factors

    try:
        await backfill_macro_factors()
    except Exception:
        logger.exception("Macro refresh failed")


async def _analyst_live_exits(runtime) -> None:
    from app.analyst import service
    from app.analyst.execution import manage_open_trades

    try:
        await manage_open_trades(runtime, {**service._latest, **service._latest_gold})
    except Exception:
        logger.exception("Analyst live exit check failed")


async def _ml_book_filter(runtime) -> None:
    from app.ml.book_filter import run_daily_job

    try:
        now = dt.datetime.now(dt.timezone.utc)
        await run_daily_job(runtime.settings, now=now, final_attempt=now.minute >= 50)
    except Exception:
        logger.exception("ML book filter daily job failed -- sizing unaffected (multipliers stay 1.0)")
    finally:
        release_free_memory()


async def _options_s7_tick(runtime) -> None:
    from app.options.s7 import run_s7_tick

    try:
        await run_s7_tick(runtime)
    except Exception:
        logger.exception("Options S7 paper tick failed")


async def _analyst_training(runtime) -> None:
    if not getattr(runtime.settings, "analyst_training_enabled", False):
        return
    from app.analyst.training import run_training_job

    try:
        await run_training_job(runtime.settings)  # CPU part runs in a worker thread
    except Exception:
        logger.exception("Analyst training failed -- weights unchanged")
    finally:
        release_free_memory()


async def _analyst_paper_exits() -> None:
    from app.analyst.service import update_paper_trades

    try:
        await update_paper_trades()
    except Exception:
        logger.exception("Analyst paper-trade update failed")


async def _halted_means_flat(runtime) -> None:
    """While the kill switch is on the account must be flat. Every supervisor
    stands down while it is on (nothing manages a position then), so anything
    still open -- an order that was already in flight when the halt came, or
    a manual trade -- would sit unwatched until the switch is turned off.
    Close it."""
    if not await is_kill_switch_on():
        return
    positions = await runtime.order_manager.get_positions()
    open_now = [p for p in positions.values() if p.size != 0]
    if not open_now:
        return
    logger.warning(
        "Kill switch is on but %d position(s) are open (%s) -- closing them",
        len(open_now), ", ".join(f"{p.symbol} {p.size:+d}" for p in open_now),
    )
    await runtime.order_manager.flatten_all(mode_override=await get_trading_mode(), reason="kill_switch")


async def _daily_loss_supervisor(runtime) -> None:
    """Proactively (not just reactively on the next order submission)
    auto-engages the kill switch the FIRST time today's mark-to-market
    equity breaches daily_loss_limit_pct -- flattening every open position
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

    with fresh_balance():  # never the shared cache here -- see live_broker.fresh_balance
        equity = await runtime.order_manager.get_equity()
    await _sync_external_flows(runtime)  # AFTER the balance read: a deposit in it is in the list
    ok, reason = await runtime.risk_manager.check_daily_loss(equity)
    if ok:
        return

    today = runtime.risk_manager.daily_pnl.day.isoformat()
    if await get_state("daily_kill_switch_auto_triggered_day") == today:
        return  # already auto-triggered (and possibly manually re-enabled since) today

    logger.warning("Daily loss limit breached -- auto-engaging kill switch: %s", reason)
    try:
        # HALT FIRST, then flatten: with the flag set, every new entry is
        # refused (OrderManager.submit + run_trading_tick check it), so a
        # trading tick already in flight cannot open a position in the gap --
        # 2026-10-05 one did, buying BNB 2 s after this flattened ETH. Halt,
        # don't switch to paper: the account stays on real money.
        await set_kill_switch(True, reason=f"daily loss limit -- {reason}")  # reason = owner alert text
        mode_to_flatten = await get_trading_mode()
        await runtime.order_manager.flatten_all(mode_override=mode_to_flatten, reason="daily_loss")
    finally:
        # Recorded even if flattening raised partway through, so a bug in
        # the flatten path can't turn into re-attempting it (and re-halting
        # trading) every 15 seconds for the rest of the day.
        await set_state("daily_kill_switch_auto_triggered_day", today)


async def _daily_profit_target_supervisor(runtime) -> None:
    """Mirror of _daily_loss_supervisor for the upside: once today's mark-
    to-market gain hits settings.daily_profit_target_pct of the day's
    starting equity, auto-engages the kill switch and flattens every open
    position -- locking in a good day instead of risking giving it back.

    Same "fires at most once per calendar day, respects a later manual
    re-enable" semantics as the loss supervisor, tracked by its OWN
    persisted flag (daily_profit_target_auto_triggered_day) so it can never
    be confused with, or blocked by, the separate loss-breach flag -- PnL
    can't be breaching both limits at once, but keeping the flags distinct
    means a target-triggered halt and a loss-triggered halt are always
    logged and re-armed independently.
    """
    if await is_kill_switch_on():
        return  # already engaged (manually, or by this job earlier today) -- nothing to do

    with fresh_balance():  # never the shared cache here -- see live_broker.fresh_balance
        equity = await runtime.order_manager.get_equity()
    await _sync_external_flows(runtime)  # AFTER the balance read: a deposit in it is in the list
    ok, reason = await runtime.risk_manager.check_daily_profit_target(equity)
    if ok:
        return

    today = runtime.risk_manager.daily_pnl.day.isoformat()
    if await get_state("daily_profit_target_auto_triggered_day") == today:
        return  # already auto-triggered (and possibly manually re-enabled since) today

    logger.warning("Daily profit target reached -- auto-engaging kill switch: %s", reason)
    try:
        # HALT FIRST, then flatten: with the flag set, every new entry is
        # refused (OrderManager.submit + run_trading_tick check it), so a
        # trading tick already in flight cannot open a position in the gap --
        # 2026-10-05 one did, buying BNB 2 s after this flattened ETH. Halt,
        # don't switch to paper: the account stays on real money.
        await set_kill_switch(True, reason=f"daily profit target -- {reason}")  # reason = owner alert text
        mode_to_flatten = await get_trading_mode()
        await runtime.order_manager.flatten_all(mode_override=mode_to_flatten, reason="profit_target")
    finally:
        await set_state("daily_profit_target_auto_triggered_day", today)


async def _drawdown_supervisor(runtime) -> None:
    """The multi-day risk control the daily supervisors structurally cannot
    provide: halts trading when equity falls settings.max_drawdown_pct below
    its persistent high-water mark, regardless of how any individual day
    looked. Three consecutive 9% down days breach nothing daily while taking
    the account 27% off its peak -- this is the check that sees that.

    Unlike the daily supervisors it uses NO once-per-day flag, because a
    drawdown isn't a daily event: instead it sets a persistent
    `drawdown_halt_active` flag that _daily_kill_switch_reset explicitly
    refuses to clear, so the halt survives day boundaries. That flag is
    cleared only by a deliberate manual kill-switch-off (see
    routes_trading.update_kill_switch), which also re-bases the high-water
    mark -- without that re-basing, acknowledging the halt would re-trigger
    it on the very next tick, since the old peak is still just as far above
    current equity as it was a moment ago.
    """
    if await is_kill_switch_on():
        return

    with fresh_balance():  # never the shared cache here -- see live_broker.fresh_balance
        equity = await runtime.order_manager.get_equity()
    await _sync_external_flows(runtime)  # AFTER the balance read: a deposit in it is in the list
    ok, reason = await runtime.risk_manager.check_max_drawdown(equity)
    if ok:
        return

    logger.warning("Max drawdown breached -- auto-engaging kill switch: %s", reason)
    try:
        # HALT FIRST, then flatten: with the flag set, every new entry is
        # refused (OrderManager.submit + run_trading_tick check it), so a
        # trading tick already in flight cannot open a position in the gap --
        # 2026-10-05 one did, buying BNB 2 s after this flattened ETH. Halt,
        # don't switch to paper: the account stays on real money.
        await set_kill_switch(True, reason=f"max drawdown -- {reason}")  # reason = owner alert text
        mode_to_flatten = await get_trading_mode()
        await runtime.order_manager.flatten_all(mode_override=mode_to_flatten, reason="max_drawdown")
    finally:
        # Set even if flattening raised partway through: a halt that failed
        # to fully flatten is MORE reason to stay halted, not less.
        await set_state("drawdown_halt_active", "on")


async def _daily_kill_switch_reset() -> None:
    """Resets the kill switch to OFF once the calendar day (UTC) it was
    engaged on is no longer today -- the loss/profit limits it protects are
    DAILY by definition, so a halt from a previous day (whether the daily
    supervisors auto-engaged it, or a human engaged it manually) must never
    silently carry over and block trading on a fresh day that hasn't
    breached anything itself yet.

    Keyed off `kill_switch_engaged_day` (stamped by app_state.set_kill_switch
    itself, every time the switch is turned ON -- see its docstring), NOT
    this job's own "have I checked before" bookkeeping: a first-run-only
    guard would (and initially did, see 2026-08-23 regression below) miss a
    switch that was ALREADY stale from a previous day the very first time
    this job ever runs, since it would have no prior state of its own to
    compare against and would wrongly assume "nothing to roll over."
    Comparing directly against WHEN THE SWITCH ITSELF was engaged has no
    such blind spot: a None/older `kill_switch_engaged_day` is unambiguously
    not-today regardless of whether this job has ever run before.

    Deliberately does NOT touch daily_kill_switch_auto_triggered_day /
    daily_profit_target_auto_triggered_day -- those already self-invalidate
    once "today" no longer matches the stored date, so the two daily
    supervisors are free to trigger fresh on the new day without any help
    from this job.

    Also deliberately does NOT clear a MAX-DRAWDOWN halt (see
    _drawdown_supervisor): that limit is peak-to-trough across days, not
    daily, so a new calendar day is not evidence the drawdown has recovered
    -- auto-clearing it here would quietly turn the one multi-day risk
    control in the system back into a daily one, which is precisely the hole
    it was added to close.
    """
    if not await is_kill_switch_on():
        return

    if (await get_state("drawdown_halt_active")) == "on":
        return

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    engaged_day = await get_state("kill_switch_engaged_day")
    if engaged_day != today:
        logger.info("Kill switch has been on since %s (not today, %s) -- resetting to OFF", engaged_day, today)
        await set_kill_switch(False)


async def _retrain_models(runtime) -> None:
    from app.data.query import load_candles_df, load_pseudo_series_df
    from app.ml.train import ML_RESOLUTIONS

    # Core symbols only when the size-aware universe is on: the small-account
    # fillers never feed an ML model, and training them would multiply the
    # nightly job on a 1-vCPU server for nothing.
    core = set(runtime.settings.core_symbol_list)
    for symbol in [s for s in runtime.registry.symbols if not core or s in core]:
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
                news = await load_pseudo_series_df(f"NEWS:{symbol}", "1h")

                # Off the event loop, like the gold retrain: run inline, the
                # feature build + LightGBM fit froze trading ticks, the
                # kill-switch supervisors and the API for the whole retrain
                # (measured 41 s on a laptop; minutes on a 1-core VPS).
                await asyncio.to_thread(
                    _train_crypto_sync, candles, symbol, resolution, cfg["horizon_bars"],
                    dict(funding=funding, mark=mark, spot=spot, fear_greed=fear_greed,
                         binance_funding=binance_funding, binance_oi=binance_oi,
                         binance_lsr=binance_lsr, dvol=dvol, news=news),
                )
            except Exception:
                logger.exception("Retrain failed for %s %s", symbol, resolution)

    await runtime.refresh_ensembles()


def _train_crypto_sync(candles, symbol: str, resolution: str, horizon_bars: int, series: dict) -> None:
    from app.features.feature_pipeline import build_feature_frame

    features = build_feature_frame(candles, **series)
    train_and_save(features, symbol=symbol, resolution=resolution, horizon_bars=horizon_bars)


async def _gold_research_tick(runtime) -> None:
    """Gold's own continuous research loop: refresh PAXG/USDT candles,
    recompute the full parameter bank + all horizon predictions, and
    persist the result for the API route to serve instantly. Entirely
    independent of trading -- no PaperBroker position, no PnL, no risk
    manager involvement -- per explicit request that this be a pure
    prediction/research page.
    """
    try:
        await gold_incremental_backfill()
        snapshot = await build_gold_snapshot()
        save_gold_snapshot(snapshot)
    except Exception:
        logger.exception("Gold research tick failed")
    finally:
        release_free_memory()


def _train_gold_horizon_sync(candles, horizon: str, cfg: dict):
    """The CPU-heavy half of a gold retrain (feature computation + LightGBM
    training) -- called via asyncio.to_thread by _gold_retrain below, never
    awaited directly on the main event loop. Same reasoning as engine.py's
    _build_gold_snapshot_sync: PAXG's much deeper history (up to ~128k rows
    for intraday) makes this take real time now, and running it inline on
    the event loop would freeze the entire app, not just gold, for however
    long it takes.
    """
    from app.gold.features import build_gold_feature_frame
    from app.gold.train import train_and_save_gold

    features = build_gold_feature_frame(candles)
    return train_and_save_gold(
        features, horizon=horizon, horizon_bars=cfg["horizon_bars"], resolution=cfg["resolution"],
        feature_columns=cfg["feature_columns"], warmup_bars=cfg["warmup_bars"], n_splits=cfg["n_splits"],
    )


async def _gold_retrain(runtime) -> None:
    from app.data.query import load_candles_df
    from app.gold.ingest import GOLD_SPOT_SYMBOL
    from app.gold.train import GOLD_HORIZONS

    for horizon, cfg in GOLD_HORIZONS.items():
        try:
            candles = await load_candles_df(GOLD_SPOT_SYMBOL, cfg["resolution"], limit=cfg["limit"])
            if len(candles) < 90:
                logger.info("Skipping gold %s retrain: only %d candles", horizon, len(candles))
                continue
            await asyncio.to_thread(_train_gold_horizon_sync, candles, horizon, cfg)
        except Exception:
            logger.exception("Gold retrain failed for %s", horizon)
