from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api import (
    routes_analyst,
    routes_gold,
    routes_market,
    routes_ml,
    routes_news,
    routes_options,
    routes_predictions,
    routes_strategy,
    routes_trading,
    routes_ws,
)
from app.api.auth import DashboardAuthMiddleware
from app.config import ALLOWED_ORIGINS, _PRIVATE_ORIGIN_RE, get_settings
from app.core.single_instance import InstanceLock, lock_path_for
from app.data.ingest_rest import backfill_all
from app.gold.ingest import gold_deep_backfill
from app.runtime import Runtime, get_or_create_runtime
from app.scheduler import build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Rotated: the unrotated file reached 255 MB in a month, which on a
        # small always-on VM ends in a full disk and a dead trading loop.
        logging.handlers.RotatingFileHandler("app.log", maxBytes=20_000_000, backupCount=5, encoding="utf-8"),
    ],
)
# Under systemd stderr IS the journal, and the unit file intends "only errors
# in the journal" (app.log keeps everything). Logging INFO there filled the
# VPS's 64 MB volatile journal in ~6 hours (17k lines/hour), so a 24 h log
# review found nothing before 01:38. A terminal still gets INFO.
if not sys.stderr.isatty():
    for _handler in logging.getLogger().handlers:
        if type(_handler) is logging.StreamHandler:
            _handler.setLevel(logging.WARNING)
# APScheduler logs "Running job" + "executed successfully" for every run of
# every 5-15 s job: 46k of 97k log lines in 5.5 h. Its warnings (missed or
# skipped runs) and job exceptions are still logged.
logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def _initial_backfill(runtime: Runtime) -> None:
    logger.info("Starting initial historical backfill...")
    try:
        await backfill_all(runtime.rest_client, runtime.registry, lookback_days=90)
        logger.info("Initial historical backfill complete")
    except Exception:
        logger.exception("Initial backfill failed (will retry incrementally)")

    logger.info("Starting initial gold historical backfill...")
    try:
        await gold_deep_backfill()
        logger.info("Initial gold historical backfill complete")
    except Exception:
        logger.exception("Initial gold backfill failed (will retry incrementally)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Refuse to become a second trading process on the same database (see
    # app/core/single_instance.py). Raising here stops uvicorn before the
    # scheduler exists, so a duplicate can never place an order.
    instance_lock = InstanceLock(lock_path_for(get_settings().database_url))
    if not instance_lock.acquire():
        logger.error(
            "Another cryptoalgo backend already holds %s -- refusing to start a second trading "
            "process on the same database. Stop the other one first.", instance_lock.path,
        )
        raise RuntimeError("cryptoalgo is already running on this database")

    runtime = await get_or_create_runtime()
    app.state.runtime = runtime

    asyncio.create_task(_initial_backfill(runtime))

    scheduler = build_scheduler(runtime)
    scheduler.start()
    app.state.scheduler = scheduler

    logger.info(
        "cryptoalgo backend ready | env=%s | symbols=%s | live_credentials=%s",
        runtime.settings.delta_env.value,
        runtime.registry.symbols,
        runtime.settings.has_credentials,
    )

    yield

    scheduler.shutdown(wait=False)
    await runtime.stop()
    instance_lock.release()


app = FastAPI(title="cryptoalgo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Same private/loopback rule the WebSocket check uses, so HTTP and WS
    # cannot disagree about who is allowed in.
    allow_origin_regex=_PRIVATE_ORIGIN_RE.pattern,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Added last so it runs FIRST -- nothing, CORS preflights included, reaches a
# route without passing it. A no-op unless DASHBOARD_PASSWORD is set.
app.add_middleware(DashboardAuthMiddleware)

app.include_router(routes_market.router)
app.include_router(routes_trading.router)
app.include_router(routes_strategy.router)
app.include_router(routes_ws.router)
app.include_router(routes_gold.router)
app.include_router(routes_news.router)
app.include_router(routes_analyst.router)
app.include_router(routes_predictions.router)
app.include_router(routes_options.router)
app.include_router(routes_ml.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


TRADING_STALE_SECONDS = 600


@app.get("/api/health/trading")
async def trading_health():
    """503 unless the trading loop has recorded paper equity in the last 10
    minutes. /api/health only proves HTTP answers; a process serving 200s with
    a dead scheduler is the failure seen on 2026-09-07. Point an uptime monitor
    here. Reveals nothing but ok/stale and the age, so it needs no password."""
    import datetime as dt

    from fastapi.responses import JSONResponse
    from sqlalchemy import func, select

    from app.data.db import SessionLocal
    from app.data.models import EquitySnapshot

    async with SessionLocal() as session:
        last = (await session.execute(
            select(func.max(EquitySnapshot.timestamp)).where(EquitySnapshot.mode == "paper")
        )).scalar()
    if last is None:
        return JSONResponse({"status": "stale", "age_seconds": None}, status_code=503)
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    age = (dt.datetime.now(dt.timezone.utc) - last).total_seconds()
    ok = age < TRADING_STALE_SECONDS
    return JSONResponse({"status": "ok" if ok else "stale", "age_seconds": round(age)}, status_code=200 if ok else 503)


# The built dashboard (`npm run build`), served by this same process when
# present, so a deploy is ONE container on ONE port: no second host, no CORS,
# and the socket is same-origin. In development the directory is normally
# absent and Vite serves the frontend as before.
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"))

if (FRONTEND_DIST / "index.html").is_file():
    _dist_root = FRONTEND_DIST.resolve()

    # Every method, not just GET: with a GET-only catch-all, a POST to a
    # mistyped /api path matched it and answered 405 Method Not Allowed,
    # which points at the wrong problem. Now it is a plain 404.
    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    async def frontend(path: str, request: Request):
        # Registered after every router, so /api and /ws always win. An
        # unknown /api path must stay a JSON 404, not quietly return HTML.
        if path.startswith(("api/", "ws/")) or request.method not in ("GET", "HEAD"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (_dist_root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(_dist_root):
            return FileResponse(candidate)
        return FileResponse(_dist_root / "index.html")  # client-side routes
