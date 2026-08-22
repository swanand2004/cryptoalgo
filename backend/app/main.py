from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_gold, routes_market, routes_strategy, routes_trading, routes_ws
from app.config import ALLOWED_ORIGINS
from app.data.ingest_rest import backfill_all
from app.gold.ingest import gold_initial_backfill
from app.runtime import Runtime, get_or_create_runtime
from app.scheduler import build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
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
        await gold_initial_backfill(runtime.rest_client)
        logger.info("Initial gold historical backfill complete")
    except Exception:
        logger.exception("Initial gold backfill failed (will retry incrementally)")


@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(title="cryptoalgo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_market.router)
app.include_router(routes_trading.router)
app.include_router(routes_strategy.router)
app.include_router(routes_ws.router)
app.include_router(routes_gold.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
