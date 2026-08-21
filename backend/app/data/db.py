from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.data.models import AppState, Base

_settings = get_settings()
engine = create_async_engine(_settings.database_url, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    # WAL lets readers and a single writer proceed concurrently instead of
    # locking the whole file -- important since the scheduler, WS ingestion,
    # and API requests all hit this DB from different tasks.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _seed_app_state()


async def _seed_app_state() -> None:
    from sqlalchemy import select

    async with SessionLocal() as session:
        result = await session.execute(select(AppState).where(AppState.key == "trading_mode"))
        if result.scalar_one_or_none() is None:
            # Always boot in paper mode -- live mode must be enabled explicitly each session.
            session.add(AppState(key="trading_mode", value="paper"))
            session.add(AppState(key="kill_switch", value="off"))
            await session.commit()


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def session_dependency() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
