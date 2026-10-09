import os

# Must happen before any `app.*` module is imported anywhere in the test
# session, since app.config.get_settings() is lru_cached and app.data.db
# reads the DB URL at import time. Isolates tests from the real dev DB file.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_cryptoalgo.db"
os.environ.setdefault("DELTA_API_KEY", "test-key")
os.environ.setdefault("DELTA_API_SECRET", "test-secret")
# Owner alerts OFF in tests even if a local .env holds a real bot token: an env
# var (even empty) wins over .env, so no test can ever message the owner.
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

import pathlib

import pytest_asyncio

TEST_DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "test_cryptoalgo.db"


@pytest_asyncio.fixture(autouse=True)
async def _clean_db():
    from app.data.db import engine
    from app.data.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
