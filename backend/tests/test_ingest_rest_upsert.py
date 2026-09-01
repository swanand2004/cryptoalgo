import pytest

from app.data.ingest_rest import upsert_candles
from app.data.query import load_candles_df


@pytest.mark.asyncio
async def test_upsert_candles_handles_a_batch_larger_than_one_sqlite_chunk():
    """Regression test: a single multi-row VALUES insert binds (columns x
    rows) SQL variables at once -- SQLite errors with 'too many SQL
    variables' well past a few thousand rows in one statement. This bit
    gold's PAXG/Binance deep backfill live (2026-08-25), which hands
    upsert_candles a single ~128k-row batch; every existing caller had
    stayed under the limit only because Delta's own REST API caps a single
    candles response to 2000. 5000 rows here is comfortably more than one
    chunk at the current _UPSERT_CHUNK_SIZE=500.
    """
    n = 5000
    candles = [
        {"time": i * 60, "open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i, "close": 100.5 + i, "volume": 1.0}
        for i in range(n)
    ]

    stored = await upsert_candles("BIGBATCHTEST", "1m", candles)

    assert stored == n
    df = await load_candles_df("BIGBATCHTEST", "1m", limit=n + 10)
    assert len(df) == n
    assert df["close"].iloc[0] == 100.5
    assert df["close"].iloc[-1] == 100.5 + (n - 1)


@pytest.mark.asyncio
async def test_upsert_candles_updates_on_conflict_across_chunk_boundaries():
    """The on-conflict-update behavior must still hold even when the
    overlapping rows land in different chunks of a large batch."""
    first = [
        {"time": i * 60, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        for i in range(1200)
    ]
    await upsert_candles("CONFLICTTEST", "1m", first)

    second = [
        {"time": i * 60, "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 2.0}
        for i in range(1200)
    ]
    await upsert_candles("CONFLICTTEST", "1m", second)

    df = await load_candles_df("CONFLICTTEST", "1m", limit=2000)
    assert len(df) == 1200  # updated in place, not duplicated
    assert (df["close"] == 2.0).all()
