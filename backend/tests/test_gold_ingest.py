import pytest

import app.gold.ingest as ingest_mod
from app.data.query import load_candles_df
from app.gold.ingest import (
    GOLD_SPOT_SYMBOL,
    gold_deep_backfill,
    gold_incremental_backfill,
)


class _FakeExchange:
    """Records every fetch_ohlcv call and serves canned batches in order --
    stands in for ccxt's real Binance client so these tests never touch the
    network.
    """

    def __init__(self, batches: list[list[list]] | None = None):
        self.enableRateLimit = False
        self.calls: list[dict] = []
        self._batches = list(batches or [])
        self.closed = False

    async def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit})
        if not self._batches:
            return []
        return self._batches.pop(0)

    async def close(self):
        self.closed = True


def _candle_row(ts_ms: int, price: float = 4500.0) -> list:
    return [ts_ms, price, price * 1.001, price * 0.999, price, 12.5]


@pytest.mark.asyncio
async def test_fetch_ohlcv_since_pages_across_multiple_full_batches(monkeypatch):
    batch1 = [_candle_row(i * 900_000) for i in range(1000)]
    batch2 = [_candle_row((1000 + i) * 900_000) for i in range(1000)]
    batch3 = [_candle_row((2000 + i) * 900_000) for i in range(200)]  # short -> stop
    fake = _FakeExchange([batch1, batch2, batch3])
    monkeypatch.setattr(ingest_mod.ccxt_async, "binance", lambda: fake)

    rows = await ingest_mod._fetch_ohlcv_since("15m", since_ms=0)

    assert len(rows) == 2200
    assert fake.closed is True
    # each successive call starts right after the previous batch's last candle
    assert fake.calls[1]["since"] == batch1[-1][0] + 1
    assert fake.calls[2]["since"] == batch2[-1][0] + 1


@pytest.mark.asyncio
async def test_fetch_ohlcv_since_stops_at_max_batches_safety_cap(monkeypatch):
    """An exchange that (hypothetically) never returns a short final page
    must not page forever -- max_batches is a hard stop."""
    infinite_full_batches = [[_candle_row(i * 60_000) for i in range(1000)] for _ in range(50)]
    fake = _FakeExchange(infinite_full_batches)
    monkeypatch.setattr(ingest_mod.ccxt_async, "binance", lambda: fake)

    rows = await ingest_mod._fetch_ohlcv_since("1m", since_ms=0, max_batches=3)

    assert len(fake.calls) == 3
    assert len(rows) == 3000


@pytest.mark.asyncio
async def test_gold_deep_backfill_skips_resolutions_that_already_have_history(monkeypatch):
    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("should never fetch a resolution that already has >=1000 candles")

    monkeypatch.setattr(ingest_mod, "_fetch_ohlcv_since", _fail_if_called)

    from app.data.ingest_rest import upsert_candles

    for resolution in ingest_mod.GOLD_RESOLUTIONS:
        candles = [
            {"time": i * 900, "open": 4500.0, "high": 4501.0, "low": 4499.0, "close": 4500.0, "volume": 1.0}
            for i in range(1000)
        ]
        await upsert_candles(GOLD_SPOT_SYMBOL, resolution, candles)

    await gold_deep_backfill()  # must not raise -- proves it never called the fetch


@pytest.mark.asyncio
async def test_gold_deep_backfill_pulls_and_stores_history_for_an_empty_resolution(monkeypatch):
    async def _fake_fetch(timeframe, since_ms, max_batches=400):
        return [{"time": i * 86400, "open": 4500.0, "high": 4510.0, "low": 4490.0, "close": 4505.0, "volume": 3.0} for i in range(50)]

    monkeypatch.setattr(ingest_mod, "_fetch_ohlcv_since", _fake_fetch)

    await gold_deep_backfill()

    stored = await load_candles_df(GOLD_SPOT_SYMBOL, "1d", limit=1000)
    assert len(stored) == 50
    assert stored["close"].iloc[0] == 4505.0


@pytest.mark.asyncio
async def test_gold_incremental_backfill_covers_every_resolution(monkeypatch):
    seen_timeframes = []

    async def _fake_fetch(timeframe, since_ms, max_batches=400):
        seen_timeframes.append(timeframe)
        return [{"time": 1_700_000_000, "open": 4500.0, "high": 4501.0, "low": 4499.0, "close": 4500.5, "volume": 2.0}]

    monkeypatch.setattr(ingest_mod, "_fetch_ohlcv_since", _fake_fetch)

    await gold_incremental_backfill()

    assert set(seen_timeframes) == {"15m", "1h", "1d"}
    for resolution in ingest_mod.GOLD_RESOLUTIONS:
        stored = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=10)
        assert len(stored) == 1
