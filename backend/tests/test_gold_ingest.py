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
async def test_gold_deep_backfill_splices_paxg_history_onto_delta_xaut(monkeypatch):
    """Empty series: PAXG bars strictly BEFORE XAUT's first Delta bar, Delta
    XAUTUSD from that bar on -- one continuous series under GOLD_SPOT_SYMBOL."""
    day = 86_400
    xaut_start = 100 * day

    async def _fake_delta(resolution, since_s, until_s=None):
        return [{"time": xaut_start + i * day, "open": 4600.0, "high": 4610.0, "low": 4590.0,
                 "close": 4605.0, "volume": 9.0} for i in range(20)]

    async def _fake_binance(timeframe, since_ms, max_batches=400, until_ms=None):
        # Binance would also return bars at/after the splice -- they must not be used
        return [{"time": i * day, "open": 4500.0, "high": 4510.0, "low": 4490.0, "close": 4505.0, "volume": 3.0}
                for i in range(120) if until_ms is None or i * day * 1000 < until_ms]

    monkeypatch.setattr(ingest_mod, "_fetch_delta_since", _fake_delta)
    monkeypatch.setattr(ingest_mod, "_fetch_ohlcv_since", _fake_binance)

    await gold_deep_backfill()

    stored = await load_candles_df(GOLD_SPOT_SYMBOL, "1d", limit=1000)
    assert len(stored) == 120
    before = stored[stored["time"] < xaut_start]
    after = stored[stored["time"] >= xaut_start]
    assert len(before) == 100 and (before["close"] == 4505.0).all()  # PAXG history
    assert len(after) == 20 and (after["close"] == 4605.0).all()  # Delta XAUT from its first bar


@pytest.mark.asyncio
async def test_gold_deep_backfill_reuses_stored_paxg_rows_instead_of_refetching(monkeypatch):
    from app.data.ingest_rest import upsert_candles

    day = 86_400
    paxg = [{"time": i * day, "open": 4400.0, "high": 4410.0, "low": 4390.0, "close": 4401.0, "volume": 1.0}
            for i in range(1200)]
    await upsert_candles(ingest_mod.PAXG_HISTORY_SYMBOL, "1d", paxg)

    async def _fake_delta(resolution, since_s, until_s=None):
        return [{"time": 1100 * day + i * day, "open": 4600.0, "high": 4610.0, "low": 4590.0,
                 "close": 4605.0, "volume": 9.0} for i in range(5)] if resolution == "1d" else []

    async def _fail_if_called(*args, **kwargs):
        if args and args[0] == "1d":
            raise AssertionError("stored PAXG daily history must be reused, not refetched")
        return []

    monkeypatch.setattr(ingest_mod, "_fetch_delta_since", _fake_delta)
    monkeypatch.setattr(ingest_mod, "_fetch_ohlcv_since", _fail_if_called)

    await gold_deep_backfill()

    stored = await load_candles_df(GOLD_SPOT_SYMBOL, "1d", limit=10_000)
    assert len(stored) == 1105  # 1100 PAXG rows before the splice + 5 XAUT
    assert stored["close"].iloc[-1] == 4605.0


@pytest.mark.asyncio
async def test_fetch_delta_since_pages_forward_and_returns_oldest_first(monkeypatch):
    import httpx

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.params
        assert q["symbol"] == "XAUTUSD"
        start, end = int(q["start"]), int(q["end"])
        seen.append((start, end))
        bars = [{"time": t, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 3}
                for t in range(start, end, 3600)]
        return httpx.Response(200, json={"result": list(reversed(bars))})  # Delta: newest first

    real_client = httpx.AsyncClient
    monkeypatch.setattr(ingest_mod.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(ingest_mod, "_DELTA_PAGE_BARS", 10)

    rows = await ingest_mod._fetch_delta_since("1h", since_s=0, until_s=25 * 3600)

    assert [r["time"] for r in rows] == [i * 3600 for i in range(25)]
    assert seen == [(0, 36_000), (36_000, 72_000), (72_000, 90_000)]


@pytest.mark.asyncio
async def test_gold_incremental_backfill_covers_every_resolution_from_delta(monkeypatch):
    seen = []

    async def _fake_delta(resolution, since_s, until_s=None):
        seen.append(resolution)
        return [{"time": 1_700_000_000, "open": 4500.0, "high": 4501.0, "low": 4499.0, "close": 4500.5, "volume": 2.0}]

    monkeypatch.setattr(ingest_mod, "_fetch_delta_since", _fake_delta)

    await gold_incremental_backfill()

    assert set(seen) == {"15m", "1h", "1d"}
    for resolution in ingest_mod.GOLD_RESOLUTIONS:
        stored = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=10)
        assert len(stored) == 1
