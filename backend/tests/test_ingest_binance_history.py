import httpx
import pytest

from app.data.ingest_binance_history import (
    BINANCE_SPOT_SYMBOL,
    HISTORY_PREFIX,
    fetch_binance_klines,
)


def _kline(open_ms: int, close_price: float = 100.0):
    # Binance kline array layout: [openTime, open, high, low, close, volume, ...]
    return [open_ms, "100.0", "101.0", "99.0", str(close_price), "5.0",
            open_ms + 3_600_000, "0", 0, "0", "0", "0"]


class _FakeClient:
    """Stands in for httpx.AsyncClient, replaying scripted kline pages."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    async def get(self, url, params=None):
        self.calls.append(params)
        rows = self._pages.pop(0) if self._pages else []
        return httpx.Response(200, json=rows, request=httpx.Request("GET", url))


@pytest.mark.asyncio
async def test_klines_are_converted_to_candle_dicts_in_seconds():
    client = _FakeClient([[_kline(1_600_000_000_000, 123.5)], []])
    out = await fetch_binance_klines("BTCUSDT", "1h", 0, end_ms=1_600_003_600_000, client=client)
    assert out == [{
        "time": 1_600_000_000,  # milliseconds -> seconds
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 123.5, "volume": 5.0,
    }]


@pytest.mark.asyncio
async def test_paging_advances_from_the_last_open_time_not_a_fixed_step():
    """Stepping by a fixed interval would silently skip gaps (halts, or a
    symbol that had not listed yet). The cursor must follow the data."""
    full = [_kline(1_000_000_000_000 + i * 3_600_000) for i in range(1000)]
    second = [_kline(1_000_000_000_000 + 1000 * 3_600_000)]
    client = _FakeClient([full, second])
    out = await fetch_binance_klines("BTCUSDT", "1h", 1_000_000_000_000,
                                     end_ms=1_900_000_000_000, client=client)
    assert len(out) == 1001
    # Second request must start just after the last bar of the first page.
    assert client.calls[1]["startTime"] == full[-1][0] + 1


@pytest.mark.asyncio
async def test_a_short_page_ends_paging():
    """Fewer than the max means the history is exhausted -- continuing would
    spin on the same final bar forever."""
    client = _FakeClient([[_kline(1_600_000_000_000)]])
    out = await fetch_binance_klines("BTCUSDT", "1h", 0, end_ms=1_900_000_000_000, client=client)
    assert len(out) == 1
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_no_forward_progress_stops_rather_than_looping_forever():
    """Defensive: if the API ever returns a page whose last open-time is at or
    behind the cursor, paging must stop instead of hanging."""
    stuck = [_kline(1_000)] * 1000  # every row has the same open time
    client = _FakeClient([stuck, stuck, stuck])
    out = await fetch_binance_klines("BTCUSDT", "1h", 1_000_000, end_ms=1_900_000_000_000, client=client)
    assert len(out) == 1000  # one page, then it gives up
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_empty_response_is_not_an_error():
    client = _FakeClient([[]])
    assert await fetch_binance_klines("BTCUSDT", "1h", 0, end_ms=1, client=client) == []


def test_history_is_namespaced_away_from_the_live_delta_symbols():
    """Live trading executes against Delta's own prints. Another venue's
    candles must never land on the symbol the trading loop reads."""
    assert HISTORY_PREFIX.endswith(":")
    for delta_symbol in BINANCE_SPOT_SYMBOL:
        assert not delta_symbol.startswith(HISTORY_PREFIX)
        assert f"{HISTORY_PREFIX}{delta_symbol}" != delta_symbol


class _FakeFundingClient:
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    async def get(self, url, params=None):
        self.calls.append(params)
        rows = self._pages.pop(0) if self._pages else []
        return httpx.Response(200, json=rows, request=httpx.Request("GET", url))


def _funding(ms, rate="0.0001"):
    return {"symbol": "BTCUSDT", "fundingTime": ms, "fundingRate": rate}


@pytest.mark.asyncio
async def test_funding_fraction_is_converted_to_percent():
    """ccxt/Binance report funding as a FRACTION; Delta and
    app/features/funding.py both work in PERCENT. Mixing them is the exact
    100x unit bug that made funding_spread meaningless until 2026-09-03."""
    from app.data.ingest_binance_history import fetch_binance_funding_history_deep

    client = _FakeFundingClient([[_funding(1_600_000_000_000, "0.0001")], []])
    out = await fetch_binance_funding_history_deep("BTCUSDT", 0, client=client)
    assert len(out) == 1
    assert out[0]["close"] == pytest.approx(0.01)  # 0.0001 fraction == 0.01 percent
    assert out[0]["time"] == 1_600_000_000


@pytest.mark.asyncio
async def test_negative_funding_keeps_its_sign():
    """Negative funding means shorts pay longs -- inverting it would flip the
    direction of every carry conclusion drawn from it."""
    from app.data.ingest_binance_history import fetch_binance_funding_history_deep

    client = _FakeFundingClient([[_funding(1_600_000_000_000, "-0.0005")], []])
    out = await fetch_binance_funding_history_deep("BTCUSDT", 0, client=client)
    assert out[0]["close"] == pytest.approx(-0.05)


@pytest.mark.asyncio
async def test_funding_paging_stops_without_forward_progress():
    from app.data.ingest_binance_history import fetch_binance_funding_history_deep

    stuck = [_funding(1_000)] * 1000
    client = _FakeFundingClient([stuck, stuck])
    out = await fetch_binance_funding_history_deep("BTCUSDT", 5_000, client=client)
    assert len(out) == 1000 and len(client.calls) == 1


def test_funding_history_is_namespaced_separately_from_the_hourly_series():
    """BFUNDHIST: (deep, 7 years) must not collide with BFUND: (shallow,
    maintained by the hourly scheduler) or with Delta's own FUNDING:."""
    from app.data.ingest_binance_history import FUNDING_PREFIX, HISTORY_PREFIX

    assert FUNDING_PREFIX == "BFUNDHIST:"
    assert FUNDING_PREFIX != HISTORY_PREFIX
    assert not FUNDING_PREFIX.startswith("BFUND:")
