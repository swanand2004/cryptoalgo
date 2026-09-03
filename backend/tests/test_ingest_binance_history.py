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
