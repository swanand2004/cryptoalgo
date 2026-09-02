import pandas as pd
import pytest

import app.api.routes_news as routes_news
from app.api.routes_news import _label, _matched_symbols, get_headlines, get_sentiment_history


def test_label_thresholds_match_vaders_own_convention():
    assert _label(0.05) == "bullish"
    assert _label(1.0) == "bullish"
    assert _label(-0.05) == "bearish"
    assert _label(-1.0) == "bearish"
    assert _label(0.0) == "neutral"
    assert _label(0.049) == "neutral"
    assert _label(-0.049) == "neutral"


def test_matched_symbols_is_case_insensitive_and_can_match_multiple():
    assert _matched_symbols("Bitcoin rallies as Ethereum lags") == ["BTCUSD", "ETHUSD"]
    assert _matched_symbols("SOLANA network outage") == ["SOLUSD"]
    assert _matched_symbols("General crypto market update") == []


@pytest.fixture(autouse=True)
def _reset_cache():
    routes_news._cache["fetched_at"] = 0.0
    routes_news._cache["headlines"] = []
    yield
    routes_news._cache["fetched_at"] = 0.0
    routes_news._cache["headlines"] = []


@pytest.mark.asyncio
async def test_get_headlines_caches_and_does_not_refetch_within_ttl(monkeypatch):
    calls = {"n": 0}

    async def fake_fetch():
        calls["n"] += 1
        return [{"time": 1_700_000_000, "title": "Bitcoin surges on ETF inflows", "source": "coindesk"}]

    monkeypatch.setattr(routes_news, "fetch_crypto_news_headlines", fake_fetch)

    first = await get_headlines(limit=10)
    second = await get_headlines(limit=10)

    assert calls["n"] == 1  # second call served from cache, not a new fetch
    assert first["headlines"][0]["title"] == "Bitcoin surges on ETF inflows"
    assert first["headlines"][0]["symbols"] == ["BTCUSD"]
    assert second["headlines"] == first["headlines"]


@pytest.mark.asyncio
async def test_get_headlines_falls_back_to_last_known_cache_on_fetch_failure(monkeypatch):
    async def working_fetch():
        return [{"time": 1_700_000_000, "title": "Ethereum upgrade goes smoothly", "source": "cointelegraph"}]

    monkeypatch.setattr(routes_news, "fetch_crypto_news_headlines", working_fetch)
    first = await get_headlines(limit=10)
    assert len(first["headlines"]) == 1

    routes_news._cache["fetched_at"] = 0.0  # force the TTL to look expired

    async def failing_fetch():
        raise RuntimeError("feed unreachable")

    monkeypatch.setattr(routes_news, "fetch_crypto_news_headlines", failing_fetch)
    second = await get_headlines(limit=10)
    assert second["headlines"] == first["headlines"]  # stale cache beats an empty page


@pytest.mark.asyncio
async def test_sentiment_history_maps_close_to_sentiment_and_volume_to_count(monkeypatch):
    fake_df = pd.DataFrame(
        {"time": [1_700_000_000, 1_700_003_600], "open": [0.1, -0.2], "high": [0.1, -0.2],
         "low": [0.1, -0.2], "close": [0.1, -0.2], "volume": [3.0, 1.0]}
    )

    async def fake_load(symbol, resolution, limit=2000):
        assert symbol == "NEWS:BTCUSD"
        return fake_df

    monkeypatch.setattr(routes_news, "load_candles_df", fake_load)

    result = await get_sentiment_history(symbol="BTCUSD", hours=48)
    assert result == [
        {"time": 1_700_000_000, "sentiment": 0.1, "headline_count": 3},
        {"time": 1_700_003_600, "sentiment": -0.2, "headline_count": 1},
    ]


@pytest.mark.asyncio
async def test_sentiment_history_global_maps_to_the_global_series_symbol(monkeypatch):
    seen = {}

    async def fake_load(symbol, resolution, limit=2000):
        seen["symbol"] = symbol
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    monkeypatch.setattr(routes_news, "load_candles_df", fake_load)

    await get_sentiment_history(symbol="GLOBAL", hours=24)
    assert seen["symbol"] == "NEWS:GLOBAL"
