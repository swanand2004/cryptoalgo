import numpy as np
import pytest

from app.data.db import SessionLocal
from app.data.models import Candle
from app.gold.cross_asset import CRYPTO_SYMBOLS, crypto_correlation_context
from app.gold.ingest import GOLD_SPOT_SYMBOL


async def _seed_correlated_series(symbol: str, resolution: str, n: int, correlation_seed_offset: float, shared: np.ndarray) -> None:
    """Builds a price series whose returns are a noisy copy of `shared` --
    lets tests construct a KNOWN, controllable correlation with gold rather
    than depending on real market data.
    """
    noise = np.random.default_rng(hash(symbol) % 1000).normal(0, 0.0005, n)
    log_returns = shared + noise * correlation_seed_offset
    close = 100 * np.exp(np.cumsum(log_returns))
    step = {"15m": 900, "1h": 3600}[resolution]
    time = np.arange(n) * step

    async with SessionLocal() as session:
        for i in range(n):
            session.add(
                Candle(
                    symbol=symbol, resolution=resolution, time=int(time[i]),
                    open=float(close[i]), high=float(close[i] * 1.001), low=float(close[i] * 0.999),
                    close=float(close[i]), volume=0.0,
                )
            )
        await session.commit()


@pytest.mark.asyncio
async def test_not_ready_with_insufficient_gold_history():
    result = await crypto_correlation_context(resolution="1h", limit=300)
    assert result["ready"] is False
    assert result["assets"] == {}


@pytest.mark.asyncio
async def test_detects_a_strong_positive_correlation_when_one_genuinely_exists():
    n = 100
    rng = np.random.default_rng(42)
    shared_signal = rng.normal(0, 0.003, n)  # the common driver both series share

    await _seed_correlated_series(GOLD_SPOT_SYMBOL, "1h", n, correlation_seed_offset=0.2, shared=shared_signal)
    for symbol in CRYPTO_SYMBOLS:
        await _seed_correlated_series(symbol, "1h", n, correlation_seed_offset=0.2, shared=shared_signal)

    result = await crypto_correlation_context(resolution="1h", limit=300)

    assert result["ready"] is True
    for symbol in CRYPTO_SYMBOLS:
        entry = result["assets"][symbol]
        assert entry["correlation"] is not None
        assert entry["correlation"] > 0.5
        assert entry["interpretation"] == "moving_with"
        assert entry["recent_return_pct"] is not None


@pytest.mark.asyncio
async def test_detects_decoupling_when_series_are_genuinely_independent():
    n = 100
    rng = np.random.default_rng(7)
    gold_only = rng.normal(0, 0.003, n)

    await _seed_correlated_series(GOLD_SPOT_SYMBOL, "1h", n, correlation_seed_offset=0.0, shared=gold_only)
    for symbol in CRYPTO_SYMBOLS:
        independent_signal = np.random.default_rng(hash(symbol) % 999).normal(0, 0.003, n)
        await _seed_correlated_series(symbol, "1h", n, correlation_seed_offset=0.0, shared=independent_signal)

    result = await crypto_correlation_context(resolution="1h", limit=300)

    assert result["ready"] is True
    for symbol in CRYPTO_SYMBOLS:
        entry = result["assets"][symbol]
        # Independently-generated random series -- correlation should be
        # small in magnitude (not exactly zero, but nowhere near "strong").
        assert entry["correlation"] is not None
        assert abs(entry["correlation"]) < 0.5


@pytest.mark.asyncio
async def test_missing_crypto_history_reports_unknown_not_a_crash():
    n = 100
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 0.003, n)
    await _seed_correlated_series(GOLD_SPOT_SYMBOL, "1h", n, correlation_seed_offset=0.2, shared=shared)
    # No BTCUSD/ETHUSD/SOLUSD candles seeded at all.

    result = await crypto_correlation_context(resolution="1h", limit=300)

    assert result["ready"] is True
    for symbol in CRYPTO_SYMBOLS:
        assert result["assets"][symbol]["correlation"] is None
        assert result["assets"][symbol]["interpretation"] == "unknown"
        assert result["assets"][symbol]["recent_return_pct"] is None
