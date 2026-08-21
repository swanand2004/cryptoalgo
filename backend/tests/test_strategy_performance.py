import numpy as np
import pytest

from app.config import Settings
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.db import SessionLocal
from app.data.models import Candle
from app.strategy.performance import STRATEGY_NAMES, evaluate_all_strategies


def _synthetic_candle_rows(symbol: str, n: int = 300, seed: int = 3) -> list[Candle]:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.004, n)
    close = 1_000 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(10, 100, n)
    time = np.arange(n) * 900

    return [
        Candle(
            symbol=symbol,
            resolution="15m",
            time=int(time[i]),
            open=float(open_[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=float(volume[i]),
        )
        for i in range(n)
    ]


def _registry_with_one_symbol() -> SymbolRegistry:
    reg = SymbolRegistry(Settings(delta_api_key="k", delta_api_secret="s"))
    reg._by_symbol["TESTUSD"] = ProductInfo(
        product_id=1,
        symbol="TESTUSD",
        contract_type="perpetual_futures",
        tick_size=0.5,
        contract_value=0.01,
        underlying_asset="TEST",
        state="live",
        default_leverage=10,
        taker_commission_rate=0.0005,
        maker_commission_rate=0.0002,
        spot_index_symbol="",
    )
    return reg


@pytest.mark.asyncio
async def test_evaluate_all_strategies_returns_one_entry_per_strategy():
    async with SessionLocal() as session:
        for row in _synthetic_candle_rows("TESTUSD"):
            session.add(row)
        await session.commit()

    registry = _registry_with_one_symbol()
    results = await evaluate_all_strategies(registry, resolution="15m", limit=500)

    assert {r.name for r in results} == set(STRATEGY_NAMES)
    for r in results:
        assert 0.0 <= r.win_rate <= 100.0
        assert len(r.by_symbol) == 1
        assert r.by_symbol[0].symbol == "TESTUSD"


@pytest.mark.asyncio
async def test_graha_carries_an_experimental_note():
    """graha is the one strategy in the current lineup with no established
    causal mechanism behind it (real astrology math, but no proven link to
    markets) -- its performance entry must carry a note saying so, so the
    number isn't read at face value without that context.
    """
    async with SessionLocal() as session:
        for row in _synthetic_candle_rows("TESTUSD"):
            session.add(row)
        await session.commit()

    registry = _registry_with_one_symbol()
    results = await evaluate_all_strategies(registry, resolution="15m", limit=500)

    graha = next(r for r in results if r.name == "graha")
    assert graha.note is not None
    assert "EXPERIMENTAL" in graha.note
