import numpy as np
import pandas as pd
import pytest

from app.core.symbols import ProductInfo, SymbolRegistry
from app.config import Settings
from app.data.db import SessionLocal
from app.data.market_cache import MarketDataCache
from app.data.models import Candle
from app.ml.train import ML_RESOLUTIONS
from app.trading_loop import get_live_ml_predictions


def _synthetic_candle_rows(symbol: str, resolution: str = "15m", n: int = 300, seed: int = 3) -> list[Candle]:
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
            symbol=symbol, resolution=resolution, time=int(time[i]),
            open=float(open_[i]), high=float(high[i]), low=float(low[i]),
            close=float(close[i]), volume=float(volume[i]),
        )
        for i in range(n)
    ]


async def _seed_all_resolutions(symbol: str) -> None:
    async with SessionLocal() as session:
        for resolution in ML_RESOLUTIONS:
            for row in _synthetic_candle_rows(symbol, resolution=resolution):
                session.add(row)
        await session.commit()


def _registry(symbols: list[str]) -> SymbolRegistry:
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    reg = SymbolRegistry(settings)
    for symbol in symbols:
        reg._by_symbol[symbol] = ProductInfo(
            product_id=1, symbol=symbol, contract_type="perpetual_futures", tick_size=0.5,
            contract_value=0.01, underlying_asset=symbol[:3], state="live", default_leverage=10,
            taker_commission_rate=0.0005, maker_commission_rate=0.0002, spot_index_symbol="",
        )
    return reg


class _FakeRuntime:
    def __init__(self, symbols: list[str]) -> None:
        self.registry = _registry(symbols)
        self.cache = MarketDataCache()


class _FakeDecidablePredictor:
    """Same nonzero value both gated and ungated -- a genuinely decidable
    prediction, not an abstention."""

    def __init__(self, symbol: str, resolution: str = "15m") -> None:
        self.is_ready = True

    def predict_score_batch(self, features: pd.DataFrame, require_confidence: bool = True) -> pd.Series:
        return pd.Series([0.6] * len(features), index=features.index)


class _FakeAbstainingPredictor:
    """Gated is zeroed out (abstaining) while ungated is genuinely nonzero --
    the model has an opinion but isn't confident enough to act on it."""

    def __init__(self, symbol: str, resolution: str = "15m") -> None:
        self.is_ready = True

    def predict_score_batch(self, features: pd.DataFrame, require_confidence: bool = True) -> pd.Series:
        value = 0.0 if require_confidence else 0.8
        return pd.Series([value] * len(features), index=features.index)


class _FakeUntrainedPredictor:
    def __init__(self, symbol: str, resolution: str = "15m") -> None:
        self.is_ready = False

    def predict_score_batch(self, features: pd.DataFrame, require_confidence: bool = True) -> pd.Series:
        raise AssertionError("should never be called when not ready")


@pytest.mark.asyncio
async def test_decidable_prediction_reports_its_score_and_decidable_true_for_every_resolution(monkeypatch):
    await _seed_all_resolutions("TESTUSD")

    import app.ml.predict as predict_module

    monkeypatch.setattr(predict_module, "MLPredictor", _FakeDecidablePredictor)

    runtime = _FakeRuntime(["TESTUSD"])
    result = await get_live_ml_predictions(runtime)

    assert set(result["TESTUSD"].keys()) == set(ML_RESOLUTIONS.keys())
    for resolution in ML_RESOLUTIONS:
        entry = result["TESTUSD"][resolution]
        assert entry["score"] == pytest.approx(0.6)
        assert entry["decidable"] is True
        assert entry["resolution"] == resolution


@pytest.mark.asyncio
async def test_abstaining_prediction_reports_zero_and_decidable_false(monkeypatch):
    await _seed_all_resolutions("TESTUSD")

    import app.ml.predict as predict_module

    monkeypatch.setattr(predict_module, "MLPredictor", _FakeAbstainingPredictor)

    runtime = _FakeRuntime(["TESTUSD"])
    result = await get_live_ml_predictions(runtime)

    for resolution in ML_RESOLUTIONS:
        assert result["TESTUSD"][resolution]["score"] == 0.0
        assert result["TESTUSD"][resolution]["decidable"] is False


@pytest.mark.asyncio
async def test_untrained_symbol_reports_none_score_not_a_crash(monkeypatch):
    await _seed_all_resolutions("TESTUSD")

    import app.ml.predict as predict_module

    monkeypatch.setattr(predict_module, "MLPredictor", _FakeUntrainedPredictor)

    runtime = _FakeRuntime(["TESTUSD"])
    result = await get_live_ml_predictions(runtime)

    for resolution in ML_RESOLUTIONS:
        assert result["TESTUSD"][resolution]["score"] is None
        assert result["TESTUSD"][resolution]["decidable"] is False


@pytest.mark.asyncio
async def test_missing_data_for_one_resolution_does_not_break_the_others(monkeypatch):
    # Only 15m has any candles at all -- 1h/4h should gracefully report
    # "no data" rather than crash the whole symbol.
    async with SessionLocal() as session:
        for row in _synthetic_candle_rows("TESTUSD", resolution="15m"):
            session.add(row)
        await session.commit()

    import app.ml.predict as predict_module

    monkeypatch.setattr(predict_module, "MLPredictor", _FakeDecidablePredictor)

    runtime = _FakeRuntime(["TESTUSD"])
    result = await get_live_ml_predictions(runtime)

    assert result["TESTUSD"]["15m"]["score"] == pytest.approx(0.6)
    assert result["TESTUSD"]["1h"]["score"] is None
    assert result["TESTUSD"]["4h"]["score"] is None


@pytest.mark.asyncio
async def test_covers_every_symbol_in_the_registry(monkeypatch):
    await _seed_all_resolutions("AAAUSD")
    await _seed_all_resolutions("BBBUSD")

    import app.ml.predict as predict_module

    monkeypatch.setattr(predict_module, "MLPredictor", _FakeDecidablePredictor)

    runtime = _FakeRuntime(["AAAUSD", "BBBUSD"])
    result = await get_live_ml_predictions(runtime)

    assert set(result.keys()) == {"AAAUSD", "BBBUSD"}
