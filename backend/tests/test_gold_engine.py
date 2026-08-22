import numpy as np
import pytest

from app.data.db import SessionLocal
from app.data.models import Candle
from app.gold.engine import GOLD_SPOT_SYMBOL, build_gold_snapshot, load_gold_snapshot, save_gold_snapshot
from app.gold.features import GOLD_FEATURE_COLUMNS


async def _seed_gold_candles(resolution: str, n: int = 1000, seed: int = 17) -> None:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.003, n)
    close = 4_500 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    step = {"15m": 900, "1h": 3600, "1d": 86400}[resolution]
    time = np.arange(n) * step

    async with SessionLocal() as session:
        for i in range(n):
            session.add(
                Candle(
                    symbol=GOLD_SPOT_SYMBOL, resolution=resolution, time=int(time[i]),
                    open=float(open_[i]), high=float(high[i]), low=float(low[i]), close=float(close[i]), volume=0.0,
                )
            )
        await session.commit()


@pytest.mark.asyncio
async def test_snapshot_not_ready_with_no_candles_at_all():
    snapshot = await build_gold_snapshot()
    assert snapshot["ready"] is False
    assert snapshot["price"] is None
    assert snapshot["indicators"] == {}


@pytest.mark.asyncio
async def test_snapshot_ready_with_seeded_candles_covers_every_indicator():
    await _seed_gold_candles("15m", n=1000)
    await _seed_gold_candles("1h", n=1000)
    await _seed_gold_candles("1d", n=200)

    snapshot = await build_gold_snapshot()

    assert snapshot["ready"] is True
    assert snapshot["price"] is not None
    assert snapshot["regime"] in ("trending", "mean_reverting", "random_walk", "unknown")
    for col in GOLD_FEATURE_COLUMNS:
        assert col in snapshot["indicators"]
    assert "events" in snapshot
    assert "seasonality" in snapshot["events"]
    assert "gold_buying_tithis" in snapshot["events"]
    assert "cross_asset" in snapshot
    assert "assets" in snapshot["cross_asset"]


@pytest.mark.asyncio
async def test_snapshot_predictions_report_all_horizons_even_when_untrained(tmp_path, monkeypatch):
    import app.gold.engine as engine_mod
    import app.gold.predict as predict_mod
    import app.gold.train as train_mod

    # Isolated, guaranteed-empty model store -- without this, a real model
    # already trained on disk (from an earlier live retrain elsewhere in
    # this project) would make this test's "untrained" premise false.
    # engine.py imported GOLD_MODEL_STORE_DIR by value, so it needs its own
    # patch too, not just train_mod's.
    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)
    monkeypatch.setattr(predict_mod, "GOLD_MODEL_STORE_DIR", tmp_path)
    monkeypatch.setattr(engine_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    await _seed_gold_candles("15m", n=1000)
    await _seed_gold_candles("1h", n=1000)
    await _seed_gold_candles("1d", n=200)

    snapshot = await build_gold_snapshot()

    assert set(snapshot["predictions"].keys()) == {"intraday", "short_term", "long_term"}
    for horizon, entry in snapshot["predictions"].items():
        # No trained model exists in the real model store for a symbol this
        # test just invented data for -- must gracefully report "not ready"
        # rather than crash.
        assert "score" in entry
        assert "decidable" in entry
        assert entry["action"] == "HOLD"  # untrained -> no call, never crashes
        assert entry["target_price"] is None
        assert entry["valid_until"] is not None


def test_snapshot_save_and_load_round_trip(tmp_path, monkeypatch):
    import app.gold.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_SNAPSHOT_PATH", tmp_path / "gold_snapshot_latest.json")

    payload = {
        "ready": True, "as_of": "2026-08-21T00:00:00+00:00", "price": 4500.0,
        "regime": "trending", "indicators": {"rsi_14": 55.0}, "predictions": {},
    }
    save_gold_snapshot(payload)
    loaded = load_gold_snapshot()

    assert loaded == payload


def test_load_snapshot_before_any_save_reports_not_ready(tmp_path, monkeypatch):
    import app.gold.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_SNAPSHOT_PATH", tmp_path / "does_not_exist.json")

    loaded = load_gold_snapshot()
    assert loaded["ready"] is False
