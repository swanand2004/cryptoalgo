import numpy as np
import pytest

from app.data.db import SessionLocal
from app.data.models import Candle
from app.gold.engine import GOLD_SPOT_SYMBOL, _risk_level, build_gold_snapshot, load_gold_snapshot, save_gold_snapshot
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
        assert entry["risk_level"] == "very_high"
        assert "no validated track record" in entry["risk_reason"]


@pytest.mark.asyncio
async def test_snapshot_loads_the_small_serving_limit_not_the_full_training_limit(monkeypatch):
    """Regression test for a real incident (2026-08-25): build_gold_snapshot
    used to load GOLD_HORIZONS' TRAINING `limit` (up to 150,000 rows for
    intraday) on every 3-minute scheduler tick, and rebuilding the full
    feature bank (in particular hurst_exponent's O(rows x window) rolling
    calculation) at that scale blocked the single asyncio event loop for
    30+ minutes -- freezing the entire app, not just gold. Live serving must
    request each resolution's much smaller `serving_limit` instead.
    """
    import app.gold.engine as engine_mod
    from app.gold.train import GOLD_HORIZONS

    seen_limits: dict[str, int] = {}
    real_load = engine_mod.load_candles_df

    async def _spy_load(symbol, resolution, limit=500):
        seen_limits[resolution] = limit
        return await real_load(symbol, resolution, limit=limit)

    monkeypatch.setattr(engine_mod, "load_candles_df", _spy_load)

    await _seed_gold_candles("15m", n=50)
    await _seed_gold_candles("1h", n=50)
    await _seed_gold_candles("1d", n=50)

    await build_gold_snapshot()

    for cfg in GOLD_HORIZONS.values():
        assert seen_limits[cfg["resolution"]] == cfg["serving_limit"]
        assert seen_limits[cfg["resolution"]] < cfg["limit"]


@pytest.mark.asyncio
async def test_snapshot_predicts_on_only_the_tail_row_not_the_full_frame(tmp_path, monkeypatch):
    """Regression test, same incident as the serving_limit test above:
    predict_score_batch used to run across the ENTIRE feature frame even
    though only its LAST value was ever used -- pure wasted computation that
    got dramatically worse once PAXG's history made frames far larger.
    """
    import app.gold.engine as engine_mod
    import app.gold.predict as predict_mod
    import app.gold.train as train_mod
    from app.gold.features import build_gold_feature_frame
    from app.gold.predict import GoldPredictor
    from app.gold.train import GOLD_HORIZONS, train_and_save_gold

    monkeypatch.setattr(train_mod, "GOLD_MODEL_STORE_DIR", tmp_path)
    monkeypatch.setattr(predict_mod, "GOLD_MODEL_STORE_DIR", tmp_path)
    monkeypatch.setattr(engine_mod, "GOLD_MODEL_STORE_DIR", tmp_path)

    await _seed_gold_candles("15m", n=1000)
    await _seed_gold_candles("1h", n=1000)
    await _seed_gold_candles("1d", n=200)

    from app.data.query import load_candles_df

    cfg = GOLD_HORIZONS["intraday"]
    candles = await load_candles_df(GOLD_SPOT_SYMBOL, "15m", limit=1000)
    features = build_gold_feature_frame(candles)
    train_and_save_gold(
        features, horizon="intraday", horizon_bars=cfg["horizon_bars"], resolution=cfg["resolution"],
        feature_columns=cfg["feature_columns"], warmup_bars=cfg["warmup_bars"], n_splits=1,
    )

    seen_lengths: list[int] = []
    real_predict = GoldPredictor.predict_score_batch

    def _spy_predict(self, feature_df, require_confidence=True):
        seen_lengths.append(len(feature_df))
        return real_predict(self, feature_df, require_confidence=require_confidence)

    monkeypatch.setattr(GoldPredictor, "predict_score_batch", _spy_predict)

    await build_gold_snapshot()

    assert seen_lengths, "predict_score_batch was never called"
    assert all(n == 1 for n in seen_lengths)


def test_risk_level_no_track_record_is_very_high():
    level, reason = _risk_level(None, None)
    assert level == "very_high"
    assert "no validated track record" in reason


def test_risk_level_thin_sample_is_very_high_even_with_good_accuracy():
    """A high accuracy number means little from a handful of samples --
    matches long_term's genuinely thin (n~58) real-world sample."""
    level, reason = _risk_level(0.70, 58)
    assert level == "very_high"
    assert "58" in reason


def test_risk_level_at_or_below_coin_flip_is_very_high():
    level, reason = _risk_level(0.48, 2766)
    assert level == "very_high"
    assert "coin flip" in reason


def test_risk_level_slight_edge_is_high():
    level, _ = _risk_level(0.517, 3786)
    assert level == "high"


def test_risk_level_modest_edge_is_medium():
    level, _ = _risk_level(0.54, 1000)
    assert level == "medium"


def test_risk_level_strong_edge_is_low():
    level, _ = _risk_level(0.60, 1000)
    assert level == "low"


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
