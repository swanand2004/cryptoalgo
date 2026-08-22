from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from app.data.query import load_candles_df
from app.gold.cross_asset import crypto_correlation_context
from app.gold.events import build_gold_events
from app.gold.features import GOLD_FEATURE_COLUMNS, build_gold_feature_frame, gold_regime
from app.gold.predict import GoldPredictor
from app.gold.train import GOLD_HORIZONS, GOLD_MODEL_STORE_DIR

logger = logging.getLogger(__name__)

GOLD_SPOT_SYMBOL = ".DEXAUTUSD"
_SNAPSHOT_PATH = GOLD_MODEL_STORE_DIR / "gold_snapshot_latest.json"

# Only what's actually used across GOLD_HORIZONS' resolutions -- a small
# local mapping rather than reaching into ingest_rest's internal
# (underscore-prefixed) one.
_RESOLUTION_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def save_gold_snapshot(snapshot: dict) -> None:
    GOLD_MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))


def load_gold_snapshot() -> dict:
    """Serves the persisted copy of the last scheduler-computed snapshot --
    the API route never recomputes the feature bank/predictions itself, so
    a request is always instant regardless of how heavy the calculation is.
    """
    if not _SNAPSHOT_PATH.exists():
        return {
            "ready": False,
            "as_of": None,
            "price": None,
            "regime": "unknown",
            "indicators": {},
            "predictions": {},
            "events": build_gold_events(),
            # crypto_correlation_context is async (queries the DB) and this
            # function is sync/instant-by-design -- this transient state only
            # exists in the brief window before the very first research tick
            # has ever completed, so a static placeholder is fine here.
            "cross_asset": {"ready": False, "resolution": None, "window_bars": None, "assets": {}},
        }
    return json.loads(_SNAPSHOT_PATH.read_text())


def _clean(value: float) -> float | None:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return float(value)


def _load_report_summary(horizon: str) -> dict:
    """Surfaces the model's own honest walk-forward numbers (sample size,
    directional accuracy, IC) alongside its live score -- per the explicit
    "understand gold's move" framing, a bare score with no sense of how much
    data or accuracy backs it would be misleading, especially for
    long_term's genuinely thin daily sample (see GOLD_HORIZONS' docstring).
    """
    report_path = GOLD_MODEL_STORE_DIR / f"gold_{horizon}_latest_report.json"
    if not report_path.exists():
        return {"n_samples": None, "mean_directional_accuracy": None, "mean_information_coefficient": None}
    report = json.loads(report_path.read_text())
    return {
        "n_samples": report.get("n_samples"),
        "mean_directional_accuracy": report.get("mean_directional_accuracy"),
        "mean_information_coefficient": report.get("mean_information_coefficient"),
    }


def _action_for(score: float | None, decidable: bool) -> str:
    if score is None or not decidable:
        return "HOLD"
    if score > 0:
        return "BUY"
    if score < 0:
        return "SELL"
    return "HOLD"


async def build_gold_snapshot() -> dict:
    """Computes gold's full research snapshot: current price, the entire
    GOLD_FEATURE_COLUMNS parameter bank's latest readings, a plain-language
    regime label, an actionable BUY/SELL/HOLD + price-target + valid-until
    call for each horizon (intraday ~1h, short-term ~1 day, long-term ~1
    week), and a real-world Events panel (documented gold seasonality +
    Hindu-calendar gold-buying tithi proximity). This is the ONE function
    the scheduler tick and the API route both call -- scheduler persists its
    result, the route serves the persisted copy, so a slow feature/model
    computation never blocks a request.
    """
    now = dt.datetime.now(dt.timezone.utc)
    resolutions_needed = {cfg["resolution"] for cfg in GOLD_HORIZONS.values()}
    frames: dict[str, pd.DataFrame] = {}
    for resolution in resolutions_needed:
        limit = max(cfg["limit"] for cfg in GOLD_HORIZONS.values() if cfg["resolution"] == resolution)
        candles = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=limit)
        if len(candles) < 30:
            continue
        frames[resolution] = build_gold_feature_frame(candles)

    # The finest-grained resolution in play (intraday, 15m) makes the
    # freshest "current reading" -- short_term/long_term still predict off
    # their OWN resolution's frame below regardless.
    display_resolution = GOLD_HORIZONS["intraday"]["resolution"]
    display_frame = frames.get(display_resolution)
    if display_frame is None or display_frame.empty:
        return {
            "ready": False,
            "as_of": now.isoformat(),
            "price": None,
            "regime": "unknown",
            "indicators": {},
            "predictions": {},
            "events": build_gold_events(now),
            "cross_asset": await crypto_correlation_context(),
        }

    last = display_frame.iloc[-1]
    current_price = float(last["close"])
    indicators = {col: _clean(last[col]) for col in GOLD_FEATURE_COLUMNS if col in display_frame.columns}
    regime = gold_regime(last.get("hurst_100"))

    predictions: dict[str, dict] = {}
    for horizon, cfg in GOLD_HORIZONS.items():
        frame = frames.get(cfg["resolution"])
        predictor = GoldPredictor(horizon)
        report_summary = _load_report_summary(horizon)
        resolution_seconds = _RESOLUTION_SECONDS[cfg["resolution"]]
        valid_until = (now + dt.timedelta(seconds=cfg["horizon_bars"] * resolution_seconds)).isoformat()

        if frame is None or frame.empty or not predictor.is_ready:
            predictions[horizon] = {
                "score": None, "decidable": False, "resolution": cfg["resolution"], "horizon_bars": cfg["horizon_bars"],
                "action": "HOLD", "target_price": None, "expected_move_pct": None, "valid_until": valid_until,
                **report_summary,
            }
            continue

        gated = predictor.predict_score_batch(frame, require_confidence=True)
        ungated = predictor.predict_score_batch(frame, require_confidence=False)
        gated_last = float(gated.iloc[-1])
        ungated_last = float(ungated.iloc[-1])
        decidable = gated_last != 0.0 or ungated_last == 0.0

        raw_return = predictor.predict_raw_return_last(frame)
        frame_price = float(frame["close"].iloc[-1])
        target_price = frame_price * (1 + raw_return) if raw_return is not None else None

        predictions[horizon] = {
            "score": gated_last, "decidable": decidable, "resolution": cfg["resolution"], "horizon_bars": cfg["horizon_bars"],
            "action": _action_for(gated_last, decidable),
            "target_price": _clean(target_price) if target_price is not None else None,
            "expected_move_pct": _clean(raw_return * 100) if raw_return is not None else None,
            "valid_until": valid_until,
            **report_summary,
        }

    return {
        "ready": True,
        "as_of": now.isoformat(),
        "price": _clean(current_price),
        "regime": regime,
        "indicators": indicators,
        "predictions": predictions,
        "events": build_gold_events(now),
        "cross_asset": await crypto_correlation_context(),
    }
