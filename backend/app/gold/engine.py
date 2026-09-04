from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from app.data.query import load_candles_df
from app.gold.cross_asset import crypto_correlation_context
from app.gold.events import build_gold_events
from app.gold.macro_factors import CFTC_SYMBOL, FRED_SERIES, blend_score, compute_macro_factors
from app.gold.micro_factors import compute_micro_factors, conviction_multiplier
from app.gold.features import GOLD_FEATURE_COLUMNS, build_gold_feature_frame, gold_regime
from app.gold.ingest import GOLD_SPOT_SYMBOL
from app.gold.predict import GoldPredictor
from app.gold.train import GOLD_HORIZONS, GOLD_MODEL_STORE_DIR

logger = logging.getLogger(__name__)

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



# Requested split: half the call from this project's own model, half from the
# macro factors. See macro_factors.COMPOSITE_FACTORS for what the factor half
# is actually made of and the measured accuracy behind that choice.
GOLD_MODEL_WEIGHT = 0.5


async def _load_macro_factors() -> list:
    """Macro series -> scored factors. Never fatal: gold prediction must keep
    working when FRED or CFTC is unreachable, so a failure yields neutral
    (zero-contribution) factors rather than taking the snapshot down."""
    try:
        frames = {}
        for _series_id, (symbol, _name) in FRED_SERIES.items():
            frames[symbol] = await load_candles_df(symbol, "1d", limit=20_000)
        frames[CFTC_SYMBOL] = await load_candles_df(CFTC_SYMBOL, "1d", limit=20_000)
        return compute_macro_factors(frames)
    except Exception:
        logger.exception("Macro factor load failed -- gold prediction continues without them")
        return compute_macro_factors({})


def _price_levels(frame, horizon_bars: int, action: str) -> dict:
    """Concrete prices at which the call is CONFIRMED or INVALIDATED.

    A target price on its own is not actionable: it says where the model
    thinks price is going but nothing about when to stop believing it. These
    are the levels to watch, derived from actual recent price structure
    rather than round numbers:

      resistance / support -- the highest high and lowest low over a window
          scaled to the prediction's own horizon. A close beyond one of these
          is a genuine break of the range the forecast was made inside.
      breakout_above / breakdown_below -- the level whose breach CONFIRMS the
          call, and the one that breaks it. Which is which flips with
          direction: for a BUY, clearing resistance confirms and losing
          support invalidates; for a SELL it is the mirror image.
      invalidation -- an ATR-based line placed against the position, so it
          adapts to how volatile gold currently is instead of using a fixed
          percentage that means different things in calm and wild markets.

    Returns empty when the frame lacks the columns or history to say anything
    honest, rather than inventing levels from a handful of bars.
    """
    needed = {"high", "low", "close"}
    if frame is None or frame.empty or not needed.issubset(frame.columns):
        return {}

    # Window scaled to the horizon: a 1-hour-ahead call should not be judged
    # against a 3-month range, and a 1-week call should not be judged against
    # the last few bars.
    window = max(20, min(int(horizon_bars) * 10, len(frame)))
    recent = frame.tail(window)
    if len(recent) < 20:
        return {}

    resistance = float(recent["high"].max())
    support = float(recent["low"].min())
    price = float(frame["close"].iloc[-1])

    atr_pct = frame["atr_pct_14"].iloc[-1] if "atr_pct_14" in frame.columns else None
    atr_abs = float(atr_pct) * price if atr_pct is not None and pd.notna(atr_pct) else None

    if action == "BUY":
        breakout_above, breakdown_below = resistance, support
        invalidation = price - 2.0 * atr_abs if atr_abs else support
    elif action == "SELL":
        breakout_above, breakdown_below = support, resistance
        invalidation = price + 2.0 * atr_abs if atr_abs else resistance
    else:
        # HOLD: no position to invalidate, but the range worth watching for a
        # break in either direction is still meaningful.
        breakout_above, breakdown_below, invalidation = resistance, support, None

    return {
        "support": _clean(support),
        "resistance": _clean(resistance),
        "breakout_above": _clean(breakout_above),
        "breakdown_below": _clean(breakdown_below),
        "invalidation": _clean(invalidation) if invalidation is not None else None,
        "atr": _clean(atr_abs) if atr_abs is not None else None,
        "range_window_bars": int(window),
        "distance_to_breakout_pct": _clean((breakout_above / price - 1) * 100),
        "distance_to_breakdown_pct": _clean((breakdown_below / price - 1) * 100),
    }


# Stop distance, in ATR. Matches the `invalidation` line already shown, so
# the trade plan and the levels panel never disagree about where the call
# stops being valid.
TRADE_STOP_ATR = 2.0
# Hard cap. Hugging a structural level is worth doing, chasing one 6 ATR away
# is not -- past this the trade is simply too wide to be the same trade.
TRADE_MAX_STOP_ATR = 3.0
# Slack beyond a structural level, so an obvious wick does not take the stop.
TRADE_STRUCTURE_BUFFER_ATR = 0.25
# Below this the stop costs more than the model's own target pays.
MIN_VIABLE_RR = 1.0


def _trade_plan(levels: dict, action: str, price: float | None,
                target_price: float | None, conviction: float = 1.0) -> dict:
    """Entry / stop / target for the call, plus whether it is worth taking.

    The take-profit is the MODEL'S OWN predicted price, not a structural
    level that happens to give a flattering ratio. That distinction is the
    whole point: resistance is where price might go, the model target is
    where this model says it goes, and only the second one has a measured
    track record attached. The structural level is returned alongside as
    context, clearly named, never substituted in when the model's number is
    unimpressive.

    Consequences of that choice are reported rather than hidden:
      - risk_reward below 1 means the 2-ATR stop costs more than the model's
        own target pays. The plan is still returned, marked not viable.
      - the gated score and the raw return regression can disagree (score
        says BUY, predicted return is negative). That is a real conflict
        inside the model and is surfaced, not resolved by quietly flipping
        one of them.

    HOLD returns no entry -- there is no position -- but keeps the two levels
    whose breach would START a trade, which is the actionable part of a
    no-trade call.
    """
    atr = levels.get("atr")
    support, resistance = levels.get("support"), levels.get("resistance")
    if price is None or not atr or atr <= 0:
        return {"action": "HOLD", "viable": False, "entry": None, "stop_loss": None,
                "take_profit": None, "risk_reward": None,
                "note": "no ATR available -- cannot place a stop honestly"}

    if action == "HOLD":
        return {
            "action": "HOLD", "viable": False, "entry": None, "stop_loss": None,
            "take_profit": None, "risk_reward": None,
            "trigger_long_above": _clean(resistance) if resistance else None,
            "trigger_short_below": _clean(support) if support else None,
            "note": "no position -- a close beyond either level is what would start one",
        }

    long_side = action == "BUY"
    direction = 1.0 if long_side else -1.0
    structure_stop_level = support if long_side else resistance

    stop = price - direction * TRADE_STOP_ATR * atr
    structure_used = False
    # If the structural level sits just beyond the ATR stop, the stop is
    # sitting exactly where a wick would run it. Step past the level instead.
    if structure_stop_level is not None:
        beyond = (structure_stop_level < stop) if long_side else (structure_stop_level > stop)
        if beyond and abs(stop - structure_stop_level) <= 0.5 * atr:
            stop = structure_stop_level - direction * TRADE_STRUCTURE_BUFFER_ATR * atr
            structure_used = True
    # Cap, applied after the structural adjustment so it binds either path.
    cap = price - direction * TRADE_MAX_STOP_ATR * atr
    stop = max(stop, cap) if long_side else min(stop, cap)

    risk = abs(price - stop)
    take_profit = target_price
    reward = (take_profit - price) * direction if take_profit is not None else None

    conflict = take_profit is not None and reward is not None and reward <= 0
    rr = (reward / risk) if (reward is not None and reward > 0 and risk > 0) else None
    structural_target = resistance if long_side else support
    structural_reward = ((structural_target - price) * direction) if structural_target else None

    if conflict:
        note = ("the model's predicted price sits AGAINST its own signal -- the classifier "
                "and the return regression disagree, so there is no honest target here")
    elif rr is None:
        note = "no model target available"
    elif rr < MIN_VIABLE_RR:
        note = (f"target pays {rr:.2f}x what the stop risks -- the model's own predicted move "
                f"does not cover a {TRADE_STOP_ATR:.0f}-ATR stop")
    else:
        note = f"target pays {rr:.2f}x the risk"
    if structure_used:
        note += "; stop placed beyond structure rather than on the flat ATR line"

    return {
        "action": action,
        "viable": bool(rr is not None and rr >= MIN_VIABLE_RR),
        "entry": _clean(price),
        "stop_loss": _clean(stop),
        "take_profit": _clean(take_profit) if take_profit is not None else None,
        "risk_per_unit": _clean(risk),
        "reward_per_unit": _clean(reward) if reward is not None else None,
        "risk_reward": _clean(rr) if rr is not None else None,
        "stop_distance_pct": _clean(risk / price * 100),
        "target_distance_pct": _clean(abs(take_profit - price) / price * 100) if take_profit else None,
        "stop_atr_multiple": _clean(risk / atr),
        # Structure shown as CONTEXT. Not the take-profit, on purpose.
        "structural_target": _clean(structural_target) if structural_target else None,
        "structural_risk_reward": _clean(structural_reward / risk)
            if (structural_reward is not None and structural_reward > 0 and risk > 0) else None,
        # Conviction sizes the trade; it never moves entry, stop or target.
        "size_multiplier": _clean(max(0.0, min(1.0, conviction))),
        "model_conflict": bool(conflict),
        "note": note,
    }


def _action_for(score: float | None, decidable: bool) -> str:
    if score is None or not decidable:
        return "HOLD"
    if score > 0:
        return "BUY"
    if score < 0:
        return "SELL"
    return "HOLD"


def _risk_level(mean_directional_accuracy: float | None, n_samples: int | None) -> tuple[str, str]:
    """Honest risk rating for THIS horizon's model, derived from its own
    walk-forward track record -- not a generic disclaimer. A call from a
    model with a real, well-sampled edge is a genuinely different risk than
    one from a model that hasn't cleared a coin flip yet (see [[project-
    trading-system-state]] for why gold's short_term/long_term horizons
    currently sit in the risky end of this scale). Thresholds are fixed
    ahead of any specific reading, not tuned to make any one horizon look
    better.
    """
    if mean_directional_accuracy is None or n_samples is None:
        return "very_high", "no validated track record yet"
    if n_samples < 100:
        return "very_high", f"only {n_samples} historical samples -- too thin to trust"
    if mean_directional_accuracy <= 0.50:
        return "very_high", f"{mean_directional_accuracy:.0%} historical accuracy -- at or below a coin flip"
    if mean_directional_accuracy < 0.53:
        return "high", f"{mean_directional_accuracy:.0%} historical accuracy -- only a slight edge"
    if mean_directional_accuracy < 0.56:
        return "medium", f"{mean_directional_accuracy:.0%} historical accuracy -- a modest, unproven edge"
    return "low", f"{mean_directional_accuracy:.0%} historical accuracy across {n_samples} samples"


def _build_gold_snapshot_sync(raw_candles: dict[str, pd.DataFrame], now: dt.datetime,
                              macro_factors: list | None = None) -> dict:
    """The actual CPU-heavy work (feature computation, prediction) -- kept
    fully synchronous and called via asyncio.to_thread by build_gold_snapshot
    below, never awaited directly on the main event loop. Real bug caught
    live (2026-08-25): running this inline inside an `async def` still
    blocks the loop for its ENTIRE duration (Python's GIL-bound CPU work
    doesn't yield just because the function is declared async) -- once
    PAXG's ~127k-row intraday history made hurst_exponent's rolling
    calculation take 30+ minutes, that froze the whole app (crypto trading's
    own risk supervisors and HTTP routes included), not just gold.
    """
    frames: dict[str, pd.DataFrame] = {
        resolution: build_gold_feature_frame(candles) for resolution, candles in raw_candles.items()
    }

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
        }

    macro_factors = macro_factors or []

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

        risk_level, risk_reason = _risk_level(
            report_summary.get("mean_directional_accuracy"), report_summary.get("n_samples")
        )

        if frame is None or frame.empty or not predictor.is_ready:
            predictions[horizon] = {
                "score": None, "decidable": False, "resolution": cfg["resolution"], "horizon_bars": cfg["horizon_bars"],
                "action": "HOLD", "levels": {}, "trade": None, "blend": None, "target_price": None, "expected_move_pct": None, "valid_until": valid_until,
                "risk_level": risk_level, "risk_reason": risk_reason,
                **report_summary,
            }
            continue

        # Only the LAST row's score is ever used below -- batch-predicting
        # across the entire (now much larger) history was pure waste, and
        # part of the same slow-tick problem this function's docstring
        # describes.
        tail = frame.tail(1)
        gated_last = float(predictor.predict_score_batch(tail, require_confidence=True).iloc[-1])
        ungated_last = float(predictor.predict_score_batch(tail, require_confidence=False).iloc[-1])
        decidable = gated_last != 0.0 or ungated_last == 0.0

        raw_return = predictor.predict_raw_return_last(frame)
        frame_price = float(frame["close"].iloc[-1])
        target_price = frame_price * (1 + raw_return) if raw_return is not None else None

        # 50/50 model / macro-factor blend, per explicit request. The ACTION
        # still follows the model's own score: the factor half informs and is
        # shown alongside, but silently letting macro flip a call would change
        # what the model's measured accuracy actually refers to.
        micro = compute_micro_factors(frame, now)
        blend = blend_score(
            gated_last, macro_factors, model_weight=GOLD_MODEL_WEIGHT,
            micro_conviction=conviction_multiplier(micro),
            micro_detail=[f.__dict__ for f in micro],
        )

        action = _action_for(gated_last, decidable)
        levels = _price_levels(frame, cfg["horizon_bars"], action)
        predictions[horizon] = {
            "score": gated_last, "decidable": decidable, "resolution": cfg["resolution"], "horizon_bars": cfg["horizon_bars"],
            "action": action,
            "blend": blend,
            "levels": levels,
            "trade": _trade_plan(levels, action, frame_price, target_price,
                                 blend.get("micro_conviction", 1.0)),
            "target_price": _clean(target_price) if target_price is not None else None,
            "expected_move_pct": _clean(raw_return * 100) if raw_return is not None else None,
            "valid_until": valid_until,
            "risk_level": risk_level, "risk_reason": risk_reason,
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
    }


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

    Loads only `serving_limit` rows per resolution (a few thousand, enough
    for the tail's features to be fully warmed up) -- NOT the much larger
    `limit` training uses -- so a live tick stays fast regardless of how
    much history PAXG has accumulated for actual model training.
    """
    now = dt.datetime.now(dt.timezone.utc)
    resolutions_needed = {cfg["resolution"] for cfg in GOLD_HORIZONS.values()}
    raw_candles: dict[str, pd.DataFrame] = {}
    for resolution in resolutions_needed:
        limit = max(cfg["serving_limit"] for cfg in GOLD_HORIZONS.values() if cfg["resolution"] == resolution)
        candles = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=limit)
        if len(candles) >= 30:
            raw_candles[resolution] = candles

    cross_asset = await crypto_correlation_context()
    macro_factors = await _load_macro_factors()
    snapshot = await asyncio.to_thread(_build_gold_snapshot_sync, raw_candles, now, macro_factors)
    snapshot["cross_asset"] = cross_asset
    return snapshot
