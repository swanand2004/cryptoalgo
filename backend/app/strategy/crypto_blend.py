from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# Crypto's blended directional call: part model, part factors.
#
# The 40/60 split is a design decision and is implemented as asked. WHICH
# factors fill the 60% is not a design decision, and this module only lets a
# factor vote if it survived measurement.
#
# MEASURED on the deep Binance history (79,170 hourly BTC/ETH bars back to
# 2017, 53,116 SOL bars back to 2020), 4 hours ahead, three-way chronological
# split -- signs fitted on TRAIN, factors selected on VALIDATION, TEST read
# once (backend/scripts/crypto_factor_study.py and crypto_factor_focus.py).
#
# EIGHTEEN factors were tested. TWO survived on all three symbols:
#
#     factor                 BTC              ETH              SOL
#                       train/val/test   train/val/test   train/val/test
#     signed_flow       51.8/52.2/52.2   51.7/52.5/52.2   52.1/51.1/50.9
#     close_location    52.5/52.2/51.6   51.7/51.3/51.6   51.2/51.6/50.9
#
# Against the long-only baseline on TEST -- the bar that matters, because
# crypto drifts up too:
#
#     the two together   BTC +1.43   ETH +1.58   SOL +1.57   all outside noise
#
# Three independent symbols, three independent splits, consistent sign and
# magnitude. That is the strongest evidence found anywhere in this project.
#
# WHAT DID NOT SURVIVE, and is therefore computed, displayed, and given ZERO
# weight -- concealing them would make the blend look better informed than it
# is, and re-testing them later should not require rebuilding them:
#
#     dollar (DXY)            49.7 - 50.3 on test
#     real rates              50.1 - 50.8
#     breakeven inflation     49.5 - 50.4
#     fear & greed            49.7 - 50.2
#     funding crowding        50.0 - 51.1
#     astrology               49.1 - 50.5
#     event proximity         regime, not direction (same finding as gold)
#     10 of 12 micro columns  inside noise of 50
#
# An equal-weight composite of the ten factors VALIDATION happened to nominate
# scored 51.32/51.36/50.33 -- WORSE than the two core factors alone at
# 52.11/52.18/51.58. Identical to gold, where averaging one real signal with
# three noisy ones scored 44.6% against a best component of 54.4%. Dilution is
# the default outcome of "use everything", not a rare failure.
#
# NEWS cannot be measured yet. The NEWS: series holds 74 hourly bars -- four
# days. It is displayed with zero weight and will stay that way until there is
# enough history to test it, which is a data problem and not a modelling one.
#
# THE MODEL HALF IS THE WEAK HALF. Out-of-fold, it scores BELOW a coin flip on
# every symbol: BTC 49.01%, ETH 47.98%, SOL 48.81%. Giving it 40% costs about
# 0.15 points against the factors alone (inside noise), and the cost is
# reported rather than hidden. The weight is configurable precisely so this is
# a decision someone can revisit against the measurement.

MODEL_WEIGHT = 0.40

# The only two factors that earned a vote. Signs frozen from TRAIN.
VOTING_FACTORS: dict[str, float] = {
    "signed_flow": -1.0,
    "close_location": -1.0,
}

Z_WINDOW = 720          # ~30 days of hourly bars
Z_CAP = 2.5
DEADBAND = 0.02         # below this the blend has no opinion worth acting on

BUCKETS = ("micro", "macro", "sentiment", "funding", "events", "astrology", "news")


@dataclass
class CryptoFactor:
    name: str
    bucket: str
    value: float | None
    z_score: float | None
    score: float          # [-1, 1], already signed
    direction: str        # bullish | bearish | neutral
    contributes: bool     # did it survive measurement?
    detail: str


def _clean(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


def _z_last(series: pd.Series, window: int = Z_WINDOW) -> float | None:
    """Trailing z of the last observation. Never full-sample -- a full-sample
    z tells every historical bar what the future distribution looks like."""
    s = pd.Series(series).dropna()
    if len(s) < max(30, window // 5):
        return None
    tail = s.tail(window)
    sd = tail.std()
    if not sd or not np.isfinite(sd) or sd == 0:
        return None
    return float((s.iloc[-1] - tail.mean()) / sd)


def _score_from_z(z: float | None, sign: float) -> float:
    if z is None:
        return 0.0
    return float(np.clip(z, -Z_CAP, Z_CAP) / Z_CAP) * sign


def _direction(score: float) -> str:
    return "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"


def _factor(name, bucket, series, sign, detail, contributes) -> CryptoFactor:
    if series is None or len(pd.Series(series).dropna()) == 0:
        return CryptoFactor(name, bucket, None, None, 0.0, "neutral", contributes,
                            f"{detail} (unavailable)")
    s = pd.Series(series).dropna()
    z = _z_last(s)
    score = _score_from_z(z, sign) if contributes else 0.0
    shown = _score_from_z(z, sign)
    return CryptoFactor(
        name=name,
        bucket=bucket,
        value=_clean(s.iloc[-1]),
        z_score=_clean(z),
        score=round(score, 4),
        direction=_direction(shown),
        contributes=contributes,
        detail=detail,
    )


def compute_crypto_factors(
    frame: pd.DataFrame,
    macro: dict[str, pd.Series] | None = None,
    news_sentiment: pd.Series | None = None,
) -> list[CryptoFactor]:
    """Every factor, voting or not.

    `frame` is a built feature frame (needs the micro_* columns plus
    macro_event_proximity / astro_all_pairs_aspect_score). `macro` optionally
    carries dxy / real_rate / breakeven / fear_greed / funding series already
    aligned to the frame. Anything missing degrades to a neutral, non-voting
    reading rather than raising -- a blend that dies because one external feed
    is down is worse than one that says so.
    """
    macro = macro or {}
    out: list[CryptoFactor] = []
    col = lambda c: frame[c] if (frame is not None and c in frame.columns) else None

    # ---- micro: the two that earned a vote ------------------------------
    out.append(_factor(
        "signed_flow", "micro", col("micro_signed_flow"), VOTING_FACTORS["signed_flow"],
        "Tick-rule signed volume over 24 bars. Contrarian: sustained one-way "
        "aggression marks a crowded push, not a continuing one. 52.2/52.2/50.9% on test.",
        contributes=True))
    out.append(_factor(
        "close_location", "micro", col("micro_close_location"), VOTING_FACTORS["close_location"],
        "Where bars close within their own range. Contrarian: repeated closes "
        "pinned at the high are exhaustion more often than continuation. "
        "51.6/51.6/50.9% on test.",
        contributes=True))

    # ---- micro: measured, not voting ------------------------------------
    for name, detail in (
        ("cs_spread", "Corwin-Schultz high-low spread estimator"),
        ("roll_spread", "Roll serial-covariance spread estimator"),
        ("amihud", "Amihud illiquidity: price move per unit volume"),
        ("kyle_lambda", "Kyle-lambda directional price impact"),
        ("volume_shock", "Volume vs its own 96-bar median"),
        ("wick_asymmetry", "Which side of the bar got rejected"),
        ("body_ratio", "Body as a share of range -- conviction of the bar"),
        ("ret_autocorr", "Lag-1 return autocorrelation: bounce vs momentum"),
        ("vol_of_vol", "Volatility of volatility"),
        ("jump_ratio", "Realized variance vs bipower variation -- jump activity"),
    ):
        out.append(_factor(name, "micro", col(f"micro_{name}"), 1.0,
                           f"{detail}. Inside noise of 50% on test.", contributes=False))

    # ---- macro ----------------------------------------------------------
    out.append(_factor("dollar", "macro", macro.get("dxy"), -1.0,
                       "Trade-weighted dollar. A stronger dollar should price risk "
                       "assets lower. 49.7-50.3% on test -- it does not, measurably.",
                       contributes=False))
    out.append(_factor("real_rates", "macro", macro.get("real_rate"), -1.0,
                       "10y TIPS real yield -- the discount rate on long-duration risk. "
                       "50.1-50.8% on test.", contributes=False))
    out.append(_factor("breakeven_inflation", "macro", macro.get("breakeven"), 1.0,
                       "10y breakeven inflation, the 'digital gold' thesis. "
                       "49.5-50.4% on test.", contributes=False))

    # ---- sentiment ------------------------------------------------------
    out.append(_factor("fear_greed", "sentiment", macro.get("fear_greed"), -1.0,
                       "Crypto Fear & Greed index, read contrarian. 49.7-50.2% on test "
                       "-- the famous contrarian signal does not survive measurement here.",
                       contributes=False))

    # ---- funding --------------------------------------------------------
    out.append(_factor("funding_crowding", "funding", macro.get("funding"), -1.0,
                       "Perp funding, read contrarian: longs paying to hold is a crowded "
                       "trade. 50.0-51.1% on test.", contributes=False))

    # ---- events ---------------------------------------------------------
    out.append(_factor("event_proximity", "events", col("macro_event_proximity"), 1.0,
                       "Proximity to FOMC / CPI / NFP. A liquidity and volatility regime, "
                       "not a direction -- the same finding as gold.", contributes=False))

    # ---- astrology ------------------------------------------------------
    out.append(_factor("astrology", "astrology", col("astro_all_pairs_aspect_score"), 1.0,
                       "Planetary aspect score. 49.1-50.5% on test, with a sign that had to "
                       "be fitted rather than predicted.", contributes=False))

    # ---- news -----------------------------------------------------------
    out.append(_factor("news_sentiment", "news", news_sentiment, 1.0,
                       "VADER sentiment over crypto headlines. NOT measurable yet: the "
                       "series holds four days of history, which cannot validate anything.",
                       contributes=False))
    return out


def factor_score(factors: list[CryptoFactor]) -> float:
    """Mean of the factors that earned a vote.

    Mean of the SURVIVORS, never of everything present. Averaging the two real
    signals with sixteen noisy ones measured 51.3% against 52.1% for the two
    alone -- dilution is what "use everything" actually buys.
    """
    votes = [f.score for f in factors if f.contributes]
    return float(np.mean(votes)) if votes else 0.0


def blend_score(
    model_score: float | None,
    factors: list[CryptoFactor],
    model_weight: float = MODEL_WEIGHT,
) -> dict:
    model = float(model_score) if model_score is not None and np.isfinite(model_score) else 0.0
    fac = factor_score(factors)
    fw = 1.0 - model_weight
    blended = float(np.clip(model_weight * model + fw * fac, -1.0, 1.0))

    contributing = [f for f in factors if f.contributes]
    return {
        "model_score": round(model, 4),
        "factor_score": round(fac, 4),
        "blended_score": round(blended, 4),
        "model_weight": model_weight,
        "factor_weight": round(fw, 4),
        "action": "BUY" if blended > DEADBAND else "SELL" if blended < -DEADBAND else "HOLD",
        "deadband": DEADBAND,
        # Disagreement between the halves is the single most useful thing to
        # show a human, so it must not be averaged away silently.
        "sides_agree": bool(model == 0.0 or fac == 0.0 or np.sign(model) == np.sign(fac)),
        "factors_total": len(factors),
        "factors_contributing": len(contributing),
        "buckets": sorted({f.bucket for f in factors}),
        "factors": [asdict(f) for f in factors],
    }


def snapshot(
    symbol: str,
    frame: pd.DataFrame,
    model_score: float | None,
    macro: dict[str, pd.Series] | None = None,
    news_sentiment: pd.Series | None = None,
    model_weight: float = MODEL_WEIGHT,
    now: dt.datetime | None = None,
) -> dict:
    factors = compute_crypto_factors(frame, macro, news_sentiment)
    blend = blend_score(model_score, factors, model_weight)
    price = None
    if frame is not None and not frame.empty and "close" in frame.columns:
        price = _clean(frame["close"].iloc[-1])
    return {
        "symbol": symbol,
        "as_of": (now or dt.datetime.now(dt.timezone.utc)).isoformat(),
        "price": price,
        "horizon_hours": 4,
        **blend,
    }
