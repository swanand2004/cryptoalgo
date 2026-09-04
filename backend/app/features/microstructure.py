from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import Optional

from app.data.market_cache import MarketDataCache, OrderbookState


@dataclass
class MicrostructureFeatures:
    symbol: str
    spread: Optional[float]
    spread_bps: Optional[float]
    book_imbalance: Optional[float]  # [-1, 1], + means bid-heavy
    trade_flow_imbalance: Optional[float]  # [-1, 1] over recent trade window, + means buy-heavy
    mid: Optional[float]


def orderbook_features(ob: Optional[OrderbookState]) -> dict:
    if ob is None or ob.best_bid is None or ob.best_ask is None:
        return {"spread": None, "spread_bps": None, "book_imbalance": None, "mid": None}
    spread = ob.best_ask - ob.best_bid
    mid = ob.mid
    spread_bps = (spread / mid) * 10_000 if mid else None
    return {
        "spread": spread,
        "spread_bps": spread_bps,
        "book_imbalance": ob.imbalance,
        "mid": mid,
    }


def trade_flow_imbalance(cache: MarketDataCache, symbol: str, window: int = 200) -> Optional[float]:
    """Fraction of recent taker volume that was buyer-initiated minus seller-initiated,
    over the last `window` trades for this symbol. "buyer_role" == "t" means the
    trade's aggressor was a taker; Delta's trades feed reports the *maker* role, so
    role == "m" implies the taker (aggressor) was on the other side.
    We approximate aggressor side from the `r` (buyer_role) field: role "t" == the
    quoted side was the taker; since Delta trades don't include an explicit side,
    we treat consecutive price upticks as buy pressure via price direction instead.
    """
    recent = [t for t in cache.trade_buffer if t.get("symbol") == symbol]
    if len(recent) < 2:
        return None
    recent = recent[-window:]
    buy_vol = 0.0
    sell_vol = 0.0
    for prev, cur in zip(recent, recent[1:]):
        if cur["price"] > prev["price"]:
            buy_vol += cur["size"]
        elif cur["price"] < prev["price"]:
            sell_vol += cur["size"]
    total = buy_vol + sell_vol
    if total == 0:
        return None
    return (buy_vol - sell_vol) / total


def compute_microstructure(cache: MarketDataCache, symbol: str) -> MicrostructureFeatures:
    ob = cache.get_orderbook(symbol)
    ob_feats = orderbook_features(ob)
    tfi = trade_flow_imbalance(cache, symbol)
    return MicrostructureFeatures(
        symbol=symbol,
        spread=ob_feats["spread"],
        spread_bps=ob_feats["spread_bps"],
        book_imbalance=ob_feats["book_imbalance"],
        trade_flow_imbalance=tfi,
        mid=ob_feats["mid"],
    )


# ---------------------------------------------------------------------------
# OHLCV microstructure
# ---------------------------------------------------------------------------
# Everything above this line needs a live order book and a trade tape. This
# block needs neither -- it is the family of estimators built specifically for
# markets observed as BARS, which is how history arrives for backtesting and
# training, and how thin venues arrive even live.
#
# Shared by crypto (feature_pipeline) and gold (gold/features) rather than
# implemented twice: they are the same estimators over the same inputs, and two
# copies would drift apart the first time one was fixed.

OHLCV_MICRO_COLUMNS: list[str] = [
    "micro_cs_spread",
    "micro_roll_spread",
    "micro_amihud",
    "micro_kyle_lambda",
    "micro_signed_flow",
    "micro_volume_shock",
    "micro_close_location",
    "micro_wick_asymmetry",
    "micro_body_ratio",
    "micro_ret_autocorr",
    "micro_vol_of_vol",
    "micro_jump_ratio",
]


def ohlcv_microstructure(out: "pd.DataFrame") -> "pd.DataFrame":
    """Adds OHLCV_MICRO_COLUMNS in place and returns the frame.

    Each is the published estimator for the quantity it names, not a loose
    proxy renamed to sound like one:

      cs_spread      Corwin & Schultz (2012). Two-bar high-low ratios separate
                     spread from volatility, because volatility scales with
                     the interval and the spread does not.
      roll_spread    Roll (1984). Bid-ask bounce makes consecutive price
                     changes negatively autocovariant; the spread falls out.
      amihud         Amihud (2002). Price move per unit of volume.
      kyle_lambda    Move regressed on SIGNED volume -- directional impact,
                     where Amihud measures undirected cost.
      signed_flow    Tick rule. With no trade tape the bar's own direction is
                     the standard stand-in for aggressor side. A PROXY for
                     order flow, and named so it cannot be mistaken for the
                     real thing.

    VOLUME IS OPTIONAL. The four volume-dependent columns read 0.0 -- no
    observable flow, no measurable impact -- when volume is absent or all
    zero. That is deliberate: a column of NaN does not merely lose itself,
    it removes every row from any training matrix that requires all features
    present, so one missing optional input can silently empty a 127,000-row
    dataset.
    """
    o, h, l, c = (out[k].astype(float) for k in ("open", "high", "low", "close"))
    ret = np.log(c.replace(0, np.nan)).diff()
    rng = (h - l).replace(0, np.nan)

    v = out["volume"].astype(float) if "volume" in out.columns else pd.Series(0.0, index=out.index)
    volume_usable = bool(v.replace(0, np.nan).notna().any())

    beta = (np.log((h / l).replace(0, np.nan)) ** 2).rolling(2).sum()
    gamma = np.log((h.rolling(2).max() / l.rolling(2).min()).replace(0, np.nan)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    out["micro_cs_spread"] = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).rolling(24).mean()

    dp = c.diff()
    out["micro_roll_spread"] = 2 * np.sqrt((-dp.rolling(48).cov(dp.shift(1))).clip(lower=0))

    if volume_usable:
        out["micro_amihud"] = (ret.abs() / v.replace(0, np.nan)).rolling(24).mean()
        signed_vol = np.sign(ret) * v
        out["micro_kyle_lambda"] = (
            ret.rolling(48).cov(signed_vol) / signed_vol.rolling(48).var().replace(0, np.nan)
        ).fillna(0.0)
        out["micro_signed_flow"] = (
            signed_vol.rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
        ).fillna(0.0)
        out["micro_volume_shock"] = np.log(
            (v / v.rolling(96).median().replace(0, np.nan)).replace(0, np.nan)
        ).fillna(0.0)
    else:
        for name in ("micro_amihud", "micro_kyle_lambda", "micro_signed_flow", "micro_volume_shock"):
            out[name] = 0.0

    out["micro_close_location"] = ((c - l) - (h - c)) / rng
    out["micro_wick_asymmetry"] = ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / rng
    out["micro_body_ratio"] = (c - o).abs() / rng

    out["micro_ret_autocorr"] = ret.rolling(96).corr(ret.shift(1))
    out["micro_vol_of_vol"] = ret.rolling(24).std().rolling(96).std()

    rv = (ret**2).rolling(24).sum()
    bpv = (ret.abs() * ret.abs().shift(1)).rolling(24).sum() * (np.pi / 2)
    out["micro_jump_ratio"] = (rv - bpv) / rv.replace(0, np.nan)

    for name in OHLCV_MICRO_COLUMNS:
        out[name] = out[name].replace([np.inf, -np.inf], np.nan)
    return out
