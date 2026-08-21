from __future__ import annotations

from dataclasses import dataclass
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
