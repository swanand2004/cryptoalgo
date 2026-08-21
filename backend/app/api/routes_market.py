from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_runtime
from app.data.query import load_candles_df
from app.runtime import Runtime

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/symbols")
async def list_symbols(runtime: Runtime = Depends(get_runtime)):
    return [
        {
            "symbol": p.symbol,
            "product_id": p.product_id,
            "underlying_asset": p.underlying_asset,
            "tick_size": p.tick_size,
            "contract_value": p.contract_value,
            "default_leverage": p.default_leverage,
            "spot_index_symbol": p.spot_index_symbol,
        }
        for p in runtime.registry.all()
    ]


@router.get("/{symbol}/ticker")
async def get_ticker(symbol: str, runtime: Runtime = Depends(get_runtime)):
    ticker = runtime.cache.ticker.get(symbol)
    if ticker is None:
        raise HTTPException(status_code=404, detail="no ticker data yet for this symbol")
    return {
        "symbol": symbol,
        "raw": ticker,
        "mark_price": runtime.cache.mark_price.get(f"MARK:{symbol}"),
        "spot_price": runtime.cache.spot_price.get(runtime.registry.get(symbol).spot_index_symbol),
        "funding_rate": runtime.cache.funding_rate.get(symbol),
        "last_price": runtime.cache.get_last_price(symbol),
    }


@router.get("/{symbol}/orderbook")
async def get_orderbook(symbol: str, runtime: Runtime = Depends(get_runtime)):
    ob = runtime.cache.get_orderbook(symbol)
    if ob is None:
        raise HTTPException(status_code=404, detail="no orderbook data yet for this symbol")
    return {
        "symbol": symbol,
        "bids": ob.bids,
        "asks": ob.asks,
        "best_bid": ob.best_bid,
        "best_ask": ob.best_ask,
        "mid": ob.mid,
        "imbalance": ob.imbalance,
        "timestamp": ob.timestamp,
    }


@router.get("/{symbol}/candles")
async def get_candles(symbol: str, resolution: str = "15m", limit: int = 300):
    df = await load_candles_df(symbol, resolution, limit=limit)
    return df.to_dict(orient="records")
