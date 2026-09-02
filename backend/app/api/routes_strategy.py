from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import get_runtime
from app.backtest.engine import BacktestConfig, run_backtest
from app.backtest.report import build_report
from app.core.time_utils import to_iso_utc
from app.data.db import SessionLocal
from app.data.models import StrategyConfig
from app.data.query import load_candles_df, load_pseudo_series_df
from app.features.feature_pipeline import build_feature_frame
from app.ml.train import MODEL_STORE_DIR
from app.runtime import Runtime
from app.strategy.performance import evaluate_all_strategies, performance_to_dict
from app.strategy.registry import build_ensemble_for_symbol

router = APIRouter(prefix="/api", tags=["strategy"])


class StrategyUpdate(BaseModel):
    enabled: bool | None = None
    weight: float | None = None


class BacktestRequest(BaseModel):
    symbol: str
    resolution: str = "1h"
    limit: int = 1000


@router.get("/strategies")
async def list_strategies():
    async with SessionLocal() as session:
        result = await session.execute(select(StrategyConfig))
        return [
            {"name": s.name, "enabled": s.enabled, "weight": s.weight, "updated_at": to_iso_utc(s.updated_at)}
            for s in result.scalars().all()
        ]


@router.get("/ml/{symbol}/report")
async def get_ml_report(symbol: str, resolution: str = "15m"):
    """The ML model's OWN honest accuracy: purged walk-forward, out-of-sample
    directional accuracy / information coefficient, computed fresh at the
    most recent daily retrain -- NOT the isolated-strategy backtest number
    (which unavoidably overlaps the data the current model was trained on,
    and should not be trusted as this model's real skill).
    """
    report_path = MODEL_STORE_DIR / f"{symbol}_{resolution}_latest_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"no training report yet for {symbol}/{resolution}")
    return json.loads(report_path.read_text())


@router.get("/strategies/performance")
async def get_strategy_performance(
    resolution: str = "15m", limit: int = 1500, runtime: Runtime = Depends(get_runtime)
):
    """Backtests each strategy in isolation (no ensemble blending) across every
    configured symbol using already-ingested history, and reports win/loss
    rate, trade count, and return per strategy -- the honest per-strategy
    breakdown, since live trades are only ever tagged "ensemble".
    """
    results = await evaluate_all_strategies(
        runtime.registry,
        resolution=resolution,
        limit=limit,
        max_leverage=runtime.settings.max_leverage,
        min_leverage=runtime.settings.min_leverage,
        margin_per_trade_pct=runtime.settings.max_margin_per_trade_pct,
        atr_stop_multiple=runtime.settings.atr_stop_multiple,
        atr_take_profit_multiple=runtime.settings.atr_take_profit_multiple,
        atr_breakeven_trigger_multiple=runtime.settings.atr_breakeven_trigger_multiple,
        atr_tightened_trail_multiple=runtime.settings.atr_tightened_trail_multiple,
    )
    return [performance_to_dict(r) for r in results]


@router.put("/strategies/{name}")
async def update_strategy(name: str, body: StrategyUpdate, runtime: Runtime = Depends(get_runtime)):
    async with SessionLocal() as session:
        result = await session.execute(select(StrategyConfig).where(StrategyConfig.name == name))
        cfg = result.scalar_one_or_none()
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"unknown strategy '{name}'")
        if body.enabled is not None:
            cfg.enabled = body.enabled
        if body.weight is not None:
            cfg.weight = body.weight
        await session.commit()
    await runtime.refresh_ensembles()
    return {"name": name, "enabled": cfg.enabled, "weight": cfg.weight}


@router.post("/backtest")
async def run_backtest_endpoint(body: BacktestRequest, runtime: Runtime = Depends(get_runtime)):
    if body.symbol not in runtime.registry.symbols:
        raise HTTPException(status_code=404, detail=f"unknown symbol '{body.symbol}'")

    candles = await load_candles_df(body.symbol, body.resolution, limit=body.limit)
    if len(candles) < 100:
        raise HTTPException(status_code=400, detail="not enough historical candles ingested yet for this symbol/resolution")

    funding = await load_pseudo_series_df(f"FUNDING:{body.symbol}", "1h")
    mark = await load_pseudo_series_df(f"MARK:{body.symbol}", body.resolution)
    product = runtime.registry.get(body.symbol)
    spot = await load_pseudo_series_df(product.spot_index_symbol, body.resolution) if product.spot_index_symbol else None
    fear_greed = await load_pseudo_series_df("FNG:GLOBAL", "1d")
    binance_funding = await load_pseudo_series_df(f"BFUND:{body.symbol}", "8h")
    binance_oi = await load_pseudo_series_df(f"BOI:{body.symbol}", "1h")
    binance_lsr = await load_pseudo_series_df(f"BLSR:{body.symbol}", "1h")
    dvol = await load_pseudo_series_df(f"DVOL:{body.symbol}", "1h")
    news = await load_pseudo_series_df(f"NEWS:{body.symbol}", "1h")

    features = build_feature_frame(
        candles,
        funding=funding,
        mark=mark,
        spot=spot,
        fear_greed=fear_greed,
        binance_funding=binance_funding,
        binance_oi=binance_oi,
        binance_lsr=binance_lsr,
        dvol=dvol,
        news=news,
    )
    ensemble = await build_ensemble_for_symbol(body.symbol)

    result = run_backtest(
        features,
        ensemble,
        BacktestConfig(
            contract_value=product.contract_value,
            max_leverage=runtime.settings.max_leverage,
            min_leverage=runtime.settings.min_leverage,
            margin_per_trade_pct=runtime.settings.max_margin_per_trade_pct,
            atr_stop_multiple=runtime.settings.atr_stop_multiple,
            atr_take_profit_multiple=runtime.settings.atr_take_profit_multiple,
            atr_breakeven_trigger_multiple=runtime.settings.atr_breakeven_trigger_multiple,
            atr_tightened_trail_multiple=runtime.settings.atr_tightened_trail_multiple,
        ),
    )
    report = build_report(result, resolution=body.resolution)

    step = max(1, len(result.equity_curve) // 500)
    equity_points = [
        {"time": int(features["time"].iloc[i]), "equity": float(result.equity_curve.iloc[i])}
        for i in range(0, len(result.equity_curve), step)
    ]

    return {"report": asdict(report), "equity_curve": equity_points, "num_bars": len(features)}
