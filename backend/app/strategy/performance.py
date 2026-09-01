from __future__ import annotations

from dataclasses import asdict, dataclass

from app.backtest.engine import BacktestConfig, Trade, run_backtest
from app.backtest.report import BacktestReport, build_report
from app.core.symbols import SymbolRegistry
from app.data.query import load_candles_df, load_pseudo_series_df
from app.features.feature_pipeline import build_feature_frame
from app.strategy.ensemble import SignalEnsemble, WeightedStrategy
from app.strategy.graha import GrahaStrategy
from app.strategy.time_series_momentum import TimeSeriesMomentumStrategy
from app.strategy.trend_momentum import TrendMomentumStrategy

# Kept in sync with registry.py's DEFAULT_WEIGHTS -- see its comment for why
# mean_reversion/funding_carry/orderbook_imbalance/ml_signal/rahu were fully
# removed rather than just left disabled.
STRATEGY_NAMES = [
    "trend_momentum",
    "graha",
    "tsmom",
]


def _make_strategy(name: str, symbol: str, resolution: str):
    return {
        "trend_momentum": TrendMomentumStrategy(),
        "graha": GrahaStrategy(symbol=symbol),
        "tsmom": TimeSeriesMomentumStrategy(),
    }[name]


@dataclass
class SymbolPerformance:
    symbol: str
    report: BacktestReport


@dataclass
class StrategyPerformance:
    name: str
    note: str | None
    win_rate: float
    total_trades: int
    avg_return_pct: float
    avg_sharpe: float
    by_symbol: list[SymbolPerformance]


async def evaluate_all_strategies(
    registry: SymbolRegistry,
    resolution: str = "15m",
    limit: int = 1500,
    max_leverage: float = 15.0,
    min_leverage: float = 10.0,
    margin_per_trade_pct: float = 0.10,
    atr_stop_multiple: float = 2.0,
    atr_take_profit_multiple: float = 3.0,
    atr_breakeven_trigger_multiple: float = 0.5,
    atr_tightened_trail_multiple: float = 0.4,
) -> list[StrategyPerformance]:
    """Backtests each strategy in isolation (weight=1, no ensemble blending)
    across every configured symbol, using whatever historical data is already
    ingested. This is the honest, on-demand version of the research done
    manually earlier -- same engine, same code path as live trading.
    """
    symbol_data: dict[str, tuple] = {}
    for product in registry.all():
        candles = await load_candles_df(product.symbol, resolution, limit=limit)
        if len(candles) < 100:
            continue
        funding = await load_pseudo_series_df(f"FUNDING:{product.symbol}", "1h")
        mark = await load_pseudo_series_df(f"MARK:{product.symbol}", resolution)
        spot = (
            await load_pseudo_series_df(product.spot_index_symbol, resolution)
            if product.spot_index_symbol
            else None
        )
        fear_greed = await load_pseudo_series_df("FNG:GLOBAL", "1d")
        binance_funding = await load_pseudo_series_df(f"BFUND:{product.symbol}", "8h")
        binance_oi = await load_pseudo_series_df(f"BOI:{product.symbol}", "1h")
        binance_lsr = await load_pseudo_series_df(f"BLSR:{product.symbol}", "1h")
        dvol = await load_pseudo_series_df(f"DVOL:{product.symbol}", "1h")
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
        )
        symbol_data[product.symbol] = (features, product.contract_value)

    results: list[StrategyPerformance] = []
    for name in STRATEGY_NAMES:
        by_symbol: list[SymbolPerformance] = []
        pooled_trades: list[Trade] = []
        returns: list[float] = []
        sharpes: list[float] = []

        for symbol, (features, contract_value) in symbol_data.items():
            strategy = _make_strategy(name, symbol, resolution)
            ensemble = SignalEnsemble([WeightedStrategy(strategy, weight=1.0)], deadband=0.15)
            cfg = BacktestConfig(
                contract_value=contract_value,
                max_leverage=max_leverage,
                min_leverage=min_leverage,
                margin_per_trade_pct=margin_per_trade_pct,
                atr_stop_multiple=atr_stop_multiple,
                atr_take_profit_multiple=atr_take_profit_multiple,
                atr_breakeven_trigger_multiple=atr_breakeven_trigger_multiple,
                atr_tightened_trail_multiple=atr_tightened_trail_multiple,
            )
            result = run_backtest(features, ensemble, cfg)
            report = build_report(result, resolution=resolution)
            by_symbol.append(SymbolPerformance(symbol=symbol, report=report))
            pooled_trades.extend(result.trades)
            if report.num_trades > 0:
                returns.append(report.total_return_pct)
                sharpes.append(report.sharpe)

        wins = [t for t in pooled_trades if t.realized_pnl > 0]
        closing_trades = [t for t in pooled_trades if t.realized_pnl != 0]
        win_rate = (len(wins) / len(closing_trades) * 100) if closing_trades else 0.0

        note = None
        if name == "graha":
            note = "EXPERIMENTAL: astrology-conditioned signal with no established link to market behavior. Judge this backtest skeptically."

        results.append(
            StrategyPerformance(
                name=name,
                note=note,
                win_rate=round(win_rate, 2),
                total_trades=len(closing_trades),
                # Average, not sum -- each symbol is an independent $10k
                # simulation, so adding their % returns together wouldn't mean
                # anything ("combined portfolio return" would need shared capital).
                avg_return_pct=round(sum(returns) / len(returns), 3) if returns else 0.0,
                avg_sharpe=round(sum(sharpes) / len(sharpes), 3) if sharpes else 0.0,
                by_symbol=by_symbol,
            )
        )

    return results


def performance_to_dict(perf: StrategyPerformance) -> dict:
    return {
        "name": perf.name,
        "note": perf.note,
        "win_rate": perf.win_rate,
        "loss_rate": round(100 - perf.win_rate, 2) if perf.total_trades > 0 else 0.0,
        "total_trades": perf.total_trades,
        "avg_return_pct": perf.avg_return_pct,
        "avg_sharpe": perf.avg_sharpe,
        "by_symbol": [
            {"symbol": s.symbol, **asdict(s.report)} for s in perf.by_symbol
        ],
    }
