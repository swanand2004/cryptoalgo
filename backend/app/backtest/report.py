from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestResult

_BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


@dataclass
class BacktestReport:
    total_return_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    turnover_usd: float
    fee_drag_usd: float
    final_equity: float


def build_report(result: BacktestResult, resolution: str = "1h") -> BacktestReport:
    equity = result.equity_curve
    returns = equity.pct_change().dropna()
    bars_per_year = _BARS_PER_YEAR.get(resolution, 365 * 24)

    total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if equity.iloc[0] else 0.0

    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    if downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(bars_per_year)
    else:
        sortino = 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)
    max_drawdown_pct = drawdown.min() * 100 if not drawdown.empty else 0.0

    wins = [t for t in result.trades if t.realized_pnl > 0]
    closing_trades = [t for t in result.trades if t.realized_pnl != 0]
    win_rate = (len(wins) / len(closing_trades) * 100) if closing_trades else 0.0

    turnover_usd = sum(t.size * t.price for t in result.trades)
    fee_drag_usd = sum(t.fee for t in result.trades)

    return BacktestReport(
        total_return_pct=round(total_return_pct, 3),
        sharpe=round(float(sharpe), 3),
        sortino=round(float(sortino), 3),
        max_drawdown_pct=round(float(max_drawdown_pct), 3),
        win_rate=round(win_rate, 2),
        num_trades=len(result.trades),
        turnover_usd=round(turnover_usd, 2),
        fee_drag_usd=round(fee_drag_usd, 2),
        final_equity=round(float(equity.iloc[-1]), 2),
    )
