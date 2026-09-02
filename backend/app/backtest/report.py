from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

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
    # -- original fields, unchanged so existing callers keep working ---------
    total_return_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    turnover_usd: float
    fee_drag_usd: float
    final_equity: float

    # -- added 2026-09-02 ---------------------------------------------------
    # Win rate on its own is close to useless: an 85%-win-rate strategy that
    # gives everything back in one loss is a losing strategy. Profit factor
    # and expectancy are what actually say whether the edge pays.
    cagr_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy_usd: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    payoff_ratio: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    # Gross = before fees, so the fee burden is visible rather than implied.
    gross_return_pct: float = 0.0
    fee_drag_pct_of_gross: float = 0.0
    # Long/short split -- a book that only makes money on one side is a
    # directional bet wearing a systematic label.
    long_trades: int = 0
    short_trades: int = 0
    long_pnl_usd: float = 0.0
    short_pnl_usd: float = 0.0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    # Risk / survival
    calmar: float = 0.0
    time_in_market_pct: float = 0.0
    worst_trade_usd: float = 0.0
    best_trade_usd: float = 0.0
    max_adverse_equity_excursion_pct: float = 0.0
    days_covered: float = 0.0
    monthly_returns_pct: dict = field(default_factory=dict)
    regime_performance: dict = field(default_factory=dict)
    # Funding is NOT simulated by the engine -- see the note in build_report.
    funding_pnl_usd: float | None = None


def _closing_trades(result: BacktestResult) -> list:
    """Only trades that actually realised P&L close a position; opening fills
    realise nothing and would otherwise dilute every per-trade statistic."""
    return [t for t in result.trades if t.realized_pnl != 0]


def _max_consecutive(flags: list[bool]) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def _monthly_returns(equity: pd.Series, times: pd.Series | None) -> dict:
    if times is None or len(times) != len(equity):
        return {}
    idx = pd.to_datetime(times.to_numpy(), unit="s", utc=True)
    s = pd.Series(equity.to_numpy(), index=idx)
    monthly = s.resample("ME").last().dropna()
    if monthly.empty:
        return {}
    first = float(s.iloc[0])
    prev = first
    out: dict[str, float] = {}
    for ts, value in monthly.items():
        if prev:
            out[ts.strftime("%Y-%m")] = round((float(value) / prev - 1) * 100, 3)
        prev = float(value)
    return out


def build_report(
    result: BacktestResult,
    resolution: str = "1h",
    regimes: pd.Series | None = None,
) -> BacktestReport:
    """Full performance breakdown.

    `regimes`, when supplied (see app/strategy/regime.py), is a per-bar label
    used to split returns by market regime -- the check for whether a strategy
    only works in one environment and is quietly dead in the others.

    Deliberately NOT reported as if it were simulated: the engine models fees
    and slippage but does NOT model funding payments on held perpetual
    positions. `funding_pnl_usd` stays None rather than 0.0 so a reader can
    tell "not modelled" from "modelled and came out flat". For a strategy
    holding positions across funding windows this is a real, unmodelled cost
    or income -- currently roughly +11%/yr on Delta for a short, -11% for a
    long (measured 2026-09-02 against 133k live funding ticks).
    """
    equity = result.equity_curve
    returns = equity.pct_change().dropna()
    bars_per_year = _BARS_PER_YEAR.get(resolution, 365 * 24)

    start_equity = float(equity.iloc[0]) if len(equity) else 0.0
    final_equity = float(equity.iloc[-1]) if len(equity) else 0.0
    total_return_pct = (final_equity / start_equity - 1) * 100 if start_equity else 0.0

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

    closing = _closing_trades(result)
    wins = [t for t in closing if t.realized_pnl > 0]
    losses = [t for t in closing if t.realized_pnl < 0]
    win_rate = (len(wins) / len(closing) * 100) if closing else 0.0

    turnover_usd = sum(t.size * t.price for t in result.trades)
    fee_drag_usd = sum(t.fee for t in result.trades)

    gross_profit = sum(t.realized_pnl for t in wins)
    gross_loss = abs(sum(t.realized_pnl for t in losses))
    # inf is the honest value for "made money and never lost any" -- 0.0 would
    # read as the worst possible score for the best possible outcome.
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    expectancy = (sum(t.realized_pnl for t in closing) / len(closing)) if closing else 0.0
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (-gross_loss / len(losses)) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss else (np.inf if avg_win > 0 else 0.0)

    # A closing SELL unwinds a long; a closing BUY unwinds a short.
    long_closes = [t for t in closing if t.side == "sell"]
    short_closes = [t for t in closing if t.side == "buy"]
    long_pnl = sum(t.realized_pnl for t in long_closes)
    short_pnl = sum(t.realized_pnl for t in short_closes)

    days = 0.0
    if result.times is not None and len(result.times) > 1:
        days = (float(result.times.iloc[-1]) - float(result.times.iloc[0])) / 86400.0
    elif len(equity) > 1:
        days = len(equity) / (bars_per_year / 365.0)

    if days > 0 and start_equity > 0 and final_equity > 0:
        cagr_pct = ((final_equity / start_equity) ** (365.0 / days) - 1) * 100
    else:
        cagr_pct = 0.0

    calmar = (cagr_pct / abs(max_drawdown_pct)) if max_drawdown_pct else 0.0
    time_in_market = float((result.position != 0).mean()) * 100 if len(result.position) else 0.0

    gross_return_pct = (
        ((final_equity + fee_drag_usd) / start_equity - 1) * 100 if start_equity else 0.0
    )
    fee_drag_pct_of_gross = (
        abs(fee_drag_usd / (gross_return_pct / 100 * start_equity)) * 100
        if start_equity and gross_return_pct
        else 0.0
    )

    regime_performance: dict = {}
    if regimes is not None and len(regimes) == len(equity):
        r = returns.reindex(equity.index).fillna(0.0)
        for label in pd.Series(regimes).dropna().unique():
            mask = (pd.Series(regimes).reset_index(drop=True) == label).to_numpy()
            block = r.to_numpy()[mask[: len(r)]] if len(r) else np.array([])
            if len(block) == 0:
                continue
            regime_performance[str(label)] = {
                "bars": int(len(block)),
                "total_return_pct": round(float((np.prod(1 + block) - 1) * 100), 3),
                "mean_bar_return_bps": round(float(np.mean(block) * 10_000), 3),
                "sharpe": round(
                    float(np.mean(block) / np.std(block) * np.sqrt(bars_per_year)) if np.std(block) > 0 else 0.0, 3
                ),
            }

    return BacktestReport(
        total_return_pct=round(total_return_pct, 3),
        sharpe=round(float(sharpe), 3),
        sortino=round(float(sortino), 3),
        max_drawdown_pct=round(float(max_drawdown_pct), 3),
        win_rate=round(win_rate, 2),
        num_trades=len(result.trades),
        turnover_usd=round(turnover_usd, 2),
        fee_drag_usd=round(fee_drag_usd, 2),
        final_equity=round(final_equity, 2),
        cagr_pct=round(float(cagr_pct), 3),
        profit_factor=round(float(profit_factor), 3) if np.isfinite(profit_factor) else float("inf"),
        expectancy_usd=round(float(expectancy), 4),
        avg_win_usd=round(float(avg_win), 2),
        avg_loss_usd=round(float(avg_loss), 2),
        payoff_ratio=round(float(payoff), 3) if np.isfinite(payoff) else float("inf"),
        max_consecutive_losses=_max_consecutive([t.realized_pnl < 0 for t in closing]),
        max_consecutive_wins=_max_consecutive([t.realized_pnl > 0 for t in closing]),
        gross_return_pct=round(float(gross_return_pct), 3),
        fee_drag_pct_of_gross=round(float(fee_drag_pct_of_gross), 2),
        long_trades=len(long_closes),
        short_trades=len(short_closes),
        long_pnl_usd=round(float(long_pnl), 2),
        short_pnl_usd=round(float(short_pnl), 2),
        long_win_rate=round(
            len([t for t in long_closes if t.realized_pnl > 0]) / len(long_closes) * 100, 2
        ) if long_closes else 0.0,
        short_win_rate=round(
            len([t for t in short_closes if t.realized_pnl > 0]) / len(short_closes) * 100, 2
        ) if short_closes else 0.0,
        calmar=round(float(calmar), 3),
        time_in_market_pct=round(time_in_market, 2),
        worst_trade_usd=round(min((t.realized_pnl for t in closing), default=0.0), 2),
        best_trade_usd=round(max((t.realized_pnl for t in closing), default=0.0), 2),
        max_adverse_equity_excursion_pct=round(float(drawdown.min() * 100) if not drawdown.empty else 0.0, 3),
        days_covered=round(days, 2),
        monthly_returns_pct=_monthly_returns(equity, result.times),
        regime_performance=regime_performance,
        funding_pnl_usd=None,  # not modelled -- see docstring
    )
