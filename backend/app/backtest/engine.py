from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.risk.position_sizing import (
    SizingInputs,
    apply_signal_hysteresis,
    fixed_margin_contracts,
    should_rebalance,
)
from app.strategy.ensemble import SignalEnsemble


@dataclass
class BacktestConfig:
    starting_equity: float = 10_000.0
    contract_value: float = 0.001
    taker_fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    margin_per_trade_pct: float = 0.10
    atr_stop_multiple: float = 2.0
    atr_take_profit_multiple: float = 3.0  # must mirror Settings/PaperBroker, or the backtest
    atr_breakeven_trigger_multiple: float = 1.0  # simulates a different protective
    atr_tightened_trail_multiple: float = 0.8  # mechanism than what live actually runs
    min_leverage: float = 10.0
    max_leverage: float = 15.0


@dataclass
class Trade:
    time: int
    side: str
    size: int
    price: float
    fee: float
    realized_pnl: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    signal: pd.Series
    position: pd.Series
    trades: list[Trade] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    ensemble: SignalEnsemble,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Event-driven single-symbol backtest that reuses the exact SignalEnsemble
    (and therefore the exact per-strategy code) used live. Signal computed on
    bar i is executed at the OPEN of bar i+1 -- never on the same bar's close --
    so the result can't benefit from lookahead.

    Also simulates the same ATR-based stop-loss/take-profit (with the
    two-stage breakeven-lock trailing stop) that PaperBroker actually runs
    live. Without this, a backtest position only ever exits on a signal flip
    and can ride a loss all the way down with no protection -- a materially
    different (and much riskier) risk profile than what live trading ever
    allows, which silently made every isolated-strategy backtest look far
    worse than the strategy would actually perform live.
    """
    cfg = config or BacktestConfig()
    df = df.reset_index(drop=True)
    signal = ensemble.smoothed_signal(df)
    warmup = ensemble.warmup_bars()

    n = len(df)
    equity = np.full(n, cfg.starting_equity, dtype=float)
    position_contracts = np.zeros(n, dtype=int)
    entry_price = 0.0
    balance = cfg.starting_equity
    trades: list[Trade] = []

    # Trailing-stop state for whatever position is currently open -- mirrors
    # Position.{stop_loss_price,take_profit_price,peak_price,stop_distance}
    # in app/data/models.py / app/execution/paper_broker.py.
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    peak_price: float | None = None
    stop_distance: float | None = None

    def open_fresh_stop_target(direction: int, price: float, atr: float) -> None:
        nonlocal stop_loss_price, take_profit_price, peak_price, stop_distance
        if atr > 0:
            stop_loss_price = price - direction * cfg.atr_stop_multiple * atr
            take_profit_price = price + direction * cfg.atr_take_profit_multiple * atr
            peak_price = price
            stop_distance = cfg.atr_stop_multiple * atr
        else:
            stop_loss_price = take_profit_price = peak_price = stop_distance = None

    def clear_stop_target() -> None:
        nonlocal stop_loss_price, take_profit_price, peak_price, stop_distance
        stop_loss_price = take_profit_price = peak_price = stop_distance = None

    for i in range(1, n):
        # Carry forward yesterday's position/equity baseline before this bar's action.
        position_contracts[i] = position_contracts[i - 1]

        if i - 1 < warmup:
            equity[i] = balance
            continue

        prev_size = int(position_contracts[i - 1])
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])

        # Stop-loss/take-profit monitored against THIS bar's high/low, for
        # whatever position was carried in, BEFORE this bar's signal-driven
        # rebalance -- mirrors PaperBroker.check_stop_loss_take_profit, which
        # polls continuously between signal ticks rather than only reacting
        # to a new signal value.
        if prev_size != 0 and stop_distance is not None and peak_price is not None:
            is_long = prev_size > 0
            favorable_extreme = bar_high if is_long else bar_low
            new_peak = max(peak_price, favorable_extreme) if is_long else min(peak_price, favorable_extreme)
            if new_peak != peak_price:
                peak_price = new_peak
                atr_at_entry = stop_distance / cfg.atr_stop_multiple
                favorable_move = (new_peak - entry_price) if is_long else (entry_price - new_peak)
                triggered = favorable_move >= cfg.atr_breakeven_trigger_multiple * atr_at_entry

                trail_distance = (
                    cfg.atr_tightened_trail_multiple * atr_at_entry if triggered else stop_distance
                )
                candidate_stop = new_peak - trail_distance if is_long else new_peak + trail_distance
                if triggered:
                    candidate_stop = (
                        max(candidate_stop, entry_price) if is_long else min(candidate_stop, entry_price)
                    )

                if stop_loss_price is None or (
                    (is_long and candidate_stop > stop_loss_price)
                    or (not is_long and candidate_stop < stop_loss_price)
                ):
                    stop_loss_price = candidate_stop

            hit_stop = stop_loss_price is not None and (
                (is_long and bar_low <= stop_loss_price) or (not is_long and bar_high >= stop_loss_price)
            )
            hit_target = take_profit_price is not None and (
                (is_long and bar_high >= take_profit_price) or (not is_long and bar_low <= take_profit_price)
            )
            if hit_stop or hit_target:
                # Conservative: if both could plausibly have been touched
                # within the same bar, assume the adverse one (stop) came first.
                exit_price = stop_loss_price if hit_stop else take_profit_price
                direction = 1 if is_long else -1
                fee = abs(prev_size) * cfg.contract_value * exit_price * cfg.taker_fee_rate
                realized = abs(prev_size) * cfg.contract_value * (exit_price - entry_price) * direction
                balance += realized - fee
                trades.append(
                    Trade(int(df["time"].iloc[i]), "sell" if is_long else "buy", abs(prev_size), exit_price, fee, realized)
                )
                position_contracts[i] = 0
                prev_size = 0
                entry_price = 0.0
                clear_stop_target()

        raw_signal = float(signal.iloc[i - 1])
        prev_sign = 1 if prev_size > 0 else (-1 if prev_size < 0 else 0)
        target_signal = apply_signal_hysteresis(raw_signal, prev_sign, ensemble.deadband, ensemble.exit_deadband)

        price = float(df["open"].iloc[i])
        atr = float(df["atr_14"].iloc[i - 1]) if not pd.isna(df["atr_14"].iloc[i - 1]) else 0.0

        current_equity = balance + prev_size * cfg.contract_value * (price - entry_price)

        target_contracts = fixed_margin_contracts(
            SizingInputs(
                equity_usd=current_equity,
                signal=target_signal,
                price=price,
                contract_value=cfg.contract_value,
                margin_per_trade_pct=cfg.margin_per_trade_pct,
                min_leverage=cfg.min_leverage,
                max_leverage=cfg.max_leverage,
            )
        )

        delta = target_contracts - prev_size
        if should_rebalance(prev_size, target_contracts):
            side = "buy" if delta > 0 else "sell"
            fill_price = price * (1 + cfg.slippage_bps / 10_000 * (1 if delta > 0 else -1))
            fee = abs(delta) * cfg.contract_value * fill_price * cfg.taker_fee_rate

            realized = 0.0
            opened_from_flat = prev_size == 0
            if prev_size == 0 or (prev_size > 0) == (delta > 0):
                new_size = prev_size + delta
                if new_size != 0:
                    entry_price = (
                        entry_price * abs(prev_size) + fill_price * abs(delta)
                    ) / abs(new_size)
                position_contracts[i] = new_size
                if opened_from_flat:
                    # Stop/target set ONCE on genuine entry from flat, exactly
                    # like PaperBroker.open_or_extend -- NOT recomputed on
                    # later same-direction extensions.
                    open_fresh_stop_target(1 if new_size > 0 else -1, entry_price, atr)
            else:
                closing_qty = min(abs(delta), abs(prev_size))
                direction = 1 if prev_size > 0 else -1
                realized = closing_qty * cfg.contract_value * (fill_price - entry_price) * direction
                new_size = prev_size + delta
                position_contracts[i] = new_size
                if new_size == 0:
                    entry_price = 0.0
                    clear_stop_target()
                elif (new_size > 0) != (direction > 0):
                    # Flipped straight through flat within one fill -- the
                    # leftover is a brand new entry, with its own fresh stop/target.
                    entry_price = fill_price
                    open_fresh_stop_target(1 if new_size > 0 else -1, fill_price, atr)
                # else: partial close, same direction remains -- entry/SL/TP/peak unchanged.

            balance += realized - fee
            trades.append(Trade(int(df["time"].iloc[i]), side, abs(delta), fill_price, fee, realized))

        close_price = float(df["close"].iloc[i])
        equity[i] = balance + position_contracts[i] * cfg.contract_value * (close_price - entry_price)

    return BacktestResult(
        equity_curve=pd.Series(equity, index=df.index),
        signal=signal,
        position=pd.Series(position_contracts, index=df.index),
        trades=trades,
    )
