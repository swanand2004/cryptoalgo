from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SizingInputs:
    equity_usd: float
    signal: float  # ensemble signal in [-1, 1]
    price: float
    contract_value: float  # units of underlying per contract
    margin_per_trade_pct: float = 0.10  # fraction of equity committed as margin, every trade, regardless of conviction; always overridden by settings.max_margin_per_trade_pct in practice
    min_leverage: float = 10.0  # always overridden by settings.min_leverage in practice
    max_leverage: float = 50.0  # always overridden by settings.max_leverage in practice


def fixed_margin_contracts(inputs: SizingInputs) -> int:
    """Every trade commits the same fixed fraction of equity as margin
    (`margin_per_trade_pct`, e.g. 10%) -- conviction (|signal|) is expressed
    entirely through the leverage tier `select_leverage` picks (10x for a
    marginal entry, up to 50x at full conviction), not through position size
    at constant leverage. So a full-conviction trade ends up with more
    notional exposure than a marginal one, but both put up the exact same
    dollar amount of margin.

    This replaces the previous ATR-risk-targeted sizing (constant $-at-risk
    at the stop, shrinking in high-vol regimes and growing in calm ones) with
    a simpler, constant-capital-per-trade model -- see
    [[feedback-validate-before-deploy]] discipline note: this was an explicit
    user request, not a backtested-and-compared change, so don't assume it
    beats the old approach empirically, just that it's what's live now.
    """
    if inputs.price <= 0 or inputs.contract_value <= 0 or inputs.signal == 0:
        return 0

    quality = min(1.0, abs(inputs.signal))
    leverage = select_leverage(quality, inputs.min_leverage, inputs.max_leverage)
    margin_usd = inputs.equity_usd * inputs.margin_per_trade_pct
    notional_usd = margin_usd * leverage

    notional_per_contract = inputs.contract_value * inputs.price
    if notional_per_contract <= 0:
        return 0

    contracts = int(notional_usd / notional_per_contract)
    return contracts if inputs.signal >= 0 else -contracts


def apply_signal_hysteresis(
    signal: float,
    current_position_sign: int,
    entry_threshold: float = 0.15,
    exit_threshold: float = 0.05,
) -> float:
    """Two-threshold (Schmitt-trigger) hysteresis on top of the raw ensemble
    signal, applied identically in backtest and live before sizing.

    Without this, a signal oscillating right around a single threshold (e.g.
    0.16 -> 0.14 -> 0.17) causes a full close-then-reopen on pure noise every
    time it dips under the line, even though conviction never actually
    reversed direction. That's exactly the overtrading pattern the isolated
    per-strategy backtests showed (up to ~70% of bars trading). The fix:
    require the *stronger* entry_threshold to open a new position or flip
    direction, but only require dropping below the *weaker* exit_threshold to
    close an existing one -- and never fully flatten on a same-direction dip
    that hasn't crossed the exit line yet.

    entry_threshold must be >= exit_threshold; the gap between them is the
    hysteresis band.
    """
    if current_position_sign == 0:
        return signal if abs(signal) >= entry_threshold else 0.0

    same_direction = (signal >= 0) == (current_position_sign > 0)
    if same_direction:
        # Still leaning the same way (even weakly) -- hold, size tracks |signal|.
        if abs(signal) >= exit_threshold:
            return signal
        # Faded past the exit line on the same side -- flatten, don't reverse.
        return 0.0

    # Signal has flipped sign relative to the current position: only act on a
    # genuine reversal (full entry conviction), otherwise hold current stance
    # rather than flatten on a marginal opposite-direction wobble.
    return signal if abs(signal) >= entry_threshold else 0.0


def select_leverage(quality: float, min_leverage: float = 10.0, max_leverage: float = 50.0) -> float:
    """Real exchanges require picking a discrete leverage tier up front, not
    a continuous notional/equity ratio -- so leverage here is a genuine,
    deliberate choice made once at entry, driven by trade quality (the
    ensemble signal's conviction, |signal| in [0, 1]): a marginal trade that
    barely cleared the entry threshold gets the floor tier, full-conviction
    trades get the ceiling, linearly in between. Fixed for the life of the
    position rather than recomputed on every resize (see
    PaperBroker.open_or_extend) -- exactly like a real "cross 10x" selection
    doesn't silently change because the position grew.
    """
    quality = max(0.0, min(1.0, quality))
    raw = min_leverage + (max_leverage - min_leverage) * quality
    return float(max(min_leverage, min(max_leverage, round(raw))))


def stop_and_target(
    direction: int, entry_price: float, atr: float | None, stop_multiple: float, target_multiple: float
) -> tuple[float | None, float | None]:
    """direction: +1 for long, -1 for short. Returns (stop_loss, take_profit)
    at fixed ATR multiples from entry. Shared by PaperBroker (simulated
    fills) and LiveBroker (real Delta bracket orders) so both size the exact
    same protective levels off the exact same signal-time ATR.
    """
    if not atr or atr <= 0:
        return None, None
    stop = entry_price - direction * stop_multiple * atr
    target = entry_price + direction * target_multiple * atr
    return stop, target


def required_margin(notional: float, funds: float) -> float:
    """Defensive clamp: margin can never exceed the funds it's measured
    against, however it was computed upstream (e.g. notional / leverage).
    Used as a last-resort safety net when reporting margin_usd, not as the
    primary margin formula -- the primary formula is notional / the
    position's genuinely selected leverage (see select_leverage).
    """
    if funds <= 0:
        return notional
    return min(notional, funds)


def should_rebalance(
    current_contracts: int,
    target_contracts: int,
    min_change_fraction: float = 0.2,
    min_shrink_contracts: int = 5,
) -> bool:
    """Guards against a second, subtler source of overtrading: even with
    direction hysteresis, `fixed_margin_contracts` still rescales
    continuously with signal *magnitude* (via the leverage tier it implies)
    -- so while holding a stable-direction
    position, ordinary bar-to-bar noise in that magnitude was generating a
    resize "trade" on almost every bar. A same-direction change is only worth
    the fee/slippage cost if it's a meaningful fraction of the current
    position; opening from flat, closing to flat, or flipping direction always
    executes regardless.

    Below min_shrink_contracts, `round(min_change_fraction * current)`
    degenerates to 0-1 regardless of min_change_fraction (e.g. 20% of 3
    contracts rounds to 1), so the "meaningful fraction" guard stops meaning
    anything and a winding-down position gets nibbled by exactly 1 contract
    every tick until it hits zero -- each one a real fee for no real change
    in stance. Once a position is already this small, further SHRINKS are
    held flat instead (growth and full closes are unaffected) until the
    signal either strengthens enough to extend it meaningfully or fades
    enough to close it outright.
    """
    if current_contracts == target_contracts:
        return False
    if current_contracts == 0 or target_contracts == 0:
        return True
    if (current_contracts > 0) != (target_contracts > 0):
        return True
    is_shrinking = abs(target_contracts) < abs(current_contracts)
    if is_shrinking and abs(current_contracts) <= min_shrink_contracts:
        return False
    change = abs(target_contracts - current_contracts)
    return change >= max(1, round(min_change_fraction * abs(current_contracts)))
