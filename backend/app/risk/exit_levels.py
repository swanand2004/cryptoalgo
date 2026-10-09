"""Price levels at which the account-level limits would close a position.

Live positions carry no exchange stop or take-profit (Settings.atr_brackets_enabled
is off by design); they are closed by the supervisors instead -- the daily loss
limit, the daily profit target (when enabled) and the drawdown halt, each a
level on EQUITY. This converts those equity levels into the PRICE of one
position at which it would be reached, so the dashboard can show where each
trade will actually exit. With several positions open each level assumes the
others stay where they are.
"""
from __future__ import annotations


def _price_for_equity(target_equity: float, equity: float, size: int, contract_value: float, price: float) -> float | None:
    units = size * contract_value
    if units == 0 or price <= 0:
        return None
    level = price + (target_equity - equity) / units
    return level if level > 0 else None


def exit_levels(
    *,
    size: int,
    contract_value: float,
    price: float,
    equity: float | None,
    day_start_equity: float | None,
    daily_loss_limit_pct: float,
    daily_profit_target_pct: float,
    high_water_mark: float | None,
    max_drawdown_pct: float,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {"loss_exit_price": None, "profit_exit_price": None, "drawdown_exit_price": None}
    if equity is None or equity <= 0:
        return out
    start = day_start_equity if day_start_equity and day_start_equity > 0 else equity
    if daily_loss_limit_pct > 0:
        out["loss_exit_price"] = _price_for_equity(start * (1 - daily_loss_limit_pct), equity, size, contract_value, price)
    if daily_profit_target_pct > 0:
        out["profit_exit_price"] = _price_for_equity(start * (1 + daily_profit_target_pct), equity, size, contract_value, price)
    if max_drawdown_pct > 0 and high_water_mark and high_water_mark > 0:
        out["drawdown_exit_price"] = _price_for_equity(
            high_water_mark * (1 - max_drawdown_pct), equity, size, contract_value, price
        )
    return out


async def position_exit_levels(runtime, position, equity: float | None) -> dict[str, float | None]:
    """exit_levels() for one open position, from the runtime's live state --
    shared by the REST positions route and the dashboard websocket so both
    always show the same numbers."""
    from app.data.app_state import get_state

    try:
        product = runtime.registry.get(position.symbol)
    except Exception:
        return {}
    settings = runtime.settings
    raw_hwm = await get_state("equity_high_water_mark")
    price = runtime.cache.get_last_price(position.symbol) or position.entry_price
    return exit_levels(
        size=position.size, contract_value=product.contract_value, price=price, equity=equity,
        day_start_equity=runtime.order_manager.risk_manager.daily_pnl.starting_equity,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        daily_profit_target_pct=settings.daily_profit_target_pct,
        high_water_mark=float(raw_hwm) if raw_hwm else None,
        max_drawdown_pct=settings.max_drawdown_pct,
    )
