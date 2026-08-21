import pytest

from app.risk.position_sizing import (
    SizingInputs,
    fixed_margin_contracts,
    required_margin,
    select_leverage,
)


def test_select_leverage_floors_at_min_for_zero_conviction():
    assert select_leverage(0.0, min_leverage=10.0, max_leverage=50.0) == 10.0


def test_select_leverage_caps_at_max_for_full_conviction():
    assert select_leverage(1.0, min_leverage=10.0, max_leverage=50.0) == 50.0


def test_select_leverage_interpolates_linearly_between_floor_and_ceiling():
    assert select_leverage(0.5, min_leverage=10.0, max_leverage=50.0) == 30.0


def test_select_leverage_clamps_out_of_range_quality():
    assert select_leverage(-1.0, min_leverage=10.0, max_leverage=50.0) == 10.0
    assert select_leverage(2.0, min_leverage=10.0, max_leverage=50.0) == 50.0


def test_select_leverage_uses_default_range_when_unspecified():
    assert select_leverage(1.0) == 50.0
    assert select_leverage(0.0) == 10.0


def test_required_margin_never_exceeds_funds():
    assert required_margin(15_000, 10_000) == 10_000
    assert required_margin(5_000, 10_000) == 5_000
    assert required_margin(1_000, 0) == 1_000


def _margin_used(contracts: int, price: float, contract_value: float, leverage: float) -> float:
    notional = abs(contracts) * contract_value * price
    return notional / leverage


def test_fixed_margin_contracts_uses_exactly_the_configured_fraction_of_equity_at_full_conviction():
    inputs = SizingInputs(
        equity_usd=10_000.0, signal=1.0, price=100.0, contract_value=1.0,
        margin_per_trade_pct=0.10, min_leverage=10.0, max_leverage=50.0,
    )
    contracts = fixed_margin_contracts(inputs)
    assert contracts > 0
    margin = _margin_used(contracts, inputs.price, inputs.contract_value, leverage=50.0)
    assert margin == pytest.approx(1_000.0, rel=0.01)  # 10% of 10,000 equity


def test_fixed_margin_contracts_uses_the_same_margin_fraction_at_lower_conviction_too():
    """Margin is fixed regardless of conviction -- only the leverage tier (and
    therefore notional/contract count) should move with |signal|."""
    weak = fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=0.2, price=100.0, contract_value=1.0,
                     margin_per_trade_pct=0.10, min_leverage=10.0, max_leverage=50.0)
    )
    strong = fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=0.9, price=100.0, contract_value=1.0,
                     margin_per_trade_pct=0.10, min_leverage=10.0, max_leverage=50.0)
    )
    weak_leverage = select_leverage(0.2, 10.0, 50.0)
    strong_leverage = select_leverage(0.9, 10.0, 50.0)
    weak_margin = _margin_used(weak, 100.0, 1.0, weak_leverage)
    strong_margin = _margin_used(strong, 100.0, 1.0, strong_leverage)

    assert weak_margin == pytest.approx(1_000.0, rel=0.01)
    assert strong_margin == pytest.approx(1_000.0, rel=0.01)
    # Higher conviction -> higher leverage tier -> strictly more contracts,
    # even though the margin committed is identical.
    assert strong > weak


def test_fixed_margin_contracts_scales_linearly_with_margin_per_trade_pct():
    base = fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=1.0, price=100.0, contract_value=1.0, margin_per_trade_pct=0.10)
    )
    doubled = fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=1.0, price=100.0, contract_value=1.0, margin_per_trade_pct=0.20)
    )
    assert doubled == pytest.approx(2 * base, rel=0.01)


def test_fixed_margin_contracts_is_negative_for_a_short_signal():
    contracts = fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=-0.8, price=100.0, contract_value=1.0)
    )
    assert contracts < 0


def test_fixed_margin_contracts_is_zero_for_a_flat_signal():
    assert fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=0.0, price=100.0, contract_value=1.0)
    ) == 0


def test_fixed_margin_contracts_is_zero_for_invalid_price_or_contract_value():
    assert fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=1.0, price=0.0, contract_value=1.0)
    ) == 0
    assert fixed_margin_contracts(
        SizingInputs(equity_usd=10_000.0, signal=1.0, price=100.0, contract_value=0.0)
    ) == 0
