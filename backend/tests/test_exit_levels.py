import pytest

from app.risk.exit_levels import exit_levels


def _levels(**kw):
    base = dict(size=1, contract_value=0.01, price=2700.0, equity=3.04, day_start_equity=2.956,
                daily_loss_limit_pct=0.20, daily_profit_target_pct=0.10, high_water_mark=3.156,
                max_drawdown_pct=0.60)
    base.update(kw)
    return exit_levels(**base)


def test_long_levels_match_the_live_eth_contract():
    lv = _levels()
    # equity must fall 3.04 -> 2.3648 (-0.6752) = $67.52 of ETH on 0.01 ETH
    assert lv["loss_exit_price"] == pytest.approx(2700 - 67.52, abs=0.01)
    # equity must rise 3.04 -> 3.2516 (+0.2116) = $21.16 of ETH
    assert lv["profit_exit_price"] == pytest.approx(2700 + 21.16, abs=0.01)
    assert lv["drawdown_exit_price"] < lv["loss_exit_price"]


def test_short_levels_are_mirrored():
    lv = _levels(size=-1)
    assert lv["loss_exit_price"] == pytest.approx(2700 + 67.52, abs=0.01)
    assert lv["profit_exit_price"] == pytest.approx(2700 - 21.16, abs=0.01)


def test_disabled_profit_target_and_unknown_equity_give_no_levels():
    assert _levels(daily_profit_target_pct=0.0)["profit_exit_price"] is None
    assert _levels(equity=None) == {"loss_exit_price": None, "profit_exit_price": None, "drawdown_exit_price": None}


def test_a_level_below_zero_is_reported_as_none():
    # a tiny position cannot lose 20% of the account before its price hits 0
    assert _levels(size=1, contract_value=0.0001, price=10.0)["loss_exit_price"] is None
