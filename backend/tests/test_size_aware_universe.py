"""Size-aware universe + one book leverage (2026-10-05: on $14.49 the first lot,
BNB, locked $9.91 of margin and ETH was squeezed to ~20x)."""
import types

import pandas as pd
import pytest

from app.config import Settings
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.app_state import set_trading_mode
from app.execution.broker_base import PositionView
from app.trading_loop import _active_universe, _close_inactive, _min_lot_target, book_margin_leverage

CORE = "BTCUSD,ETHUSD,SOLUSD,BNBUSD"
FILLERS = "HBARUSD,MANAUSD,XLMUSD,ADAUSD,TRXUSD,STXUSD"
CONTRACT = {"BTCUSD": (0.001, 85_000), "ETHUSD": (0.01, 2_700), "SOLUSD": (1, 122), "BNBUSD": (0.1, 790),
            "HBARUSD": (1, 0.10), "MANAUSD": (1, 0.11), "XLMUSD": (1, 0.22), "ADAUSD": (1, 0.25),
            "TRXUSD": (1, 0.34), "STXUSD": (1, 0.39)}


def _settings(**kw):
    base = dict(delta_api_key="k", delta_api_secret="s", symbols=f"{CORE},{FILLERS}", core_symbols=CORE)
    base.update(kw)
    return Settings(**base)


def _runtime(equity, settings=None, positions=None, closed=None):
    settings = settings or _settings()
    reg = SymbolRegistry(settings)
    for i, s in enumerate(settings.symbol_list):
        cv, _ = CONTRACT[s]
        reg._by_symbol[s] = ProductInfo(
            product_id=i, symbol=s, contract_type="perpetual_futures", tick_size=0.01, contract_value=cv,
            underlying_asset=s[:-3], state="live", default_leverage=10, taker_commission_rate=0.0005,
            maker_commission_rate=0.0002, spot_index_symbol="",
        )

    async def get_equity():
        if equity is None:
            raise RuntimeError("balance unavailable")
        return equity

    async def get_positions():
        return positions or {}

    async def close_position(symbol, strategy):
        closed.append((symbol, strategy))

    om = types.SimpleNamespace(get_equity=get_equity, get_positions=get_positions, close_position=close_position)
    cache = types.SimpleNamespace(get_last_price=lambda s: CONTRACT[s][1])
    return types.SimpleNamespace(settings=settings, registry=reg, order_manager=om, cache=cache)


@pytest.mark.asyncio
async def test_feature_off_trades_every_symbol():
    rt = _runtime(14.49, _settings(core_symbols=""))
    assert await _active_universe(rt, {}) == set(rt.settings.symbol_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("equity, expected", [
    (14.49, set(FILLERS.split(","))),                         # nothing core fits: fillers only
    (60.0, {"ETHUSD"} | set(FILLERS.split(","))),              # $27 ETH <= $30: joins the fillers
    (240.0, {"ETHUSD", "BNBUSD", "BTCUSD"} | set(FILLERS.split(","))),  # SOL $122 > $120: fillers stay
    (300.0, set(CORE.split(","))),                             # every core symbol fits: core only
])
async def test_universe_follows_the_balance(equity, expected):
    await set_trading_mode("live")
    assert await _active_universe(_runtime(equity), {}) == expected


@pytest.mark.asyncio
async def test_universe_is_decided_once_a_day_and_not_when_equity_is_unknown():
    await set_trading_mode("live")
    assert await _active_universe(_runtime(None), {}) is None
    rt = _runtime(14.49)
    first = await _active_universe(rt, {})
    rt.order_manager.get_equity = lambda: _value(5_000.0)  # balance jumps intraday: no churn until tomorrow
    assert await _active_universe(rt, {}) == first


async def _value(v):
    return v


def test_one_book_leverage_fits_the_whole_book_in_the_margin_share():
    s = _settings()
    signals = {"A": pd.Series([0.5]), "B": pd.Series([-0.4]), "C": pd.Series([0.0])}
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    # gross = (0.5*0.5 + 0.4*0.3) * k=3.7 = 1.369x equity -> /0.85 -> 1.61 -> 2x for every position
    assert book_margin_leverage(s, signals, weights, 3.7) == 2.0
    assert book_margin_leverage(s, signals, weights, 0.5) == 1.0  # never below 1x


@pytest.mark.asyncio
async def test_every_lot_gets_the_same_cap_leverage_so_later_lots_still_fit():
    await set_trading_mode("live")
    rt = types.SimpleNamespace(settings=_settings(live_min_lot_max_leverage=25.0))
    product = types.SimpleNamespace(contract_value=1.0)
    # $14 account, cap 25x = $350: BNB $79 then SOL $122 then BTC $85 then ETH $27 all fit at ceil(25/0.85)=30x
    held, positions = 0.0, {}
    for sym, notional in (("BNBUSD", 79.0), ("SOLUSD", 122.0), ("BTCUSD", 85.0), ("ETHUSD", 27.0)):
        target, leverage, min_lot = await _min_lot_target(
            rt, product, notional, positions, 14.0, symbol=sym, target_contracts=0, latest_signal=0.4,
        )
        assert (target, leverage, min_lot) == (1, 30.0, True), sym
        positions[f"{sym}:consolidated"] = PositionView(sym, 1, notional, 0, 0, strategy="consolidated",
                                                        notional_usd=notional, margin_usd=notional / leverage)
        held += notional / leverage
    assert held <= 0.85 * 14.0  # $10.43 of margin on $14


@pytest.mark.asyncio
async def test_a_symbol_outside_the_universe_is_closed():
    closed = []
    pos = {"BNBUSD:consolidated": PositionView("BNBUSD", 1, 790, 0, 0, strategy="consolidated")}
    rt = _runtime(14.49, positions=pos, closed=closed)
    await _close_inactive(rt, "BNBUSD")
    await _close_inactive(rt, "ETHUSD")  # nothing open: nothing to do
    assert closed == [("BNBUSD", "consolidated")]
