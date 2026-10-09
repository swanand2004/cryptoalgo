import types

import pytest

from app.config import Settings
from app.core.delta_client import DeltaApiError
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.app_state import set_state, set_trading_mode
from app.data.market_cache import MarketDataCache
from app.execution.broker_base import OrderRequest, PositionView
from app.execution.live_broker import LIVE_POSITION_STRATEGY, LiveBroker
from app.trading_loop import CONSOLIDATED_STRATEGY, _min_lot_target


def _registry(contract_value=0.01, symbol="ETHUSD", product_id=3136) -> SymbolRegistry:
    reg = SymbolRegistry(Settings(delta_api_key="k", delta_api_secret="s"))
    reg._by_symbol[symbol] = ProductInfo(
        product_id=product_id, symbol=symbol, contract_type="perpetual_futures", tick_size=0.05,
        contract_value=contract_value, underlying_asset="ETH", state="live", default_leverage=10,
        taker_commission_rate=0.0005, maker_commission_rate=0.0002, spot_index_symbol="",
    )
    return reg


class _FakeDelta:
    """Stands in for Delta's REST API -- no network, no real orders."""

    def __init__(self, remote_positions=None, fail_leverage=False):
        self.remote_positions = remote_positions or []
        self.fail_leverage = fail_leverage
        self.calls = []

    async def get_positions_margined(self, product_ids=None):
        return list(self.remote_positions)

    async def request(self, method, path, *, json_body=None, auth=False, params=None):
        self.calls.append((method, path, json_body))
        if path == "/v2/fills":
            return list(getattr(self, "fills", []))
        if self.fail_leverage and path.endswith("/orders/leverage"):
            raise DeltaApiError(400, {"error": "nope"})
        return {"leverage": (json_body or {}).get("leverage")}

    async def get_balances(self):
        if getattr(self, "balance_error", False):
            raise DeltaApiError(401, {"error": {"code": "expired_signature"}})
        return [{"asset_symbol": "USD", "balance": str(getattr(self, "balance", 3.67))}]

    async def place_order(self, body):
        self.calls.append(("ORDER", body))
        self.remote_positions = [{"size": body["size"] if body["side"] == "buy" else -body["size"],
                                  "entry_price": "2750", "margin": "3.06", "liquidation_price": "2480",
                                  "realized_pnl": "0"}]
        return {"id": 99, "state": "closed", "unfilled_size": 0}


def _broker(fake, settings=None):
    settings = settings or Settings(delta_api_key="k", delta_api_secret="s")
    cache = MarketDataCache()
    cache.update_orderbook("ETHUSD", bids=[[2749.5, 10]], asks=[[2750.5, 10]], timestamp=1)
    return LiveBroker(fake, _registry(), cache, None, settings)


def test_live_positions_use_the_same_name_the_trading_loop_looks_up():
    assert LIVE_POSITION_STRATEGY == CONSOLIDATED_STRATEGY


@pytest.mark.asyncio
async def test_sync_records_a_position_the_loop_can_find():
    # Regression: rows had no strategy (NOT NULL -> every sync failed) and were
    # keyed by bare symbol, so the loop always saw itself flat and re-bought.
    fake = _FakeDelta([{"size": 1, "entry_price": "2750", "margin": "3.06", "liquidation_price": "2480",
                        "realized_pnl": "0"}])
    broker = _broker(fake)
    await broker.sync_fills_and_positions()
    positions = await broker.get_positions()
    assert list(positions) == ["ETHUSD:consolidated"]
    pos = positions["ETHUSD:consolidated"]
    assert pos.size == 1 and pos.entry_time is not None
    assert pos.leverage == pytest.approx(27.5 / 3.06, rel=0.01)


@pytest.mark.asyncio
async def test_flat_on_delta_clears_the_local_position():
    fake = _FakeDelta([{"size": 1, "entry_price": "2750", "margin": "3", "liquidation_price": "0", "realized_pnl": "0"}])
    broker = _broker(fake)
    await broker.sync_fills_and_positions()
    fake.remote_positions = []
    await broker.sync_fills_and_positions()
    assert await broker.get_positions() == {}


@pytest.mark.asyncio
async def test_order_sets_leverage_first_then_reflects_the_fill_immediately():
    fake = _FakeDelta()
    broker = _broker(fake)
    result = await broker.place_order(OrderRequest(symbol="ETHUSD", side="buy", size=1, strategy="consolidated",
                                                   leverage=9))
    assert result.state == "closed"
    leverage_calls = [c for c in fake.calls if c[0] == "POST"]
    assert leverage_calls == [("POST", "/v2/products/3136/orders/leverage", {"leverage": "9"})]
    assert fake.calls.index(leverage_calls[0]) < [c[0] for c in fake.calls].index("ORDER")
    # No waiting for the 15 s sync: the very next tick sees the position.
    assert (await broker.get_positions())["ETHUSD:consolidated"].size == 1
    # Same leverage again: not re-sent.
    await broker.place_order(OrderRequest(symbol="ETHUSD", side="buy", size=1, strategy="consolidated", leverage=9))
    assert len([c for c in fake.calls if c[0] == "POST"]) == 1


@pytest.mark.asyncio
async def test_order_is_refused_if_leverage_cannot_be_set():
    fake = _FakeDelta(fail_leverage=True)
    result = await _broker(fake).place_order(OrderRequest(symbol="ETHUSD", side="buy", size=1, leverage=9))
    assert result.state == "rejected"
    assert not [c for c in fake.calls if c[0] == "ORDER"]  # never sent at the wrong leverage


@pytest.mark.asyncio
async def test_reducing_orders_do_not_touch_leverage():
    fake = _FakeDelta([{"size": 1, "entry_price": "2750", "margin": "3", "liquidation_price": "0", "realized_pnl": "0"}])
    await _broker(fake).place_order(OrderRequest(symbol="ETHUSD", side="sell", size=1, reduce_only=True))
    assert not [c for c in fake.calls if c[0] == "POST"]


# --------------------------------------------------------------- small-account lot

def _rt(cap=10.0, max_leverage=50.0):
    return types.SimpleNamespace(settings=Settings(delta_api_key="k", delta_api_secret="s",
                                                   live_min_lot_max_leverage=cap, max_leverage=max_leverage))


def _product(contract_value):
    return types.SimpleNamespace(contract_value=contract_value)


@pytest.mark.asyncio
async def test_min_lot_is_off_by_default_and_in_paper():
    await set_trading_mode("live")
    out = await _min_lot_target(_rt(cap=0.0), _product(0.01), 2750, {}, 3.67,
                                symbol="ETHUSD", target_contracts=0, latest_signal=0.4)
    assert out == (0, None, False)
    await set_trading_mode("paper")
    out = await _min_lot_target(_rt(), _product(0.01), 2750, {}, 3.67,
                                symbol="ETHUSD", target_contracts=0, latest_signal=0.4)
    assert out == (0, None, False)


@pytest.mark.asyncio
async def test_tiny_account_takes_one_eth_contract_at_the_book_lot_leverage():
    await set_trading_mode("live")
    target, leverage, min_lot = await _min_lot_target(
        _rt(), _product(0.01), 2750, {}, 3.67, symbol="ETHUSD", target_contracts=0, latest_signal=0.42)
    assert (target, min_lot) == (1, True)
    # one leverage for every lot: ceil(cap 10 / 0.85) = 12x -- the whole capped book fits the margin
    assert leverage == 12.0


@pytest.mark.asyncio
async def test_coins_whose_one_contract_breaks_the_cap_stay_flat():
    await set_trading_mode("live")
    # BTC: 1 contract = 0.001 * $86,670 = $86.67 = 23.6x of $3.67 > 10x cap.
    out = await _min_lot_target(_rt(), _product(0.001), 86670, {}, 3.67,
                                symbol="BTCUSD", target_contracts=0, latest_signal=0.47)
    assert out == (0, None, False)


@pytest.mark.asyncio
async def test_existing_positions_count_against_the_cap():
    await set_trading_mode("live")
    eth = PositionView("ETHUSD", 1, 2750, 0, 0, strategy="consolidated", notional_usd=27.5, margin_usd=3.06)
    # $8 account: a BNB contract ($78) alone would fit 10x... with ETH's $27.5 already open it would not.
    out = await _min_lot_target(_rt(), _product(0.1), 780, {"ETHUSD:consolidated": eth}, 8.0,
                                symbol="BNBUSD", target_contracts=0, latest_signal=0.4)
    assert out == (0, None, False)


@pytest.mark.asyncio
async def test_short_signal_takes_a_short_lot_and_normal_sizing_is_left_alone():
    await set_trading_mode("live")
    target, _, _ = await _min_lot_target(_rt(), _product(0.01), 2750, {}, 3.67,
                                         symbol="ETHUSD", target_contracts=0, latest_signal=-0.3)
    assert target == -1
    # A big enough account sized normally: never overridden.
    out = await _min_lot_target(_rt(), _product(0.01), 2750, {}, 5000, symbol="ETHUSD",
                                target_contracts=62, latest_signal=0.4)
    assert out == (62, None, False)


@pytest.mark.asyncio
async def test_switching_paper_to_live_does_not_read_as_a_99pct_drawdown():
    from app.risk.risk_manager import RiskManager

    rm = RiskManager(Settings(delta_api_key="k", delta_api_secret="s"))
    await set_state("equity_high_water_mark", "5732.12")  # the paper book's peak
    rm.daily_pnl.starting_equity = 5230.0
    assert (await rm.check_max_drawdown(3.67))[0] is False  # what a raw switch would do
    assert (await rm.check_daily_loss(3.67))[0] is False

    await rm.rebase_for_mode_switch(3.67)
    assert (await rm.check_max_drawdown(3.67))[0] is True
    assert (await rm.check_daily_loss(3.67))[0] is True
    # The limits still bite on the new account: a real 70% fall still halts.
    assert (await rm.check_max_drawdown(1.0))[0] is False


@pytest.mark.asyncio
async def test_min_lot_order_is_not_shrunk_to_zero_by_the_per_trade_margin_cap():
    # On $3.67 the 10%-of-equity per-trade margin cap can never fit a whole
    # contract; the min-lot path has already applied its own account-level cap.
    from app.execution.order_manager import OrderManager
    from app.execution.paper_broker import PaperBroker
    from app.risk.risk_manager import RiskManager

    settings = Settings(delta_api_key="k", delta_api_secret="s")
    reg = _registry()
    cache = MarketDataCache()
    cache.update_orderbook("ETHUSD", bids=[[2749.5, 10]], asks=[[2750.5, 10]], timestamp=1)
    paper = PaperBroker(reg, cache, settings)
    om = OrderManager(paper, LiveBroker(_FakeDelta(), reg, cache), RiskManager(settings), reg, cache)
    await set_trading_mode("paper")
    await set_state("paper_balance", "3.67")

    capped = await om.submit(OrderRequest(symbol="ETHUSD", side="buy", size=1, strategy="consolidated"))
    assert capped.state == "rejected"  # the normal path refuses it
    lot = await om.submit(OrderRequest(symbol="ETHUSD", side="buy", size=1, strategy="consolidated", min_lot=True))
    assert lot.filled_size == 1


# --------------------------------------------------------------- 2026-10-03 fixes

@pytest.mark.asyncio
async def test_expired_signature_is_re_signed_and_retried():
    # Real incident: a slow path made Delta see a 15 s-old signature and reject it.
    import httpx

    from app.core.delta_client import DeltaRestClient

    seen = []

    def handler(request):
        seen.append(request.headers.get("timestamp"))
        if len(seen) == 1:
            return httpx.Response(401, json={"success": False, "error": {"code": "expired_signature"}})
        return httpx.Response(200, json={"success": True, "result": [{"asset_symbol": "USD", "balance": "2.9"}]})

    client = DeltaRestClient(Settings(delta_api_key="k", delta_api_secret="s"))
    client._client = httpx.AsyncClient(base_url="https://x", transport=httpx.MockTransport(handler))
    result = await client.request("GET", "/v2/wallet/balances", auth=True)
    assert result[0]["balance"] == "2.9" and len(seen) == 2 and all(seen)


@pytest.mark.asyncio
async def test_unreadable_balance_is_never_reported_as_zero():
    # Real incident: get_equity returned 0.0 on an API error and the daily-loss
    # breaker read it as a 100% loss, closing a real position on a false alarm.
    from app.execution.live_broker import LiveEquityUnavailable

    fake = _FakeDelta()
    broker = _broker(fake)
    fake.balance_error = True
    with pytest.raises(LiveEquityUnavailable):
        await broker.get_equity()  # nothing known yet: raise, don't guess
    fake.balance_error = False
    assert await broker.get_equity() == pytest.approx(3.67)
    fake.balance_error = True
    assert await broker.get_equity() == pytest.approx(3.67)  # recent good value, not 0


@pytest.mark.asyncio
async def test_daily_loss_check_skips_when_balance_is_unknown():
    from app.data.app_state import is_kill_switch_on, set_kill_switch
    from app.execution.live_broker import LiveEquityUnavailable
    from app.scheduler import _daily_loss_supervisor

    await set_kill_switch(False)

    async def boom():
        raise LiveEquityUnavailable("delta down")

    runtime = types.SimpleNamespace(order_manager=types.SimpleNamespace(get_equity=boom))
    with pytest.raises(LiveEquityUnavailable):
        await _daily_loss_supervisor(runtime)
    assert not await is_kill_switch_on()


@pytest.mark.asyncio
async def test_daily_loss_breach_halts_with_kill_switch_and_stays_live():
    from app.data.app_state import get_trading_mode, is_kill_switch_on, set_kill_switch
    from app.risk.risk_manager import RiskManager
    from app.scheduler import _daily_loss_supervisor

    await set_kill_switch(False)
    await set_trading_mode("live")
    rm = RiskManager(Settings(delta_api_key="k", delta_api_secret="s"))
    await rm.rebase_for_mode_switch(3.67)
    flattened = []

    async def equity():
        return 2.90  # -21%: over the 20% daily limit

    async def flatten_all(mode_override=None, reason="kill_switch"):
        flattened.append((mode_override, reason))

    runtime = types.SimpleNamespace(
        risk_manager=rm, order_manager=types.SimpleNamespace(get_equity=equity, flatten_all=flatten_all))
    await _daily_loss_supervisor(runtime)
    assert flattened == [("live", "daily_loss")]
    assert await is_kill_switch_on()
    assert await get_trading_mode() == "live"  # halted on real money, not flipped to paper


@pytest.mark.asyncio
async def test_trading_tick_does_nothing_while_the_kill_switch_is_on(monkeypatch):
    import app.trading_loop as tl
    from app.data.app_state import set_kill_switch

    called = []

    async def spy(runtime):
        called.append(True)
        return {}, {}

    monkeypatch.setattr(tl, "_build_ladder_frames", spy)
    await set_kill_switch(True)
    await tl.run_trading_tick(types.SimpleNamespace())
    assert called == []


@pytest.mark.asyncio
async def test_closed_live_trade_is_logged_with_delta_exit_price_and_fees():
    import datetime as dt

    from sqlalchemy import select

    from app.data.db import SessionLocal
    from app.data.models import ClosedTrade

    now = dt.datetime.now(dt.timezone.utc)
    fake = _FakeDelta([{"size": 1, "entry_price": "2678.2", "margin": "2.5", "liquidation_price": "2440",
                        "realized_pnl": "0"}])
    broker = _broker(fake)
    await broker.sync_fills_and_positions()  # opens the local row
    fake.fills = [
        {"side": "buy", "size": "1", "price": "2678.2", "commission": "0.01580138",
         "created_at": (now - dt.timedelta(seconds=5)).isoformat().replace("+00:00", "Z")},
        {"side": "sell", "size": "1", "price": "2683.4", "commission": "0.01583206",
         "created_at": now.isoformat().replace("+00:00", "Z")},
    ]
    broker._exit_reason["ETHUSD"] = "daily_loss"
    fake.remote_positions = []
    await broker.sync_fills_and_positions()  # flat on Delta -> closed trade

    async with SessionLocal() as session:
        [t] = (await session.execute(select(ClosedTrade).where(ClosedTrade.mode == "live"))).scalars().all()
    assert (t.side, t.size, t.exit_reason) == ("long", 1, "daily_loss")
    assert t.exit_price == pytest.approx(2683.4)
    assert t.realized_pnl == pytest.approx((2683.4 - 2678.2) * 0.01)
    assert t.fees == pytest.approx(0.01580138 + 0.01583206)
