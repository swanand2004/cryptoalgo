import pytest

from app.config import Settings
from app.core.delta_client import DeltaApiError
from app.core.symbols import ProductInfo, SymbolRegistry
from app.data.market_cache import MarketDataCache
from app.execution.broker_base import OrderRequest
from app.execution.live_broker import LiveBroker


def _registry() -> SymbolRegistry:
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    reg = SymbolRegistry(settings)
    reg._by_symbol["TESTUSD"] = ProductInfo(
        product_id=7, symbol="TESTUSD", contract_type="perpetual_futures", tick_size=0.5,
        contract_value=0.01, underlying_asset="TEST", state="live", default_leverage=10,
        taker_commission_rate=0.0005, maker_commission_rate=0.0002, spot_index_symbol="",
    )
    return reg


class _FakeDeltaClient:
    """Simulates just enough of Delta's real behaviour for LiveBroker's
    bracket-order wiring: an order fill moves a single tracked position, and
    get_positions_margined always reflects its current state -- exactly the
    authoritative-position read LiveBroker relies on instead of the locally
    polled (up to 15s stale) copy.
    """

    def __init__(self, fill_price: float = 50_000.0) -> None:
        self.size = 0
        self.entry_price = 0.0
        self.fill_price = fill_price
        self.bracket_calls: list[dict] = []
        self.cancel_all_calls: list[dict] = []
        self.bracket_should_fail = False

    async def place_order(self, body: dict) -> dict:
        signed = body["size"] if body["side"] == "buy" else -body["size"]
        opening_or_extending = self.size == 0 or (self.size > 0) == (signed > 0)
        if opening_or_extending:
            self.entry_price = self.fill_price
        self.size += signed
        if self.size == 0:
            self.entry_price = 0.0
        return {"id": 999, "state": "closed", "unfilled_size": 0}

    async def get_positions_margined(self, product_ids: str | None = None) -> list[dict]:
        if self.size == 0:
            return []
        return [{"size": self.size, "entry_price": self.entry_price}]

    async def place_bracket_order(self, body: dict) -> dict:
        if self.bracket_should_fail:
            raise DeltaApiError(400, {"error": "boom"})
        self.bracket_calls.append(body)
        return {}

    async def cancel_all_orders(self, body: dict | None = None) -> dict:
        self.cancel_all_calls.append(body)
        return {}


def _broker(client: _FakeDeltaClient) -> LiveBroker:
    settings = Settings(delta_api_key="k", delta_api_secret="s")
    return LiveBroker(client, _registry(), MarketDataCache(), risk_manager=None, settings=settings)


@pytest.mark.asyncio
async def test_new_long_entry_from_flat_attaches_a_bracket_order():
    client = _FakeDeltaClient(fill_price=50_000.0)
    broker = _broker(client)

    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=100, order_type="market", atr=100.0, quality=1.0)
    )

    assert len(client.bracket_calls) == 1
    body = client.bracket_calls[0]
    assert body["product_id"] == 7
    assert body["product_symbol"] == "TESTUSD"
    # Settings defaults: atr_stop_multiple=2.0, atr_take_profit_multiple=3.0.
    # Long: stop = entry - 2*atr = 49_800, target = entry + 3*atr = 50_300.
    assert body["stop_loss_order"]["stop_price"] == "49800"
    assert body["stop_loss_order"]["trail_amount"] == "200"
    assert body["take_profit_order"]["stop_price"] == "50300"
    assert body["bracket_stop_trigger_method"] == "mark_price"


@pytest.mark.asyncio
async def test_new_short_entry_computes_stop_above_and_target_below_entry():
    client = _FakeDeltaClient(fill_price=50_000.0)
    broker = _broker(client)

    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="sell", size=100, order_type="market", atr=100.0, quality=1.0)
    )

    assert len(client.bracket_calls) == 1
    body = client.bracket_calls[0]
    # Short: stop = entry + 2*atr = 50_200, target = entry - 3*atr = 49_700.
    assert body["stop_loss_order"]["stop_price"] == "50200"
    assert body["take_profit_order"]["stop_price"] == "49700"


@pytest.mark.asyncio
async def test_extending_an_existing_position_does_not_attach_a_second_bracket():
    """Delta allows only ONE bracket order per open position -- so a resize
    (not a genuine flat->open transition) must never try to attach another.
    """
    client = _FakeDeltaClient(fill_price=50_000.0)
    broker = _broker(client)

    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=100, order_type="market", atr=100.0, quality=1.0)
    )
    assert len(client.bracket_calls) == 1

    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=50, order_type="market", atr=100.0, quality=1.0)
    )
    assert len(client.bracket_calls) == 1  # unchanged


@pytest.mark.asyncio
async def test_reduce_only_order_that_fully_flattens_cleans_up_resting_stop_orders():
    client = _FakeDeltaClient(fill_price=50_000.0)
    broker = _broker(client)

    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=100, order_type="market", atr=100.0, quality=1.0)
    )
    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="sell", size=100, order_type="market", reduce_only=True)
    )

    assert len(client.cancel_all_calls) == 1
    assert client.cancel_all_calls[0] == {"product_id": 7, "cancel_stop_orders": True}


@pytest.mark.asyncio
async def test_partial_reduce_that_stays_open_does_not_touch_the_bracket():
    client = _FakeDeltaClient(fill_price=50_000.0)
    broker = _broker(client)

    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=100, order_type="market", atr=100.0, quality=1.0)
    )
    await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="sell", size=40, order_type="market", reduce_only=True)
    )

    assert client.cancel_all_calls == []


@pytest.mark.asyncio
async def test_bracket_attach_failure_does_not_fail_the_entry_order():
    """The entry itself already succeeded on the exchange -- a bracket-attach
    failure must be loud (logged) but must not make place_order look like the
    entry itself failed.
    """
    client = _FakeDeltaClient(fill_price=50_000.0)
    client.bracket_should_fail = True
    broker = _broker(client)

    result = await broker.place_order(
        OrderRequest(symbol="TESTUSD", side="buy", size=100, order_type="market", atr=100.0, quality=1.0)
    )

    assert result.state == "closed"
    assert result.filled_size == 100


@pytest.mark.asyncio
async def test_no_atr_means_no_bracket_attach_attempt():
    """A signal with no ATR (e.g. a manual order with no live strategy
    context) can't size a stop -- must not attempt a bracket call at all."""
    client = _FakeDeltaClient(fill_price=50_000.0)
    broker = _broker(client)

    await broker.place_order(OrderRequest(symbol="TESTUSD", side="buy", size=100, order_type="market"))

    assert client.bracket_calls == []
