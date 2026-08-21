from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.core.delta_client import DeltaRestClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductInfo:
    product_id: int
    symbol: str
    contract_type: str
    tick_size: float
    contract_value: float
    underlying_asset: str
    state: str
    default_leverage: float
    taker_commission_rate: float
    maker_commission_rate: float
    spot_index_symbol: str


class SymbolRegistry:
    """Resolves and caches product metadata for the configured trading universe.

    Product IDs are never hardcoded -- they're looked up from /v2/products at
    startup (and can be refreshed) so the app keeps working if Delta re-lists
    or re-numbers a contract.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._by_symbol: dict[str, ProductInfo] = {}

    async def refresh(self, client: DeltaRestClient) -> None:
        wanted = set(self.settings.symbol_list)
        products = await client.get_products(contract_types="perpetual_futures", states="live")
        found: dict[str, ProductInfo] = {}
        for p in products:
            symbol = p.get("symbol")
            if symbol not in wanted:
                continue
            found[symbol] = ProductInfo(
                product_id=p["id"],
                symbol=symbol,
                contract_type=p.get("contract_type", "perpetual_futures"),
                tick_size=float(p.get("tick_size", "0.5")),
                contract_value=float(p.get("contract_value", "0.001")),
                underlying_asset=(p.get("underlying_asset") or {}).get("symbol", ""),
                state=p.get("state", "live"),
                default_leverage=float(p.get("default_leverage", "1")),
                taker_commission_rate=float(p.get("taker_commission_rate", "0.0005")),
                maker_commission_rate=float(p.get("maker_commission_rate", "0.0002")),
                spot_index_symbol=(p.get("spot_index") or {}).get("symbol", ""),
            )
        missing = wanted - found.keys()
        if missing:
            logger.warning("Could not resolve live perpetual products for: %s", missing)
        self._by_symbol = found
        logger.info("Resolved %d products: %s", len(found), list(found.keys()))

    def get(self, symbol: str) -> ProductInfo:
        try:
            return self._by_symbol[symbol]
        except KeyError as exc:
            raise KeyError(f"Symbol {symbol} not resolved yet -- call refresh() first") from exc

    def all(self) -> list[ProductInfo]:
        return list(self._by_symbol.values())

    @property
    def symbols(self) -> list[str]:
        return list(self._by_symbol.keys())


_registry: SymbolRegistry | None = None


def get_symbol_registry() -> SymbolRegistry:
    global _registry
    if _registry is None:
        _registry = SymbolRegistry()
    return _registry
