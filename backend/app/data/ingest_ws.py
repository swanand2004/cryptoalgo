from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.core.delta_ws_public import DeltaPublicWebSocket
from app.core.symbols import SymbolRegistry
from app.data.market_cache import get_market_cache

logger = logging.getLogger(__name__)


def build_channels(registry: SymbolRegistry) -> list[dict]:
    symbols = registry.symbols
    mark_symbols = [f"MARK:{s}" for s in symbols]
    index_symbols = [p.spot_index_symbol for p in registry.all() if p.spot_index_symbol]

    # ob_l2 rejects more than one symbol per channel entry ("subscription
    # forbidden on this channel with more than 1 symbol") -- confirmed against
    # the live testnet socket, contrary to what the general "up to 100 symbols
    # per connection" doc note implies. One channel object per symbol instead.
    ob_l2_channels = [{"name": "ob_l2", "symbols": [s]} for s in symbols]

    return [
        {"name": "ticker", "symbols": symbols},
        *ob_l2_channels,
        {"name": "trades", "symbols": symbols},
        {"name": "mark_price", "symbols": mark_symbols},
        {"name": "funding_rate", "symbols": symbols},
        {"name": "spot_price", "symbols": index_symbols},
        {"name": "candlestick_1m", "symbols": symbols + mark_symbols},
        {"name": "system_status"},
    ]


class WebSocketIngestor:
    def __init__(self, registry: SymbolRegistry, settings: Settings | None = None) -> None:
        self.registry = registry
        self.settings = settings or get_settings()
        self.cache = get_market_cache()
        self._ws = DeltaPublicWebSocket(
            channels=build_channels(registry), on_message=self._handle_message, settings=self.settings
        )

    async def run(self) -> None:
        await self._ws.run()

    def stop(self) -> None:
        self._ws.stop()

    async def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        handler = getattr(self, f"_on_{msg_type}", None)
        if handler is None:
            return
        try:
            await handler(msg)
        except Exception:
            logger.exception("Error handling %s message: %s", msg_type, msg)

    async def _on_ticker(self, msg: dict) -> None:
        for item in msg.get("d", []):
            symbol = item.get("s")
            if not symbol:
                continue
            self.cache.update_ticker(symbol, item)

    async def _on_ob_l2(self, msg: dict) -> None:
        symbol = msg.get("sy")
        if not symbol:
            return
        bids = [[float(p), float(s)] for p, s in msg.get("b", [])]
        asks = [[float(p), float(s)] for p, s in msg.get("a", [])]
        self.cache.update_orderbook(symbol, bids, asks, msg.get("ts", 0))

    async def _on_trades(self, msg: dict) -> None:
        symbol = msg.get("sy")
        if not symbol:
            return
        self.cache.add_trade(
            {
                "symbol": symbol,
                "price": float(msg["p"]),
                "size": float(msg["s"]),
                "buyer_role": msg.get("r", "t"),
                "timestamp": msg.get("t", msg.get("ts", 0)),
            }
        )

    async def _on_mark_price(self, msg: dict) -> None:
        symbol = msg.get("sy")
        if not symbol:
            return
        self.cache.update_mark_price(symbol, float(msg["p"]))

    async def _on_spot_price(self, msg: dict) -> None:
        symbol = msg.get("sy")
        if not symbol:
            return
        self.cache.update_spot_price(symbol, float(msg["p"]))

    async def _on_funding_rate(self, msg: dict) -> None:
        symbol = msg.get("sy")
        if not symbol:
            return
        self.cache.update_funding_rate(
            symbol,
            {
                "funding_rate": msg.get("fr"),
                "next_funding_realization": msg.get("nfr"),
                "timestamp": msg.get("ts"),
            },
        )

    async def _on_candlestick_1m(self, msg: dict) -> None:
        symbol = msg.get("sy")
        if not symbol:
            return
        ts_seconds = int(msg.get("ts", 0)) // 1_000_000
        bar_time = ts_seconds - (ts_seconds % 60)
        self.cache.update_candle(
            symbol,
            "1m",
            {
                "time": bar_time,
                "open": msg.get("o"),
                "high": msg.get("h"),
                "low": msg.get("l"),
                "close": msg.get("c"),
                "volume": msg.get("v", 0.0) or 0.0,
            },
        )

    async def _on_system_status(self, msg: dict) -> None:
        self.cache.system_status = msg
