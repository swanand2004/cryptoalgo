from __future__ import annotations

import asyncio
import logging

from app.config import Settings, get_settings
from app.core.deadman_switch import DeadmanSwitch
from app.core.delta_client import DeltaRestClient
from app.core.symbols import SymbolRegistry
from app.data.db import init_db
from app.data.ingest_ws import WebSocketIngestor
from app.data.market_cache import MarketDataCache, get_market_cache
from app.execution.live_broker import LiveBroker
from app.execution.order_manager import OrderManager
from app.execution.paper_broker import PaperBroker
from app.risk.risk_manager import RiskManager
from app.strategy.ensemble import SignalEnsemble
from app.strategy.registry import build_ensemble_for_symbol, seed_default_strategy_configs

logger = logging.getLogger(__name__)


class Runtime:
    """Process-wide singleton holding every long-lived component. Built once
    at FastAPI startup and stashed on `app.state.runtime`; routes, the
    scheduler, and the WS ingestor all share it instead of re-constructing
    clients/registries per request.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.rest_client = DeltaRestClient(settings)
        self.registry = SymbolRegistry(settings)
        self.cache: MarketDataCache = get_market_cache()
        self.risk_manager = RiskManager(settings)
        self.paper_broker = PaperBroker(self.registry, self.cache, settings)
        self.live_broker = LiveBroker(self.rest_client, self.registry, self.cache, self.risk_manager, settings)
        self.order_manager = OrderManager(
            self.paper_broker, self.live_broker, self.risk_manager, self.registry, self.cache
        )
        self.deadman_switch = DeadmanSwitch(self.rest_client)
        self.ensembles: dict[str, SignalEnsemble] = {}
        # Latest blended signal per symbol, BEFORE hysteresis, written by the
        # trading tick. This -- not a price -- is what ends a trade (there are
        # no stop/take-profit orders, see Settings.atr_brackets_enabled), so
        # the dashboard shows it where a stop would normally sit.
        self.live_signals: dict[str, float] = {}
        self.ws_ingestor: WebSocketIngestor | None = None
        self._ws_task: asyncio.Task | None = None

    async def start(self) -> None:
        await init_db()
        await seed_default_strategy_configs()
        await self.risk_manager.load_daily_pnl_state()
        await self.registry.refresh(self.rest_client)

        if not self.registry.symbols:
            logger.warning(
                "No products resolved for configured symbols %s -- check network/DELTA_ENV",
                self.settings.symbol_list,
            )

        for symbol in self.registry.symbols:
            self.ensembles[symbol] = await build_ensemble_for_symbol(symbol)

        self.ws_ingestor = WebSocketIngestor(self.registry, self.settings)
        self._ws_task = asyncio.create_task(self.ws_ingestor.run())
        logger.info("Runtime started for symbols: %s", self.registry.symbols)

    async def refresh_ensembles(self) -> None:
        for symbol in self.registry.symbols:
            self.ensembles[symbol] = await build_ensemble_for_symbol(symbol)

    async def stop(self) -> None:
        if self.ws_ingestor:
            self.ws_ingestor.stop()
        if self._ws_task:
            self._ws_task.cancel()
        await self.deadman_switch.stop(disable_remote=False)
        await self.rest_client.aclose()


_runtime: Runtime | None = None


async def get_or_create_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime(get_settings())
        await _runtime.start()
    return _runtime


def get_runtime_sync() -> Runtime:
    if _runtime is None:
        raise RuntimeError("Runtime not started yet")
    return _runtime
