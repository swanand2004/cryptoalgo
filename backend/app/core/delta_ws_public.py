from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict], Awaitable[None]]


class DeltaPublicWebSocket:
    """Resilient client for Delta's new public WebSocket endpoint.

    Subscribes once on connect/reconnect to whatever channel list was configured,
    enables the server heartbeat and watches it locally (per docs: 30s cadence,
    reconnect if none seen for 35s), and hands every parsed message to a single
    dispatcher callback -- ingestion code decides what to do with each type.
    """

    def __init__(
        self,
        channels: list[dict],
        on_message: MessageHandler,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.channels = channels
        self.on_message = on_message
        self._stop = asyncio.Event()
        self._last_heartbeat = 0.0

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with connect(
                    self.settings.ws_public_url, open_timeout=10, close_timeout=5
                ) as ws:
                    logger.info("Connected to public WS %s", self.settings.ws_public_url)
                    backoff = 1.0
                    await self._subscribe(ws)
                    await self._listen(ws)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                logger.warning("Public WS disconnected (%s), reconnecting in %.1fs", exc, backoff)
            except Exception:
                logger.exception("Unexpected public WS error, reconnecting in %.1fs", backoff)

            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _subscribe(self, ws: ClientConnection) -> None:
        await ws.send(json.dumps({"type": "enable_heartbeat"}))
        await ws.send(
            json.dumps({"type": "subscribe", "payload": {"channels": self.channels}})
        )

    async def _listen(self, ws: ClientConnection) -> None:
        watchdog = asyncio.create_task(self._heartbeat_watchdog(ws))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "heartbeat":
                    self._last_heartbeat = asyncio.get_event_loop().time()
                    continue
                if msg_type == "pong":
                    continue
                await self.on_message(msg)
        finally:
            watchdog.cancel()

    async def _heartbeat_watchdog(self, ws: ClientConnection) -> None:
        self._last_heartbeat = asyncio.get_event_loop().time()
        while True:
            await asyncio.sleep(5)
            if asyncio.get_event_loop().time() - self._last_heartbeat > 35:
                logger.warning("No heartbeat in 35s, forcing reconnect")
                await ws.close()
                return
