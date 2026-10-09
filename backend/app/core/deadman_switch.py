from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.delta_client import DeltaApiError, DeltaRestClient

logger = logging.getLogger(__name__)

NETWORK_RETRY_SECONDS = 3.0


class DeadmanSwitch:
    """Wraps Delta's Heartbeat/Deadman Switch API: if this process stops
    sending heartbeats (crash, lost network, host goes to sleep), Delta itself
    cancels all our open orders. This is the safety net that makes running a
    live trading bot on a home machine acceptable -- without it, a dropped
    connection could leave resting orders live with nobody watching them.
    """

    def __init__(
        self,
        client: DeltaRestClient,
        heartbeat_id: str = "cryptoalgo_bot",
        contract_types: list[str] | None = None,
        ttl_ms: int = 30_000,
        send_interval_seconds: float = 20.0,
    ) -> None:
        self.client = client
        self.heartbeat_id = heartbeat_id
        self.contract_types = contract_types or ["perpetual_futures"]
        self.ttl_ms = ttl_ms
        self.send_interval_seconds = send_interval_seconds
        self._task: asyncio.Task | None = None
        self._registered = False

    async def _register(self) -> bool:
        """Returns False only when the request never reached Delta."""
        try:
            await self.client.create_heartbeat(
                {
                    "heartbeat_id": self.heartbeat_id,
                    "impact": "contracts",
                    "contract_types": self.contract_types,
                    "config": [{"action": "cancel_orders", "unhealthy_count": 1}],
                }
            )
            self._registered = True
            logger.info("Deadman switch registered: %s", self.heartbeat_id)
        except DeltaApiError as exc:
            logger.error("Failed to register deadman switch: %s", exc)
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            # Network blip (connect timeout, DNS): nothing was registered;
            # the loop retries shortly. Must not escape -- see _loop.
            logger.warning("Deadman switch registration did not reach Delta (%s); retrying", type(exc).__name__)
            return False
        return True

    async def _loop(self) -> None:
        while True:
            # Retried every iteration, not just once at task start -- a
            # transient failure on the FIRST attempt used to leave
            # _registered permanently False, so the loop kept "sending
            # heartbeats" to an id Delta never actually registered (each
            # call failing, silently) with no automatic recovery. This
            # self-heals: it keeps retrying registration at the normal
            # heartbeat cadence until it succeeds.
            network_miss = False
            if not self._registered:
                network_miss = not await self._register()
            if self._registered:
                try:
                    await self.client.send_heartbeat(self.heartbeat_id, self.ttl_ms)
                except DeltaApiError as exc:
                    logger.error("Deadman switch heartbeat failed: %s", exc)
                    # A failed heartbeat may mean the registration itself
                    # was lost or expired server-side -- don't keep
                    # assuming it's still valid; re-register next iteration.
                    self._registered = False
                except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                    # A network error used to escape and END this task
                    # ("Task exception was never retrieved", 21x on
                    # 2026-10-06), leaving no heartbeat until the 10 s
                    # supervisor noticed. The registration itself is still
                    # valid, so keep it and just resend soon.
                    logger.warning("Deadman switch heartbeat did not reach Delta (%s); retrying", type(exc).__name__)
                    network_miss = True
            # After a network miss, retry well inside the TTL rather than a
            # full interval later (20 s interval + a miss overruns a 30 s TTL).
            # API rejections keep the normal cadence, as before.
            await asyncio.sleep(
                min(self.send_interval_seconds, NETWORK_RETRY_SECONDS) if network_miss else self.send_interval_seconds
            )

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info("Deadman switch loop started")

    async def stop(self, disable_remote: bool = True) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if disable_remote and self._registered:
            try:
                await self.client.send_heartbeat(self.heartbeat_id, 0)
            except (DeltaApiError, httpx.HTTPError, asyncio.TimeoutError) as exc:
                logger.warning("Failed to disable deadman switch remotely: %r", exc)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()
