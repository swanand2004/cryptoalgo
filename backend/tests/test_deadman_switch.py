import asyncio

import pytest

from app.core.delta_client import DeltaApiError
from app.core.deadman_switch import DeadmanSwitch


class _FlakyClient:
    """create_heartbeat fails the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times: int = 2) -> None:
        self.fail_times = fail_times
        self.register_attempts = 0
        self.heartbeats_sent = 0

    async def create_heartbeat(self, body: dict) -> None:
        self.register_attempts += 1
        if self.register_attempts <= self.fail_times:
            raise DeltaApiError(500, {"error": "transient"})

    async def send_heartbeat(self, heartbeat_id: str, ttl_ms: int) -> None:
        self.heartbeats_sent += 1


async def _run_loop_briefly(switch: DeadmanSwitch, duration: float) -> None:
    task = asyncio.create_task(switch._loop())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=duration)
    except asyncio.TimeoutError:
        pass
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_registration_is_retried_until_it_succeeds_not_just_once():
    """Regression test: a failed FIRST registration attempt used to leave
    _registered permanently False for the life of the task -- the loop would
    then just keep calling send_heartbeat forever against an id Delta never
    actually registered, with no automatic recovery. Registration must be
    retried on subsequent loop iterations until it succeeds.
    """
    client = _FlakyClient(fail_times=2)
    switch = DeadmanSwitch(client, send_interval_seconds=0.01)

    await _run_loop_briefly(switch, duration=0.2)

    assert client.register_attempts >= 3  # 2 failures + the eventual success
    assert switch._registered is True
    assert client.heartbeats_sent >= 1  # started sending real heartbeats once registered


@pytest.mark.asyncio
async def test_a_failed_heartbeat_triggers_re_registration_next_iteration():
    """If a heartbeat call fails after registration succeeded, the switch
    should not keep assuming it's still validly registered -- it should
    re-attempt registration on the next loop tick.
    """
    client = _FlakyClient(fail_times=0)  # registers immediately

    class _HeartbeatFailsOnce(_FlakyClient):
        def __init__(self) -> None:
            super().__init__(fail_times=0)
            self._heartbeat_failed = False

        async def send_heartbeat(self, heartbeat_id: str, ttl_ms: int) -> None:
            if not self._heartbeat_failed:
                self._heartbeat_failed = True
                raise DeltaApiError(500, {"error": "transient"})
            await super().send_heartbeat(heartbeat_id, ttl_ms)

    client = _HeartbeatFailsOnce()
    switch = DeadmanSwitch(client, send_interval_seconds=0.01)

    await _run_loop_briefly(switch, duration=0.2)

    # Registered once, heartbeat failed once (forcing _registered=False),
    # then re-registered again before the next successful heartbeat.
    assert client.register_attempts >= 2
    assert switch._registered is True
    assert client.heartbeats_sent >= 1
