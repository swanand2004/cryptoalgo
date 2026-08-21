from __future__ import annotations

import asyncio
import time


class WeightedRateLimiter:
    """Client-side guard for Delta's fixed 5-minute weighted quota (10000 per window).

    Mirrors the documented cost structure so we throttle ourselves before Delta
    returns a 429, rather than discovering the limit by hitting it.
    """

    def __init__(self, quota: int = 10000, window_seconds: float = 300.0) -> None:
        self.quota = quota
        self.window_seconds = window_seconds
        self._used = 0
        self._window_start = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int = 1) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._window_start
            if elapsed >= self.window_seconds:
                self._used = 0
                self._window_start = now
                elapsed = 0.0

            if self._used + weight > self.quota:
                wait_for = self.window_seconds - elapsed
                if wait_for > 0:
                    await asyncio.sleep(wait_for)
                self._used = 0
                self._window_start = time.monotonic()

            self._used += weight

    async def note_429(self, reset_ms: int | None) -> None:
        """Called when the server itself returns 429 -- back off for the reset window."""
        async with self._lock:
            self._used = self.quota
            if reset_ms:
                await asyncio.sleep(reset_ms / 1000.0)
