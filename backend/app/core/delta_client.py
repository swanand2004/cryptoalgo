from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from app.config import Settings, get_settings
from app.core.rate_limiter import WeightedRateLimiter

logger = logging.getLogger(__name__)

# Endpoint weight table, taken from the Rate Limits section of the API docs.
# Any endpoint not listed here defaults to weight 1.
_WEIGHT_3 = {
    ("GET", "/v2/products"),
    ("GET", "/v2/l2orderbook"),
    ("GET", "/v2/tickers"),
    ("GET", "/v2/orders"),
    ("GET", "/v2/positions/margined"),
    ("GET", "/v2/wallet/balances"),
    ("GET", "/v2/history/candles"),
}
_WEIGHT_5 = {
    ("POST", "/v2/orders"),
    ("PUT", "/v2/orders"),
    ("DELETE", "/v2/orders"),
    ("POST", "/v2/positions/change_margin"),
}
_WEIGHT_10 = {
    ("GET", "/v2/orders/history"),
    ("GET", "/v2/fills"),
    ("GET", "/v2/wallet/transactions"),
}
_WEIGHT_25 = {
    ("POST", "/v2/orders/batch"),
    ("PUT", "/v2/orders/batch"),
    ("DELETE", "/v2/orders/batch"),
}


def _is_expired_signature(response) -> bool:
    try:
        return (response.json().get("error") or {}).get("code") == "expired_signature"
    except (ValueError, AttributeError):
        return False


def _endpoint_weight(method: str, path: str) -> int:
    # Match by prefix so path params (e.g. /v2/orders/{id}) still resolve.
    for table, weight in ((_WEIGHT_25, 25), (_WEIGHT_10, 10), (_WEIGHT_5, 5), (_WEIGHT_3, 3)):
        for m, p in table:
            if method == m and path.startswith(p):
                return weight
    return 1


class DeltaApiError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Delta API error {status_code}: {payload}")


class DeltaRestClient:
    """Signed async REST client for Delta Exchange India.

    Signing follows the documented scheme exactly:
        signature = HMAC_SHA256(secret, method + timestamp + path + query_string + body)
    where query_string includes the leading '?' when present and body is the exact
    JSON string sent on the wire (empty string for bodyless requests).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        local_address = {"ipv6": "::", "ipv4": "0.0.0.0"}.get(self.settings.delta_ip_family)
        transport = httpx.AsyncHTTPTransport(local_address=local_address) if local_address else None
        self._client = httpx.AsyncClient(
            base_url=self.settings.rest_url, timeout=httpx.Timeout(3.0, read=27.0), transport=transport
        )
        self._rate_limiter = WeightedRateLimiter()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DeltaRestClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _sign(self, method: str, timestamp: str, path: str, query_string: str, body: str) -> str:
        message = method + timestamp + path + query_string + body
        return hmac.new(
            self.settings.delta_api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        auth: bool = False,
        max_retries: int = 3,
    ) -> Any:
        method = method.upper()
        query_string = f"?{urlencode(params)}" if params else ""
        body = json.dumps(json_body, separators=(",", ":")) if json_body is not None else ""

        weight = _endpoint_weight(method, path)
        await self._rate_limiter.acquire(weight)

        headers = {
            "Accept": "application/json",
            "User-Agent": "cryptoalgo-python-client",
        }
        if body:
            headers["Content-Type"] = "application/json"

        if auth and not self.settings.has_credentials:
            raise RuntimeError(
                f"Authenticated request to {path} requires DELTA_API_KEY/DELTA_API_SECRET "
                "in backend/.env"
            )

        url = path + query_string
        for attempt in range(max_retries):
            if auth:
                # Signed per ATTEMPT, right before sending. Delta rejects a
                # signature a few seconds old ("expired_signature"), and both a
                # rate-limit back-off and a slow network path can age one that
                # was made earlier -- the VPS measured ~10-15 s request times.
                timestamp = str(int(time.time()))
                headers.update(
                    {
                        "api-key": self.settings.delta_api_key,
                        "signature": self._sign(method, timestamp, path, query_string, body),
                        "timestamp": timestamp,
                    }
                )
            try:
                response = await self._client.request(method, url, content=body or None, headers=headers)
            except (httpx.ConnectTimeout, httpx.ConnectError):
                # The connection was never established, so nothing reached
                # Delta. Retried for GETs only (reads are idempotent); order
                # POST/DELETEs keep failing fast exactly as before. On the VPS
                # the 3 s IPv6 connect timed out hundreds of times a day
                # (2026-10-06 journal: ~550 balance fallbacks in 10 h), each one
                # surfacing as a stale-balance fallback, a skipped position
                # sync, a dead deadman loop or (once) a failed startup.
                if method != "GET" or attempt >= max_retries - 1:
                    raise
                logger.info("Delta connect failed on %s %s; retrying (attempt %d)", method, path, attempt + 2)
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code == 401 and auth and attempt < max_retries - 1 and _is_expired_signature(response):
                # Rejected before Delta acted on it, so a fresh signature is
                # safe to resend -- even for an order.
                logger.warning("Delta signature expired in transit on %s %s; re-signing and retrying", method, path)
                continue
            if response.status_code == 429:
                reset_ms = int(response.headers.get("X-RATE-LIMIT-RESET", "1000"))
                logger.warning("Rate limited on %s %s, backing off %sms", method, path, reset_ms)
                await self._rate_limiter.note_429(reset_ms)
                continue
            if response.status_code >= 500 and attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

            try:
                payload = response.json()
            except ValueError:
                response.raise_for_status()
                return None

            if response.status_code >= 400 or payload.get("success") is False:
                raise DeltaApiError(response.status_code, payload)
            return payload.get("result")

        raise DeltaApiError(429, {"error": "rate_limited_retries_exhausted"})

    # ---- Public market data -------------------------------------------------

    async def get_products(
        self, contract_types: Optional[str] = None, states: Optional[str] = None
    ) -> list[dict]:
        params: dict[str, Any] = {"page_size": 500}
        if contract_types:
            params["contract_types"] = contract_types
        if states:
            params["states"] = states
        return await self.request("GET", "/v2/products", params=params)

    async def get_ticker(self, symbol: str) -> dict:
        return await self.request("GET", f"/v2/tickers/{symbol}")

    async def get_tickers(self, contract_types: Optional[str] = None) -> list[dict]:
        params = {"contract_types": contract_types} if contract_types else None
        return await self.request("GET", "/v2/tickers", params=params)

    async def get_l2_orderbook(self, symbol: str, depth: Optional[int] = None) -> dict:
        params = {"depth": depth} if depth else None
        return await self.request("GET", f"/v2/l2orderbook/{symbol}", params=params)

    async def get_trades(self, symbol: str) -> dict:
        return await self.request("GET", f"/v2/trades/{symbol}")

    async def get_candles(self, symbol: str, resolution: str, start: int, end: int) -> list[dict]:
        params = {"symbol": symbol, "resolution": resolution, "start": start, "end": end}
        return await self.request("GET", "/v2/history/candles", params=params)

    async def get_stats(self) -> dict:
        return await self.request("GET", "/v2/stats")

    # ---- Private: account / trading ------------------------------------------

    async def get_balances(self) -> list[dict]:
        return await self.request("GET", "/v2/wallet/balances", auth=True)

    async def get_positions_margined(self, product_ids: Optional[str] = None) -> list[dict]:
        params = {"product_ids": product_ids} if product_ids else None
        return await self.request("GET", "/v2/positions/margined", params=params, auth=True)

    async def get_open_orders(self, product_ids: Optional[str] = None) -> list[dict]:
        params = {"product_ids": product_ids, "states": "open,pending"}
        params = {k: v for k, v in params.items() if v is not None}
        return await self.request("GET", "/v2/orders", params=params, auth=True)

    async def place_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/orders", json_body=body, auth=True)

    async def edit_order(self, body: dict) -> dict:
        return await self.request("PUT", "/v2/orders", json_body=body, auth=True)

    async def cancel_order(self, body: dict) -> dict:
        return await self.request("DELETE", "/v2/orders", json_body=body, auth=True)

    async def place_bracket_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/orders/bracket", json_body=body, auth=True)

    async def cancel_all_orders(self, body: Optional[dict] = None) -> dict:
        return await self.request("DELETE", "/v2/orders/all", json_body=body or {}, auth=True)

    async def create_heartbeat(self, body: dict) -> dict:
        return await self.request("POST", "/v2/heartbeat/create", json_body=body, auth=True)

    async def send_heartbeat(self, heartbeat_id: str, ttl_ms: int) -> dict:
        return await self.request(
            "POST", "/v2/heartbeat", json_body={"heartbeat_id": heartbeat_id, "ttl": ttl_ms}, auth=True
        )
