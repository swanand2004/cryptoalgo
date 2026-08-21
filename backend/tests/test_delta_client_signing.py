import hashlib
import hmac

from app.config import Settings
from app.core.delta_client import DeltaRestClient


def _settings() -> Settings:
    return Settings(delta_api_key="test-key", delta_api_secret="test-secret", delta_env="testnet")


def test_signature_matches_documented_formula():
    """Delta's docs define: signature = HMAC_SHA256(secret, method + timestamp +
    path + query_string + body). Verify our implementation follows that exact
    formula (independently recomputed here, not copy-pasted from the client).
    """
    client = DeltaRestClient(_settings())
    method, timestamp, path, query_string, body = "GET", "1542110948", "/v2/orders", "?product_id=1&state=open", ""

    expected = hmac.new(
        b"test-secret", (method + timestamp + path + query_string + body).encode(), hashlib.sha256
    ).hexdigest()

    assert client._sign(method, timestamp, path, query_string, body) == expected
    assert len(expected) == 64


def test_signature_changes_with_body():
    client = DeltaRestClient(_settings())
    sig_no_body = client._sign("POST", "123", "/v2/orders", "", "")
    sig_with_body = client._sign("POST", "123", "/v2/orders", "", '{"size":1}')
    assert sig_no_body != sig_with_body


def test_endpoint_weight_lookup():
    from app.core.delta_client import _endpoint_weight

    assert _endpoint_weight("GET", "/v2/products") == 3
    assert _endpoint_weight("POST", "/v2/orders") == 5
    assert _endpoint_weight("GET", "/v2/fills") == 10
    assert _endpoint_weight("POST", "/v2/orders/batch") == 25
    assert _endpoint_weight("GET", "/v2/assets") == 1
