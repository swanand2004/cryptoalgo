import pytest

from app.config import ALLOWED_ORIGINS, is_allowed_origin


@pytest.mark.parametrize("origin", [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://[::1]:5173",
])
def test_loopback_origins_are_allowed(origin):
    assert is_allowed_origin(origin)


@pytest.mark.parametrize("origin", [
    "http://192.168.0.101:5173",  # the exact origin that was 403ing
    "http://192.168.1.42:5173",
    "http://10.0.0.5:5173",
    "http://172.16.3.9:5173",
    "http://172.31.255.1:5173",
])
def test_private_lan_origins_are_allowed(origin):
    """Regression for a real 403 on 2026-09-03. Binding the dev server to all
    interfaces (needed because 'localhost' resolved IPv6-only and left the
    dashboard unreachable over IPv4) made Vite advertise a LAN URL. Opening it
    sent an Origin the static allowlist rejected, and every dashboard socket
    was refused with no indication of why."""
    assert is_allowed_origin(origin)


@pytest.mark.parametrize("origin", [
    "https://evil.example.com",
    "http://example.com:5173",
    "http://172.32.0.1:5173",   # just outside the 172.16-31 private block
    "http://11.0.0.1:5173",     # adjacent to 10.x but public
    "http://193.168.0.101:5173",  # one digit off 192.168
])
def test_public_origins_are_still_refused(origin):
    """The protection that matters is unchanged: an arbitrary web page must
    not be able to open this socket and read live equity and positions."""
    assert not is_allowed_origin(origin)


@pytest.mark.parametrize("origin", [
    "http://192.168.0.101.evil.com",       # private-looking prefix, public host
    "http://127.0.0.1.attacker.net:5173",
    "http://localhost.evil.com",
])
def test_hosts_that_merely_look_private_are_refused(origin):
    """The pattern is anchored, so a public domain cannot smuggle itself in by
    starting with something that resembles a private address."""
    assert not is_allowed_origin(origin)


def test_missing_origin_is_allowed():
    """Non-browser clients (this project's own tests, a CLI websocket client)
    send no Origin at all. The header is only meaningful when a browser sets
    it, so its absence must not be treated as hostile."""
    assert is_allowed_origin(None)


def test_the_static_allowlist_still_passes():
    for origin in ALLOWED_ORIGINS:
        assert is_allowed_origin(origin)


def test_same_origin_public_host_is_allowed():
    # The deployed dashboard is served by the backend itself, so its Origin
    # host equals the request Host -- whatever the tunnel URL happens to be.
    assert is_allowed_origin("https://dash.example.com", host="dash.example.com")


def test_public_origin_for_a_different_host_is_still_refused():
    assert not is_allowed_origin("https://evil.example.net", host="dash.example.com")
    assert not is_allowed_origin("https://dash.example.com.evil.net", host="dash.example.com")
