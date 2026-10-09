import datetime as dt

import pytest

import app.options.chain as chain
from app.options.chain import build_chain_view, parse_expiry, parse_ticker

UTC = dt.timezone.utc


def _raw(symbol, ctype, strike, *, mark="100", ask="105", bid="95", delta="0.4", spot="85000", cv="0.001"):
    return {
        "symbol": symbol, "contract_type": ctype, "strike_price": str(strike), "mark_price": mark,
        "spot_price": spot, "contract_value": cv, "oi_contracts": "12", "oi": "0.012", "volume": 1.0,
        "mark_vol": "0.30",
        "quotes": {"best_ask": ask, "best_bid": bid, "mark_iv": "0.31", "ask_iv": "0.33", "bid_iv": "0.29"},
        "greeks": {"delta": delta, "gamma": "0.0001", "theta": "-50", "vega": "20", "spot": spot},
    }


def test_parse_expiry_reads_ddmmyy_at_noon_utc():
    assert parse_expiry("C-BTC-90000-161026") == dt.datetime(2026, 10, 16, 12, tzinfo=UTC)
    assert parse_expiry("P-ETH-2500-251226") == dt.datetime(2026, 12, 25, 12, tzinfo=UTC)
    assert parse_expiry("BTCUSD") is None
    assert parse_expiry("C-BTC-1-991399") is None


def test_parse_ticker_types_and_treats_zero_quotes_as_missing():
    r = parse_ticker(_raw("C-BTC-90000-161026", "call_options", 90000, ask="0", bid=None))
    assert r["type"] == "call" and r["strike"] == 90000.0 and r["delta"] == 0.4
    assert r["ask"] is None and r["bid"] is None and r["mark"] == 100.0
    assert r["iv"] == 0.31 and r["contract_value"] == 0.001 and r["oi_contracts"] == 12.0
    assert parse_ticker({"symbol": "BTCUSD", "contract_type": "perpetual_futures"}) is None


def test_build_chain_view_pairs_calls_puts_and_windows_strikes_around_spot():
    now = dt.datetime(2026, 10, 9, 12, tzinfo=UTC)
    raw = []
    for k in range(80000, 90001, 1000):
        for exp in ("091026", "161026"):
            raw.append(_raw(f"C-BTC-{k}-{exp}", "call_options", k))
            raw.append(_raw(f"P-BTC-{k}-{exp}", "put_options", k, delta="-0.4"))
    rows = [parse_ticker(x) for x in raw]
    view = build_chain_view(rows, width=2, now=now)
    # 09 Oct 12:00 has expired at `now`, so only 16 Oct is listed and chosen.
    assert [e["code"] for e in view["expiries"]] == ["161026"]
    assert view["expiry"] == "161026"
    assert [s["strike"] for s in view["strikes"]] == [83000, 84000, 85000, 86000, 87000]
    assert view["strikes"][0]["call"]["symbol"] == "C-BTC-83000-161026"
    assert view["strikes"][0]["put"]["symbol"] == "P-BTC-83000-161026"
    assert view["spot"] == 85000.0


@pytest.mark.asyncio
async def test_get_chain_caches_and_serves_stale_copy_on_failure(monkeypatch):
    chain._cache.clear()
    calls = {"n": 0}

    async def fake_fetch(underlying):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("down")
        return [_raw("C-BTC-90000-161026", "call_options", 90000)]

    monkeypatch.setattr(chain, "_fetch_raw", fake_fetch)
    rows, t1 = await chain.get_chain("BTC")
    rows2, t2 = await chain.get_chain("btc")
    assert calls["n"] == 1 and t1 == t2 and len(rows2) == 1
    rows3, t3 = await chain.get_chain("BTC", force=True)  # refresh fails -> old copy
    assert calls["n"] == 2 and t3 == t1 and len(rows3) == 1
    with pytest.raises(ValueError):
        await chain.get_chain("SOL")
    chain._cache.clear()
