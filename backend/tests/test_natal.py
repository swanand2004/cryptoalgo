import datetime as dt

import numpy as np
import pytest

from app.astro import natal

skyfield = pytest.importorskip("skyfield")


def _unix(y, m, d, h=0, mi=0, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp()


def test_genesis_natal_chart_covers_all_9_navagraha():
    chart = natal.genesis_natal_chart("BTCUSD")
    assert set(chart.keys()) == set(natal.NATAL_BODIES)
    for lon in chart.values():
        assert 0.0 <= lon < 360.0


def test_genesis_natal_chart_is_cached_and_stable():
    first = natal.genesis_natal_chart("ETHUSD")
    second = natal.genesis_natal_chart("ETHUSD")
    assert first == second  # exact same dict values -- computed once, cached


def test_different_symbols_have_different_natal_charts():
    btc = natal.genesis_natal_chart("BTCUSD")
    eth = natal.genesis_natal_chart("ETHUSD")
    sol = natal.genesis_natal_chart("SOLUSD")
    # Genuinely different genesis moments -- their Sun positions (which move
    # ~1deg/day, so different dates years apart must differ) should not
    # coincide.
    assert btc["sun"] != eth["sun"] != sol["sun"]


def test_unknown_symbol_returns_empty_chart_and_zero_score():
    assert natal.genesis_natal_chart("DOGEUSD") == {}
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    score = natal.natal_transit_score("DOGEUSD", [now])
    assert list(score) == [0.0]


def test_natal_transit_score_is_vectorized_and_bounded():
    times = [_unix(2024, m, 1) for m in range(1, 13)]
    score = natal.natal_transit_score("BTCUSD", times)
    assert len(score) == 12
    assert np.all(score >= -2.0) and np.all(score <= 4.0)


def test_genesis_timestamps_match_documented_values():
    # BTC Genesis Block: 2009-01-03 18:15:05 UTC. ETH block #0: Unix
    # 1438269973 (2015-07-30 15:26:13 UTC, per Etherscan). SOL Mainnet Beta
    # genesis: 2020-03-16 14:29:00 UTC.
    assert natal.GENESIS_UTC["BTCUSD"] == dt.datetime(2009, 1, 3, 18, 15, 5, tzinfo=dt.timezone.utc)
    assert natal.GENESIS_UTC["ETHUSD"].timestamp() == 1438269973
    assert natal.GENESIS_UTC["SOLUSD"] == dt.datetime(2020, 3, 16, 14, 29, 0, tzinfo=dt.timezone.utc)
