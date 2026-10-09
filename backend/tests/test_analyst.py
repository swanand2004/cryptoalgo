import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.analyst.factors import Inputs, analyse, derivatives_factors, macro_factors, sentiment_factors


def _hourly(drift: float, n_days: int = 320, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_days * 24
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close}, index=idx)


def test_uptrend_reads_buy_with_a_quick_one_to_one_plan():
    a = analyse("BTCUSD", Inputs(hourly=_hourly(0.0006), strategy_signal=0.8))
    assert a.decision == "BUY" and a.conviction >= 3
    assert a.stop_loss < a.entry < a.take_profit
    assert a.take_profit > a.entry > a.stop_loss and a.plan_note
    # nothing else supplied -> only price-based families are scored
    assert {f.family for f in a.factors} == {"technical", "structure"}


def test_downtrend_reads_sell_and_mirror():
    a = analyse("BTCUSD", Inputs(hourly=_hourly(-0.0006), strategy_signal=-0.8))
    assert a.decision == "SELL" and a.stop_loss > a.entry > a.take_profit


def test_too_little_history_gives_no_analysis():
    assert analyse("BTCUSD", Inputs(hourly=_hourly(0.0, n_days=30))) is None


def test_crowded_funding_and_one_sided_positioning_lean_bearish():
    h = _hourly(0.0)
    now = h.index[-1]
    fund = pd.Series([0.0008] * 6, index=pd.date_range(now - pd.Timedelta(hours=40), periods=6, freq="8h"))
    lsr = pd.Series([2.6], index=[now])
    f = {x.key: x for x in derivatives_factors(Inputs(hourly=h, funding=fund, long_short_ratio=lsr), now)}
    assert f["funding"].score == -1.0 and "crowded long" in f["funding"].reading  # 0.08%/8h = +88%/yr
    assert f["long_short"].score == -0.5


def test_stale_inputs_drop_out_instead_of_voting():
    h = _hourly(0.0)
    now = h.index[-1]
    old_fg = pd.Series([10.0], index=[now - pd.Timedelta(days=10)])
    assert sentiment_factors(Inputs(hourly=h, fear_greed=old_fg), now) == []
    fresh = pd.Series([10.0], index=[now - pd.Timedelta(hours=5)])
    [fg] = sentiment_factors(Inputs(hourly=h, fear_greed=fresh), now)
    assert fg.score == 0.5 and "extreme fear" in fg.reading


def test_event_within_24h_halves_size_without_voting():
    h = _hourly(0.0)
    now = h.index[-1]
    _, size, notes = macro_factors(Inputs(hourly=h, events=[(now + pd.Timedelta(hours=5), "CPI")]), now)
    assert size == 0.5 and "CPI" in notes[0]


def test_plan_is_recentred_on_the_live_price_with_the_same_distances():
    from app.analyst.service import _to_live_price

    r = {"entry": 100.0, "stop_loss": 97.0, "take_profit": 103.0, "price": 100.0}
    _to_live_price(r, 101.0)
    assert (r["entry"], r["stop_loss"], r["take_profit"]) == (101.0, 98.0, 104.0)


def test_analyst_covers_only_the_four_major_coins():
    import types

    from app.analyst.service import analyst_symbols
    from app.config import Settings

    settings = Settings(delta_api_key="k", delta_api_secret="s")
    registry = types.SimpleNamespace(symbols=["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "ADAUSD", "TRXUSD"])
    assert analyst_symbols(types.SimpleNamespace(settings=settings, registry=registry)) == [
        "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"]


def test_real_yield_reading_for_gold():
    h = _hourly(0.0)
    now = h.index[-1]
    days = pd.date_range(now - pd.Timedelta(days=60), periods=60, freq="D")
    falling = pd.Series(np.linspace(2.0, 1.5, 60), index=days)
    out, _, _ = macro_factors(Inputs(hourly=h, real_yield=falling), now)
    ry = {f.key: f for f in out}["real_yield"]
    assert ry.score == 0.4 and "support gold" in ry.reading
