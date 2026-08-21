import datetime as dt

import pandas as pd
import pytest

from app.astro import panchang
from app.strategy.graha import GrahaStrategy

skyfield = pytest.importorskip("skyfield")


def _df(times: list[float], ema_spread_atrs: float = 4.0) -> pd.DataFrame:
    n = len(times)
    # macd_hist's sign always matches ema_spread_atrs's -- i.e. MACD always
    # AGREES with the trend direction -- so the base (pre-astro) signal has
    # the same magnitude regardless of which direction is under test. Tests
    # that flip ema_spread_atrs to compare "agrees with astro factor X" vs.
    # "disagrees" rely on this: a fixed macd_hist sign would silently also
    # flip MACD agreement between the two cases, confounding the comparison
    # with a second, unrelated effect.
    macd_hist = 0.5 if ema_spread_atrs >= 0 else -0.5
    return pd.DataFrame(
        {
            "time": times,
            "ema_9": [100.0 + ema_spread_atrs] * n,
            "ema_21": [100.0] * n,
            "atr_14": [1.0] * n,
            "macd_hist": [macd_hist] * n,
        }
    )


def _unix(y, m, d, h=0, mi=0, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp()


def _find_date(predicate, start: dt.date, days: int = 60) -> dt.date:
    """Scans forward from `start` for the first date (checked at 10:00 UTC)
    satisfying `predicate(rahu_kaal, vishti, yoga_severity)`. Real Panchang
    gates can't be fabricated the way a plain calendar date can -- they're
    genuine ephemeris output -- so tests find a real date with the needed
    property instead of assuming one, which stays correct even if the
    underlying astro math changes slightly.
    """
    for offset in range(days):
        date = start + dt.timedelta(days=offset)
        t = _unix(date.year, date.month, date.day, 10)
        rk = panchang.is_rahu_kaal([t])[0]
        vk = panchang.is_vishti_karana([t])[0]
        ys = panchang.yoga_severity([t])[0]
        if predicate(rk, vk, ys):
            return date
    raise AssertionError("no matching date found in range -- widen the search")


def _clean_date() -> dt.date:
    return _find_date(lambda rk, vk, ys: not rk and not vk and ys == 0, dt.date(2024, 1, 1))


def test_flat_during_rahu_kaal():
    date = _find_date(lambda rk, vk, ys: rk and not vk and ys == 0, dt.date(2024, 1, 1))
    start, end = panchang.rahu_kaal_window_utc(date)
    mid_rahu_kaal = (start + end) / 2.0

    strat = GrahaStrategy()
    signal = strat.generate_signal(_df([mid_rahu_kaal]))
    assert signal.iloc[-1] == 0.0


def test_vishti_karana_gate_is_off_by_default_but_available():
    """gate_vishti_karana defaults to False -- an ablation study (see
    graha.py's docstring) found this specific gate consistently hurt Sharpe
    on all 3 symbols, confirmed on a holdout split, unlike every other
    mechanism. It's still fully implemented and available via
    gate_vishti_karana=True, just not applied unless explicitly asked for.
    """
    date = _find_date(lambda rk, vk, ys: vk and not rk and ys == 0, dt.date(2024, 1, 1))
    t = _unix(date.year, date.month, date.day, 10)

    default_signal = GrahaStrategy().generate_signal(_df([t])).iloc[-1]
    assert default_signal != 0.0  # NOT gated by default

    gated_signal = GrahaStrategy(gate_vishti_karana=True).generate_signal(_df([t])).iloc[-1]
    assert gated_signal == 0.0  # still available on request


def test_flat_during_severe_yoga():
    date = _find_date(lambda rk, vk, ys: ys == 2 and not rk and not vk, dt.date(2024, 1, 1))
    t = _unix(date.year, date.month, date.day, 10)

    strat = GrahaStrategy()
    signal = strat.generate_signal(_df([t]))
    assert signal.iloc[-1] == 0.0


def test_moderate_yoga_discounts_but_does_not_gate_flat():
    date = _find_date(lambda rk, vk, ys: ys == 1 and not rk and not vk, dt.date(2024, 1, 1))
    t = _unix(date.year, date.month, date.day, 10)

    strat = GrahaStrategy()
    clean_date = _clean_date()
    clean_t = _unix(clean_date.year, clean_date.month, clean_date.day, 10)

    moderate_signal = abs(strat.generate_signal(_df([t])).iloc[-1])
    clean_signal = abs(strat.generate_signal(_df([clean_t])).iloc[-1])
    assert moderate_signal != 0.0  # discounted, not gated
    # Can't compare magnitudes directly (different real astro conditions on
    # each date) -- just confirm the moderate-yoga path doesn't zero it out.


def test_trades_outside_all_gates():
    date = _clean_date()
    t = _unix(date.year, date.month, date.day, 10)

    strat = GrahaStrategy()
    signal = strat.generate_signal(_df([t]))
    assert signal.iloc[-1] != 0.0


def test_missing_time_column_fails_closed():
    strat = GrahaStrategy()
    df = _df([_unix(2024, 6, 15, 10)]).drop(columns=["time"])
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] == 0.0


def test_disagreement_with_lunar_phase_discounts_but_does_not_flip_sign():
    date = _clean_date()
    t = _unix(date.year, date.month, date.day, 10)
    waxing = panchang.is_waxing_moon([t])[0]
    # Long bias (positive ema_spread) agrees with a waxing moon per the
    # strategy's bullish-on-waxing convention; force the disagreeing case by
    # flipping the trend direction to short when waxing (or long when not).
    ema_spread_atrs = -4.0 if waxing else 4.0

    df_agree = _df([t], ema_spread_atrs=-ema_spread_atrs)
    df_disagree = _df([t], ema_spread_atrs=ema_spread_atrs)

    # Isolate the paksha mechanism specifically -- the OTHER independent
    # tilts (nakshatra lord, aspects, retrograde) are real, separate Vedic
    # factors that don't necessarily point the same way at any given real
    # timestamp, so they're neutralized here rather than left to confound
    # this test of paksha alone.
    strat = GrahaStrategy(nakshatra_lord_weight=0.0, aspect_weight=0.0, mercury_retrograde_discount=1.0, malefic_retrograde_discount=1.0)
    agree_signal = strat.generate_signal(df_agree).iloc[-1]
    disagree_signal = strat.generate_signal(df_disagree).iloc[-1]

    assert (agree_signal > 0) != (disagree_signal > 0)
    assert abs(disagree_signal) < abs(agree_signal)


def test_nakshatra_lord_bias_tilts_conviction():
    """Independent of paksha: a call agreeing with the Moon's current
    nakshatra-lord classification (benefic/malefic) should carry MORE
    conviction than one disagreeing, holding everything else about the bar
    fixed."""
    date = _clean_date()
    t = _unix(date.year, date.month, date.day, 10)
    lord_bias = panchang.nakshatra_lord_bias([t])[0]
    assert lord_bias in (-1.0, 1.0)

    ema_spread_atrs = 4.0 if lord_bias > 0 else -4.0  # agrees with the lord's bias
    # Isolate the nakshatra-lord mechanism -- paksha flips its own agreement
    # in lockstep with direction the same way this one does, so it's
    # neutralized here (disagreement_discount=1.0 means "no paksha effect")
    # rather than left to confound which factor caused the difference.
    strat = GrahaStrategy(disagreement_discount=1.0, aspect_weight=0.0, mercury_retrograde_discount=1.0, malefic_retrograde_discount=1.0)
    agree_signal = abs(strat.generate_signal(_df([t], ema_spread_atrs=ema_spread_atrs)).iloc[-1])
    disagree_signal = abs(strat.generate_signal(_df([t], ema_spread_atrs=-ema_spread_atrs)).iloc[-1])
    assert agree_signal > disagree_signal


def test_retrograde_discounts_actually_reduce_conviction_vs_no_discount():
    """Turning the retrograde discounts off (multiplier=1.0) must never
    produce a SMALLER signal than leaving them on -- i.e. the discount only
    ever shrinks conviction, and does so strictly whenever any tracked
    planet (Mercury, or Mars/Saturn) is actually retrograde right now.
    """
    from app.astro.ephemeris import is_retrograde

    date = _clean_date()
    t = _unix(date.year, date.month, date.day, 10)

    n_malefic_retro = panchang.malefic_retrograde_count([t])[0]
    mercury_retro_now = is_retrograde("mercury", [t])[0]

    with_discount = GrahaStrategy()
    no_discount = GrahaStrategy(mercury_retrograde_discount=1.0, malefic_retrograde_discount=1.0)

    df = _df([t])
    discounted_signal = abs(with_discount.generate_signal(df).iloc[-1])
    undiscounted_signal = abs(no_discount.generate_signal(df).iloc[-1])

    assert discounted_signal <= undiscounted_signal
    if mercury_retro_now or n_malefic_retro > 0:
        assert discounted_signal < undiscounted_signal


def test_dasha_weight_is_off_by_default_but_available():
    """dasha_weight defaults to 0.0 -- backtested at several weights and
    both signs across all 3 symbols, it consistently hurt SOLUSD with only
    small/inconsistent effects on BTC/ETH (see graha.py's docstring). Still
    fully implemented (app/astro/dasha.py) and available via
    dasha_weight=<non-zero>, just not applied unless explicitly asked for.
    """
    from app.astro.dasha import dasha_bias

    date = _clean_date()
    t = _unix(date.year, date.month, date.day, 10)
    assert dasha_bias("BTCUSD", [t])[0] != 0.0  # every Navagraha classifies benefic/malefic -- never neutral

    default_signal = GrahaStrategy(symbol="BTCUSD").generate_signal(_df([t])).iloc[-1]
    no_symbol_signal = GrahaStrategy(symbol=None).generate_signal(_df([t])).iloc[-1]
    assert default_signal == pytest.approx(no_symbol_signal)  # dasha_weight=0.0 -> identical to no symbol at all

    tilted_signal = GrahaStrategy(symbol="BTCUSD", dasha_weight=0.3).generate_signal(_df([t])).iloc[-1]
    assert tilted_signal != pytest.approx(default_signal)  # active once explicitly enabled


def test_hora_weight_is_off_by_default_but_available():
    """hora_weight defaults to 0.0 -- same finding and same discipline as
    dasha_weight above (see graha.py's docstring)."""
    from app.astro.panchang import hora_bias

    date = _clean_date()
    t = _unix(date.year, date.month, date.day, 10)
    assert hora_bias([t])[0] != 0.0  # every Hora lord classifies benefic/malefic -- never neutral

    default_signal = GrahaStrategy().generate_signal(_df([t])).iloc[-1]
    tilted_signal = GrahaStrategy(hora_weight=0.3).generate_signal(_df([t])).iloc[-1]
    assert tilted_signal != pytest.approx(default_signal)  # active once explicitly enabled


def test_required_warmup_bars_is_positive():
    assert GrahaStrategy().required_warmup_bars() > 0
