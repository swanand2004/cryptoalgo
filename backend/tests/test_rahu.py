import pandas as pd
import pytest

from app.strategy.rahu import RahuStrategy

IST_OFFSET_SECONDS = 5.5 * 3600
DAY_ANCHOR = 10 * 86400  # arbitrary day, kept positive after subtracting the IST offset


def _time_for_ist_hour(hour: float) -> int:
    """A unix epoch (seconds) whose IST time-of-day is `hour`."""
    return int(hour * 3600 - IST_OFFSET_SECONDS + DAY_ANCHOR)


def _df(times: list[int], ema_spread_atrs: float = 4.0) -> pd.DataFrame:
    """ema_9/ema_21/atr_14 chosen so (ema_9-ema_21)/atr_14 == ema_spread_atrs,
    i.e. a long bias of that many ATRs -- 4.0 comfortably saturates
    RahuStrategy's magnitude (which reaches 1.0 at spread=3/greed=2.0 ATRs)
    but would ALSO saturate trend_momentum's (which needs spread>=3.0 ATRs),
    so tests that need to distinguish "greedy reaches full sooner" use a
    smaller, deliberately sub-3.0 spread instead.
    """
    n = len(times)
    return pd.DataFrame(
        {
            "time": times,
            "ema_9": [100.0 + ema_spread_atrs] * n,
            "ema_21": [100.0] * n,
            "atr_14": [1.0] * n,
            "macd_hist": [0.5] * n,  # agrees with the long bias
            "adx_14": [5.0] * n,  # deliberately LOW -- would fully gate trend_momentum
        }
    )


def test_trades_at_night_ist():
    strat = RahuStrategy()
    df = _df([_time_for_ist_hour(22.0)])  # 10 PM IST -- squarely in the night window
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] > 0


def test_stays_flat_during_the_day_regardless_of_signal_strength():
    strat = RahuStrategy()
    df = _df([_time_for_ist_hour(12.0)])  # noon IST -- squarely daytime
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] == 0.0


def test_flat_right_at_sunrise_and_active_right_before_it():
    strat = RahuStrategy(sunset_hour_ist=18.5, sunrise_hour_ist=6.0)
    just_before_sunrise = _df([_time_for_ist_hour(5.99)])
    at_sunrise = _df([_time_for_ist_hour(6.0)])
    assert strat.generate_signal(just_before_sunrise).iloc[-1] > 0
    assert strat.generate_signal(at_sunrise).iloc[-1] == 0.0


def test_active_right_at_sunset_and_flat_right_before_it():
    strat = RahuStrategy(sunset_hour_ist=18.5, sunrise_hour_ist=6.0)
    just_before_sunset = _df([_time_for_ist_hour(18.49)])
    at_sunset = _df([_time_for_ist_hour(18.5)])
    assert strat.generate_signal(just_before_sunset).iloc[-1] == 0.0
    assert strat.generate_signal(at_sunset).iloc[-1] > 0


def test_window_wraps_correctly_across_midnight():
    strat = RahuStrategy()  # default window: 8 PM -> 3 AM IST
    for hour in [23.0, 0.0, 2.0]:
        signal = strat.generate_signal(_df([_time_for_ist_hour(hour)]))
        assert signal.iloc[-1] > 0, f"expected active at {hour}:00 IST"


def test_is_greedier_than_trend_momentum_reaching_full_conviction_sooner():
    """The whole point of "greedy": a spread too small to fully convince
    trend_momentum (which needs >=3.0 ATRs and a passing ADX gate) should
    still fully convince rahu during its active window.
    """
    from app.strategy.trend_momentum import TrendMomentumStrategy

    small_spread_df = _df([_time_for_ist_hour(22.0)], ema_spread_atrs=1.6)

    rahu_signal = RahuStrategy().generate_signal(small_spread_df).iloc[-1]
    trend_signal = TrendMomentumStrategy().generate_signal(small_spread_df).iloc[-1]

    assert rahu_signal > 0.7  # comfortably past the 0.15 entry threshold
    assert trend_signal < rahu_signal  # gated by both a smaller magnitude AND the (low) ADX


def test_no_adx_gate_unlike_trend_momentum():
    """A low ADX (adx_14=5, well under trend_momentum's own gate floor of
    15) would fully zero trend_momentum's signal -- rahu has no such gate.
    """
    df = _df([_time_for_ist_hour(22.0)])  # adx_14=5.0 baked into the fixture
    signal = RahuStrategy().generate_signal(df)
    assert signal.iloc[-1] > 0.9


def test_missing_time_column_fails_closed_not_open():
    strat = RahuStrategy()
    df = _df([_time_for_ist_hour(22.0)]).drop(columns=["time"])
    signal = strat.generate_signal(df)
    assert signal.iloc[-1] == 0.0
