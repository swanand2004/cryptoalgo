import numpy as np
import pandas as pd

from app.strategy.gann_angle import DAY_SECONDS, GannAngleStrategy, daily_angle_signal, swing_pivots
from app.strategy.registry import DEFAULT_WEIGHTS


def _hourly(daily_closes: list[float], start_day: int = 19_000) -> pd.DataFrame:
    """24 hourly bars per day, drifting linearly from the previous day's close
    to this day's close, with a small high/low envelope."""
    rows, prev = [], daily_closes[0]
    for d, close in enumerate(daily_closes):
        for h in range(24):
            px = prev + (close - prev) * (h + 1) / 24
            t = (start_day + d) * DAY_SECONDS + h * 3600
            rows.append({"time": t, "open": px, "high": px * 1.001, "low": px * 0.999, "close": px})
        prev = close
    return pd.DataFrame(rows)


def _random_walk(n_days: int, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n_days))))


def test_swing_chart_marks_the_top_of_an_up_swing_once_two_lower_lows_confirm_it():
    high = np.array([10, 11, 12, 13, 12, 11, 10], dtype=float)
    low = high - 0.5
    pivots = swing_pivots(high, low, n=2)
    top = [p for p in pivots if p[2] == "top"]
    assert top and top[0][0] == 3 and top[0][1] == 13.0
    assert top[0][3] == 5  # known only on the second lower low, never before


def test_strong_rally_off_a_bottom_is_long_and_breaking_the_1x1_goes_flat():
    # Down, swing bottom, then a steep rally (above the 1x1) that later stalls.
    closes = [100, 97, 94, 91, 88, 90, 95, 101, 108, 116] + [116] * 25
    daily = pd.DataFrame({"high": np.array(closes) * 1.01, "low": np.array(closes) * 0.99, "close": closes})
    sig = daily_angle_signal(daily)
    assert sig.iloc[9] == 1.0     # riding above the angle from the bottom
    assert sig.iloc[-1] == 0.0    # flat price eventually falls under a rising 1x1


def test_signal_is_causal_appending_future_bars_never_changes_the_past():
    full = _hourly(_random_walk(120))
    head = full.iloc[: 24 * 80].copy()
    strat = GannAngleStrategy()
    a = strat.generate_signal(head).to_numpy()
    b = strat.generate_signal(full).to_numpy()[: len(head)]
    np.testing.assert_array_equal(a, b)


def test_todays_unfinished_bars_cannot_move_todays_signal():
    # The signal for day D is decided at the close of day D-1, so a violent
    # move inside today's partial day must not repaint today's value.
    closes = _random_walk(90)
    base = _hourly(closes)
    today = base.iloc[-24:-12].copy()
    shocked = today.copy()
    shocked[["open", "high", "low", "close"]] *= 1.5
    strat = GannAngleStrategy()
    calm = strat.generate_signal(pd.concat([base.iloc[:-24], today], ignore_index=True))
    wild = strat.generate_signal(pd.concat([base.iloc[:-24], shocked], ignore_index=True))
    assert calm.iloc[-12:].tolist() == wild.iloc[-12:].tolist()


def test_values_are_in_range_and_aligned():
    df = _hourly(_random_walk(100))
    sig = GannAngleStrategy().generate_signal(df)
    assert sig.index.equals(df.index)
    assert set(np.unique(sig)) <= {-1.0, 0.0, 1.0}


def test_missing_columns_give_a_neutral_signal():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    assert GannAngleStrategy().generate_signal(df).tolist() == [0.0, 0.0]


def test_live_lineup_is_the_tested_50_50_blend():
    # Equal weights -> the ensemble's weight-normalised average is exactly the
    # 50/50 that was validated. Unequal weights would run an untested blend.
    assert DEFAULT_WEIGHTS == {"trend_ladder": 1.0, "gann_angle": 1.0}
