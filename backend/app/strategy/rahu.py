from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.base import Strategy

# India's actual sunset/sunrise vary by ~1-2 hours across seasons and by
# roughly half an hour across the country's longitude -- this is a fixed,
# documented approximation of "the Indian night session" as clock-time
# bounds, not a claim of astronomical precision. Adjust via the constructor
# if a tighter/looser or season-shifted window is wanted.
IST_OFFSET_SECONDS = 5.5 * 3600
SECONDS_PER_DAY = 86400


class RahuStrategy(Strategy):
    """Trades ONLY during the Indian night session (8 PM -> 3 AM IST, by
    default) and is deliberately more aggressive than every other strategy within
    that window: no ADX/trend-confirmation gate (trend_momentum requires
    one), and a smaller normalization divisor so it reaches full conviction
    on a much smaller EMA spread -- quick to commit rather than patient.

    Outside the window it is flat: exactly 0.0 signal, every bar, no
    exceptions. Since fixed_margin_contracts sizes a 0-signal target
    to 0 contracts and should_rebalance always executes a move to a 0
    target, this doesn't just block NEW entries during the day -- the
    existing position-management pipeline automatically closes out
    whatever this strategy is holding the moment daytime starts, with no
    separate close-out logic needed here.
    """

    name = "rahu"

    def __init__(
        self,
        ema_fast: str = "ema_9",
        ema_slow: str = "ema_21",
        sunset_hour_ist: float = 20.0,  # 8:00 PM IST
        sunrise_hour_ist: float = 3.0,  # 3:00 AM IST
        greed: float = 1.5,  # amplifies conviction magnitude before clipping
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sunset_hour_ist = sunset_hour_ist
        self.sunrise_hour_ist = sunrise_hour_ist
        self.greed = greed

    def _in_night_window(self, df: pd.DataFrame) -> pd.Series:
        if "time" not in df.columns:
            # Fail CLOSED (never trade), not open -- consistent with this
            # project's default-safe posture elsewhere (paper mode on every
            # restart, kill switch, etc.). A missing time column should
            # never actually happen (build_feature_frame always keeps it),
            # but if it somehow did, "silently trade all the time" is the
            # wrong direction to fail in for a strategy defined by WHEN it
            # trades.
            return pd.Series(False, index=df.index)
        ist_seconds = (df["time"].astype(float) + IST_OFFSET_SECONDS) % SECONDS_PER_DAY
        sunset_s = self.sunset_hour_ist * 3600
        sunrise_s = self.sunrise_hour_ist * 3600
        # Window wraps past midnight (e.g. 18:30 -> 24:00 -> 06:00), so it's
        # an OR of "at/after sunset" and "before sunrise", not a plain range.
        return (ist_seconds >= sunset_s) | (ist_seconds < sunrise_s)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        ema_spread = (df[self.ema_fast] - df[self.ema_slow]) / df["atr_14"].replace(0, np.nan)
        direction = np.sign(ema_spread)
        macd_confirm = np.sign(df["macd_hist"])
        agreement = (direction == macd_confirm).astype(float)

        # Reaches full conviction (clipped at 1.0) on roughly 1/(greed) as
        # large an EMA spread as trend_momentum needs -- and unlike
        # trend_momentum, there's no ADX gate holding it back in a
        # not-yet-confirmed trend at all.
        magnitude = (ema_spread.abs() * self.greed / 3.0).clip(0, 1)
        raw = direction * magnitude * (0.5 + 0.5 * agreement)

        night = self._in_night_window(df)
        raw = raw.where(night, 0.0)

        return raw.clip(-1, 1).fillna(0.0)

    def required_warmup_bars(self) -> int:
        return 30
