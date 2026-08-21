from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.strategy.base import Strategy


@dataclass
class WeightedStrategy:
    strategy: Strategy
    weight: float
    enabled: bool = True


class SignalEnsemble:
    """Combines multiple strategies' signals into one target-direction score via
    a confidence(weight)-weighted average, smooths it to damp bar-to-bar noise,
    then applies a minimum-conviction deadband so the system doesn't churn on
    noise near zero.

    Smoothing matters more than it looks: an unsmoothed signal that oscillates
    right around the deadband line causes a full close-then-reopen every time
    it dips under, even when conviction never really reversed -- exactly the
    overtrading pattern (up to ~70% of bars trading) seen before this was
    added. `smoothing_span=1` disables smoothing (raw weighted average).
    """

    def __init__(
        self,
        members: list[WeightedStrategy],
        deadband: float = 0.15,
        smoothing_span: int = 3,
        exit_deadband: float | None = None,
    ) -> None:
        self.members = members
        self.deadband = deadband
        self.smoothing_span = max(1, smoothing_span)
        # Hysteresis exit line for stateful execution (see position_sizing.
        # apply_signal_hysteresis) -- deliberately looser than `deadband`
        # (the entry line) so a position isn't flattened by the same noise
        # that wouldn't have been enough to open it.
        self.exit_deadband = exit_deadband if exit_deadband is not None else deadband * 0.4

    def per_strategy_signals(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            m.strategy.name: m.strategy.generate_signal(df)
            for m in self.members
            if m.enabled
        }

    def raw_combined_signal(self, df: pd.DataFrame) -> pd.Series:
        """Weighted average before smoothing or the deadband -- the signal a
        single instant would imply, useful for diagnostics.
        """
        active = [m for m in self.members if m.enabled and m.weight > 0]
        if not active:
            return pd.Series(0.0, index=df.index)

        total_weight = sum(m.weight for m in active)
        combined = pd.Series(0.0, index=df.index)
        for m in active:
            sig = m.strategy.generate_signal(df)
            combined = combined.add(sig * (m.weight / total_weight), fill_value=0.0)
        return combined

    def smoothed_signal(self, df: pd.DataFrame) -> pd.Series:
        """Weighted average + EMA smoothing, WITHOUT the hard deadband. This is
        what stateful execution (backtest engine, live trading loop) should
        use, feeding it through `position_sizing.apply_signal_hysteresis` with
        the previous bar's position sign -- hysteresis needs the graduated
        signal, not one already collapsed to exactly 0 below the threshold.
        """
        combined = self.raw_combined_signal(df)
        if self.smoothing_span > 1:
            combined = combined.ewm(span=self.smoothing_span, adjust=False).mean()
        return combined.clip(-1, 1)

    def combined_signal(self, df: pd.DataFrame) -> pd.Series:
        """Smoothed + hard-deadbanded. Stateless (no position awareness) --
        appropriate for diagnostics/display, not for driving execution.
        """
        combined = self.smoothed_signal(df)
        return combined.where(combined.abs() >= self.deadband, 0.0)

    def warmup_bars(self) -> int:
        if not self.members:
            return 0
        return max(m.strategy.required_warmup_bars() for m in self.members)
