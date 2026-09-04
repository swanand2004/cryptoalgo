from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.config import get_settings
from app.data.db import SessionLocal
from app.data.models import StrategyConfig
from app.strategy.ensemble import SignalEnsemble, WeightedStrategy
from app.strategy.graha import GrahaStrategy
from app.strategy.time_series_momentum import TimeSeriesMomentumStrategy
from app.strategy.trend_momentum import TrendMomentumStrategy

# Final, deliberately narrow lineup (decided 2026-08-21): exactly 3
# strategies, chosen to be genuinely DIFFERENT mechanisms rather than
# near-duplicates, each backed by its own backtest:
#   - graha: the deep, full-Panchang astrology mechanism (all 9 Navagraha,
#     5-anga Panchang, planetary aspects, multi-planet retrograde -- see
#     app/strategy/graha.py's docstring for the complete rule list). Its own
#     backtest is the strongest of any strategy tested, on every symbol
#     (e.g. BTC 4h Sharpe +4.5, SOL 4h +4.6) -- an explicit, INFORMED user
#     override, not an oversight: there is no established causal link
#     between planetary positions and markets, and a strong backtest with no
#     plausible mechanism is a classic overfitting red flag, but the user
#     chose to judge it purely on the numbers rather than exclude it on
#     priors. Flagged here so a future reader doesn't mistake this for the
#     honesty framing having been forgotten.
#   - trend_momentum: classic EMA(9/21)-crossover trend-following, ADX-gated.
#   - tsmom: time-series momentum, a barbell of a short + long lookback
#     horizon voting independently, grounded in Moskowitz/Ooi/Pedersen (2012)
#     and AQR's 2017 "Century of Evidence on Trend-Following Investing" --
#     genuinely different signal construction from trend_momentum (raw
#     multi-horizon price displacement vs. one EMA pair's spread), not a
#     re-parameterization of the same idea.
#
# Everything else -- mean_reversion, funding_carry, orderbook_imbalance,
# ml_signal, rahu -- was backtested and cut on RESULTS, then fully removed
# from this wiring (not just disabled) per an explicit request to stop
# computing/showing/backtesting strategies nobody's using:
#   - mean_reversion: never cleared breakeven even at its best timeframe.
#   - funding_carry: mixed at best, net not clearly good enough.
#   - ml_signal: its isolated-backtest numbers (800%+ returns) are
#     untrustworthy (near-certainly in-sample/overfit); its OWN honest
#     purged-walk-forward accuracy (see app/ml/train.py) was separately
#     measured at barely-above-coin-flip on average and NEGATIVE in the
#     most recent out-of-sample fold on all 3 symbols.
#   - orderbook_imbalance: can't be backtested at all (no stored order-book
#     history), so its live-only value was never actually proven.
#   - rahu: a simpler, single-factor time-gate made redundant by graha's much
#     deeper mechanism (which already includes its own Rahu-Kaal gate).
#
# Their files USED to be kept on disk on the grounds that "this project has no
# git, so deleting them would be unrecoverable". That premise no longer holds
# -- the project is version-controlled -- so as of 2026-09-04 the five cut
# strategies (mean_reversion, ml_signal, orderbook_imbalance, rahu,
# residual_momentum) are deleted rather than left as unwired modules that look
# live to anyone reading the directory. Recovering one is a git checkout of
# this commit's parent, not a rewrite.
#
# funding_carry is NOT in that list despite being cut from the ensemble: it is
# still active research, priced live by funding_carry_options.py, and is kept
# deliberately. ensemble.py is core machinery, not a strategy.
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_momentum": 1.0,
    "graha": 0.4,
    "tsmom": 0.8,
}


async def seed_default_strategy_configs() -> None:
    async with SessionLocal() as session:
        for name, weight in DEFAULT_WEIGHTS.items():
            stmt = sqlite_insert(StrategyConfig).values(name=name, enabled=True, weight=weight, params={})
            stmt = stmt.on_conflict_do_nothing(index_elements=["name"])
            await session.execute(stmt)
        # Removed strategies' leftover config rows (from before they were
        # cut) are deleted outright, not just left disabled -- so they stop
        # showing up in /api/strategies entirely, matching that they're no
        # longer part of the running system at all.
        await session.execute(delete(StrategyConfig).where(StrategyConfig.name.notin_(DEFAULT_WEIGHTS.keys())))
        await session.commit()


async def load_strategy_configs() -> dict[str, StrategyConfig]:
    async with SessionLocal() as session:
        result = await session.execute(select(StrategyConfig))
        return {row.name: row for row in result.scalars().all()}


async def build_ensemble_for_symbol(symbol: str, ml_resolution: str = "15m") -> SignalEnsemble:
    configs = await load_strategy_configs()

    def weight_of(name: str) -> tuple[float, bool]:
        cfg = configs.get(name)
        if cfg is None:
            return DEFAULT_WEIGHTS.get(name, 0.0), True
        return cfg.weight, cfg.enabled

    members = []
    for name, strategy in [
        ("trend_momentum", TrendMomentumStrategy()),
        ("graha", GrahaStrategy(symbol=symbol)),
        ("tsmom", TimeSeriesMomentumStrategy()),
    ]:
        weight, enabled = weight_of(name)
        members.append(WeightedStrategy(strategy=strategy, weight=weight, enabled=enabled))

    # Thresholds come from settings rather than SignalEnsemble's constructor
    # defaults, so the line that decides whether anything trades at all is
    # configurable without a code edit. See Settings.signal_deadband.
    settings = get_settings()
    return SignalEnsemble(
        members,
        deadband=settings.signal_deadband,
        smoothing_span=settings.signal_smoothing_span,
        exit_deadband=settings.signal_exit_deadband,
    )
