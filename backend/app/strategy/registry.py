from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.config import get_settings
from app.data.db import SessionLocal
from app.data.models import StrategyConfig
from app.strategy.ensemble import SignalEnsemble, WeightedStrategy
from app.strategy.gann_angle import GannAngleStrategy
from app.strategy.graha import GrahaStrategy
from app.strategy.time_series_momentum import TimeSeriesMomentumStrategy
from app.strategy.trend_ladder import TrendLadderStrategy
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
# --------------------------------------------------------------------------
# 2026-09-07: the lineup above is superseded by a single strategy.
#
# `trend_ladder` (app/strategy/trend_ladder.py) is the configuration selected
# by the nine-year cross-symbol study: chosen on a 2022-2023 validation window,
# confirmed by rolling annual walk-forward, and evaluated once on a 2024-2026
# holdout. On BTC/ETH/SOL as one book, net of 7bps a side and real funding, at
# a matched 15% vol target it scores Sharpe 1.33 / PF 2.31 / maxDD -16.3% and
# is positive in all nine calendar years.
#
# It runs ALONE rather than blended with the previous three, and that is the
# deliberate part. The study's core finding is that averaging a validated
# signal with unvalidated ones destroys it -- the same dilution effect already
# recorded for the gold blend, where an equal-weight composite scored 44.6%
# against its own best component's 54.4%. trend_momentum, graha and tsmom have
# never been measured on this book, these costs, or this three-way split, so
# giving them weight would trade a measured 1.33 Sharpe for an unmeasured one.
#
# The three are NOT deleted -- their modules and imports are untouched and
# reverting is this dict, nothing else:
#
#     DEFAULT_WEIGHTS = {"trend_momentum": 1.0, "graha": 0.4, "tsmom": 0.8}
#
# Note that seed_default_strategy_configs deletes config rows for any strategy
# not named here, so switching this dict genuinely stops the others trading
# rather than leaving them enabled in the database.
#
# graha in particular was an explicit, informed user override kept on its
# backtest numbers alone. It is set aside here on the same principle that
# justified keeping it -- judge on measurement -- not on priors about
# astrology: its backtest predates the 2026-08-31 engine fixes, which inflated
# every result by 30-50x, and it has never been re-measured since.
#
# 2026-10-02: gann_angle (W.D. Gann's 1x1 angle off his 2-day swing chart)
# joins at EQUAL weight -- a 50/50 signal blend, since the ensemble normalises
# by total weight. Equal rather than 0.5/0.5 because an existing database row
# keeps trend_ladder's stored 1.0 (seeding never overwrites), and 1.0/0.5
# would silently run a 67/33 blend nobody tested. Shipped code path, hourly,
# live thresholds, 7 bps + funding, 50% vol book:
#
#                    TRAIN <=2021      VAL 2022-23      TEST 2024-26
#   ladder alone     1.58 / -34%       0.85 / -45%      0.92 / -43%    (Sharpe / maxDD)
#   ladder + gann    1.56 / -31%       1.10 / -31%      1.14 / -35%
#
# 50/50 was fixed before testing, not tuned. See app/strategy/gann_angle.py for
# the Gann methods that were tested and rejected. Reverting is this dict.
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_ladder": 1.0,
    "gann_angle": 1.0,
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

    available: dict[str, object] = {
        "trend_ladder": TrendLadderStrategy(),
        "gann_angle": GannAngleStrategy(),
        # Retained and constructible so flipping DEFAULT_WEIGHTS back is a
        # one-line change, but not wired in unless named there.
        "trend_momentum": TrendMomentumStrategy(),
        "graha": GrahaStrategy(symbol=symbol),
        "tsmom": TimeSeriesMomentumStrategy(),
    }

    members = []
    for name in DEFAULT_WEIGHTS:
        strategy = available.get(name)
        if strategy is None:
            continue
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
