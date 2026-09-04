"""Which non-ML factors deserve a vote in the crypto blend?

The request is a gold-style blend: 40% model, 60% from micro / macro / news /
events / astrology. The 40/60 split is a design decision and is implemented as
asked. WHICH factors fill the 60% is not a design decision -- it is a
measurement, and this is it.

The gold work is the reason. There, an equal-weight composite of all four macro
factors scored 44.6% -- WORSE than its own best component at 54.4% -- because
averaging one real signal with three noisy ones destroys it. Wiring five
buckets at equal weight because they were listed in a request would repeat that
exactly.

DISCIPLINE, three-way and chronological, each stage used once:

    TRAIN       fit the sign of factors that have no a-priori direction.
    VALIDATION  decide which factors survive.
    TEST        touched once, to report.

Where theory gives a sign it is COMMITTED before looking:
    stronger dollar        -> bearish  (risk asset priced in USD)
    rising real rates      -> bearish  (discount rate on long-duration risk)
    extreme greed          -> bearish  (contrarian; crowded positioning)
    high funding           -> bearish  (longs paying to hold = crowded trade)
Astrology and the microstructure block get their signs from TRAIN, because
there is no theory to commit to -- which is itself worth stating rather than
pretending a direction was known in advance.

NEWS IS ABSENT and cannot be fixed here: the NEWS: series holds 74 hourly bars,
four days. Four days cannot validate anything, so news is shown on the page and
given zero weight, the same treatment gold's breakeven and CFTC factors get.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from app.data.query import load_candles_df
from app.features.feature_pipeline import FEATURE_COLUMNS, build_feature_frame
from app.features.microstructure import OHLCV_MICRO_COLUMNS
from app.ml.dataset import make_forward_return_labels, purged_walk_forward_splits
from app.ml.train import LGBM_PARAMS

SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD")
HORIZON_BARS = 4          # ~4h ahead, matching the live 1h config
WARMUP = 250
Z_WINDOW = 720            # ~30 days of hourly bars
Z_CAP = 2.5
FRED_LAG_DAYS = 1         # FRED publishes the following business day
MODEL_WEIGHT = 0.40       # as requested

# Committed before measuring. See module docstring.
COMMITTED_SIGNS = {
    "dollar_change": -1.0,
    "real_rate_change": -1.0,
    "breakeven_inflation": +1.0,
    "fear_greed": -1.0,
    "funding_crowding": -1.0,
}


def z(s: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    m = s.rolling(window, min_periods=window // 5).mean()
    sd = s.rolling(window, min_periods=window // 5).std()
    return ((s - m) / sd.replace(0, np.nan)).clip(-Z_CAP, Z_CAP) / Z_CAP


def acc(sig: pd.Series, fwd: pd.Series, mask, min_abs=0.05):
    both = pd.DataFrame({"s": sig, "f": fwd})[mask].dropna()
    both = both[(both["s"].abs() > min_abs) & (both["f"] != 0)]
    if len(both) < 200:
        return None, len(both)
    return float((np.sign(both["s"]) == np.sign(both["f"])).mean()) * 100, len(both)


def band(n):
    return 50.0 / np.sqrt(max(n, 1)) * 2


async def daily_series(symbol: str) -> pd.Series:
    df = await load_candles_df(symbol, "1d", limit=50_000)
    if df.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df["time"], unit="s", utc=True).dt.normalize()
    return pd.Series(df["close"].astype(float).to_numpy(), index=idx).groupby(level=0).last()


async def series_on(symbol: str, resolution: str, grid: pd.DatetimeIndex) -> pd.Series:
    """Any series, forward-filled onto the hourly grid. Never interpolated
    forward -- only the last ALREADY-PUBLISHED value is carried."""
    df = await load_candles_df(symbol, resolution, limit=200_000)
    if df.empty:
        return pd.Series(np.nan, index=grid)
    idx = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s", utc=True))
    s = pd.Series(df["close"].astype(float).to_numpy(), index=idx).groupby(level=0).last()
    return s.reindex(s.index.union(grid)).ffill().reindex(grid)


async def build_factors(symbol: str, feat: pd.DataFrame, grid: pd.DatetimeIndex) -> dict[str, pd.Series]:
    factors: dict[str, pd.Series] = {}

    # --- macro (daily FRED, lagged for publication) ----------------------
    dxy = (await daily_series("MACRO:DXY")).shift(FRED_LAG_DAYS)
    real = (await daily_series("MACRO:REALRATE10Y")).shift(FRED_LAG_DAYS)
    be = (await daily_series("MACRO:BREAKEVEN10Y")).shift(FRED_LAG_DAYS)

    def onto(daily: pd.Series) -> pd.Series:
        if daily.empty:
            return pd.Series(np.nan, index=grid)
        return daily.reindex(daily.index.union(grid)).ffill().reindex(grid)

    factors["dollar_change"] = z(onto(dxy).pct_change(24 * 5) * 100)
    factors["real_rate_change"] = z(onto(real).diff(24 * 5))
    factors["breakeven_inflation"] = z(onto(be))

    # --- sentiment -------------------------------------------------------
    fng = await daily_series("FNG:GLOBAL")
    factors["fear_greed"] = z(onto(fng))

    # --- funding crowding ------------------------------------------------
    funding = await series_on(f"BFUNDHIST:{symbol}", "8h", grid)
    factors["funding_crowding"] = z(funding)

    # --- events ----------------------------------------------------------
    if "macro_event_proximity" in feat.columns:
        factors["event_proximity"] = z(feat["macro_event_proximity"])

    # --- astrology -------------------------------------------------------
    if "astro_all_pairs_aspect_score" in feat.columns:
        factors["astrology"] = z(feat["astro_all_pairs_aspect_score"])

    # --- microstructure, as ONE bucket (mean of the 12, each signed on TRAIN)
    for col in OHLCV_MICRO_COLUMNS:
        if col in feat.columns:
            factors[f"micro::{col}"] = z(feat[col])
    return factors


async def study(symbol: str):
    raw = await load_candles_df(f"BINANCE:{symbol}", "1h", limit=200_000)
    if raw.empty or len(raw) < 10_000:
        print(f"{symbol}: not enough history")
        return
    feat = build_feature_frame(raw, symbol=symbol)
    grid = pd.DatetimeIndex(pd.to_datetime(feat["time"], unit="s", utc=True))
    feat = feat.set_index(grid)

    close = feat["close"].astype(float)
    fwd = close.shift(-HORIZON_BARS) / close - 1.0
    factors = await build_factors(symbol, feat, grid)

    n = len(feat)
    a, b = int(n * 0.5), int(n * 0.75)
    train = np.zeros(n, bool); train[WARMUP:a] = True
    val = np.zeros(n, bool); val[a:b] = True
    test = np.zeros(n, bool); test[b:] = True

    print("=" * 96)
    print(f"{symbol}   {n:,} hourly bars   {grid[0]:%Y-%m-%d} -> {grid[-1]:%Y-%m-%d}   ({HORIZON_BARS}h ahead)")
    print(f"  TRAIN {int(train.sum()):,} | VALIDATION {int(val.sum()):,} | TEST {int(test.sum()):,}")
    print("=" * 96)

    long_only, n_lo = acc(pd.Series(1.0, index=grid), fwd, test, min_abs=0.0)
    print(f"\n  long-only baseline on TEST: {long_only:.2f}%  (n={n_lo:,}, band +-{band(n_lo):.2f})\n")

    print(f"  {'factor':30s}{'sign':>6s}{'src':>8s}{'TRAIN':>9s}{'VALID':>9s}{'TEST':>9s}{'n':>9s}")
    signs, rows = {}, []
    for name, series in factors.items():
        a_raw, _ = acc(series, fwd, train)
        if a_raw is None:
            continue
        if name in COMMITTED_SIGNS:
            sign, src = COMMITTED_SIGNS[name], "theory"
        else:
            sign, src = (1.0 if a_raw >= 50 else -1.0), "train"
        signs[name] = sign
        s = series * sign
        a_tr, _ = acc(s, fwd, train)
        a_v, _ = acc(s, fwd, val)
        a_te, n_te = acc(s, fwd, test)
        rows.append((name, a_v, a_te, n_te))
        f = lambda x: "n/a" if x is None else f"{x:.1f}%"
        print(f"  {name:30s}{sign:>+6.0f}{src:>8s}{f(a_tr):>9s}{f(a_v):>9s}{f(a_te):>9s}{n_te:>9,}")

    # --- survivors chosen on VALIDATION only ------------------------------
    survivors = [r[0] for r in rows if r[1] is not None and r[1] > 50.0]
    print(f"\n  survivors on validation ({len(survivors)}): {', '.join(survivors) if survivors else 'NONE'}")
    if not survivors:
        print("  nothing earns a vote -- the 60% has no honest content\n")
        return

    signed = pd.DataFrame({nm: factors[nm] * signs[nm] for nm in survivors})
    composite = signed.mean(axis=1)
    a_v, _ = acc(composite, fwd, val, min_abs=0.02)
    a_te, n_te = acc(composite, fwd, test, min_abs=0.02)
    f = lambda x: "n/a" if x is None else f"{x:.2f}%"
    print(f"  {'factor composite':30s} VALID {f(a_v):>8s}  TEST {f(a_te):>8s}  n={n_te:,}")

    # --- the model half, out of fold --------------------------------------
    labels = make_forward_return_labels(feat.reset_index(drop=True), HORIZON_BARS)
    fr = feat.reset_index(drop=True)
    ok = labels.notna() & fr[FEATURE_COLUMNS].notna().all(axis=1)
    X = fr.loc[ok, FEATURE_COLUMNS].reset_index(drop=True)
    y = labels.loc[ok].reset_index(drop=True)
    times = grid[ok.to_numpy()]

    oof = pd.Series(np.nan, index=range(len(X)), dtype=float)
    for tr_i, te_i in purged_walk_forward_splits(len(X), n_splits=5, embargo=HORIZON_BARS):
        mdl = lgb.train(LGBM_PARAMS, lgb.Dataset(X.iloc[tr_i], label=y.iloc[tr_i]), num_boost_round=200)
        oof.iloc[te_i] = mdl.predict(X.iloc[te_i])
    cov = oof.notna().to_numpy()
    model = pd.Series((oof[cov] / oof[cov].std()).clip(-Z_CAP, Z_CAP).to_numpy() / Z_CAP, index=times[cov])
    model = model.reindex(grid)

    a_v, _ = acc(model, fwd, val, min_abs=0.02)
    a_te, n_te = acc(model, fwd, test, min_abs=0.02)
    print(f"  {'model (out of fold)':30s} VALID {f(a_v):>8s}  TEST {f(a_te):>8s}  n={n_te:,}")

    blend = MODEL_WEIGHT * model.fillna(0.0) + (1 - MODEL_WEIGHT) * composite.fillna(0.0)
    a_v, _ = acc(blend, fwd, val, min_abs=0.02)
    a_te, n_te = acc(blend, fwd, test, min_abs=0.02)
    print(f"  {f'blend {MODEL_WEIGHT:.0%} model / {1-MODEL_WEIGHT:.0%} factors':30s} "
          f"VALID {f(a_v):>8s}  TEST {f(a_te):>8s}  n={n_te:,}")
    if a_te is not None and long_only is not None:
        d = a_te - long_only
        print(f"\n  blend vs long-only: {d:+.2f} points "
              f"({'INSIDE noise' if abs(d) < band(n_te) else 'OUTSIDE noise'})\n")


async def main():
    for sym in SYMBOLS:
        await study(sym)


if __name__ == "__main__":
    asyncio.run(main())
