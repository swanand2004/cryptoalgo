"""Which weights actually improve gold's accuracy?

The blend is currently 50% model / 50% macro, with macro being an equal-weight
mix of real rates and the dollar. Nothing about those numbers was measured --
50/50 was requested, and equal-weight was the obvious default. This measures
what the weights should be.

DISCIPLINE, because a weight sweep is exactly where overfitting lives:

  OUT-OF-FOLD model scores. The model's score series comes from purged
      walk-forward folds, so every score was produced by a model that never
      saw that bar. Using a model's in-sample predictions here would make any
      weight that leans on the model look good.
  PUBLICATION LAGS on the macro side (FRED +1 day, CFTC +4).
  TRAILING z-scores only.
  SPLIT SELECTION FROM REPORTING. The OOF period is cut in half: the weight
      is chosen on the first half and reported on the second, which it never
      touched. The best in-sample weight is reported alongside so the gap
      between the two is visible -- that gap IS the overfitting.
"""
import sys

sys.path.insert(0, r"C:\Users\swana\Desktop\cryptoalgo\backend")

import asyncio
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from app.data.query import load_candles_df
from app.gold.features import GOLD_FEATURE_COLUMNS, GOLD_WARMUP_BARS, build_gold_feature_frame
from app.gold.ingest import GOLD_SPOT_SYMBOL
from app.gold.macro_factors import CFTC_SYMBOL, FACTOR_ROC_DAYS, FACTOR_Z_CAP, FACTOR_Z_WINDOW
from app.gold.train import GOLD_HORIZONS, LGBM_PARAMS
from app.ml.dataset import make_forward_return_labels, purged_walk_forward_splits

FRED_LAG, CFTC_LAG = 1, 4
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def trailing_z(s, window=FACTOR_Z_WINDOW):
    mean = s.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 4)).std()
    return (s - mean) / sd.replace(0, np.nan)


def to_score(z, sign):
    return (z.clip(-FACTOR_Z_CAP, FACTOR_Z_CAP) / FACTOR_Z_CAP) * sign


async def daily(symbol, resolution="1d"):
    df = await load_candles_df(symbol, resolution, limit=50_000)
    if df.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df["time"], unit="s", utc=True).dt.normalize()
    return pd.Series(df["close"].astype(float).to_numpy(), index=idx).groupby(level=0).last()


def acc(signal, fwd, min_abs=1e-9):
    both = pd.DataFrame({"s": signal, "f": fwd}).dropna()
    both = both[both["s"].abs() > min_abs]
    if len(both) < 60:
        return None, len(both)
    return float((np.sign(both["s"]) == np.sign(both["f"])).mean()) * 100, len(both)


def band(n):
    return 50.0 / np.sqrt(max(n, 1)) * 2


async def main():
    cfg = GOLD_HORIZONS["long_term"]
    hb = cfg["horizon_bars"]

    raw = await load_candles_df(GOLD_SPOT_SYMBOL, "1d", limit=cfg["limit"])
    feat = build_gold_feature_frame(raw)
    labels = make_forward_return_labels(feat, hb)

    df = feat.iloc[GOLD_WARMUP_BARS:].copy()
    lab = labels.iloc[GOLD_WARMUP_BARS:]
    valid = lab.notna() & df[GOLD_FEATURE_COLUMNS].notna().all(axis=1)
    X = df.loc[valid, GOLD_FEATURE_COLUMNS].reset_index(drop=True)
    y = lab.loc[valid].reset_index(drop=True)
    times = pd.to_datetime(df.loc[valid, "time"], unit="s", utc=True).dt.normalize().reset_index(drop=True)
    print(f"daily gold: {len(X):,} usable bars, {times.iloc[0]:%Y-%m-%d} -> {times.iloc[-1]:%Y-%m-%d}")
    print(f"label: {hb}-day forward return\n")

    # ---- out-of-fold model scores ---------------------------------------
    oof = pd.Series(np.nan, index=range(len(X)), dtype=float)
    for fold_i, (tr, te) in enumerate(purged_walk_forward_splits(len(X), n_splits=cfg["n_splits"], embargo=hb)):
        model = lgb.train(LGBM_PARAMS, lgb.Dataset(X.iloc[tr], label=y.iloc[tr]), num_boost_round=200)
        oof.iloc[te] = model.predict(X.iloc[te])
        print(f"  fold {fold_i}: trained {len(tr):,} -> predicted {len(te):,}")

    covered = oof.notna()
    oof_t = pd.Series(oof[covered].to_numpy(), index=times[covered].to_numpy())
    fwd_t = pd.Series(y[covered].to_numpy(), index=times[covered].to_numpy())
    # Scale to [-1,1] the way the live score is, so the weights mean the same
    # thing here as they do in blend_score.
    model_score = (oof_t / oof_t.std()).clip(-2.5, 2.5) / 2.5

    a, n = acc(model_score, fwd_t)
    print(f"\nMODEL alone (out-of-fold): {a:.1f}% on {n:,} bars, noise band +-{band(n):.1f}\n")

    # ---- macro factor scores on the same dates ---------------------------
    real = (await daily("MACRO:REALRATE10Y")).shift(FRED_LAG)
    dxy = (await daily("MACRO:DXY")).shift(FRED_LAG)
    be = (await daily("MACRO:BREAKEVEN10Y")).shift(FRED_LAG)
    cftc = (await daily(CFTC_SYMBOL)).shift(CFTC_LAG)

    fac = pd.DataFrame({
        "real_rate": to_score(trailing_z(real.diff(FACTOR_ROC_DAYS)), -1.0),
        "dollar": to_score(trailing_z(dxy.pct_change(FACTOR_ROC_DAYS) * 100), -1.0),
        "breakeven": to_score(trailing_z(be), 1.0),
        "cftc": to_score(trailing_z(cftc, window=104), -1.0),
    })
    fac = fac.reindex(fac.index.union(oof_t.index)).ffill().reindex(oof_t.index)

    print("=" * 76)
    print("EACH PIECE ALONE, on the same bars (so the numbers are comparable)")
    print("=" * 76)
    print(f"{'signal':16s}{'accuracy':>11s}{'n':>9s}")
    for c in fac.columns:
        a, n = acc(fac[c], fwd_t, min_abs=0.05)
        print(f"{c:16s}{('n/a' if a is None else f'{a:.1f}%'):>11s}{n:>9,}")
    a, n = acc(model_score, fwd_t)
    print(f"{'model (OOF)':16s}{a:>10.1f}%{n:>9,}")

    # ---- how should the macro half be weighted internally? ---------------
    print(f"\n{'=' * 76}")
    print("MACRO INTERNAL WEIGHT: real_rate vs dollar (currently 50/50)")
    print("=" * 76)
    print(f"{'real_rate wt':>13s}{'dollar wt':>11s}{'accuracy':>11s}{'n':>9s}")
    macro_rows = {}
    for w in WEIGHTS:
        combo = w * fac["real_rate"] + (1 - w) * fac["dollar"]
        a, n = acc(combo, fwd_t, min_abs=0.05)
        macro_rows[w] = a
        print(f"{w:>13.1f}{1 - w:>11.1f}{('n/a' if a is None else f'{a:.1f}%'):>11s}{n:>9,}")

    # ---- the headline sweep, split so selection != reporting -------------
    cut = oof_t.index[len(oof_t) // 2]
    print(f"\n{'=' * 76}")
    print("MODEL WEIGHT SWEEP (currently 0.5)")
    print("=" * 76)
    print(f"selection half: up to {cut:%Y-%m-%d}   |   report half: after it\n")
    print(f"{'model wt':>9s}{'macro wt':>10s}{'SELECT':>10s}{'REPORT':>10s}{'n rep':>8s}")

    macro_eq = fac[["real_rate", "dollar"]].mean(axis=1)
    sel_mask, rep_mask = oof_t.index <= cut, oof_t.index > cut
    rows = []
    for w in WEIGHTS:
        blend = w * model_score + (1 - w) * macro_eq
        a_s, _ = acc(blend[sel_mask], fwd_t[sel_mask], min_abs=0.02)
        a_r, n_r = acc(blend[rep_mask], fwd_t[rep_mask], min_abs=0.02)
        rows.append((w, a_s, a_r, n_r))
        mark = "  <- current" if abs(w - 0.5) < 1e-9 else ""
        print(f"{w:>9.1f}{1 - w:>10.1f}"
              f"{('n/a' if a_s is None else f'{a_s:.1f}%'):>10s}"
              f"{('n/a' if a_r is None else f'{a_r:.1f}%'):>10s}{n_r:>8,}{mark}")

    valid_rows = [r for r in rows if r[1] is not None and r[2] is not None]
    if valid_rows:
        best_sel = max(valid_rows, key=lambda r: r[1])
        best_rep = max(valid_rows, key=lambda r: r[2])
        cur = next(r for r in valid_rows if abs(r[0] - 0.5) < 1e-9)
        nb = band(best_sel[3])
        print(f"\n  chosen on SELECT half:  w={best_sel[0]:.1f} -> {best_sel[1]:.1f}% there, "
              f"{best_sel[2]:.1f}% on the held-out half")
        print(f"  current setting:        w=0.5 -> {cur[2]:.1f}% on the held-out half")
        print(f"  best possible in HINDSIGHT on the report half: w={best_rep[0]:.1f} "
              f"-> {best_rep[2]:.1f}%  (not available in advance)")
        print(f"\n  noise band on the report half: +-{nb:.1f} points.")
        print(f"  honest gain from re-weighting: {best_sel[2] - cur[2]:+.1f} points "
              f"({'INSIDE' if abs(best_sel[2] - cur[2]) < nb else 'outside'} the noise band)")


if __name__ == "__main__":
    asyncio.run(main())
