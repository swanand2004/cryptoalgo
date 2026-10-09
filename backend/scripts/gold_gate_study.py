"""The weight sweep found nothing. Do FILTERS do better than WEIGHTS?

Study 1 showed that tuning the model/macro weight buys +0.4 points on held-out
data inside a +-3.6 band -- i.e. nothing. That is the expected result when you
turn a continuous knob on two signals that are both near the noise floor.

A filter is a different kind of change. Instead of asking "how much of each
signal", it asks "when should we act at all". It trades COVERAGE for accuracy,
and unlike a weight it has an a-priori reason to work: two weak independent
signals agreeing is genuinely more informative than either alone, and that is
true before any measurement.

Tested here, all under the same split (choose on the first half, report on the
second, which the choice never touched):

  AGREEMENT   act only when the model and the macro composite point the same
              way. Halves coverage, should raise accuracy if the two are
              genuinely independent.
  CONVICTION  act only when |blend| clears a threshold.
  BOTH        agreement AND conviction.
  MACRO TILT  the dollar-vs-real-rate weight, run through the SAME split
              discipline -- the in-sample table in study 1 liked a dollar
              tilt, which is exactly the kind of thing that does not hold up.

Coverage is reported for every row. An accuracy that only exists on 4% of bars
is not an edge, it is a small sample, and the two are easy to confuse.
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


def trailing_z(s, window=FACTOR_Z_WINDOW):
    m = s.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 4)).std()
    return (s - m) / sd.replace(0, np.nan)


def to_score(z, sign):
    return (z.clip(-FACTOR_Z_CAP, FACTOR_Z_CAP) / FACTOR_Z_CAP) * sign


async def daily(sym):
    df = await load_candles_df(sym, "1d", limit=50_000)
    if df.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df["time"], unit="s", utc=True).dt.normalize()
    return pd.Series(df["close"].astype(float).to_numpy(), index=idx).groupby(level=0).last()


def stat(sig, fwd, mask, total):
    """accuracy, n, and coverage as a share of ALL bars in the period."""
    both = pd.DataFrame({"s": sig, "f": fwd})[mask].dropna()
    if len(both) < 40:
        return None, len(both), len(both) / max(total, 1) * 100
    a = float((np.sign(both["s"]) == np.sign(both["f"])).mean()) * 100
    return a, len(both), len(both) / max(total, 1) * 100


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

    oof = pd.Series(np.nan, index=range(len(X)), dtype=float)
    for tr, te in purged_walk_forward_splits(len(X), n_splits=cfg["n_splits"], embargo=hb):
        m = lgb.train(LGBM_PARAMS, lgb.Dataset(X.iloc[tr], label=y.iloc[tr]), num_boost_round=200)
        oof.iloc[te] = m.predict(X.iloc[te])

    cov = oof.notna()
    oof_t = pd.Series(oof[cov].to_numpy(), index=times[cov].to_numpy())
    fwd = pd.Series(y[cov].to_numpy(), index=times[cov].to_numpy())
    model = (oof_t / oof_t.std()).clip(-2.5, 2.5) / 2.5

    real = (await daily("MACRO:REALRATE10Y")).shift(FRED_LAG)
    dxy = (await daily("MACRO:DXY")).shift(FRED_LAG)
    fac = pd.DataFrame({
        "real_rate": to_score(trailing_z(real.diff(FACTOR_ROC_DAYS)), -1.0),
        "dollar": to_score(trailing_z(dxy.pct_change(FACTOR_ROC_DAYS) * 100), -1.0),
    })
    fac = fac.reindex(fac.index.union(oof_t.index)).ffill().reindex(oof_t.index)
    macro = fac.mean(axis=1)
    blend = 0.5 * model + 0.5 * macro

    cut = oof_t.index[len(oof_t) // 2]
    sel, rep = oof_t.index <= cut, oof_t.index > cut
    n_sel, n_rep = int(sel.sum()), int(rep.sum())
    print(f"bars: {len(oof_t):,} out-of-fold, {oof_t.index[0]:%Y-%m-%d} -> {oof_t.index[-1]:%Y-%m-%d}")
    print(f"selection half <= {cut:%Y-%m-%d} ({n_sel:,} bars) | report half ({n_rep:,} bars)\n")

    base = pd.Series(True, index=oof_t.index)
    agree = np.sign(model) == np.sign(macro)

    print("=" * 84)
    print("FILTERS -- accuracy vs COVERAGE (share of all bars the filter lets through)")
    print("=" * 84)
    print(f"{'filter':30s}{'SELECT acc':>12s}{'cov':>7s}{'REPORT acc':>12s}{'cov':>7s}{'n':>7s}")

    rows = []
    tests = [("none (current: every bar)", base, 0.0)]
    tests += [(f"conviction |blend| > {t:.2f}", base, t) for t in (0.10, 0.20, 0.30, 0.40)]
    tests += [("agreement only", agree, 0.0)]
    tests += [(f"agreement + |blend| > {t:.2f}", agree, t) for t in (0.10, 0.20, 0.30)]

    for name, cond, thr in tests:
        m = cond & (blend.abs() > thr) if thr else cond
        a_s, _, c_s = stat(blend, fwd, m & sel, n_sel)
        a_r, n_r, c_r = stat(blend, fwd, m & rep, n_rep)
        rows.append((name, a_s, a_r, n_r, c_r))
        f = lambda v: "n/a" if v is None else f"{v:.1f}%"
        print(f"{name:30s}{f(a_s):>12s}{c_s:>6.0f}%{f(a_r):>12s}{c_r:>6.0f}%{n_r:>7,}")

    print(f"\n{'=' * 84}")
    print("MACRO TILT -- the dollar lean study 1 liked, run through the SAME split")
    print("=" * 84)
    print(f"{'real_rate wt':>13s}{'SELECT':>10s}{'REPORT':>10s}{'n':>8s}")
    tilt = []
    for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
        combo = w * fac["real_rate"] + (1 - w) * fac["dollar"]
        b2 = 0.5 * model + 0.5 * combo
        a_s, _, _ = stat(b2, fwd, sel & (b2.abs() > 0.02), n_sel)
        a_r, n_r, _ = stat(b2, fwd, rep & (b2.abs() > 0.02), n_rep)
        tilt.append((w, a_s, a_r, n_r))
        f = lambda v: "n/a" if v is None else f"{v:.1f}%"
        mark = "  <- current" if abs(w - 0.5) < 1e-9 else ""
        print(f"{w:>13.1f}{f(a_s):>10s}{f(a_r):>10s}{n_r:>8,}{mark}")

    print(f"\n{'=' * 84}")
    print("VERDICT")
    print("=" * 84)
    good = [r for r in rows if r[1] is not None and r[2] is not None]
    cur = good[0]
    best = max(good[1:], key=lambda r: r[1]) if len(good) > 1 else None
    if best:
        nb = band(best[3])
        print(f"  current (no filter):  {cur[2]:.1f}% on {cur[4]:.0f}% of bars")
        print(f"  best on SELECT half:  '{best[0]}'")
        print(f"     -> {best[2]:.1f}% on the report half, {best[4]:.0f}% coverage, n={best[3]:,}")
        print(f"     noise band at that n: +-{nb:.1f} points")
        d = best[2] - cur[2]
        print(f"     change vs current: {d:+.1f} points -- "
              f"{'INSIDE the noise band, i.e. not a real gain' if abs(d) < nb else 'outside the noise band'}")
    t_good = [r for r in tilt if r[1] is not None and r[2] is not None]
    if t_good:
        bt = max(t_good, key=lambda r: r[1])
        ct = next(r for r in t_good if abs(r[0] - 0.5) < 1e-9)
        print(f"\n  macro tilt chosen on SELECT: real_rate wt {bt[0]:.1f} -> {bt[2]:.1f}% on report half")
        print(f"  current 0.5 -> {ct[2]:.1f}%   ({bt[2] - ct[2]:+.1f} points, band +-{band(bt[3]):.1f})")


if __name__ == "__main__":
    asyncio.run(main())
