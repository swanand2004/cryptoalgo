"""Does the crypto composite dilute its own best factors, the way gold's did?

The full study found exactly two factors that hold up TRAIN -> VALIDATION ->
TEST on all three symbols independently:

    micro_signed_flow      BTC 51.8/52.2/52.2   ETH 51.7/52.5/52.2   SOL 52.1/51.1/50.9
    micro_close_location   BTC 52.5/52.2/51.6   ETH 51.7/51.3/51.6   SOL 51.2/51.6/50.9

Everything else -- dollar, real rates, breakevens, fear & greed, astrology, and
ten of the twelve microstructure columns -- sits inside noise of 50%.

Gold's lesson was that averaging one real signal with three noisy ones scored
44.6%, worse than the best component alone at 54.4%. This checks whether the
same dilution is happening here, and therefore what the 60% non-model weight
should actually contain: the ten survivors the validation split nominated, or
only the two that survived on every symbol.

Also measured: whether giving the model 40% helps or hurts, given its
out-of-fold accuracy is BELOW 50% on all three symbols.
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
from app.ml.dataset import make_forward_return_labels, purged_walk_forward_splits
from app.ml.train import LGBM_PARAMS

SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD")
HORIZON_BARS = 4
WARMUP = 250
Z_WINDOW = 720
Z_CAP = 2.5

# Signs fitted on TRAIN in the full study and frozen here.
CORE = {"micro_signed_flow": -1.0, "micro_close_location": -1.0}


def z(s, window=Z_WINDOW):
    m = s.rolling(window, min_periods=window // 5).mean()
    sd = s.rolling(window, min_periods=window // 5).std()
    return ((s - m) / sd.replace(0, np.nan)).clip(-Z_CAP, Z_CAP) / Z_CAP


def acc(sig, fwd, mask, min_abs=0.02):
    both = pd.DataFrame({"s": sig, "f": fwd})[mask].dropna()
    both = both[(both["s"].abs() > min_abs) & (both["f"] != 0)]
    if len(both) < 200:
        return None, len(both)
    return float((np.sign(both["s"]) == np.sign(both["f"])).mean()) * 100, len(both)


def band(n):
    return 50.0 / np.sqrt(max(n, 1)) * 2


async def study(symbol):
    raw = await load_candles_df(f"BINANCE:{symbol}", "1h", limit=200_000)
    feat = build_feature_frame(raw, symbol=symbol)
    grid = pd.DatetimeIndex(pd.to_datetime(feat["time"], unit="s", utc=True))
    feat = feat.set_index(grid)
    close = feat["close"].astype(float)
    fwd = close.shift(-HORIZON_BARS) / close - 1.0

    n = len(feat)
    b = int(n * 0.75)
    test = np.zeros(n, bool); test[b:] = True

    core = pd.DataFrame({k: z(feat[k]) * s for k, s in CORE.items()}).mean(axis=1)

    # model, out of fold
    fr = feat.reset_index(drop=True)
    labels = make_forward_return_labels(fr, HORIZON_BARS)
    ok = labels.notna() & fr[FEATURE_COLUMNS].notna().all(axis=1)
    X = fr.loc[ok, FEATURE_COLUMNS].reset_index(drop=True)
    y = labels.loc[ok].reset_index(drop=True)
    times = grid[ok.to_numpy()]
    oof = pd.Series(np.nan, index=range(len(X)), dtype=float)
    for tr_i, te_i in purged_walk_forward_splits(len(X), n_splits=5, embargo=HORIZON_BARS):
        m = lgb.train(LGBM_PARAMS, lgb.Dataset(X.iloc[tr_i], label=y.iloc[tr_i]), num_boost_round=200)
        oof.iloc[te_i] = m.predict(X.iloc[te_i])
    cov = oof.notna().to_numpy()
    model = pd.Series((oof[cov] / oof[cov].std()).clip(-Z_CAP, Z_CAP).to_numpy() / Z_CAP,
                      index=times[cov]).reindex(grid)

    lo, n_lo = acc(pd.Series(1.0, index=grid), fwd, test, min_abs=0.0)
    print(f"\n{symbol}  (TEST n={n_lo:,}, band +-{band(n_lo):.2f}, long-only {lo:.2f}%)")
    print(f"  {'variant':44s}{'TEST':>9s}{'vs long-only':>14s}")

    variants = {
        "signed_flow alone": z(feat["micro_signed_flow"]) * CORE["micro_signed_flow"],
        "close_location alone": z(feat["micro_close_location"]) * CORE["micro_close_location"],
        "the two core factors (equal weight)": core,
        "model alone (out of fold)": model,
    }
    for w in (0.4, 0.25, 0.1):
        variants[f"{w:.0%} model / {1-w:.0%} core"] = w * model.fillna(0.0) + (1 - w) * core.fillna(0.0)

    for name, sig in variants.items():
        a, nn = acc(sig, fwd, test)
        if a is None:
            print(f"  {name:44s}{'n/a':>9s}")
            continue
        d = a - lo
        flag = "" if abs(d) < band(nn) else "  *"
        print(f"  {name:44s}{a:>8.2f}%{d:>+13.2f}{flag}")
    print("  * = outside the noise band")


async def main():
    print("Does the composite dilute the two factors that actually survived?")
    for s in SYMBOLS:
        await study(s)


if __name__ == "__main__":
    asyncio.run(main())
