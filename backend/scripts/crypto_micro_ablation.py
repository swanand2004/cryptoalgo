"""Do the microstructure features help the CRYPTO models?

The same ablation run for gold, on the same terms, against the deep Binance
history (79,170 hourly bars per symbol -- nine years) rather than the ~2,700
hourly bars Delta itself retains.

Identical purged walk-forward folds, identical LightGBM parameters, identical
labels. The only difference between the two runs is whether the 12 OHLCV
microstructure columns are in the feature matrix.

Two things this prints that the crypto pipeline has never printed before:

  long-only baseline -- the share of moved bars that closed UP. BTC is not a
      coin flip either: it drifts, and over these samples 51-53% of bars close
      higher. A model at 52% has not beaten anything. This is the number the
      gold work should have been compared against from the start, and getting
      it wrong there is exactly why it is printed here.

  flat %  -- the share of exactly-zero forward returns, which used to be
      scored as errors. Small for crypto, unlike gold; shown so the claim
      "small for crypto" is visible rather than asserted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from app.data.query import load_candles_df
from app.features.feature_pipeline import FEATURE_COLUMNS, FEATURE_COLUMNS_WITH_MICRO, build_feature_frame
from app.ml.dataset import make_forward_return_labels
from app.ml.train import walk_forward_validate

SYMBOLS = ("BINANCE:BTCUSD", "BINANCE:ETHUSD", "BINANCE:SOLUSD")
RESOLUTION = "1h"
HORIZON_BARS = 4  # ~4 hours ahead, matching the live 1h config
WARMUP = 250


def frame_for(feat, cols, horizon_bars):
    labels = make_forward_return_labels(feat, horizon_bars)
    df = feat.iloc[WARMUP:]
    lab = labels.iloc[WARMUP:]
    valid = lab.notna() & df[cols].notna().all(axis=1)
    return (
        df.loc[valid, cols].reset_index(drop=True),
        lab.loc[valid].reset_index(drop=True),
    )


async def main():
    print(f"{len(FEATURE_COLUMNS)} base features, "
          f"{len(FEATURE_COLUMNS_WITH_MICRO) - len(FEATURE_COLUMNS)} microstructure added")
    print(f"{RESOLUTION} bars, {HORIZON_BARS} ahead, deep Binance history\n")

    for symbol in SYMBOLS:
        raw = await load_candles_df(symbol, RESOLUTION, limit=200_000)
        if raw.empty or len(raw) < 5_000:
            print(f"{symbol}: only {len(raw)} bars, skipping")
            continue
        feat = build_feature_frame(raw, symbol=symbol.split(":")[-1])

        print("=" * 74)
        print(f"{symbol}  ({len(raw):,} bars)")
        print("=" * 74)

        accs = {}
        for label, cols in (("without micro", FEATURE_COLUMNS), ("with micro", FEATURE_COLUMNS_WITH_MICRO)):
            X, y = frame_for(feat, cols, HORIZON_BARS)
            folds, _ = walk_forward_validate(X, y, HORIZON_BARS)
            acc = float(np.mean([f.directional_accuracy for f in folds])) * 100
            dec = float(np.mean([f.decidable_directional_accuracy for f in folds])) * 100
            ic = float(np.mean([f.information_coefficient for f in folds]))
            accs[label] = acc
            print(f"  {label:16s} n={len(X):>7,}  dir_acc {acc:5.2f}%   decidable {dec:5.2f}%   IC {ic:+.4f}")

        moved = y[np.sign(y) != 0]
        long_only = float((moved > 0).mean()) * 100
        flat = (1 - len(moved) / len(y)) * 100
        band = 50.0 / np.sqrt(len(moved)) * 2
        print(f"  {'long-only':16s} n={len(moved):>7,}  dir_acc {long_only:5.2f}%   "
              f"(flat {flat:.2f}%, 2-sigma band +-{band:.2f})")

        d = accs["with micro"] - accs["without micro"]
        best = max(accs.values())
        print(f"\n  micro contributes {d:+.2f} points "
              f"({'inside' if abs(d) < band else 'OUTSIDE'} the noise band)")
        print(f"  best model vs long-only: {best - long_only:+.2f} points\n")


if __name__ == "__main__":
    asyncio.run(main())
