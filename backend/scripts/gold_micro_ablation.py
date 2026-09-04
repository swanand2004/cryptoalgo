"""Do the microstructure features actually make the gold models better?

An ablation, not a demonstration. Identical purged walk-forward folds, identical
LightGBM parameters, identical labels -- the ONLY difference between the two
runs is whether the 12 microstructure columns are in the feature matrix. Any
difference in the result is therefore attributable to them and to nothing else.

Reported per horizon:
  dir_acc     directional accuracy, FLAT bars excluded (see train.py: PAXG
              prints 10.7% exactly-zero 4-bar forward returns at 15m, and
              counting them as errors understated every intraday figure by
              roughly ten points)
  decidable   accuracy on the bars the model is confident enough to call
  IC          Spearman rank correlation of prediction against outcome
  long-only   the share of moved bars that went UP -- the number to beat.
              Gold drifts. A model that cannot beat "always long" has not
              earned the right to pick sides, however far above 50% it looks.

The last row is the one that matters and is the reason this script prints it
next to the model: 50% is not the bar for a drifting asset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from app.data.query import load_candles_df
from app.gold.features import GOLD_FEATURE_COLUMNS, build_gold_feature_frame
from app.gold.ingest import GOLD_SPOT_SYMBOL
from app.gold.train import GOLD_HORIZONS, build_gold_training_frame, gold_walk_forward_validate

MICRO = [c for c in GOLD_FEATURE_COLUMNS if c.startswith("micro_")]
BASE = [c for c in GOLD_FEATURE_COLUMNS if not c.startswith("micro_")]


def summarise(folds):
    return (
        float(np.mean([f.directional_accuracy for f in folds])) * 100,
        float(np.mean([f.decidable_directional_accuracy for f in folds])) * 100,
        float(np.mean([f.information_coefficient for f in folds])),
    )


async def main():
    print(f"{len(BASE)} base features, {len(MICRO)} microstructure features added\n")
    for horizon, cfg in GOLD_HORIZONS.items():
        raw = await load_candles_df(GOLD_SPOT_SYMBOL, cfg["resolution"], limit=cfg["limit"])
        if raw.empty:
            print(f"{horizon}: no data")
            continue
        feat = build_gold_feature_frame(raw)
        hb = cfg["horizon_bars"]

        print("=" * 78)
        print(f"{horizon}  ({cfg['resolution']}, {hb} bars ahead)")
        print("=" * 78)

        results = {}
        for label, cols in (("without micro", BASE), ("with micro", GOLD_FEATURE_COLUMNS)):
            X, y = build_gold_training_frame(feat, hb, cfg["warmup_bars"], cols)
            folds, _ = gold_walk_forward_validate(X, y, hb, n_splits=cfg["n_splits"])
            results[label] = summarise(folds)
            print(f"  {label:16s} n={len(X):>7,}  dir_acc {results[label][0]:5.2f}%   "
                  f"decidable {results[label][1]:5.2f}%   IC {results[label][2]:+.4f}")

            moved = y[np.sign(y) != 0]
            if label == "with micro":
                long_only = float((moved > 0).mean()) * 100
                band = 50.0 / np.sqrt(len(moved)) * 2
                print(f"  {'long-only baseline':16s} n={len(moved):>7,}  dir_acc {long_only:5.2f}%"
                      f"   (2-sigma band +-{band:.2f})")
                d = results["with micro"][0] - results["without micro"][0]
                print(f"\n  micro contributes {d:+.2f} points of directional accuracy "
                      f"({'inside' if abs(d) < band else 'OUTSIDE'} the noise band)")
                beat = results["with micro"][0] - long_only
                print(f"  best model vs long-only baseline: {beat:+.2f} points")
        print()


if __name__ == "__main__":
    asyncio.run(main())
