"""Micro factors for gold, done properly, at the SHORT horizons.

The earlier pass concluded "no order book, therefore no micro alpha" after
testing a single factor. That was too quick. There is an established family
of microstructure estimators built specifically for the case where you have
OHLCV and nothing else -- Roll's serial-covariance spread, the Corwin-Schultz
high-low spread, Amihud illiquidity, Kyle-lambda price impact. Dismissing
microstructure because the book is missing skipped the literature that exists
precisely for that situation.

So: 13 factors, on 15m and 1h bars, at the horizons that actually care.

DISCIPLINE. Most of these have no a-priori directional sign for gold -- unlike
"rising real rates are bearish", there is no theory saying which way a wide
Roll spread points. Choosing the sign from the data is legitimate ONLY if the
choice never touches the number being reported, so the sample is cut three
ways, chronologically, and each stage is used once:

    TRAIN       fix each factor's sign. Nothing is reported from here.
    VALIDATION  decide which factors survive, and how to combine them.
    TEST        touched exactly once, at the end, to report.

A factor that looks good on TRAIN and dies on VALIDATION is noise that got
lucky. A combination that looks good on VALIDATION and dies on TEST is the
selection itself overfitting -- which is what happened to the conviction
filter in the previous study (56.0% -> 46.6%), so it is the specific failure
this layout is built to catch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from app.data.query import load_candles_df
from app.gold.ingest import GOLD_SPOT_SYMBOL

Z_WINDOW = 500
Z_CAP = 2.5
MIN_ABS = 0.05


def zscore(s: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Trailing z-score. Never full-sample: a full-sample z tells every
    historical bar what the future distribution looks like."""
    m = s.rolling(window, min_periods=window // 5).mean()
    sd = s.rolling(window, min_periods=window // 5).std()
    return ((s - m) / sd.replace(0, np.nan)).clip(-Z_CAP, Z_CAP) / Z_CAP


def build_micro_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Everything computable from OHLCV that has a microstructure meaning."""
    o, h, l, c = (df[k].astype(float) for k in ("open", "high", "low", "close"))
    v = df["volume"].astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)
    ret = np.log(c).diff()
    rng = (h - l).replace(0, np.nan)
    f = {}

    # --- published spread / liquidity estimators ------------------------
    # Corwin & Schultz (2012): two-day high-low ratios separate the spread
    # from volatility, because volatility scales with the interval and the
    # spread does not.
    beta = (np.log(h / l) ** 2).rolling(2).sum()
    h2 = h.rolling(2).max()
    l2 = l.rolling(2).min()
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    cs = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)
    f["cs_spread"] = cs.rolling(24).mean()

    # Roll (1984): with bid-ask bounce, consecutive price changes are
    # negatively autocovariant, and the spread falls out of that covariance.
    dp = c.diff()
    cov = dp.rolling(48).cov(dp.shift(1))
    f["roll_spread"] = 2 * np.sqrt((-cov).clip(lower=0))

    # Amihud (2002): price move per unit of volume -- how much it costs to
    # push this market.
    f["amihud"] = (ret.abs() / v.replace(0, np.nan)).rolling(24).mean()

    # Kyle-lambda proxy: |move| regressed on signed volume. Same idea as
    # Amihud but using the SIGNED flow, so it measures directional impact.
    signed_vol = np.sign(ret) * v
    win = 48
    cov_kl = ret.rolling(win).cov(signed_vol)
    var_kl = signed_vol.rolling(win).var()
    f["kyle_lambda"] = cov_kl / var_kl.replace(0, np.nan)

    # --- flow proxies ---------------------------------------------------
    # Tick rule: with no trade-level data, the bar's own direction is the
    # standard stand-in for aggressor side. It is a PROXY, not order flow.
    f["signed_flow"] = (signed_vol.rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan))
    f["volume_shock"] = np.log(v / v.rolling(96).median().replace(0, np.nan))

    # --- intrabar structure ---------------------------------------------
    # Where in its own range the bar closed: the classic pressure read.
    f["close_location"] = ((c - l) - (h - c)) / rng
    # Which side got rejected -- long upper wick means sellers defended.
    f["wick_asymmetry"] = ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / rng
    f["body_ratio"] = (c - o).abs() / rng

    # --- return dynamics ------------------------------------------------
    # Lag-1 autocorrelation: negative is bid-ask bounce (mean-reverting
    # microstructure), positive is genuine short-horizon momentum.
    f["ret_autocorr"] = ret.rolling(96).corr(ret.shift(1))
    rv = ret.rolling(24).std()
    f["vol_of_vol"] = rv.rolling(96).std()
    # Realized variance vs bipower variation: the gap is jump activity,
    # which continuous-diffusion models mis-price.
    bpv = (ret.abs() * ret.abs().shift(1)).rolling(24).sum() * (np.pi / 2)
    f["jump_ratio"] = ((ret**2).rolling(24).sum() - bpv) / (ret**2).rolling(24).sum().replace(0, np.nan)
    f["vol_term_structure"] = ret.rolling(24).std() / ret.rolling(168).std().replace(0, np.nan)

    out = pd.DataFrame(f, index=df.index)
    return out.apply(zscore)


def acc(sig: pd.Series, fwd: pd.Series, mask, min_abs=MIN_ABS):
    """Directional accuracy, with FLAT bars removed.

    7.2% of PAXG's hourly forward returns are exactly zero -- stale prints on
    a thinly traded token. np.sign(0) is 0, so a flat bar scores as WRONG for
    a signal and for its exact opposite at the same time. Left in, they
    subtract a constant ~7 points from every factor, which silently moves the
    coin-flip line from 50% down to ~46% and makes every "below 50%, therefore
    no edge" reading meaningless. They are also untradeable: there is no
    direction to get right on a bar that did not move.
    """
    both = pd.DataFrame({"s": sig, "f": fwd})[mask].dropna()
    both = both[(both["s"].abs() > min_abs) & (both["f"] != 0)]
    if len(both) < 200:
        return None, len(both)
    return float((np.sign(both["s"]) == np.sign(both["f"])).mean()) * 100, len(both)


def band(n):
    return 50.0 / np.sqrt(max(n, 1)) * 2


async def study(resolution: str, limit: int, horizons: tuple[int, ...]):
    df = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=limit)
    if df.empty:
        print(f"no {resolution} data")
        return
    idx = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s", utc=True))
    df = df.set_index(idx)
    close = df["close"].astype(float)
    fac = build_micro_factors(df)

    n = len(df)
    a, b = int(n * 0.4), int(n * 0.7)
    train = np.zeros(n, bool); train[:a] = True
    val = np.zeros(n, bool); val[a:b] = True
    test = np.zeros(n, bool); test[b:] = True

    print("=" * 92)
    print(f"{resolution} bars: {n:,}   {idx[0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}")
    print(f"  TRAIN {a:,} (sign only) | VALIDATION {b-a:,} (selection) | TEST {n-b:,} (reported once)")
    print("=" * 92)

    for hb in horizons:
        fwd = close.shift(-hb) / close - 1.0
        print(f"\n--- {hb} bars ahead ---")
        print(f"{'factor':22s}{'sign':>6s}{'TRAIN':>9s}{'VALID':>9s}{'TEST':>9s}{'n test':>9s}")

        signs, rows = {}, []
        for name in fac.columns:
            a_tr, _ = acc(fac[name], fwd, train)
            if a_tr is None:
                continue
            # Sign fixed on TRAIN, then frozen. Everything after this uses it.
            sign = 1.0 if a_tr >= 50 else -1.0
            signs[name] = sign
            s = fac[name] * sign
            a_tr2, _ = acc(s, fwd, train)
            a_v, n_v = acc(s, fwd, val)
            a_te, n_te = acc(s, fwd, test)
            rows.append((name, sign, a_tr2, a_v, a_te, n_te))
            fm = lambda x: "n/a" if x is None else f"{x:.1f}%"
            print(f"{name:22s}{sign:>+6.0f}{fm(a_tr2):>9s}{fm(a_v):>9s}{fm(a_te):>9s}{n_te:>9,}")

        # --- combination, selected on VALIDATION only -------------------
        survivors = [r[0] for r in rows if r[3] is not None and r[3] > 50.0]
        if not survivors:
            print("  no factor cleared 50% on validation -- nothing to combine")
            continue

        signed = pd.DataFrame({nm: fac[nm] * signs[nm] for nm in survivors})
        combo = signed.mean(axis=1)
        a_v, _ = acc(combo, fwd, val, min_abs=0.02)
        a_te, n_te = acc(combo, fwd, test, min_abs=0.02)
        base_te, base_n = acc(pd.Series(1.0, index=close.index), fwd, test, min_abs=0.0)
        flat = float((fwd[test] == 0).mean()) * 100

        # Conviction-weighted variant: agreement across factors, not average
        # magnitude. Weak factors agreeing is the thing worth testing.
        agree = np.sign(signed).mean(axis=1)
        a_v2, _ = acc(agree, fwd, val, min_abs=0.02)
        a_te2, n_te2 = acc(agree, fwd, test, min_abs=0.02)

        fm = lambda x: "n/a" if x is None else f"{x:.1f}%"
        print(f"\n  survivors on validation ({len(survivors)}): {', '.join(survivors)}")
        print(f"  {'equal-weight combo':24s} VALID {fm(a_v):>7s}   TEST {fm(a_te):>7s}  n={n_te:,}  band +-{band(n_te):.1f}")
        print(f"  {'sign-agreement combo':24s} VALID {fm(a_v2):>7s}   TEST {fm(a_te2):>7s}  n={n_te2:,}  band +-{band(n_te2):.1f}")
        print(f"  {'always-up baseline':24s} {'':14s}   TEST {fm(base_te):>7s}  n={base_n:,}"
              f"   ({flat:.1f}% of bars were flat and are excluded)")
        if a_te is not None and base_te is not None:
            d = a_te - base_te
            print(f"  -> equal-weight vs baseline: {d:+.1f} pts "
                  f"({'INSIDE noise' if abs(d) < band(n_te) else 'OUTSIDE noise'})")


async def main():
    await study("15m", 150_000, (4, 8, 24, 96))   # 1h, 2h, 6h, 1d ahead
    await study("1h", 60_000, (4, 12, 24, 72))    # 4h, 12h, 1d, 3d ahead


if __name__ == "__main__":
    asyncio.run(main())
