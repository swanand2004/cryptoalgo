"""ML book filter v1: a learned daily size multiplier on the live trend book.

WHAT IT IS. Once a day, on the hourly bar that closes at 00:00 UTC, a frozen
LightGBM model scores each major (BTC/ETH/SOL/BNB) with P(the trend book's
position in that coin earns money over the next 72 hours). The score maps to a
multiplier on that coin's book weight: 0.5 below `lo`, 1.5 above `hi`, 1.0 in
between. It applies to positions from 01:00 UTC (one-hour delay) for 24 hours.

WHY IT EXISTS. Pre-registered study, cycle 5 AI-PATTERN-ML (scratchpad
team/ai_ml/PREREG.txt): trained on 2018-2022 only, validated ONCE. Book Sharpe
net of 7 bps/side + funding: VAL 2023-24 1.61 -> 1.79, TEST 2025-26 1.19 ->
1.39; better on 10/10 never-seen coins; above the p95 of random-timing and
shuffled-label controls; deflated Sharpe 0.96. The machine mostly sizes up when
the coin is near its 30-day high with broad market breadth and a strong signal,
and down when the majors are moving in lock-step or the book just had a hot
month. The margins over the controls are THIN, which is why this ships as
"shadow" (computed and logged, zero effect on sizing) by default.

MODES (Settings.ml_book_filter): "off" -- nothing runs; "shadow" -- the daily
job computes and stores multipliers, sizing untouched; "live" -- the trading
loop multiplies each major's weight by its fresh multiplier, rescaled so the
book never exceeds book_max_leverage, and 1.0 whenever the record is missing,
stale or flagged stale at compute time.

Every feature is strictly past-only and computed EXACTLY as in training: the
feature code below is a verbatim port of the research pipeline, and
tests/test_ml_book_filter.py checks parity against that pipeline's predictions
on stored history.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "model_store" / "book_filter_v1"
MAJORS = ("BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD")
STATE_KEY = "ml_book_filter_v1_history"
HISTORY_DAYS = 400
DAY = 86400
HOUR = 3600
# Rows of hourly history the rolling features need at the decision row (longest window 8760h + lags).
FEATURE_TAIL = 20_000
MODES = ("off", "shadow", "live")


@lru_cache(maxsize=1)
def load_model():
    import lightgbm as lgb

    spec = json.loads((MODEL_DIR / "spec.json").read_text())
    booster = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
    if list(booster.feature_name()) != list(spec["features"]):
        raise RuntimeError("book_filter_v1: model feature names do not match spec.json")
    return booster, spec


# ------------------------------------------------------------------ book replica (as trained)
def _atr14(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean()


def _to_grid(series_by_time: pd.Series, grid: np.ndarray, start_time) -> np.ndarray:
    s = series_by_time.reindex(grid).ffill()
    s[grid < start_time] = np.nan
    return s.to_numpy(dtype=float)


def _hysteresis(sig: np.ndarray, entry: float = 0.15, exit_: float = 0.05) -> np.ndarray:
    """Schmitt trigger identical to position_sizing.apply_signal_hysteresis (NaN resets to flat)."""
    col = sig.tolist()
    res = [0.0] * len(col)
    cur = 0
    for t, s in enumerate(col):
        if s != s:
            cur = 0
            continue
        if cur == 0:
            v = s if abs(s) >= entry else 0.0
        elif (s >= 0) == (cur > 0):
            v = s if abs(s) >= exit_ else 0.0
        else:
            v = s if abs(s) >= entry else 0.0
        res[t] = v
        cur = 1 if v > 0 else (-1 if v < 0 else 0)
    return np.asarray(res, dtype=float)


def _members(df: pd.DataFrame, grid: np.ndarray, warmup: int):
    from app.strategy.gann_angle import GannAngleStrategy
    from app.strategy.trend_ladder import TrendLadderStrategy

    d = df.copy()
    d["atr_14"] = _atr14(d)
    s1 = pd.Series(TrendLadderStrategy().generate_signal(d).to_numpy(), index=d["time"])
    s2 = pd.Series(GannAngleStrategy().generate_signal(d).to_numpy(), index=d["time"])
    s1.iloc[:warmup] = 0.0
    s2.iloc[:warmup] = 0.0
    t0 = d["time"].min()
    return _to_grid(s1, grid, t0), _to_grid(s2, grid, t0)


def _on_grid(df: pd.DataFrame, grid: np.ndarray):
    d = df.set_index("time")
    t0, t1 = d.index.min(), d.index.max()
    c = d["close"].reindex(grid).ffill()
    inside = (grid >= t0) & (grid <= t1)
    c[~inside] = np.nan
    h = d["high"].reindex(grid)
    lo = d["low"].reindex(grid)
    v = d["volume"].reindex(grid)
    miss = h.isna()
    h[miss] = c[miss]
    lo[miss] = c[miss]
    v[miss & inside] = 0.0
    return c.to_numpy(float), h.to_numpy(float), lo.to_numpy(float), v.to_numpy(float), int(t0), int(t1)


def build_hourly(frames: dict[str, pd.DataFrame], funding: dict[str, pd.Series], warmup: int = 6400) -> dict:
    """frames: symbol -> hourly OHLCV (time = bar OPEN, unix s), full history up to and including the
    decision bar. funding: symbol -> DECIMAL rate indexed by settlement unix time."""
    t0 = min(int(f["time"].min()) for f in frames.values())
    t1 = max(int(f["time"].max()) for f in frames.values())
    grid = np.arange(t0, t1 + 1, HOUR, dtype=np.int64)
    coins = {}
    for c in MAJORS:
        df = frames[c].drop_duplicates("time").sort_values("time").reset_index(drop=True)
        df = df.astype({"time": "int64", "open": float, "high": float, "low": float, "close": float, "volume": float})
        C, Hh, L, V, ct0, ct1 = _on_grid(df, grid)
        lad, gan = _members(df, grid, warmup)
        R = np.zeros(len(grid))
        R[:-1] = C[1:] / C[:-1] - 1.0
        R = np.nan_to_num(R, nan=0.0)
        S = _hysteresis((lad + gan) / 2)
        coins[c] = dict(C=C, H=Hh, L=L, V=V, R=R, S=S, lad=lad, gan=gan, t0=ct0, t1=ct1,
                        fr=funding.get(c, pd.Series(dtype=float)))
    return {"grid": grid, "coins": coins}


# ------------------------------------------------------------------ features (verbatim port of training)
def _roll_corr(a, b, w):
    return a.rolling(w, min_periods=w // 2).corr(b)


def market_feats(coins: dict, sl: slice) -> dict:
    lrs = pd.DataFrame({c: np.log(pd.Series(coins[c]["C"][sl])).diff() for c in MAJORS})
    cl = pd.DataFrame({c: pd.Series(coins[c]["C"][sl]) for c in MAJORS})
    ew = lrs.mean(axis=1, skipna=True)
    esig = ew.rolling(720, min_periods=360).std()
    mk = {}
    mk["mk_ew24n"] = ew.rolling(24).sum() / (esig * np.sqrt(24))
    mk["mk_ew168n"] = ew.rolling(168).sum() / (esig * np.sqrt(168))
    mk["mk_disp24"] = np.log(cl / cl.shift(24)).std(axis=1) / (esig * np.sqrt(24))
    mk["mk_disp168"] = np.log(cl / cl.shift(168)).std(axis=1) / (esig * np.sqrt(168))
    pcs = []
    for i in range(4):
        for j in range(i + 1, 4):
            pcs.append(_roll_corr(lrs.iloc[:, i], lrs.iloc[:, j], 168))
    mk["mk_pcorr168"] = pd.concat(pcs, axis=1).mean(axis=1)
    mk["mk_above720"] = (cl > cl.rolling(720, min_periods=360).mean()).where(cl.notna()).mean(axis=1)
    up = pd.DataFrame({c: (pd.Series(coins[c]["S"][sl]) * pd.Series(coins[c]["R"][sl])).shift(1) for c in MAJORS})
    act = pd.DataFrame({c: (pd.Series(coins[c]["S"][sl]).abs() > 0).astype(float) for c in MAJORS}).sum(axis=1).clip(lower=1)
    bk = up.sum(axis=1) / act
    mk["mk_book7"] = bk.rolling(168, min_periods=1).sum() / (esig * np.sqrt(168))
    mk["mk_book30"] = bk.rolling(720, min_periods=1).sum() / (esig * np.sqrt(720))
    return mk


def coin_features(d: dict, grid: np.ndarray, btc: np.ndarray, mk: dict, fng: pd.Series, sl: slice,
                  hrs_since_flip_full: np.ndarray) -> pd.DataFrame:
    c = pd.Series(d["C"][sl])
    h = pd.Series(d["H"][sl])
    lo = pd.Series(d["L"][sl])
    v = pd.Series(d["V"][sl])
    g = grid[sl]
    lr = np.log(c).diff()
    sig = lr.rolling(720, min_periods=360).std()
    f = {}
    for k in (1, 4, 12, 24, 72, 168, 336, 720, 1440):
        f[f"ret{k}n"] = np.log(c / c.shift(k)) / (sig * np.sqrt(k))
    f["ret24raw"] = np.log(c / c.shift(24))
    f["ret168raw"] = np.log(c / c.shift(168))
    rv = {k: lr.rolling(k, min_periods=k // 2).std() * np.sqrt(8760) for k in (24, 72, 168, 720)}
    for k, s in rv.items():
        f[f"rv{k}"] = s
    f["rvr24_720"] = rv[24] / rv[720]
    f["rvr168_720"] = rv[168] / rv[720]
    f["rvr72_720"] = rv[72] / rv[720]
    f["volvol"] = rv[24].rolling(720, min_periods=360).std() / rv[24].rolling(720, min_periods=360).mean()
    pk = np.log(h / lo).clip(lower=0) ** 2 / (4 * np.log(2))
    for k in (24, 168):
        f[f"park{k}"] = np.sqrt(pk.rolling(k, min_periods=k // 2).mean() * 8760)
    f["park_rv168"] = f["park168"] / rv[168]
    for k in (24, 168, 720, 2160):
        mx = h.rolling(k, min_periods=k // 2).max()
        mn = lo.rolling(k, min_periods=k // 2).min()
        f[f"rpos{k}"] = (c - mn) / (mx - mn).replace(0, np.nan)
        if k in (24, 168):
            f[f"rwid{k}"] = np.log(mx / mn) / (sig * np.sqrt(k))
    clv = ((c - lo) / (h - lo).replace(0, np.nan)).fillna(0.5)
    f["clv1"] = clv
    f["clv24"] = clv.rolling(24).mean()
    f["clv72"] = clv.rolling(72).mean()
    for k in (24, 168, 720, 2160):
        sma = c.rolling(k, min_periods=k // 2).mean()
        f[f"sma{k}d"] = np.log(c / sma) / (sig * np.sqrt(k))
        if k == 720:
            f["sma720slope"] = np.log(sma / sma.shift(168)) / (sig * np.sqrt(168))
    for k in (720, 2160, 8760):
        f[f"dd{k}"] = c / c.rolling(k, min_periods=k // 4).max() - 1
    f["rally720"] = c / c.rolling(720, min_periods=360).min() - 1
    lv24 = np.log(v.rolling(24).sum() + 1)
    f["volz24"] = (lv24 - lv24.rolling(720, min_periods=360).mean()) / lv24.rolling(720, min_periods=360).std()
    f["volr168_720"] = np.log((v.rolling(168).sum() + 1) / (v.rolling(720).sum() / (720 / 168) + 1))
    f["sgnvol24"] = (np.sign(lr) * v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    f["sgnvol168"] = (np.sign(lr) * v).rolling(168).sum() / v.rolling(168).sum().replace(0, np.nan)
    for k in (168, 720):
        f[f"skew{k}"] = lr.rolling(k, min_periods=k // 2).skew()
        f[f"kurt{k}"] = lr.rolling(k, min_periods=k // 2).kurt()
    f["ac1_168"] = _roll_corr(lr, lr.shift(1), 168)
    r24 = np.log(c / c.shift(24))
    f["ac24_720"] = _roll_corr(r24, r24.shift(24), 720)
    f["maxup24"] = lr.rolling(24).max() / sig
    f["maxdn24"] = lr.rolling(24).min() / sig
    b = pd.Series(btc[sl])
    blr = np.log(b).diff()
    bsig = blr.rolling(720, min_periods=360).std()
    for k in (24, 72, 168):
        f[f"btc{k}n"] = np.log(b / b.shift(k)) / (bsig * np.sqrt(k))
    for k in (24, 168, 720):
        f[f"relbtc{k}"] = (np.log(c / c.shift(k)) - np.log(b / b.shift(k))) / (sig * np.sqrt(k))
    f["corrbtc168"] = _roll_corr(lr, blr, 168)
    f["corrbtc720"] = _roll_corr(lr, blr, 720)
    f["betabtc720"] = lr.rolling(720, min_periods=360).cov(blr) / blr.rolling(720, min_periods=360).var()
    f["btc_rvr24_720"] = blr.rolling(24).std() / blr.rolling(720, min_periods=360).std()
    for key in ("mk_ew24n", "mk_ew168n", "mk_disp24", "mk_disp168", "mk_pcorr168", "mk_above720",
                "mk_book7", "mk_book30"):
        f[key] = mk[key]
    f["relmk168"] = f["ret168n"] - mk["mk_ew168n"]
    # funding: a settlement stamped T is usable iff T <= decision instant (bar open + 1h)
    fr = d["fr"].sort_index()
    fr.index.name = "t"
    fs = pd.DataFrame({"last": fr, "m3": fr.rolling(3).mean(), "m21": fr.rolling(21).mean(),
                       "m90": fr.rolling(90).mean()})
    fs["z21"] = (fs["m21"] - fr.rolling(1095, min_periods=200).mean()) / fr.rolling(1095, min_periods=200).std()
    fs["d21"] = fs["m21"] - fs["m21"].shift(21)
    inst = pd.DataFrame({"t": g + HOUR})
    fsr = fs.reset_index()
    fsr["t"] = fsr["t"].astype("int64")
    fm = pd.merge_asof(inst, fsr.sort_values("t"), on="t", direction="backward")
    for k in ("last", "m3", "m21", "m90", "z21", "d21"):
        f[f"fund_{k}"] = pd.Series(fm[k].to_numpy() * (1e4 if k != "z21" else 1.0))
    # Fear & Greed: the value stamped day D is used only from the decision at 00:00 of D+2
    fd = pd.DataFrame({"lvl": fng, "ch7": fng - fng.shift(7),
                       "z90": (fng - fng.rolling(90, min_periods=30).mean()) / fng.rolling(90, min_periods=30).std(),
                       "m30": fng.rolling(30, min_periods=10).mean()})
    fd.index = fd.index.astype("int64") + 2 * DAY
    fd.index.name = "t"
    fd = fd.reset_index().sort_values("t")
    fgm = pd.merge_asof(inst, fd, on="t", direction="backward")
    for k in ("lvl", "ch7", "z90", "m30"):
        f[f"fng_{k}"] = pd.Series(fgm[k].to_numpy())
    ts = pd.to_datetime(g + HOUR, unit="s", utc=True)
    f["dow"] = pd.Series(ts.dayofweek.to_numpy(float))
    f["weekend"] = pd.Series((ts.dayofweek >= 5).astype(float))
    doy = ts.dayofyear.to_numpy(float)
    f["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25))
    f["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25))
    S = pd.Series(d["S"][sl])
    lad = pd.Series(d["lad"][sl])
    gan = pd.Series(d["gan"][sl])
    f["S"] = S
    f["absS"] = S.abs()
    f["lad"] = lad
    f["gan"] = gan
    f["blend"] = (lad + gan) / 2
    f["agree"] = np.sign(lad) * np.sign(gan)
    sg = np.sign(S.fillna(0))
    chg = (sg != sg.shift(1)).astype(float)
    f["hrs_since_flip"] = pd.Series(hrs_since_flip_full[sl])
    f["flips720"] = chg.rolling(720, min_periods=1).sum()
    upnl = (S * pd.Series(d["R"][sl])).shift(1)
    for k in (24, 72, 168, 720):
        f[f"bpnl{k}"] = upnl.rolling(k, min_periods=1).sum() / (sig * np.sqrt(k))
    eq = upnl.fillna(0).cumsum()
    f["bdd2160"] = (eq - eq.rolling(2160, min_periods=1).max()) / (sig * np.sqrt(2160))
    f["S_x_ret168"] = np.sign(S) * f["ret168n"]
    f["S_x_btc168"] = np.sign(S) * f["btc168n"]
    out = pd.DataFrame({k: pd.Series(np.asarray(vv, dtype=float)) for k, vv in f.items()})
    return out.replace([np.inf, -np.inf], np.nan)


def _hrs_since_flip(S: np.ndarray) -> np.ndarray:
    s = pd.Series(S)
    sg = np.sign(s.fillna(0))
    grp = (sg != sg.shift(1)).astype(float).cumsum()
    return np.log1p(sg.groupby(grp).cumcount().astype(float)).to_numpy()


def feature_rows(hr: dict, fng: pd.Series, rows: np.ndarray, tail: int | None = None) -> pd.DataFrame:
    """Model feature matrix (one row per (coin, decision row)). tail=None computes on the whole grid
    (research parity); tail=N restricts rolling computations to the last N rows before max(rows)."""
    _, spec = load_model()
    grid = hr["grid"]
    rows = np.asarray(rows, dtype=int)
    if tail is None:
        sl = slice(0, len(grid))
    else:
        sl = slice(max(0, int(rows.max()) + 1 - tail), int(rows.max()) + 1)
    off = sl.start
    mk = market_feats(hr["coins"], sl)
    btc = hr["coins"]["BTCUSD"]["C"]
    parts = []
    for c in MAJORS:
        d = hr["coins"][c]
        X = coin_features(d, grid, btc, mk, fng, sl, _hrs_since_flip(d["S"]))
        Xr = X.iloc[rows - off].reset_index(drop=True)
        Xr.insert(0, "coin", c)
        Xr.insert(1, "row", rows)
        parts.append(Xr)
    X = pd.concat(parts, ignore_index=True)
    core = list(spec["core_mu"])
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            a, b = core[i], core[j]
            X[f"x_{a}__{b}"] = ((X[a] - spec["core_mu"][a]) / spec["core_sd"][a]) * \
                               ((X[b] - spec["core_mu"][b]) / spec["core_sd"][b])
    return X


def multiplier(q: float, lo: float, hi: float) -> float:
    return 0.5 if q < lo else (1.5 if q > hi else 1.0)


# ------------------------------------------------------------------ the daily decision (pure)
def decide(frames: dict[str, pd.DataFrame], funding: dict[str, pd.Series], fng: pd.Series,
           decision_ts: int) -> dict:
    """decision_ts: the 00:00 UTC instant. Uses bars with OPEN time <= decision_ts - 1h only.
    Inputs are checked for freshness; any failure yields m = 1.0 for every coin with stale=True."""
    booster, spec = load_model()
    reasons = []
    bar = decision_ts - HOUR
    trimmed = {}
    for c in MAJORS:
        df = frames.get(c)
        if df is None or df.empty:
            reasons.append(f"{c}: no price history")
            continue
        times = set(int(t) for t in df["time"].iloc[-60:])
        if bar not in times:
            reasons.append(f"{c}: decision bar missing")
        if decision_ts not in times:
            # the bar opening at the decision instant proves the 23:00 bar was refreshed after it closed
            reasons.append(f"{c}: decision bar not yet final")
        trimmed[c] = df[df["time"] <= bar]
    for c in MAJORS:
        fr = funding.get(c)
        if fr is None or fr.empty or int(fr.index.max()) < decision_ts - 16 * HOUR:
            reasons.append(f"{c}: funding stale")
    if fng is None or fng.empty or int(fng.index.max()) < decision_ts - 3 * DAY:
        reasons.append("fear&greed stale")
    rec = {
        "model": spec.get("name", "book_filter_v1"),
        "decision_ts": int(decision_ts),
        "decision_time": dt.datetime.fromtimestamp(decision_ts, dt.timezone.utc).isoformat(),
        "applies_from_ts": int(decision_ts + HOUR),
        "applies_until_ts": int(decision_ts + HOUR + DAY),
        "lo": spec["lo"], "hi": spec["hi"],
        "stale": bool(reasons), "reasons": reasons,
        "coins": {c: {"p": None, "m": 1.0, "S": None} for c in MAJORS},
    }
    if reasons:
        return rec
    hr = build_hourly(trimmed, funding, spec.get("warmup_bars", 6400))
    row = int(np.where(hr["grid"] == bar)[0][0])
    X = feature_rows(hr, fng, np.array([row]), tail=FEATURE_TAIL)
    p = booster.predict(X[spec["features"]])
    for i, c in enumerate(X["coin"]):
        rec["coins"][c] = {"p": float(p[i]), "m": multiplier(float(p[i]), spec["lo"], spec["hi"]),
                           "S": float(X["S"].iloc[i])}
    return rec


# ------------------------------------------------------------------ DB I/O
async def _load_inputs():
    from sqlalchemy import select

    from app.data.db import SessionLocal
    from app.data.models import Candle

    frames, funding = {}, {}
    async with SessionLocal() as session:
        for c in MAJORS:
            res = await session.execute(
                select(Candle.time, Candle.open, Candle.high, Candle.low, Candle.close, Candle.volume)
                .where(Candle.symbol == f"BINANCE:{c}", Candle.resolution == "1h").order_by(Candle.time))
            frames[c] = pd.DataFrame(res.all(), columns=["time", "open", "high", "low", "close", "volume"])
            res = await session.execute(
                select(Candle.time, Candle.close)
                .where(Candle.symbol == f"BFUNDHIST:{c}", Candle.resolution == "8h").order_by(Candle.time))
            fd = pd.DataFrame(res.all(), columns=["time", "rate"]).drop_duplicates("time")
            funding[c] = pd.Series(fd["rate"].astype(float).to_numpy() / 100.0,  # stored in PERCENT
                                   index=fd["time"].astype("int64").to_numpy())
        res = await session.execute(
            select(Candle.time, Candle.close)
            .where(Candle.symbol == "FNG:GLOBAL", Candle.resolution == "1d").order_by(Candle.time))
        fg = pd.DataFrame(res.all(), columns=["time", "close"]).drop_duplicates("time")
        fng = pd.Series(fg["close"].astype(float).to_numpy(), index=fg["time"].astype("int64").to_numpy())
    return frames, funding, fng


async def get_history() -> list[dict]:
    from app.data.app_state import get_state

    raw = await get_state(STATE_KEY)
    if not raw:
        return []
    try:
        hist = json.loads(raw)
        return hist if isinstance(hist, list) else []
    except (TypeError, ValueError):
        return []


async def _store(rec: dict) -> None:
    from app.data.app_state import set_state

    hist = [h for h in await get_history() if h.get("decision_ts") != rec["decision_ts"]]
    hist.append(rec)
    hist = sorted(hist, key=lambda h: h["decision_ts"])[-HISTORY_DAYS:]
    await set_state(STATE_KEY, json.dumps(hist))


def mode_of(settings) -> str:
    m = str(getattr(settings, "ml_book_filter", "shadow")).lower()
    return m if m in MODES else "off"


async def run_daily_job(settings, now: dt.datetime | None = None, final_attempt: bool = False) -> dict | None:
    """Computes and stores today's record. Retried by the scheduler during 00:05-00:50 UTC; a stale
    attempt is stored only on the final one (so a late price refresh does not lose the day)."""
    mode = mode_of(settings)
    if mode == "off":
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    decision_ts = int(now.timestamp()) // DAY * DAY
    for h in await get_history():
        if h.get("decision_ts") == decision_ts and not h.get("stale"):
            return h  # already done today
    frames, funding, fng = await _load_inputs()
    rec = await asyncio.to_thread(decide, frames, funding, fng, decision_ts)
    rec["mode"] = mode
    rec["computed_at"] = now.isoformat()
    if rec["stale"] and not final_attempt:
        logger.info("ML book filter: inputs not ready (%s) -- will retry", "; ".join(rec["reasons"]))
        return rec
    await _store(rec)
    logger.info("ML book filter [%s] %s: %s%s", mode, rec.get("decision_time", rec["decision_ts"]),
                ", ".join(f"{c} p={v['p'] if v['p'] is None else round(v['p'], 3)} m={v['m']}"
                          for c, v in rec["coins"].items()),
                " (STALE -> 1.0)" if rec["stale"] else "")
    return rec


async def current_multipliers(now: dt.datetime | None = None) -> dict[str, float]:
    """The multipliers in force now: from the record whose window [applies_from, applies_until)
    contains now and that was not stale. Anything else -> {} (callers treat a missing coin as 1.0)."""
    ts = int((now or dt.datetime.now(dt.timezone.utc)).timestamp())
    for h in reversed(await get_history()):
        if h.get("applies_from_ts", 0) <= ts < h.get("applies_until_ts", 0):
            if h.get("stale"):
                return {}
            return {c: float(v.get("m", 1.0)) for c, v in h.get("coins", {}).items()}
    return {}


def apply_to_weights(weights: dict[str, float], signals: dict, book_scalar: float,
                     mult: dict[str, float], max_book_leverage: float) -> dict[str, float]:
    """Live mode only. w_i -> w_i * m_i, then the whole book is scaled down (never up) so its gross
    target notional |s| * w * k stays inside book_max_leverage."""
    eff = {s: w * float(mult.get(s, 1.0)) for s, w in weights.items()}
    gross = sum(abs(float(sig.iloc[-1])) * eff.get(s, 0.0) for s, sig in signals.items() if len(sig)) * book_scalar
    if gross > max_book_leverage > 0:
        f = max_book_leverage / gross
        eff = {s: w * f for s, w in eff.items()}
    return eff


async def book_filter_weights(settings, weights: dict[str, float], signals: dict,
                              book_scalar: float) -> dict[str, float]:
    """Trading-loop hook. "off"/"shadow": returns `weights` ITSELF (same object, nothing read, nothing
    computed) so sizing is byte-identical. "live": fresh multipliers applied, 1.0 when unavailable."""
    if mode_of(settings) != "live":
        return weights
    try:
        mult = await current_multipliers()
    except Exception:
        logger.exception("ML book filter: could not read multipliers -- using 1.0")
        mult = {}
    if not mult:
        logger.warning("ML book filter LIVE: no fresh multipliers -- every coin at 1.0")
    return apply_to_weights(weights, signals, book_scalar, mult, getattr(settings, "book_max_leverage", 8.0))
