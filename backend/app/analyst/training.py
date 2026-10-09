"""Opt-in nightly training of the Analyst's family weights
(Settings.analyst_training_enabled, default off; scheduler job 02:40 UTC).

What one run does:
1. Rebuilds the dataset: the Analyst's per-family scores every 4 hours over the
   last WINDOW_DAYS of stored BINANCE: hourly candles, computed by the live
   `factors.analyse` on exactly the history the live service would have had
   (the last 330 days of closed bars, side series filtered to <= that time,
   daily series shifted a day, the trend-book signal recomputed causally).
   Each row also stores the live trade plan (stop / TP1) for BOTH sides.
   Family scores of a past step never change, so rows are cached and only new
   steps are analysed -- at most MAX_NEW_ANALYSES per run, so the first fill
   is spread over a few nights instead of an hour of CPU on the 1-vCPU server.
2. Outcomes: each side's plan is walked forward on hourly bars with the live
   exit rules (execution.update_exit), net of 7 bps a side and real Binance
   funding, in R. Outcome bars are always strictly AFTER the decision bar.
3. Split: the last HOLDOUT_DAYS of decisions are the holdout; the fit uses only
   decisions before it whose trades had already closed before it.
4. Fit: non-negative weights summing to 1 (random points on the simplex plus
   the current weights), objective = net R per step on the fit set, then
   shrunk halfway back to the current weights.
5. Adopt only if the candidate beats the current weights on the holdout by the
   margins below. They are FIXED here, set before any result was seen
   (scratchpad PREREG.txt), and must never be tuned on results.

Adopted weights apply only to the symbols they were trained on (never gold)
and only while the setting is on; anything missing or malformed falls back to
factors.FAMILY_WEIGHTS.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STORE_DIR = Path(__file__).resolve().parent / "model_store"
WEIGHTS_FILE = "analyst_weights.json"
RUNS_FILE = "analyst_training_runs.json"
ROWS_FILE = "analyst_training_rows.csv"
ROWS_META_FILE = "analyst_training_rows.meta.json"
MAX_RUNS_KEPT = 60

# --- pre-set, never tuned on results ----------------------------------------
WINDOW_DAYS = 730
HOLDOUT_DAYS = 90
STEP_HOURS = 4
LOOKBACK_BARS = 24 * 330  # what the live service passes (service.HOURLY_BARS)
LADDER_WARMUP_BARS = 12_000  # the live ladder frame (query.LADDER_BARS)
MAX_HOLD_HOURS = 24 * 30
COST_PER_SIDE = 0.0007
BOOK_LINE = 0.15  # execution.agrees_with_book
N_CANDIDATES = 2000
SEED = 20261006
SHRINK = 0.5  # candidate = SHRINK * current + (1 - SHRINK) * best fit
MIN_FIT_ROWS = 1000
MIN_HOLDOUT_CALLS = 30
MARGIN_R = 0.05  # holdout expectancy improvement required, R per call
MAX_NEW_ANALYSES = 3000  # per run (~0.15-0.3 s each)
_CHUNK = 50  # candidates scored at once (keeps the fit at a few tens of MB)

FEATURE_SOURCES = ("analyst/factors.py", "analyst/structure.py", "strategy/gann_angle.py",
                   "strategy/trend_ladder.py", "strategy/ensemble.py", "gold/signal_board.py")


def _families() -> tuple[str, ...]:
    from app.analyst.factors import FAMILY_WEIGHTS

    return tuple(FAMILY_WEIGHTS)


# ============================================================ stored weights
_record_cache: dict = {"key": None, "record": None}


def _valid_weights(w) -> dict[str, float] | None:
    fams = _families()
    if not isinstance(w, dict) or set(w) != set(fams):
        return None
    try:
        vals = {f: float(w[f]) for f in fams}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v < 0 for v in vals.values()) or abs(sum(vals.values()) - 1.0) > 1e-6:
        return None
    return vals


def adopted_record(store_dir: Path | None = None) -> dict | None:
    """The adopted-weights record, validated (cached by file mtime)."""
    path = Path(store_dir or STORE_DIR) / WEIGHTS_FILE
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _record_cache["key"] != key:
        record = None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            w = _valid_weights(raw.get("weights"))
            if w is not None:
                record = {**raw, "weights": w, "symbols": [str(s) for s in raw.get("symbols") or []]}
            else:
                logger.warning("Analyst training: %s is malformed -- using the fixed weights", path)
        except Exception:
            logger.warning("Analyst training: cannot read %s -- using the fixed weights", path)
        _record_cache.update(key=key, record=record)
    return _record_cache["record"]


def active_weights(symbol: str | None = None, store_dir: Path | None = None) -> dict[str, float] | None:
    """Adopted weights for this symbol, or None (= use FAMILY_WEIGHTS). Only
    while analyst_training_enabled is on, so turning it off restores the
    constants at once."""
    from app.config import get_settings

    if not getattr(get_settings(), "analyst_training_enabled", False):
        return None
    rec = adopted_record(store_dir)
    if rec is None or (symbol is not None and symbol not in rec["symbols"]):
        return None
    return dict(rec["weights"])


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy -> python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_runs(store_dir: Path | None = None) -> list[dict]:
    path = Path(store_dir or STORE_DIR) / RUNS_FILE
    try:
        runs = json.loads(path.read_text(encoding="utf-8"))
        return runs if isinstance(runs, list) else []
    except Exception:
        return []


def record_run(result: dict, store_dir: Path | None = None) -> None:
    """Appends the run to the log; writes the weights file only on adoption."""
    store = Path(store_dir or STORE_DIR)
    result = _clean(result)
    if result.get("decision") == "adopted":
        _write_json(store / WEIGHTS_FILE, {
            "weights": result["candidate"], "symbols": result["symbols"], "adopted_at": result["run_at"],
            "previous": result["current"], "holdout": result["holdout"], "fit": result.get("fit"),
        })
    runs = (load_runs(store) + [result])[-MAX_RUNS_KEPT:]
    _write_json(store / RUNS_FILE, runs)


def training_status(store_dir: Path | None = None, limit: int = 20) -> dict:
    from app.analyst.factors import FAMILY_WEIGHTS, family_weights
    from app.config import get_settings

    return {
        "enabled": bool(getattr(get_settings(), "analyst_training_enabled", False)),
        "fixed_weights": dict(FAMILY_WEIGHTS),
        "weights_in_use": family_weights("BTCUSD"),
        "adopted": adopted_record(store_dir),
        "rules": {"window_days": WINDOW_DAYS, "holdout_days": HOLDOUT_DAYS, "step_hours": STEP_HOURS,
                  "shrink_to_current": SHRINK, "min_holdout_calls": MIN_HOLDOUT_CALLS, "margin_r": MARGIN_R},
        "runs": list(reversed(load_runs(store_dir)))[:limit],
    }


# ============================================================ data
@dataclass
class SymbolData:
    symbol: str
    hourly: pd.DataFrame  # open/high/low/close, UTC index = bar OPEN time
    book_signal: pd.Series | None = None  # trend-book signal, same index
    funding: pd.Series | None = None  # fraction per 8h settlement
    open_interest: pd.Series | None = None
    long_short_ratio: pd.Series | None = None
    fear_greed: pd.Series | None = None
    stablecoin_supply: pd.Series | None = None
    news: pd.Series | None = None
    dollar: pd.Series | None = None


def book_signal_series(frame: pd.DataFrame, settings) -> pd.Series:
    """The live trend book (trend_ladder + gann_angle, registry weights),
    causal, indexed by bar open time; NaN during the ladder's warm-up."""
    from app.strategy.ensemble import SignalEnsemble, WeightedStrategy
    from app.strategy.gann_angle import GannAngleStrategy
    from app.strategy.registry import DEFAULT_WEIGHTS
    from app.strategy.trend_ladder import TrendLadderStrategy

    members = [WeightedStrategy(TrendLadderStrategy(), weight=DEFAULT_WEIGHTS.get("trend_ladder", 1.0)),
               WeightedStrategy(GannAngleStrategy(), weight=DEFAULT_WEIGHTS.get("gann_angle", 1.0))]
    ens = SignalEnsemble(members, deadband=settings.signal_deadband, smoothing_span=settings.signal_smoothing_span,
                         exit_deadband=settings.signal_exit_deadband)
    sig = ens.smoothed_signal(frame).astype(float).to_numpy().copy()
    sig[: ens.warmup_bars()] = np.nan
    return pd.Series(sig, index=pd.to_datetime(frame["time"].to_numpy(), unit="s", utc=True))


async def load_symbol_data(symbol: str, settings) -> SymbolData:
    from app.analyst.service import STABLE_SYMBOL, _series, _shift_day
    from app.data.query import load_candles_df, load_ladder_frame

    bars = WINDOW_DAYS * 24 + max(LOOKBACK_BARS, LADDER_WARMUP_BARS) + 48
    frame = await load_ladder_frame(symbol, limit=bars)
    if frame.empty:
        return SymbolData(symbol, pd.DataFrame(columns=["open", "high", "low", "close"]))
    hourly = frame.set_index(pd.to_datetime(frame["time"].to_numpy(), unit="s", utc=True))[["open", "high", "low", "close"]]
    book = book_signal_series(frame, settings)
    del frame
    if len(hourly) and hourly.index[-1] + pd.Timedelta(hours=1) > pd.Timestamp.now(tz="UTC"):
        hourly, book = hourly.iloc[:-1], book.iloc[:-1]  # never an unfinished bar
    days = WINDOW_DAYS + LOOKBACK_BARS // 24 + 10
    # Binance funding: the deep history (stored in PERCENT) under the recent
    # BFUND series the live service reads (a fraction); BFUND wins on overlap.
    hist = _series(await load_candles_df(f"BFUNDHIST:{symbol}", "8h", limit=days * 3), scale=0.01)
    recent = _series(await load_candles_df(f"BFUND:{symbol}", "8h", limit=days * 3))
    funding = None
    if hist is not None or recent is not None:
        funding = pd.concat([s for s in (hist, recent) if s is not None])
        funding = funding[~funding.index.duplicated(keep="last")].sort_index()
    return SymbolData(
        symbol=symbol, hourly=hourly, book_signal=book, funding=funding,
        open_interest=_series(await load_candles_df(f"BOI:{symbol}", "1h", limit=days * 24)),
        long_short_ratio=_series(await load_candles_df(f"BLSR:{symbol}", "1h", limit=days * 24)),
        fear_greed=_shift_day(_series(await load_candles_df("FNG:GLOBAL", "1d", limit=days))),
        stablecoin_supply=_shift_day(_series(await load_candles_df(STABLE_SYMBOL, "1d", limit=days + 800))),
        news=_series(await load_candles_df(f"NEWS:{symbol}", "1h", limit=days * 24)),
        dollar=_shift_day(_series(await load_candles_df("MACRO:DXY", "1d", limit=days))),
    )


# ============================================================ dataset
def decision_steps(d: SymbolData, start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    """Positions of the hourly bars whose CLOSE is a 4h boundary, inside
    [start, end], with the full live lookback behind them."""
    idx = d.hourly.index
    secs = (idx.asi8 // 10**9) + 3600
    ok = (secs % (STEP_HOURS * 3600) == 0) & (idx >= start) & (idx <= end)
    ok &= np.arange(len(idx)) >= 24 * 60  # analyse()'s own minimum
    return [int(i) for i in np.flatnonzero(ok)]


def feature_row(d: SymbolData, i: int) -> dict:
    """Family scores + both sides' live plan at hourly bar i, from data up to
    and including bar i only (its close is the decision time)."""
    from app.analyst.factors import STOP_ATR, TARGET_ATR, Inputs, analyse
    from app.analyst.structure import levels_plan

    t = d.hourly.index[i]
    row: dict = {"symbol": d.symbol, "time": int(t.value // 10**9)}
    sig = d.book_signal.iloc[i] if d.book_signal is not None and i < len(d.book_signal) else np.nan
    row["book"] = float(sig) if np.isfinite(sig) else np.nan
    inp = Inputs(
        hourly=d.hourly.iloc[max(0, i + 1 - LOOKBACK_BARS): i + 1],
        strategy_signal=row["book"] if np.isfinite(row["book"]) else None,
        funding=d.funding, open_interest=d.open_interest, long_short_ratio=d.long_short_ratio,
        fear_greed=d.fear_greed, stablecoin_supply=d.stablecoin_supply, news=d.news, dollar=d.dollar,
    )
    a = analyse(d.symbol, inp, now=t)
    for f in _families():
        row[f"fam_{f}"] = a.families.get(f, np.nan) if a is not None else np.nan
    row["price"] = a.price if a is not None else np.nan
    row["atr"] = a.atr if a is not None and a.atr else np.nan
    for side, tag in (("BUY", "long"), ("SELL", "short")):
        stop = target = np.nan
        if a is not None and np.isfinite(row["atr"]) and row["atr"] > 0:
            stop, target, _ = levels_plan(side, a.price, row["atr"], a.supports, a.resistances, STOP_ATR,
                                          default_rr=TARGET_ATR / STOP_ATR)
        row[f"stop_{tag}"], row[f"tp_{tag}"] = stop, target
    return row


def build_feature_rows(d: SymbolData, positions: list[int]) -> pd.DataFrame:
    return pd.DataFrame([feature_row(d, i) for i in positions])


def simulate_trade(side: str, entry: float, stop: float, target: float, atr: float,
                   opens, highs, lows, closes, settings) -> tuple[float, int]:
    """(gross R, bars held) for the live exit rules walked over the bars AFTER
    entry. A bar is judged against the stop in force at its open, stop first
    (fills at the worse of open and stop); then TP1; then the trail moves."""
    from app.analyst.execution import update_exit

    d = 1 if side == "BUY" else -1
    risk = abs(entry - stop)
    if not (risk > 0 and np.isfinite(target) and atr > 0) or len(closes) == 0:
        return np.nan, 0
    s, tp, peak = stop, target, entry
    for k in range(len(closes)):
        o, h, lo = opens[k], highs[k], lows[k]
        if (lo <= s) if d > 0 else (h >= s):
            px = min(o, s) if d > 0 else max(o, s)
            return d * (px - entry) / risk, k + 1
        fav = h if d > 0 else lo
        u = update_exit(side=side, entry=entry, stop=s, initial_stop=stop, atr=atr, peak=peak, price=fav,
                        take_profit=tp, extended=False, still_strong=False, settings=settings)
        if u.action == "TP":
            return d * (tp - entry) / risk, k + 1
        s, peak = u.stop, u.peak
    return d * (closes[-1] - entry) / risk, len(closes)  # max hold / end of data: marked at the close


def attach_outcomes(rows: pd.DataFrame, data: dict[str, SymbolData], settings) -> pd.DataFrame:
    """Net R and exit time (unix s) of each side's plan, from bars strictly
    after the decision bar."""
    out = rows.copy()
    for col in ("r_long", "r_short", "exit_long", "exit_short"):
        out[col] = np.nan
    for sym, grp in out.groupby("symbol"):
        d = data.get(sym)
        if d is None or d.hourly.empty:
            continue
        secs = d.hourly.index.asi8 // 10**9
        o, h, lo, c = (d.hourly[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        if d.funding is not None and len(d.funding):
            f_t = d.funding.index.asi8 // 10**9
            f_cum = np.concatenate([[0.0], np.cumsum(d.funding.to_numpy(float))])
        else:
            f_t, f_cum = np.array([], dtype=np.int64), np.array([0.0])
        for ix, r in grp.iterrows():
            i = int(np.searchsorted(secs, r["time"]))
            if i >= len(secs) or secs[i] != r["time"]:
                continue
            j = min(len(c), i + 1 + MAX_HOLD_HOURS)
            for side, tag in (("BUY", "long"), ("SELL", "short")):
                gross, held = simulate_trade(side, r["price"], r[f"stop_{tag}"], r[f"tp_{tag}"], r["atr"],
                                             o[i + 1:j], h[i + 1:j], lo[i + 1:j], c[i + 1:j], settings)
                if not np.isfinite(gross):
                    continue
                risk = abs(r["price"] - r[f"stop_{tag}"])
                exit_t = int(secs[i + held] + 3600) if held else int(r["time"] + 3600)
                entry_t = int(r["time"] + 3600)
                paid = f_cum[np.searchsorted(f_t, exit_t, "right")] - f_cum[np.searchsorted(f_t, entry_t, "right")]
                d_ = 1 if side == "BUY" else -1
                net = gross - 2 * COST_PER_SIDE * r["price"] / risk - d_ * paid * r["price"] / risk
                out.at[ix, f"r_{tag}"], out.at[ix, f"exit_{tag}"] = net, exit_t
    return out


# ============================================================ fit / evaluate
def family_matrix(rows: pd.DataFrame) -> np.ndarray:
    return rows[[f"fam_{f}" for f in _families()]].to_numpy(float)


def composite(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    """analyse()'s score: weighted mean over the families present (NaN = absent).
    W is (k,) or (k, m)."""
    present = np.isfinite(F)
    num = np.where(present, F, 0.0) @ W
    den = present.astype(float) @ W
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def call_returns(score, book, r_long, r_short, min_conviction: int):
    """Per-row net R of the live gate (decision line, conviction, trend-book
    agreement) and the side taken (+1/-1/0). Works on (n,) or (n, m) scores."""
    from app.analyst.factors import DECISION_LINE

    score = np.asarray(score, float)
    expand = (lambda a: a[:, None]) if score.ndim == 2 else (lambda a: a)
    book, rl, rs = expand(np.asarray(book, float)), expand(np.asarray(r_long, float)), expand(np.asarray(r_short, float))
    conv_ok = np.minimum(10, np.round(np.abs(score) * 10)) >= min_conviction
    with np.errstate(invalid="ignore"):
        buy = (score >= DECISION_LINE) & conv_ok & (book >= BOOK_LINE) & np.isfinite(rl)
        sell = (score <= -DECISION_LINE) & conv_ok & (book <= -BOOK_LINE) & np.isfinite(rs)
    r = np.where(buy, np.nan_to_num(rl), np.where(sell, np.nan_to_num(rs), 0.0))
    return r, np.where(buy, 1, np.where(sell, -1, 0))


def sequential(times, score, side, r_long, r_short, exit_long, exit_short) -> list[float]:
    """The live rule: one trade at a time across the coins, the highest-|score|
    call at a step, the next only after the last one closed."""
    order = np.lexsort((-np.abs(score), times))
    busy_until, out = -np.inf, []
    for k in order:
        if side[k] == 0 or times[k] < busy_until:
            continue
        out.append(float(r_long[k] if side[k] > 0 else r_short[k]))
        busy_until = exit_long[k] if side[k] > 0 else exit_short[k]
    return out


def evaluate(rows: pd.DataFrame, w: dict[str, float], min_conviction: int) -> dict:
    W = np.array([w[f] for f in _families()], float)
    score = composite(family_matrix(rows), W)
    r, side = call_returns(score, rows["book"], rows["r_long"], rows["r_short"], min_conviction)
    called = side != 0
    seq = sequential(rows["time"].to_numpy(), score, side, rows["r_long"].to_numpy(), rows["r_short"].to_numpy(),
                     rows["exit_long"].to_numpy(), rows["exit_short"].to_numpy())
    n = int(called.sum())
    return {
        "rows": int(len(rows)), "calls": n,
        "expectancy_r": float(r[called].mean()) if n else None,
        "win_rate": float((r[called] > 0).mean()) if n else None,
        "total_r": float(r.sum()), "r_per_row": float(r.mean()) if len(r) else 0.0,
        "seq_trades": len(seq), "seq_total_r": float(np.sum(seq)) if seq else 0.0,
        "seq_expectancy_r": float(np.mean(seq)) if seq else None,
    }


def split(rows: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(fit, holdout, holdout_start): holdout = decisions after holdout_start;
    fit = decisions at/before it whose trades (both sides) closed by then."""
    hold_start = int(rows["time"].max()) - holdout_days * 86400
    exits = np.fmax(rows["exit_long"].to_numpy(float), rows["exit_short"].to_numpy(float))
    exits = np.where(np.isfinite(exits), exits, rows["time"].to_numpy(float))
    fit = rows[(rows["time"] <= hold_start) & (exits <= hold_start)]
    hold = rows[rows["time"] > hold_start]
    return fit, hold, hold_start


def fit_weights(fit_rows: pd.DataFrame, current: dict[str, float], min_conviction: int) -> tuple[dict, dict]:
    """(candidate, best_fit): best of N_CANDIDATES simplex points + current on
    net R per row, then shrunk SHRINK of the way back to current."""
    fams = _families()
    cur = np.array([current[f] for f in fams], float)
    cands = np.vstack([cur, np.random.default_rng(SEED).dirichlet(np.ones(len(fams)), N_CANDIDATES)])
    F = family_matrix(fit_rows)
    book, rl, rs = fit_rows["book"].to_numpy(float), fit_rows["r_long"].to_numpy(float), fit_rows["r_short"].to_numpy(float)
    best_val, best = -np.inf, cur
    for k in range(0, len(cands), _CHUNK):
        W = cands[k:k + _CHUNK].T
        r, _ = call_returns(composite(F, W), book, rl, rs, min_conviction)
        vals = r.mean(axis=0)
        j = int(np.argmax(vals))
        if vals[j] > best_val:  # strict: ties keep the earlier one (current first)
            best_val, best = float(vals[j]), cands[k + j]
    cand = SHRINK * cur + (1 - SHRINK) * best
    cand = np.round(cand / cand.sum(), 4)
    cand[int(np.argmax(cand))] += 1.0 - cand.sum()  # exact sum after rounding
    return ({f: float(v) for f, v in zip(fams, cand)}, {f: float(v) for f, v in zip(fams, np.round(best, 4))})


def decide(cand: dict, cur: dict) -> tuple[bool, list[str]]:
    """Adopt only if every pre-set check passes on the holdout."""
    ce, ue = cand["expectancy_r"], cur["expectancy_r"] if cur["expectancy_r"] is not None else 0.0
    checks = [
        (cand["calls"] >= MIN_HOLDOUT_CALLS, f"holdout calls {cand['calls']} (need >= {MIN_HOLDOUT_CALLS})"),
        (ce is not None and ce > 0, f"candidate expectancy {_r(ce)} (need > 0)"),
        (ce is not None and ce >= ue + MARGIN_R, f"expectancy {_r(ce)} vs current {_r(ue)} (need +{MARGIN_R}R)"),
        (cand["r_per_row"] >= cur["r_per_row"], f"R per step {cand['r_per_row']:+.4f} vs current {cur['r_per_row']:+.4f}"),
        (cand["seq_total_r"] >= cur["seq_total_r"],
         f"one-at-a-time total {cand['seq_total_r']:+.2f}R vs current {cur['seq_total_r']:+.2f}R"),
    ]
    return all(ok for ok, _ in checks), [("PASS " if ok else "FAIL ") + why for ok, why in checks]


def _r(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}R"


def train(rows: pd.DataFrame, current: dict[str, float], min_conviction: int) -> dict:
    """Pure: split, fit, evaluate, decide. `rows` must carry outcomes."""
    rows = rows.dropna(subset=["r_long", "r_short"]).sort_values(["time", "symbol"]).reset_index(drop=True)
    if rows.empty:
        return {"decision": "skipped", "reasons": ["no rows with outcomes"], "current": current}
    fit, hold, hold_start = split(rows)
    info = {"fit_rows": len(fit), "holdout_rows": len(hold),
            "fit_from": _iso(fit["time"].min()) if len(fit) else None, "fit_to": _iso(fit["time"].max()) if len(fit) else None,
            "holdout_from": _iso(hold["time"].min()) if len(hold) else None, "holdout_to": _iso(hold["time"].max()) if len(hold) else None}
    if len(fit) < MIN_FIT_ROWS or hold.empty:
        return {"decision": "skipped", "reasons": [f"too little data: {len(fit)} fit rows (need {MIN_FIT_ROWS})"],
                "current": current, "data": info}
    cand, best = fit_weights(fit, current, min_conviction)
    result = {
        "current": current, "candidate": cand, "best_fit": best, "data": info,
        "fit": {"candidate": evaluate(fit, cand, min_conviction), "current": evaluate(fit, current, min_conviction)},
        "holdout": {"candidate": evaluate(hold, cand, min_conviction), "current": evaluate(hold, current, min_conviction)},
    }
    if all(abs(cand[f] - current[f]) < 1e-4 for f in cand):
        result.update(decision="kept", reasons=["the fit found nothing better than the current weights"])
        return result
    adopt, reasons = decide(result["holdout"]["candidate"], result["holdout"]["current"])
    result.update(decision="adopted" if adopt else "kept", reasons=reasons)
    return result


def _iso(t) -> str:
    return dt.datetime.fromtimestamp(int(t), dt.timezone.utc).isoformat()


# ============================================================ cache + job
def feature_cache_key() -> str:
    """Changes whenever the code that produces the family scores changes, so
    cached rows are never mixed across factor versions."""
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256(f"{LOOKBACK_BARS}|{STEP_HOURS}|{_families()}".encode())
    for rel in FEATURE_SOURCES:
        p = root / rel
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()[:16]


def load_cached_rows(store_dir: Path | None = None) -> pd.DataFrame:
    store = Path(store_dir or STORE_DIR)
    try:
        meta = json.loads((store / ROWS_META_FILE).read_text(encoding="utf-8"))
        if meta.get("key") != feature_cache_key():
            return pd.DataFrame()
        return pd.read_csv(store / ROWS_FILE)
    except Exception:
        return pd.DataFrame()


def save_cached_rows(rows: pd.DataFrame, store_dir: Path | None = None) -> None:
    store = Path(store_dir or STORE_DIR)
    store.mkdir(parents=True, exist_ok=True)
    keep = [c for c in rows.columns if c not in ("r_long", "r_short", "exit_long", "exit_short")]
    tmp = store / (ROWS_FILE + ".tmp")
    rows[keep].to_csv(tmp, index=False)
    os.replace(tmp, store / ROWS_FILE)
    _write_json(store / ROWS_META_FILE, {"key": feature_cache_key(), "rows": len(rows)})


def update_rows(data: dict[str, SymbolData], cached: pd.DataFrame, max_new: int | None) -> tuple[pd.DataFrame, int, int]:
    """(rows inside the window, analysed now, still missing)."""
    ends = [d.hourly.index[-1] for d in data.values() if len(d.hourly)]
    if not ends:
        return pd.DataFrame(), 0, 0
    end = max(ends)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    have = set(zip(cached["symbol"], cached["time"])) if len(cached) else set()
    todo = []
    for sym, d in data.items():
        for i in decision_steps(d, start, end):
            if (sym, int(d.hourly.index[i].value // 10**9)) not in have:
                todo.append((int(d.hourly.index[i].value // 10**9), sym, i))
    todo.sort()  # oldest first
    batch = todo if max_new is None else todo[:max_new]
    new = [feature_row(data[sym], i) for _, sym, i in batch]
    rows = pd.concat([cached, pd.DataFrame(new)], ignore_index=True) if new else cached.copy()
    if len(rows):
        rows = rows[rows["time"] >= int(start.value // 10**9)].drop_duplicates(["symbol", "time"], keep="last")
    return rows.reset_index(drop=True), len(batch), len(todo) - len(batch)


def _train_sync(data: dict[str, SymbolData], current: dict, settings, store_dir, max_new) -> dict:
    t0 = time.monotonic()
    rows, analysed, missing = update_rows(data, load_cached_rows(store_dir), max_new)
    if len(rows):
        save_cached_rows(rows, store_dir)
    base = {"analysed_now": analysed, "steps_missing": missing, "dataset_rows": int(len(rows))}
    if missing:
        result = {"decision": "skipped", "current": current,
                  "reasons": [f"dataset still building: {missing} steps left (at most {max_new} per run)"]}
    else:
        rows = attach_outcomes(rows, data, settings)
        result = train(rows, current, settings.analyst_min_conviction)
    result["data"] = {**base, **result.get("data", {})}
    result["compute_s"] = round(time.monotonic() - t0, 1)
    return result


async def run_training_job(settings, *, store_dir: Path | None = None, max_new: int | None = MAX_NEW_ANALYSES) -> dict:
    """One full training pass; CPU work off the event loop. Always logged."""
    from app.analyst.factors import FAMILY_WEIGHTS

    started = dt.datetime.now(dt.timezone.utc)
    symbols = list(settings.analyst_symbol_list)
    rec = adopted_record(store_dir)
    current = dict(rec["weights"]) if rec and set(symbols) <= set(rec["symbols"]) else dict(FAMILY_WEIGHTS)
    data = {s: await load_symbol_data(s, settings) for s in symbols}
    result = await asyncio.to_thread(_train_sync, data, current, settings, store_dir, max_new)
    del data
    result.update(run_at=started.isoformat(), symbols=symbols, min_conviction=settings.analyst_min_conviction,
                  duration_s=round((dt.datetime.now(dt.timezone.utc) - started).total_seconds(), 1))
    record_run(result, store_dir)
    h = result.get("holdout") or {}
    logger.info("Analyst training: %s | candidate %s | holdout cand %s vs cur %s | %s",
                result["decision"], result.get("candidate"),
                _r((h.get("candidate") or {}).get("expectancy_r")), _r((h.get("current") or {}).get("expectancy_r")),
                "; ".join(result.get("reasons", [])))
    return result
