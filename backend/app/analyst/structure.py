"""Chart patterns, Gann and support/resistance levels for the Analyst (user,
2026-10-06: "technical chart patterns, trend, Gann and levels analysis").

Mechanical definitions only -- every pattern is a rule on daily swing points
(Gann's 2-bar swing chart, app.strategy.gann_angle.swing_pivots), so the same
code reads history and live data the same way:

- market structure: the last two swing highs and lows -- higher highs and higher
  lows = up, lower highs and lower lows = down;
- double top / bottom: the last two swing highs (lows) within 0.5 ATR of each
  other, confirmed by a close through the swing low (high) between them;
- compression breakout: a 20-day range under half the 60-day range, then a
  close beyond it;
- levels: supports/resistances from recent swing points, the 20/50-day
  extremes and classic daily pivots; the score leans with a bounce off a level
  and against a test of one;
- Gann: the 1x1 angle (as in the live strategy), the swing-chart trend and the
  Square-of-9 breakout state.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from app.strategy.gann_angle import daily_angle_signal, swing_pivots


def _px(x: float) -> str:
    """Readable price: 85,912 / 2,701.25 / 0.2743."""
    x = float(x)
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4f}"


_PIVOT_MEMO: dict[bytes, list] = {}
_PIVOT_MEMO_MAX = 16
_PIVOT_LOCK = threading.Lock()


def _pivots(daily: pd.DataFrame) -> list[tuple[int, float, str, int]]:
    """swing_pivots(high, low, 2). One Analyst read asks for the same daily
    frame's pivots three times (structure, double pattern, levels), so the
    result is remembered by the exact high/low bytes; a copy is returned."""
    high, low = daily["high"].to_numpy(), daily["low"].to_numpy()
    if high.dtype.kind != "f" or low.dtype != high.dtype:
        return swing_pivots(high, low, 2)
    key = high.dtype.str.encode() + np.ascontiguousarray(high).tobytes() + b"|" + np.ascontiguousarray(low).tobytes()
    with _PIVOT_LOCK:
        hit = _PIVOT_MEMO.get(key)
    if hit is None:
        hit = swing_pivots(high, low, 2)
        with _PIVOT_LOCK:
            while len(_PIVOT_MEMO) >= _PIVOT_MEMO_MAX:
                _PIVOT_MEMO.pop(next(iter(_PIVOT_MEMO)))
            _PIVOT_MEMO[key] = hit
    return list(hit)


def market_structure(daily: pd.DataFrame) -> tuple[float, str] | None:
    piv = [p for p in _pivots(daily) if p[3] <= len(daily) - 1]
    tops = [p[1] for p in piv if p[2] == "top"][-2:]
    bots = [p[1] for p in piv if p[2] == "bottom"][-2:]
    if len(tops) < 2 or len(bots) < 2:
        return None
    hh, hl = tops[1] > tops[0], bots[1] > bots[0]
    if hh and hl:
        return 1.0, "higher highs and higher lows -- uptrend structure"
    if not hh and not hl:
        return -1.0, "lower highs and lower lows -- downtrend structure"
    return 0.0, ("higher highs but lower lows -- expanding, undecided" if hh
                 else "lower highs but higher lows -- contracting, undecided")


def double_pattern(daily: pd.DataFrame, atr: float) -> tuple[float, str] | None:
    piv = [p for p in _pivots(daily) if p[3] <= len(daily) - 1]
    close = float(daily["close"].iloc[-1])
    tops = [p for p in piv if p[2] == "top"]
    bots = [p for p in piv if p[2] == "bottom"]
    if len(tops) >= 2:
        t1, t2 = tops[-2], tops[-1]
        between = [p[1] for p in bots if t1[0] < p[0] < t2[0]]
        if between and abs(t1[1] - t2[1]) <= 0.5 * atr and t2[0] >= len(daily) - 30:
            neck = min(between)
            if close < neck:
                return -1.0, f"double top at {_px(t2[1])} confirmed -- closed below the {_px(neck)} neckline"
            return -0.3, f"possible double top at {_px(t2[1])} (neckline {_px(neck)} not broken yet)"
    if len(bots) >= 2:
        b1, b2 = bots[-2], bots[-1]
        between = [p[1] for p in tops if b1[0] < p[0] < b2[0]]
        if between and abs(b1[1] - b2[1]) <= 0.5 * atr and b2[0] >= len(daily) - 30:
            neck = max(between)
            if close > neck:
                return 1.0, f"double bottom at {_px(b2[1])} confirmed -- closed above the {_px(neck)} neckline"
            return 0.3, f"possible double bottom at {_px(b2[1])} (neckline {_px(neck)} not broken yet)"
    return None


def compression_breakout(daily: pd.DataFrame) -> tuple[float, str] | None:
    if len(daily) < 62:
        return None
    h, l, c = daily["high"], daily["low"], float(daily["close"].iloc[-1])
    r20 = h.iloc[-21:-1].max() - l.iloc[-21:-1].min()
    r60 = h.iloc[-61:-1].max() - l.iloc[-61:-1].min()
    if r60 <= 0 or r20 / r60 >= 0.5:
        return None
    hi, lo = h.iloc[-21:-1].max(), l.iloc[-21:-1].min()
    if c > hi:
        return 1.0, f"broke UP out of a tight range ({r20 / r60:.0%} of the 60-day range)"
    if c < lo:
        return -1.0, f"broke DOWN out of a tight range ({r20 / r60:.0%} of the 60-day range)"
    return 0.0, f"coiling: 20-day range only {r20 / r60:.0%} of the 60-day range -- breakout pending"


def key_levels(daily: pd.DataFrame, price: float) -> tuple[list[float], list[float]]:
    """(supports below price, resistances above), nearest first."""
    piv = [p for p in _pivots(daily) if p[3] <= len(daily) - 1][-12:]
    cands = [p[1] for p in piv]
    for n in (20, 50):
        if len(daily) > n:
            cands += [float(daily["high"].iloc[-n - 1:-1].max()), float(daily["low"].iloc[-n - 1:-1].min())]
    prev = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
    p = (prev["high"] + prev["low"] + prev["close"]) / 3
    cands += [p, 2 * p - prev["low"], 2 * p - prev["high"], p + (prev["high"] - prev["low"]), p - (prev["high"] - prev["low"])]
    sup = sorted({round(float(x), 10) for x in cands if x < price}, reverse=True)
    res = sorted({round(float(x), 10) for x in cands if x > price})
    return sup[:4], res[:4]


def level_position(price: float, sup: list[float], res: list[float], atr: float) -> tuple[float, str] | None:
    if not sup and not res:
        return None
    ds = (price - sup[0]) / atr if sup else np.inf
    dr = (res[0] - price) / atr if res else np.inf
    near_s = f"support {_px(sup[0])} ({ds:.1f} ATR below)" if sup else "no support nearby"
    near_r = f"resistance {_px(res[0])} ({dr:.1f} ATR above)" if res else "no resistance nearby"
    if ds <= 0.5 < dr:
        return 0.5, f"sitting on {near_s} -- bounce zone; {near_r}"
    if dr <= 0.5 < ds:
        return -0.5, f"pressing {near_r} -- rejection zone; {near_s}"
    return 0.0, f"{near_s}; {near_r}"


def gann_readings(daily: pd.DataFrame) -> list[tuple[str, str, float, str]]:
    from app.gold.signal_board import _sq9_state, _swing_trend

    out = []
    angle = float(daily_angle_signal(daily[["high", "low", "close"]].reset_index(drop=True)).iloc[-1])
    out.append(("gann_angle", "Gann 1x1 angle", angle,
                "holding above the 1x1 from the last bottom" if angle > 0 else
                "below the 1x1 from the last top" if angle < 0 else "between angles -- no Gann trend"))
    swing = float(_swing_trend(daily).iloc[-1])
    out.append(("gann_swing", "Gann swing chart", swing,
                "took out the last swing top" if swing > 0 else "broke the last swing bottom" if swing < 0 else "no swing break yet"))
    sq9 = float(_sq9_state(daily["close"], 0.5).iloc[-1])
    root = np.sqrt(float(daily["close"].iloc[-1]))
    up, dn = (np.floor(root / 0.5) + 1) * 0.5, np.floor(root / 0.5) * 0.5
    out.append(("gann_sq9", "Gann Square of 9 (90°)", sq9,
                f"last break {'up' if sq9 > 0 else 'down' if sq9 < 0 else 'none'}; next levels {_px(dn ** 2)} / {_px(up ** 2)}"))
    return out


def levels_plan(side: str, price: float, atr: float, sup: list[float], res: list[float],
                stop_atr: float, default_rr: float = 1.0) -> tuple[float, float, str]:
    """(stop, target, how) -- the stop just beyond the nearest level when that
    level is 1.0-2.5 ATR away, else stop_atr ATRs; TP1 just before the next
    level when it is 1.0-2.5R away, else default_rr R."""
    d = 1 if side == "BUY" else -1
    near = (sup if d > 0 else res)
    lvl = next((x for x in near if 1.0 * atr <= abs(price - x) <= 2.5 * atr), None)
    stop = (lvl - d * 0.25 * atr) if lvl is not None else price - d * stop_atr * atr
    risk = abs(price - stop)
    far = (res if d > 0 else sup)
    tgt_lvl = next((x for x in far if 1.0 * risk <= abs(x - price) <= 2.5 * risk), None)
    target = (tgt_lvl - d * 0.1 * atr) if tgt_lvl is not None else price + d * default_rr * risk
    how = ("stop beyond " + (f"level {_px(lvl)}" if lvl is not None else f"{stop_atr} ATR")
           + "; TP1 " + (f"before level {_px(tgt_lvl)}" if tgt_lvl is not None else f"at {default_rr:g}R"))
    return stop, target, how
