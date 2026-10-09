from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from app.strategy.gann_angle import daily_angle_signal, swing_pivots

# Gold Signal Board: a direction call from classic technicals + W.D. Gann, with
# the time each signal started and each voice's own track record.
#
# Direction-neutral by construction: every indicator votes +1 / 0 / -1 with
# equal weight and nothing is tilted toward "up". That is deliberate (it was
# asked for), and it is also why each vote carries its MEASURED hit rate beside
# the always-up rate on the same bars: the 2026-10-02 study of 45 variations on
# 26 years of COMEX gold found none that beat always-long out of sample, so the
# board must show the evidence next to the opinion rather than instead of it.
#
# Equal weights, fixed thresholds, nothing fitted -- a board tuned to its own
# history would show flattering hit rates that mean nothing.

BUY_LINE = 0.20  # consensus score at/above which the board says BUY (below -0.20: SELL)

FAMILY_TREND, FAMILY_OSC, FAMILY_GANN = "Trend", "Oscillator", "Gann"


def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False).mean()


def _three_way(cond_up: pd.Series, cond_down: pd.Series) -> pd.Series:
    return pd.Series(np.where(cond_up, 1.0, np.where(cond_down, -1.0, 0.0)), index=cond_up.index)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up, down = _wilder(delta.clip(lower=0), n), _wilder(-delta.clip(upper=0), n)
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _sq9_state(close: pd.Series, step: float) -> pd.Series:
    """Square-of-9 breakout state: +1 after a close through the next level up
    (in sqrt-price space, `step` = 0.5 is a 90-degree turn), -1 after a close
    through the next level down."""
    level = np.floor(np.sqrt(close.to_numpy()) / step)
    out, ref, state = np.zeros(len(close)), level[0], 0.0
    for i, lv in enumerate(level):
        if lv >= ref + 1:
            state, ref = 1.0, lv
        elif lv <= ref - 1:
            state, ref = -1.0, lv
        out[i] = state
    return pd.Series(out, index=close.index)


def _swing_trend(df: pd.DataFrame) -> pd.Series:
    """Gann's swing-chart trend: up once price takes out the last swing top,
    down once it breaks the last swing bottom."""
    pivots = swing_pivots(df["high"].to_numpy(), df["low"].to_numpy(), 2)
    close = df["close"].to_numpy()
    out, k, top, bottom, trend = np.zeros(len(df)), 0, None, None, 0.0
    for i in range(len(df)):
        while k < len(pivots) and pivots[k][3] <= i:
            if pivots[k][2] == "top":
                top = pivots[k][1]
            else:
                bottom = pivots[k][1]
            k += 1
        if top is not None and close[i] > top:
            trend = 1.0
        elif bottom is not None and close[i] < bottom:
            trend = -1.0
        out[i] = trend
    return pd.Series(out, index=df.index)


def indicator_votes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Every indicator's vote on every bar, plus a short live reading of each."""
    close, high, low = df["close"], df["high"], df["low"]
    votes: dict[str, pd.Series] = {}
    meta: dict[str, dict] = {}

    def add(key, name, family, vote, detail):
        votes[key] = vote.fillna(0.0)
        meta[key] = {"name": name, "family": family, "detail": detail}

    ema20, ema50 = close.ewm(span=20, adjust=False).mean(), close.ewm(span=50, adjust=False).mean()
    add("ema_20_50", "EMA 20 / 50 cross", FAMILY_TREND, np.sign(ema20 - ema50),
        f"EMA20 {ema20.iloc[-1]:,.2f} vs EMA50 {ema50.iloc[-1]:,.2f}")

    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    add("sma_50_200", "Golden / death cross (50/200)", FAMILY_TREND, np.sign(sma50 - sma200),
        f"SMA50 {sma50.iloc[-1]:,.2f} vs SMA200 {sma200.iloc[-1]:,.2f}")

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    add("macd", "MACD histogram (12,26,9)", FAMILY_TREND, np.sign(hist), f"histogram {hist.iloc[-1]:+.3f}")

    for n in (20, 120):
        mom = close / close.shift(n) - 1
        add(f"mom_{n}", f"Momentum ({n} bars)", FAMILY_TREND, np.sign(mom), f"{mom.iloc[-1] * 100:+.2f}%")

    hi20, lo20 = high.rolling(20).max().shift(1), low.rolling(20).min().shift(1)
    donchian = pd.Series(np.where(close > hi20, 1.0, np.where(close < lo20, -1.0, np.nan)), index=df.index).ffill()
    add("donchian_20", "Donchian 20 breakout", FAMILY_TREND, donchian,
        f"channel {lo20.iloc[-1]:,.2f} - {hi20.iloc[-1]:,.2f}")

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr14 = _wilder(tr, 14)
    plus_di, minus_di = 100 * _wilder(plus_dm, 14) / atr14, 100 * _wilder(minus_dm, 14) / atr14
    adx = _wilder(100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan), 14)
    add("adx_di", "ADX / DI direction", FAMILY_TREND, _three_way((adx > 20) & (plus_di > minus_di), (adx > 20) & (minus_di > plus_di)),
        f"ADX {adx.iloc[-1]:.1f}, +DI {plus_di.iloc[-1]:.1f} / -DI {minus_di.iloc[-1]:.1f}")

    rsi14, rsi2 = _rsi(close, 14), _rsi(close, 2)
    add("rsi_14", "RSI 14 (30 / 70)", FAMILY_OSC, _three_way(rsi14 < 30, rsi14 > 70), f"RSI {rsi14.iloc[-1]:.1f}")
    add("rsi_2", "RSI 2 (10 / 90)", FAMILY_OSC, _three_way(rsi2 < 10, rsi2 > 90), f"RSI {rsi2.iloc[-1]:.1f}")

    mid, sd = close.rolling(20).mean(), close.rolling(20).std()
    z = (close - mid) / sd
    add("bollinger", "Bollinger bands (20, 2)", FAMILY_OSC, _three_way(z < -2, z > 2), f"z-score {z.iloc[-1]:+.2f}")

    lowest, highest = low.rolling(14).min(), high.rolling(14).max()
    stoch = (100 * (close - lowest) / (highest - lowest).replace(0, np.nan)).rolling(3).mean()
    add("stochastic", "Stochastic (14, 3)", FAMILY_OSC, _three_way(stoch < 20, stoch > 80), f"%K {stoch.iloc[-1]:.1f}")

    add("gann_swing", "Gann swing chart (2-bar)", FAMILY_GANN, _swing_trend(df), "trend from swing tops / bottoms")
    angle = daily_angle_signal(df[["high", "low", "close"]].reset_index(drop=True))
    add("gann_angle", "Gann 1x1 angle", FAMILY_GANN, pd.Series(angle.to_numpy(), index=df.index),
        "1x1 = 1 ATR per bar from the last swing pivot")
    add("gann_sq9", "Gann Square of 9 (90-degree)", FAMILY_GANN, _sq9_state(close, 0.5),
        "breakout through the next 90-degree level")

    return pd.DataFrame(votes), meta


def _since(series: pd.Series) -> int:
    """Position of the bar on which the current value began."""
    values = series.to_numpy()
    i = len(values) - 1
    while i > 0 and values[i - 1] == values[-1]:
        i -= 1
    return i


def _track_record(vote: pd.Series, forward: pd.Series, warmup: int) -> dict:
    """Hit rate of this voice's non-neutral calls over the frame, against what
    always-up scored on the SAME bars. Flat outcomes are excluded (PAXG prints
    many; counting them as misses distorts every figure)."""
    v, f = vote.iloc[warmup:], forward.iloc[warmup:]
    mask = (v != 0) & f.notna() & (f != 0)
    n = int(mask.sum())
    if n < 20:
        return {"hit": None, "base": None, "n": n}
    hit = float((np.sign(v[mask]) == np.sign(f[mask])).mean())
    return {"hit": hit, "base": float((f[mask] > 0).mean()), "n": n}


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()


def _gann_levels(df: pd.DataFrame, bar_seconds: int) -> dict:
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    times = df["time"].to_numpy(dtype=float)
    pivots = swing_pivots(high, low, 2)
    last_top = next((p for p in reversed(pivots) if p[2] == "top"), None)
    last_bottom = next((p for p in reversed(pivots) if p[2] == "bottom"), None)
    price, now_i = float(close[-1]), len(df) - 1
    atr = pd.Series(np.maximum.reduce([high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))])).ewm(
        alpha=1 / 14, adjust=False).mean().to_numpy()

    def pivot_info(p):
        return None if p is None else {"price": p[1], "time": _iso(times[p[0]] + bar_seconds), "bars_ago": now_i - p[0]}

    angles = []
    for p, sign in ((last_bottom, 1), (last_top, -1)):
        if p is None:
            continue
        for label, ratio in (("1x2", 0.5), ("1x1", 1.0), ("2x1", 2.0)):
            line = p[1] + sign * ratio * atr[p[0]] * (now_i - p[0])
            angles.append({
                "name": f"{label} {'up from bottom' if sign > 0 else 'down from top'}",
                "price": float(line),
                "price_is": "above" if price > line else "below",
            })

    # Square of 9 from the current price: each 45 degrees is 0.25 in sqrt space.
    root = np.sqrt(price)
    sq9 = []
    for k in range(1, 9):
        deg = k * 45
        sq9.append({"price": float((root + k * 0.25) ** 2), "angle": deg, "kind": "resistance"})
        sq9.append({"price": float((root - k * 0.25) ** 2), "angle": deg, "kind": "support"})
    sq9 = sorted([lv for lv in sq9 if lv["angle"] in (45, 90, 180, 360)], key=lambda lv: lv["price"])

    # Gann time cycles counted from the MAJOR top and bottom -- the highest high
    # and lowest low of the last 200 bars. Shown as dates to watch, not a vote:
    # on 26 years of gold, Gann's cycle lengths did no better than random ones.
    cycles, w = [], 20
    # Daily only: Gann counted calendar days; 90 HOURS from a top is not his idea.
    if bar_seconds >= 86_400 and len(df) > 2 * w:
        major_top = int(np.argmax(high[-(w * 10):])) + len(df) - min(len(df), w * 10)
        major_bot = int(np.argmin(low[-(w * 10):])) + len(df) - min(len(df), w * 10)
        for idx, kind in ((major_top, "top"), (major_bot, "bottom")):
            for n in (30, 45, 60, 90, 120, 144, 180, 360):
                due = times[idx] + n * bar_seconds
                if due > times[-1]:
                    cycles.append({"due": _iso(due), "bars": n, "from": kind, "pivot_time": _iso(times[idx]),
                                   "pivot_price": float(high[idx] if kind == "top" else low[idx])})
        cycles = sorted(cycles, key=lambda c: c["due"])[:6]

    return {
        "swing": {"last_top": pivot_info(last_top), "last_bottom": pivot_info(last_bottom)},
        "angles": angles,
        "square_of_nine": sq9,
        "time_cycles": cycles,
    }



# ---------------------------------------------------------------- trade plan
# One take-profit (user rule, 2026-10-05): stop and target both 1.5 ATR from
# entry (1:1), and nothing else closes a trade. The old TP1/TP2 pair is gone.
STOP_ATR, TP_ATR = 1.5, 1.5


HISTORY_ROWS = 60  # closed trades returned per timeframe (stats use all of them)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def simulate_trade(df: pd.DataFrame, atr: np.ndarray, start: int, direction: int, close_times: np.ndarray) -> dict:
    """One trade from entry to exit, walked bar by bar.

    Entry at the close of bar `start`; stop and the single target in ATRs
    from entry. It ends ONLY at the stop or the target -- a bar that touches
    both counts as stopped (the conservative reading: bar data cannot say
    which came first). With neither hit yet, the trade is open and marked to
    the last close.
    """
    close, high, low = (df[c].to_numpy(dtype=float) for c in ("close", "high", "low"))
    entry, a = float(close[start]), float(atr[start])
    risk = STOP_ATR * a
    sl, tp = entry - direction * risk, entry + direction * TP_ATR * a
    exit_i, exit_price, result = None, None, None
    for i in range(start + 1, len(df)):
        if low[i] <= sl if direction > 0 else high[i] >= sl:
            exit_i, exit_price, result = i, sl, "SL"
            break
        if high[i] >= tp if direction > 0 else low[i] <= tp:
            exit_i, exit_price, result = i, tp, "TP"
            break

    is_open = exit_i is None
    mark = float(close[-1]) if is_open else exit_price
    return {
        "side": "BUY" if direction > 0 else "SELL",
        "entry_time": _iso(close_times[start]), "entry": entry,
        "stop_loss": sl, "take_profit": tp,
        "risk_reward": TP_ATR / STOP_ATR,
        "exit_time": None if is_open else _iso(close_times[exit_i]),
        "exit_price": None if is_open else exit_price,
        "result": result,
        "status": "ACTIVE" if is_open else {"TP": "TP HIT", "SL": "STOPPED OUT"}[result],
        "open": is_open,
        "closed_time": None if is_open else _iso(close_times[exit_i]),
        "pnl_pct": direction * (mark / entry - 1) * 100,
        "r_multiple": direction * (mark - entry) / risk if risk > 0 else 0.0,
        "bars_held": (len(df) - 1 if is_open else exit_i) - start,
    }


def trade_plan(df: pd.DataFrame, start: int, direction: int, close_times: np.ndarray) -> dict | None:
    """The trade opened at bar `start`, followed to now (no signal exit)."""
    if direction == 0:
        return None
    return simulate_trade(df, _atr(df).to_numpy(), start, direction, close_times)


def trade_history(df: pd.DataFrame, direction: pd.Series, close_times: np.ndarray, warmup: int) -> list[dict]:
    """Every trade the board would have taken. One trade at a time: it opens
    on the first bar of a BUY/SELL run of the consensus and closes ONLY at its
    stop or its target (user rule, 2026-10-05) -- the signal changing does not close
    it, and runs that begin while it is still open are skipped. Oldest first;
    the last one may still be open.

    Measured before shipping (scratchpad gold_board_study.py, 1% risk per
    trade, Delta XAUT costs): no exit rule makes this board profitable -- per
    trade, both sides SL/TP-only -0.23R/-0.18R hourly (TRAIN/VAL), worse than
    buying at random times (-0.16R/-0.09R); moving the stop to entry after TP1
    was worse still (-0.30R/-0.32R); a single 1:1 target -0.18R/-0.26R vs
    random-long -0.10R/-0.07R (gold_web_study.py). The rule is the user's
    choice of exit, not an edge.
    """
    d = direction.to_numpy()
    atr = _atr(df).to_numpy()
    trades: list[dict] = []
    busy_until = -1
    for i in range(max(warmup, 1), len(d)):
        if i <= busy_until or d[i] == 0 or d[i] == d[i - 1]:
            continue
        trade = simulate_trade(df, atr, i, int(d[i]), close_times)
        trades.append(trade)
        if trade["open"]:
            break
        busy_until = i + trade["bars_held"]
    return trades


def history_stats(closed: list[dict]) -> dict:
    if not closed:
        return {"trades": 0}
    pnl = [t["pnl_pct"] for t in closed]
    by = lambda *names: sum(1 for t in closed if t["result"] in names)  # noqa: E731
    return {
        "trades": len(closed),
        "since": closed[0]["entry_time"],
        "tp": by("TP"), "sl": by("SL"),
        "win_rate": sum(1 for x in pnl if x > 0) / len(pnl),
        "net_pct": float(np.sum(pnl)),
        "avg_pct": float(np.mean(pnl)),
    }


def _pivots(df: pd.DataFrame, period_seconds: int, offset_days: int = 0) -> dict | None:
    """Classic floor pivots from the last COMPLETED higher period (previous UTC
    day for the hourly view, previous Monday-start week for the daily view)."""
    days = df["time"].astype("int64") // 86_400
    key = (days + offset_days) // (period_seconds // 86_400)
    grouped = df.assign(_k=key).groupby("_k").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    if len(grouped) < 2:
        return None
    h, l, c = (float(grouped.iloc[-2][k]) for k in ("high", "low", "close"))
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l),
            "R3": h + 2 * (p - l), "S3": l - 2 * (h - p)}


def key_levels(df: pd.DataFrame, gann: dict, bar_seconds: int) -> list[dict]:
    """Every level worth drawing, labelled by source, sorted by price."""
    levels: list[dict] = []
    piv = _pivots(df, 86_400 * 7, offset_days=3) if bar_seconds >= 86_400 else _pivots(df, 86_400)
    period = "weekly" if bar_seconds >= 86_400 else "daily"
    for name, value in (piv or {}).items():
        levels.append({"price": value, "label": f"{period.title()} pivot {name}", "source": "pivot"})
    for lv in gann["square_of_nine"]:
        if lv["angle"] in (45, 90, 180):
            levels.append({"price": lv["price"], "label": f"Gann Sq9 {lv['angle']}°", "source": "gann"})
    for which, label in (("last_top", "Swing high"), ("last_bottom", "Swing low")):
        if gann["swing"][which]:
            levels.append({"price": gann["swing"][which]["price"], "label": label, "source": "swing"})
    levels.append({"price": float(df["high"].iloc[-20:].max()), "label": "20-bar high", "source": "range"})
    levels.append({"price": float(df["low"].iloc[-20:].min()), "label": "20-bar low", "source": "range"})
    price = float(df["close"].iloc[-1])
    for lv in levels:
        lv["kind"] = "resistance" if lv["price"] > price else "support"
    return sorted(levels, key=lambda lv: lv["price"], reverse=True)


def _family_summary(votes: pd.DataFrame, meta: dict) -> list[dict]:
    out = []
    for family in (FAMILY_TREND, FAMILY_OSC, FAMILY_GANN):
        cols = [k for k in votes.columns if meta[k]["family"] == family]
        last = votes[cols].iloc[-1]
        mean = float(last.mean())
        label = ("Strong Buy" if mean >= 0.5 else "Buy" if mean >= 0.1 else
                 "Strong Sell" if mean <= -0.5 else "Sell" if mean <= -0.1 else "Neutral")
        out.append({"family": family, "label": label, "score": mean,
                    "buy": int((last > 0).sum()), "sell": int((last < 0).sum()), "neutral": int((last == 0).sum())})
    return out


def build_board(candles: pd.DataFrame, *, bar_seconds: int, horizon_bars: int, label: str, now_ts: float | None = None) -> dict | None:
    """One timeframe's board from raw OHLC candles (time = bar OPEN, seconds)."""
    df = candles.dropna(subset=["close"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    now_ts = now_ts if now_ts is not None else dt.datetime.now(dt.timezone.utc).timestamp()
    if len(df) and df["time"].iloc[-1] + bar_seconds > now_ts:
        df = df.iloc[:-1].reset_index(drop=True)  # never vote on an unfinished bar
    warmup = 200
    if len(df) < warmup + 50:
        return None

    votes, meta = indicator_votes(df)
    forward = df["close"].shift(-horizon_bars) / df["close"] - 1
    close_times = df["time"].to_numpy(dtype=float) + bar_seconds

    indicators = []
    for key, col in votes.items():
        start = _since(col)
        indicators.append({
            "key": key, **meta[key],
            "vote": int(col.iloc[-1]),
            "since": _iso(close_times[start]),
            "price_at_since": float(df["close"].iloc[start]),
            "bars_since": len(df) - 1 - start,
            **_track_record(col, forward, warmup),
        })

    score = votes.mean(axis=1)
    direction = pd.Series(np.where(score >= BUY_LINE, 1.0, np.where(score <= -BUY_LINE, -1.0, 0.0)), index=df.index)
    start = _since(direction)
    price = float(df["close"].iloc[-1])
    counts = votes.iloc[-1]
    gann = _gann_levels(df, bar_seconds)
    signal_dir = int(direction.iloc[-1])
    day_ago = df["time"].iloc[-1] - 86_400
    ref = df.loc[df["time"] <= day_ago, "close"]
    hi20, lo20 = float(df["high"].iloc[-21:-1].max()), float(df["low"].iloc[-21:-1].min())
    # Trades: the open one (until its stop or target -- it can outlive the signal
    # that opened it) and the closed ones. A trade that hit its target or stop is
    # CLOSED and leaves the card; the next trade comes with the next signal.
    trades = trade_history(df, direction, close_times, warmup)
    closed = [t for t in trades if not t["open"]]
    current_trade = trades[-1] if trades and trades[-1]["open"] else None
    if not trades and signal_dir != 0:  # the run began inside the warm-up window the log skips
        mine = trade_plan(df, start, signal_dir, close_times)
        if mine and mine["open"]:
            current_trade = mine
        elif mine:
            closed.append(mine)
    # A BUY/SELL signal with no open trade (its trade already closed, or it
    # began while another trade was still open): show the levels a trade
    # entered NOW would carry -- entry at the last close, the same stop and
    # TP1 distances -- so the card always has an entry / SL / TP1.
    plan = None
    if current_trade is None and signal_dir != 0:
        a = float(_atr(df).iloc[-1])
        plan = {
            "side": "BUY" if signal_dir > 0 else "SELL",
            "entry": price,
            "stop_loss": price - signal_dir * STOP_ATR * a,
            "take_profit": price + signal_dir * TP_ATR * a,
        }
    return {
        "timeframe": label,
        "horizon_bars": horizon_bars,
        "bar_close_time": _iso(close_times[-1]),
        "price": price,
        "consensus": {
            "direction": {1.0: "BUY", -1.0: "SELL", 0.0: "NEUTRAL"}[float(direction.iloc[-1])],
            "score": float(score.iloc[-1]),
            "buy": int((counts > 0).sum()), "sell": int((counts < 0).sum()), "neutral": int((counts == 0).sum()),
            "since": _iso(close_times[start]),
            "price_at_signal": float(df["close"].iloc[start]),
            "change_since_pct": float(price / df["close"].iloc[start] - 1) * 100,
            "bars_since": len(df) - 1 - start,
            **_track_record(direction, forward, warmup),
        },
        "indicators": indicators,
        "gann": gann,
        "change_24h_pct": float(price / ref.iloc[-1] - 1) * 100 if len(ref) else None,
        "trade": current_trade,
        "plan": plan,
        "last_trade": closed[-1] if closed else None,
        "history": list(reversed(closed))[:HISTORY_ROWS],
        "history_stats": history_stats(closed),
        "triggers": None if current_trade or plan else {"buy_above": hi20, "sell_below": lo20},
        "levels": key_levels(df, gann, bar_seconds),
        "families": _family_summary(votes, meta),
    }
