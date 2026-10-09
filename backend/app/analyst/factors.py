"""The Analyst: every free input the system has, read the same way every time.

Each factor turns one kind of evidence into a score in [-1, +1] (bearish ..
bullish) and a sentence saying why. Factors are grouped into four families
-- technical, derivatives, sentiment/fundamental, macro -- and the families
are combined with FIXED weights into one decision per coin. Nothing here is
fitted: thresholds are conventional round numbers chosen before any test, so
the same rules can be run over history (app/analyst/backtest) and live, and
the paper record shows what "analysing everything" is actually worth next to
the validated trend strategy.

Pure functions only (no I/O): callers pass pandas objects indexed by UTC time.
A factor whose input is missing returns None and simply drops out of its
family -- most derivatives/news series only exist for recent months.
"""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

# "structure" = chart patterns, Gann and support/resistance levels (user,
# 2026-10-06). Added after the backtests above -- its weight is a judgement,
# not a measurement; the live record is what tests it.
FAMILY_WEIGHTS = {"technical": 0.40, "structure": 0.20, "derivatives": 0.15, "sentiment": 0.15, "macro": 0.10}
DECISION_LINE = 0.20  # |overall score| at/above which the analyst takes a side (was 0.25; see config.analyst_min_score)


def family_weights(symbol: str | None = None) -> dict[str, float]:
    """FAMILY_WEIGHTS -- or, only while Settings.analyst_training_enabled is on,
    the weights the nightly training (app/analyst/training.py) adopted for this
    symbol after they beat these out of sample. Any problem: the constants."""
    try:
        from app.analyst.training import active_weights

        return active_weights(symbol) or FAMILY_WEIGHTS
    except Exception:
        return FAMILY_WEIGHTS
# Daily ATRs from entry. 1:1 "quick profit": with the BEST-opportunity rule
# (one trade at a time, the single highest-|score| coin each day) this was the
# only variant positive in all three windows -- +0.12R / +0.11R / +0.09R per
# trade TRAIN/VAL/TEST, 55-57% wins, ~3-day holds, max fall -5/-6/-9% at 1%
# risk; random direction -0.06/-0.05/+0.03R, live-signal picker +0.06/+0.04/
# -0.05R. A 1:2 target on every coin was +0.26/+0.12/0.00R (scratchpad
# analyst_quick.py, 2026-10-05). Modest edge, one of 6 variants: paper first.
STOP_ATR, TARGET_ATR = 1.5, 1.5


@dataclass
class Factor:
    key: str
    family: str
    name: str
    score: float  # -1 bearish .. +1 bullish
    reading: str  # the plain-language reason


@dataclass
class Analysis:
    symbol: str
    time: str
    price: float
    decision: str  # BUY / SELL / WAIT
    score: float
    conviction: int  # 0-10
    size_factor: float  # 1.0 normal; reduced around events / high volatility
    families: dict[str, float]
    factors: list[Factor] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    atr: float | None = None  # daily ATR the plan is measured in
    supports: list[float] = field(default_factory=list)  # nearest first
    resistances: list[float] = field(default_factory=list)
    plan_note: str = ""  # how the stop and TP1 were placed

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = [asdict(f) for f in self.factors]
        return d


@dataclass
class Inputs:
    """Everything known at decision time. Series must END at or before `now`."""
    hourly: pd.DataFrame  # open/high/low/close, UTC DatetimeIndex
    strategy_signal: float | None = None  # live trend_ladder + gann ensemble, [-1, 1]
    funding: pd.Series | None = None  # per-settlement rate as a fraction, 8h cadence
    open_interest: pd.Series | None = None  # hourly
    long_short_ratio: pd.Series | None = None  # hourly
    fear_greed: pd.Series | None = None  # daily 0-100
    stablecoin_supply: pd.Series | None = None  # daily USD
    news: pd.Series | None = None  # hourly sentiment [-1, 1]
    dollar: pd.Series | None = None  # daily dollar index level
    implied_vol: pd.Series | None = None  # hourly, annualised percent (DVOL)
    real_yield: pd.Series | None = None  # daily US 10y real yield (gold only)
    events: list[tuple[dt.datetime, str]] = field(default_factory=list)  # upcoming scheduled events


# ----------------------------------------------------------------- helpers
def _sign(x: float) -> float:
    return float(np.sign(x)) if np.isfinite(x) else 0.0


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    v = float((100 - 100 / (1 + rs)).iloc[-1])
    return v if np.isfinite(v) else 50.0


_OHLC = ("open", "high", "low", "close")
_RULE_NS = {"1D": 86_400 * 10**9, "4h": 4 * 3_600 * 10**9}


def _resample_pandas(hourly: pd.DataFrame, rule: str) -> pd.DataFrame:
    return hourly.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def _resample(hourly: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLC bars of `rule` -- the same numbers as pandas'
    resample(rule).agg(first/max/min/last).dropna(), ~20x faster (it was half
    of every Analyst run). UTC/naive, sorted, float frames go through numpy;
    anything else falls back to pandas. Bins start at UTC midnight, as
    pandas' default origin ("start_day") puts them for 1D and 4h."""
    step = _RULE_NS.get(rule)
    idx = hourly.index
    if (step is None or not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0
            or not (idx.tz is None or str(idx.tz) == "UTC") or not idx.is_monotonic_increasing
            or any(hourly[c].dtype != np.float64 for c in _OHLC)):
        return _resample_pandas(hourly, rule)
    t = idx.as_unit("ns").asi8
    b = t // step
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    n = len(t)
    pos = np.arange(n)
    o, h, l, c = (hourly[k].to_numpy() for k in _OHLC)
    ov, cv = ~np.isnan(o), ~np.isnan(c)
    first = np.minimum.reduceat(np.where(ov, pos, n), starts)  # first non-NaN open per bin
    last = np.maximum.reduceat(np.where(cv, pos, -1), starts)  # last non-NaN close per bin
    out_o = np.where(first < n, o[np.minimum(first, n - 1)], np.nan)
    out_c = np.where(last >= 0, c[np.maximum(last, 0)], np.nan)
    out_h = np.fmax.reduceat(h, starts)  # NaN only when the whole bin is NaN
    out_l = np.fmin.reduceat(l, starts)
    keep = ~(np.isnan(out_o) | np.isnan(out_h) | np.isnan(out_l) | np.isnan(out_c))
    labels = pd.DatetimeIndex(b[starts][keep] * step, tz=idx.tz, name=idx.name)
    return pd.DataFrame({"open": out_o[keep], "high": out_h[keep], "low": out_l[keep], "close": out_c[keep]},
                        index=labels)


def daily_atr(daily: pd.DataFrame, n: int = 14) -> float:
    h, l, c = (daily[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    # = pandas concat([...], axis=1).max(axis=1): NaN-skipping row max.
    tr = np.fmax(np.fmax(h - l, np.abs(h - pc)), np.abs(l - pc))
    return float(pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def _last(s: pd.Series | None, now: pd.Timestamp, max_age: pd.Timedelta) -> pd.Series | None:
    """The series up to `now`, or None if it has no reading fresh enough."""
    if s is None or len(s) == 0:
        return None
    idx = s.index
    if isinstance(idx, pd.DatetimeIndex) and idx.is_monotonic_increasing:
        s = s.iloc[:idx.searchsorted(now, side="right")].dropna()  # = s[s.index <= now], sorted
    else:
        s = s[s.index <= now].dropna()
    if len(s) == 0 or now - s.index[-1] > max_age:
        return None
    return s


# ----------------------------------------------------------------- technical
def technical_factors(inp: Inputs) -> list[Factor]:
    out, scored = _price_technical(inp.hourly)
    return _with_strategy(out, scored, inp.strategy_signal)


def _with_strategy(out: list[Factor], scored: bool, signal: float | None) -> list[Factor]:
    if scored and signal is not None and np.isfinite(signal):
        s = float(np.clip(signal, -1, 1))
        out.append(Factor("strategy", "technical", "Live trend strategy (ladder + Gann)", s,
                          f"signal {s:+.2f} -- the system's validated trend reading"))
    return out


def _price_technical(hourly: pd.DataFrame, daily: pd.DataFrame | None = None) -> tuple[list[Factor], bool]:
    """The price-only technical factors, and whether there was enough daily
    history to score them (without it the strategy reading is left out too)."""
    out: list[Factor] = []
    if daily is None:
        daily = _resample(hourly, "1D")
    if len(daily) < 60:
        return out, False
    c = daily["close"]
    ema20, ema50 = _ema(c, 20).iloc[-1], _ema(c, 50).iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
    trend = 0.5 * _sign(ema20 - ema50) + (0.5 * _sign(c.iloc[-1] - sma200) if np.isfinite(sma200) else 0.0)
    out.append(Factor("trend", "technical", "Daily trend (EMA 20/50, 200-day)", trend,
                      f"EMA20 {'above' if ema20 > ema50 else 'below'} EMA50"
                      + (f"; price {'above' if c.iloc[-1] > sma200 else 'below'} the 200-day average" if np.isfinite(sma200) else "")))

    macd = _ema(c, 12) - _ema(c, 26)
    hist = float((macd - _ema(macd, 9)).iloc[-1])
    rsi = _rsi(c)
    mom = 0.5 * _sign(hist) + (0.5 if rsi > 55 else -0.5 if rsi < 45 else 0.0)
    stretched = rsi > 75 or rsi < 25
    if stretched:
        mom *= 0.5
    out.append(Factor("momentum", "technical", "Momentum (MACD, RSI 14)", mom,
                      f"MACD histogram {hist:+.4g}, RSI {rsi:.0f}" + (" -- stretched, momentum discounted" if stretched else "")))

    signs = []
    for rule in ("1h", "4h", "1D"):
        f = hourly if rule == "1h" else daily if rule == "1D" else _resample(hourly, rule)
        if len(f) >= 60:
            signs.append(_sign(_ema(f["close"], 20).iloc[-1] - _ema(f["close"], 50).iloc[-1]))
    if signs:
        agree = float(np.mean(signs))
        out.append(Factor("timeframes", "technical", "Timeframe alignment (1h / 4h / daily)", agree,
                          " / ".join("up" if s > 0 else "down" for s in signs) + " (EMA 20 vs 50)"))

    hi20, lo20 = daily["high"].iloc[-21:-1].max(), daily["low"].iloc[-21:-1].min()
    last = c.iloc[-1]
    if last > hi20:
        struct, text = 1.0, f"broke above the 20-day high {hi20:,.4g}"
    elif last < lo20:
        struct, text = -1.0, f"broke below the 20-day low {lo20:,.4g}"
    else:
        pos = (last - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5
        struct, text = float(np.clip((pos - 0.5) * 2, -1, 1)) * 0.5, f"{pos:.0%} of the way up its 20-day range"
    out.append(Factor("structure", "technical", "Price structure (20-day range)", struct, text))
    return out, True


# ----------------------------------------------------------------- derivatives
def derivatives_factors(inp: Inputs, now: pd.Timestamp) -> list[Factor]:
    out: list[Factor] = []
    fr = _last(inp.funding, now, pd.Timedelta(hours=26))
    if fr is not None:
        recent = fr.iloc[-3:].mean()  # last ~24h of 8h settlements
        ann = recent * 3 * 365
        score = -1.0 if ann > 0.6 else -0.5 if ann > 0.3 else 0.5 if ann < -0.1 else 0.0
        mood = ("longs paying a lot -- crowded long" if score < 0 else "shorts paying -- crowded short" if score > 0
                else "normal")
        out.append(Factor("funding", "derivatives", "Funding rate (crowding)", score, f"{ann:+.0%}/yr annualised, {mood}"))

    oi = _last(inp.open_interest, now, pd.Timedelta(hours=3))
    if oi is not None and len(oi) >= 25:
        d_oi = oi.iloc[-1] / oi.iloc[-25] - 1
        px = inp.hourly["close"]
        d_px = px.iloc[-1] / px.iloc[-25] - 1
        if d_px >= 0:
            score, why = (0.5, "price and open interest both rising -- new longs") if d_oi > 0 else (0.2, "price up, OI down -- shorts covering")
        else:
            score, why = (-0.5, "price falling while OI rises -- new shorts") if d_oi > 0 else (-0.2, "price down, OI down -- longs exiting")
        out.append(Factor("open_interest", "derivatives", "Open interest vs price (24h)", score,
                          f"OI {d_oi:+.1%}, price {d_px:+.1%}: {why}"))

    lsr = _last(inp.long_short_ratio, now, pd.Timedelta(hours=3))
    if lsr is not None:
        r = float(lsr.iloc[-1])
        score = -0.5 if r > 2.0 else 0.5 if r < 0.8 else 0.0
        out.append(Factor("long_short", "derivatives", "Long/short account ratio", score,
                          f"{r:.2f} longs per short" + (" -- one-sided, contrarian" if score else "")))
    return out


# ----------------------------------------------------------------- sentiment / fundamentals
def sentiment_factors(inp: Inputs, now: pd.Timestamp) -> list[Factor]:
    out: list[Factor] = []
    fg = _last(inp.fear_greed, now, pd.Timedelta(days=3))
    if fg is not None:
        v = float(fg.iloc[-1])
        score = 0.5 if v <= 20 else -0.5 if v >= 80 else 0.0
        label = "extreme fear" if v <= 20 else "extreme greed" if v >= 80 else "fear" if v < 45 else "greed" if v > 55 else "neutral"
        out.append(Factor("fear_greed", "sentiment", "Fear & Greed index", score,
                          f"{v:.0f} ({label})" + (" -- contrarian signal" if score else "")))

    sc = _last(inp.stablecoin_supply, now, pd.Timedelta(days=4))
    if sc is not None and len(sc) > 400:
        g30 = np.log(sc).diff(30)
        z = (g30.iloc[-1] - g30.iloc[-365:].mean()) / (g30.iloc[-365:].std() or np.nan)
        if np.isfinite(z):
            score = 0.5 if z > 0.5 else -0.5 if z < -0.5 else 0.0
            out.append(Factor("stablecoins", "sentiment", "Stablecoin supply (liquidity)", score,
                              f"30-day growth {g30.iloc[-1]:+.1%} ({z:+.1f} vs its 1-year norm)"))

    news = _last(inp.news, now, pd.Timedelta(hours=6))
    if news is not None:
        v = float(news[news.index > now - pd.Timedelta(hours=24)].mean())
        if np.isfinite(v):
            out.append(Factor("news", "sentiment", "News sentiment (24h)", float(np.clip(v, -1, 1)) * 0.5,
                              f"headline tone {v:+.2f}"))
    return out


# ----------------------------------------------------------------- macro
_COMPUTE = object()


def _realised_vol_pct(hourly: pd.DataFrame) -> float | None:
    """Where the 7-day realised volatility sits in its last 120 days (0..1),
    or None with too little history."""
    r = np.log(hourly["close"]).diff().dropna()
    if len(r) > 24 * 120:
        rv = r.rolling(24 * 7).std()
        return float((rv.iloc[-24 * 120:] < rv.iloc[-1]).mean())
    return None


def macro_factors(inp: Inputs, now: pd.Timestamp, realised_vol_pct=_COMPUTE) -> tuple[list[Factor], float, list[str]]:
    """`realised_vol_pct`: _realised_vol_pct(inp.hourly) when the caller already
    has it (analyse memoises it per closed bar); computed here otherwise."""
    out: list[Factor] = []
    size, notes = 1.0, []
    dx = _last(inp.dollar, now, pd.Timedelta(days=7))
    if dx is not None and len(dx) >= 50:
        below = dx.iloc[-1] < dx.iloc[-50:].mean()
        out.append(Factor("dollar", "macro", "US dollar trend", 0.3 if below else -0.3,
                          f"dollar {'below' if below else 'above'} its 50-day average -- "
                          + ("supportive for crypto and gold" if below else "a headwind for crypto and gold")))
    ry = _last(inp.real_yield, now, pd.Timedelta(days=7))
    if ry is not None and len(ry) >= 50:
        falling = ry.iloc[-1] < ry.iloc[-50:].mean()
        out.append(Factor("real_yield", "macro", "US real yields (10y TIPS)", 0.4 if falling else -0.4,
                          f"{ry.iloc[-1]:.2f}% vs its 50-day average {ry.iloc[-50:].mean():.2f}% -- "
                          + ("falling real yields support gold" if falling else "rising real yields weigh on gold")))
    soon = [(t, name) for t, name in inp.events if now <= pd.Timestamp(t) <= now + pd.Timedelta(hours=24)]
    if soon:
        size *= 0.5
        notes.append("Scheduled event within 24h (" + ", ".join(n for _, n in soon) + ") -- size halved")
    iv = _last(inp.implied_vol, now, pd.Timedelta(hours=6))
    vol_text = None
    if iv is not None and len(iv) > 24 * 30:
        pct = float((iv.iloc[-24 * 90:] < iv.iloc[-1]).mean())
        if pct > 0.8:
            size *= 0.7
            vol_text = f"implied volatility {iv.iloc[-1]:.0f} (top 20% of 90 days) -- size reduced"
    else:
        pct = _realised_vol_pct(inp.hourly) if realised_vol_pct is _COMPUTE else realised_vol_pct
        if pct is not None and pct > 0.8:
            size *= 0.7
            vol_text = "7-day realised volatility in the top 20% of 120 days -- size reduced"
    if vol_text:
        notes.append(vol_text)
    return out, size, notes


# ----------------------------------------------------------------- per-closed-bar memo
# Everything derived from price alone -- daily/4h bars, the technical factors,
# the daily ATR, patterns, levels and Gann -- changes only when a new hourly
# bar closes, but the Analyst runs every minute. It is computed once per
# distinct hourly frame (a hit needs its index and OHLC values to be bit-for-bit
# equal to the stored copy, so a changed, extended or trimmed frame is always a
# miss) and re-used until then;
# the live inputs (strategy signal, funding, OI, sentiment, events) are still
# read fresh on every run. Callers get copies, so nothing can alter the memo.
PRICE_MEMO_ENABLED = True
_PRICE_MEMO: dict[tuple, tuple] = {}
_PRICE_MEMO_MAX = 32
_PRICE_MEMO_LOCK = threading.Lock()


def _fingerprint(hourly: pd.DataFrame) -> tuple[tuple, np.ndarray] | None:
    """(cheap key, exact content). A memo entry is used only when the stored
    content is bit-for-bit equal, so a hit can never be a different frame."""
    try:
        if any(hourly[k].dtype.kind not in "fiu" for k in _OHLC):
            return None  # object columns: bytes would be pointers, not values
        t = hourly.index.as_unit("ns").asi8
        content = np.concatenate([t.view(np.uint8)] + [np.ascontiguousarray(hourly[k].to_numpy()).view(np.uint8)
                                                         for k in _OHLC])
        # Several coins share the same bar times, so the key also carries the
        # first/last closes (as raw bytes); equality of `content` decides a hit.
        c = np.ascontiguousarray(hourly["close"].to_numpy())
        key = (len(hourly), int(t[0]), int(t[-1]), c[0].tobytes(), c[-1].tobytes(), str(hourly.index.tz),
               tuple(str(hourly[k].dtype) for k in _OHLC))
        return key, content
    except Exception:  # not a plain OHLC frame on a DatetimeIndex: just compute
        return None


def _price_block(hourly: pd.DataFrame) -> tuple:
    daily = _resample(hourly, "1D")
    tech, scored = _price_technical(hourly, daily)
    atr = daily_atr(daily)
    price_now = float(hourly["close"].iloc[-1])
    struct, supports, resistances = structure_factors(daily, price_now, atr)
    return tech, scored, atr, struct, supports, resistances, _realised_vol_pct(hourly)


def _price_block_cached(hourly: pd.DataFrame) -> tuple:
    fp = _fingerprint(hourly) if PRICE_MEMO_ENABLED else None
    block = None
    if fp is not None:
        with _PRICE_MEMO_LOCK:
            hit = _PRICE_MEMO.get(fp[0])
        if hit is not None and np.array_equal(hit[0], fp[1]):
            block = hit[1]
    if block is None:
        block = _price_block(hourly)
        if fp is not None:
            with _PRICE_MEMO_LOCK:
                _PRICE_MEMO.pop(fp[0], None)
                while len(_PRICE_MEMO) >= _PRICE_MEMO_MAX:
                    _PRICE_MEMO.pop(next(iter(_PRICE_MEMO)))  # oldest first
                _PRICE_MEMO[fp[0]] = (fp[1], block)
    tech, scored, atr, struct, supports, resistances, rv_pct = block
    return ([replace(f) for f in tech], scored, atr, [replace(f) for f in struct], list(supports), list(resistances),
            rv_pct)


def clear_price_memo() -> None:
    with _PRICE_MEMO_LOCK:
        _PRICE_MEMO.clear()


# ----------------------------------------------------------------- decision
def analyse(symbol: str, inp: Inputs, now: pd.Timestamp | None = None) -> Analysis | None:
    hourly = inp.hourly.dropna(subset=["close"])
    if now is None:
        now = hourly.index[-1]
    hourly = hourly[hourly.index <= now]
    if len(hourly) < 24 * 60:
        return None
    inp = Inputs(**{**inp.__dict__, "hourly": hourly})
    tech, scored, atr, struct, supports, resistances, rv_pct = _price_block_cached(hourly)
    factors = _with_strategy(tech, scored, inp.strategy_signal)
    factors += derivatives_factors(inp, now) + sentiment_factors(inp, now)
    macro, size, notes = macro_factors(inp, now, realised_vol_pct=rv_pct)
    factors += macro
    factors += struct

    families: dict[str, float] = {}
    weights = family_weights(symbol)
    for fam in weights:
        vals = [f.score for f in factors if f.family == fam]
        if vals:
            families[fam] = float(np.mean(vals))
    weight = sum(weights[f] for f in families)
    score = sum(weights[f] * v for f, v in families.items()) / weight if weight else 0.0
    decision = "BUY" if score >= DECISION_LINE else "SELL" if score <= -DECISION_LINE else "WAIT"
    price = float(hourly["close"].iloc[-1])
    a = Analysis(
        symbol=symbol, time=now.isoformat(), price=price, decision=decision, score=score,
        conviction=int(min(10, round(abs(score) * 10))), size_factor=size, families=families,
        factors=factors, notes=notes,
    )
    a.atr, a.supports, a.resistances = atr, supports, resistances
    if decision != "WAIT":
        from app.analyst.structure import levels_plan

        stop, target, how = levels_plan(decision, price, atr, supports, resistances, STOP_ATR,
                                        default_rr=TARGET_ATR / STOP_ATR)
        a.entry, a.stop_loss, a.take_profit, a.plan_note = price, stop, target, how
    return a


def structure_factors(daily: pd.DataFrame, price: float, atr: float) -> tuple[list[Factor], list[float], list[float]]:
    """Chart patterns, Gann and levels (app/analyst/structure.py) as factors."""
    from app.analyst import structure as st

    out: list[Factor] = []
    if len(daily) < 30 or not np.isfinite(atr) or atr <= 0:
        return out, [], []
    for key, name, fn in (("structure", "Market structure (swing highs / lows)", lambda: st.market_structure(daily)),
                          ("pattern", "Chart pattern (double top / bottom)", lambda: st.double_pattern(daily, atr)),
                          ("compression", "Range compression / breakout", lambda: st.compression_breakout(daily))):
        r = fn()
        if r is not None:
            out.append(Factor(key, "structure", name, r[0], r[1]))
    sup, res = st.key_levels(daily, price)
    lp = st.level_position(price, sup, res, atr)
    if lp is not None:
        out.append(Factor("levels", "structure", "Support / resistance levels", lp[0], lp[1]))
    for key, name, score, reading in st.gann_readings(daily):
        out.append(Factor(key, "structure", name, score, reading))
    return out, sup, res
