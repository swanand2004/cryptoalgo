from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    z_score = (close - mid) / std.replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_z": z_score})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index -- measures trend STRENGTH (not direction).
    Low ADX (~<20) means a choppy/ranging market where trend-following
    whipsaws; high ADX means a genuine trend is underway. Used to gate
    trend-following signals so they sit out chop instead of trading through it.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    smoothed_tr = tr.ewm(alpha=1 / period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman's Efficiency Ratio (Perry J. Kaufman, "Trading Systems and
    Methods"): net directional displacement over `period` bars divided by
    the total bar-by-bar path length covered getting there. 1.0 = price
    moved in a straight line (maximally efficient trend); near 0 = price
    round-tripped back and forth without going anywhere (chop).

    Deliberately NOT an EMA-smoothed measure like ADX -- it's a direct ratio
    of two rolling sums, so it has no inherent lag of its own the way a
    smoothed indicator does. Used as an INDEPENDENT second opinion alongside
    ADX (which measures something related but not identical, and lags by
    construction): two different ways of asking "is this actually trending"
    agreeing is stronger evidence than either alone.
    """
    net_change = (close - close.shift(period)).abs()
    path_length = close.diff().abs().rolling(period).sum()
    return (net_change / path_length.replace(0, np.nan)).clip(0, 1)


def realized_vol(close: pd.Series, period: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(period)


def _hurst_rs(log_returns: np.ndarray, min_chunk: int) -> float:
    x = log_returns[~np.isnan(log_returns)]
    if len(x) < min_chunk * 2:
        return np.nan
    chunk_sizes = []
    size = min_chunk
    while size <= len(x) // 2:
        chunk_sizes.append(size)
        size *= 2
    if len(chunk_sizes) < 2:
        return np.nan

    log_n, log_rs = [], []
    for size in chunk_sizes:
        n_chunks = len(x) // size
        rs_values = []
        for i in range(n_chunks):
            chunk = x[i * size : (i + 1) * size]
            deviations = np.cumsum(chunk - chunk.mean())
            r = deviations.max() - deviations.min()
            s = chunk.std()
            if s > 0:
                rs_values.append(r / s)
        if rs_values:
            log_n.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_values)))
    if len(log_n) < 2:
        return np.nan
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(slope)


def hurst_exponent(close: pd.Series, window: int = 100, min_chunk: int = 8) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range (R/S) analysis (Hurst,
    1951) -- a real, independently well-tested measure of a series' own
    trending (H > 0.5) vs. mean-reverting (H < 0.5) tendency; H = 0.5 is a
    pure random walk. Specifically validated for exactly this use case in
    crypto research (Sidhu et al., "Short Term Trading Models Using Hurst
    Exponent and Machine Learning" -- tested on Bitcoin among other
    assets, using the Hurst regime to segment a short-term ML model).

    Computed over the trailing `window` log-returns: log(R/S) vs.
    log(chunk_size), averaged across several geometrically-spaced chunk
    sizes from `min_chunk` up to window/2 -- the slope of that line IS the
    Hurst exponent, by construction (not an approximation of one).
    """
    log_returns = np.log(close / close.shift(1)).to_numpy()
    n = len(log_returns)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = _hurst_rs(log_returns[i - window + 1 : i + 1], min_chunk)
    return pd.Series(out, index=close.index)


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_vol = volume.cumsum().replace(0, np.nan)
    return (typical_price * volume).cumsum() / cum_vol


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the standard indicator set to an OHLCV dataframe in-place-style
    (returns a new dataframe). Expects columns: open, high, low, close, volume.
    """
    out = df.copy()
    out["ema_9"] = ema(out["close"], 9)
    out["ema_21"] = ema(out["close"], 21)
    out["ema_50"] = ema(out["close"], 50)
    out["rsi_14"] = rsi(out["close"], 14)
    out = out.join(macd(out["close"]))
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    out["adx_14"] = adx(out["high"], out["low"], out["close"], 14)
    out["efficiency_ratio_10"] = efficiency_ratio(out["close"], 10)
    out["hurst_100"] = hurst_exponent(out["close"], window=100)
    out = out.join(bollinger(out["close"]))
    out["realized_vol_20"] = realized_vol(out["close"], 20)
    out["vwap"] = vwap(out["high"], out["low"], out["close"], out["volume"])
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_5"] = out["close"].pct_change(5)
    out["ret_15"] = out["close"].pct_change(15)
    return out
