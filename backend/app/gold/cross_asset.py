from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.query import load_candles_df

CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
CORRELATION_WINDOW_BARS = 50  # ~2 days at 1h resolution

# Genuinely researched, not assumed (2026-08-22): gold and BTC/ETH/SOL show
# a real, strongly significant CONTEMPORANEOUS correlation at 1h/15m
# granularity (IC 0.20-0.32 over thousands of bars, both resolutions) --
# both react to the same real-time risk-sentiment/liquidity/USD-strength
# conditions. Checked explicitly for a LEAD/LAG version of this (does
# crypto's past move forecast gold's future one, or vice versa, at lags
# from 15 minutes to 4 hours): none survived -- the relationship is
# same-instant only, with no exploitable lead time. So this is context for
# a human interpreting the CURRENT move ("gold is moving with/against the
# broader risk-asset complex right now"), never fed to the prediction
# models as a forecasting feature -- it has no forecasting value, honestly
# confirmed by that lead-lag check, only explanatory value.
CORRELATION_STRONG = 0.5
CORRELATION_WEAK = 0.15


def _interpret(corr: float) -> str:
    if pd.isna(corr):
        return "unknown"
    if corr >= CORRELATION_STRONG:
        return "moving_with"
    if corr <= -CORRELATION_STRONG:
        return "moving_against"
    if abs(corr) <= CORRELATION_WEAK:
        return "decoupled"
    return "loosely_with" if corr > 0 else "loosely_against"


async def crypto_correlation_context(resolution: str = "1h", limit: int = 300) -> dict:
    """Rolling contemporaneous correlation between gold's and each crypto
    symbol's returns over the trailing CORRELATION_WINDOW_BARS -- real-time
    CONTEXT for a human ("is gold moving with or against risk assets right
    now"), not a prediction. See module docstring for why this is
    deliberately never used as a model feature.
    """
    from app.gold.ingest import GOLD_SPOT_SYMBOL

    gold_candles = await load_candles_df(GOLD_SPOT_SYMBOL, resolution, limit=limit)
    if len(gold_candles) < CORRELATION_WINDOW_BARS + 1:
        return {"ready": False, "resolution": resolution, "window_bars": CORRELATION_WINDOW_BARS, "assets": {}}

    gold_candles = gold_candles.set_index("time")
    gold_ret = np.log(gold_candles["close"] / gold_candles["close"].shift(1))

    assets: dict[str, dict] = {}
    for symbol in CRYPTO_SYMBOLS:
        crypto_candles = await load_candles_df(symbol, resolution, limit=limit)
        if len(crypto_candles) < CORRELATION_WINDOW_BARS + 1:
            assets[symbol] = {"correlation": None, "interpretation": "unknown", "recent_return_pct": None}
            continue
        crypto_candles = crypto_candles.set_index("time")
        crypto_ret = np.log(crypto_candles["close"] / crypto_candles["close"].shift(1))

        joined = pd.DataFrame({"gold": gold_ret, "crypto": crypto_ret}).dropna()
        window = joined.iloc[-CORRELATION_WINDOW_BARS:]
        corr = float(window["gold"].corr(window["crypto"])) if len(window) >= 10 else float("nan")
        # The crypto asset's own most recent completed-bar return -- genuine,
        # already-observable information (no lookahead), used by the
        # frontend to say which way each asset is CURRENTLY moving, so a
        # prediction card can show "does crypto currently agree with this
        # call" alongside the correlation strength.
        recent_return_pct = float(crypto_ret.iloc[-1] * 100) if len(crypto_ret) and pd.notna(crypto_ret.iloc[-1]) else None
        assets[symbol] = {
            "correlation": None if pd.isna(corr) else round(corr, 3),
            "interpretation": _interpret(corr),
            "recent_return_pct": recent_return_pct,
        }

    return {"ready": True, "resolution": resolution, "window_bars": CORRELATION_WINDOW_BARS, "assets": assets}
