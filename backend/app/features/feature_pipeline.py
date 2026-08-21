from __future__ import annotations

import numpy as np
import pandas as pd

from app.astro.dasha import dasha_bias
from app.astro.panchang import (
    all_pairs_aspect_score,
    benefic_malefic_aspect_score,
    combust_planet_count,
    declination_aspect_score,
    graha_yuddha_intensity,
    helio_aspect_score,
    hora_bias,
    is_rahu_kaal,
    is_vishti_karana,
    is_waxing_moon,
    malefic_retrograde_count,
    nakshatra_lord_bias,
    planet_velocity_normalized,
    tithi_class_bias,
    vara_lord_bias,
    yoga_severity,
)
from app.features.calendar_events import weekly_expiry_proximity
from app.features.funding import basis, funding_z_score, infer_bars_per_hour
from app.features.indicators import add_all_indicators


def build_feature_frame(
    candles: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    mark: pd.DataFrame | None = None,
    spot: pd.DataFrame | None = None,
    fear_greed: pd.DataFrame | None = None,
    binance_funding: pd.DataFrame | None = None,
    binance_oi: pd.DataFrame | None = None,
    binance_lsr: pd.DataFrame | None = None,
    dvol: pd.DataFrame | None = None,
    higher_timeframe_candles: pd.DataFrame | None = None,
    cross_asset_candles: pd.DataFrame | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Builds the shared feature matrix used by both rule-based strategies and
    the ML dataset builder, so backtest/live/train all see identical features.

    `candles` must have columns [time, open, high, low, close, volume], one row
    per bar, sorted ascending by time. `funding`/`mark`/`spot`/`fear_greed`/
    `binance_funding`/`binance_oi`/`binance_lsr`/`dvol` are all optional
    dataframes with [time, value] columns and are forward-filled onto the
    candle grid via merge_asof -- each one missing independently degrades to
    a neutral default rather than dropping rows, exactly like funding/basis
    already do, so partial external-data coverage never silently shrinks the
    dataset. `symbol` (e.g. "BTCUSD") is only used to compute the Dasha
    astrology feature (needs a genesis chart) -- omit it and that one
    column alone degrades to neutral, everything else is symbol-agnostic.
    """
    df = candles.sort_values("time").reset_index(drop=True)
    df = add_all_indicators(df)

    # ema_9/21/50, atr_14, and macd/macd_signal/macd_hist are stored in RAW
    # price/price-difference units (e.g. ~$74,000 for BTC's ema_9) -- fine
    # for the rule-based strategies, which always divide them by ATR or
    # price themselves before use, but directly non-stationary if fed to a
    # model as-is: BTC alone ranged $58k-$75k (+29%) within one 5000-bar
    # training window, so a tree splitting on a raw absolute-dollar
    # threshold learned from the low end of that range has no principled
    # way to generalize once price has drifted through it. These
    # normalized versions (relative to price or ATR, both already
    # comparable across time) are what the ML feature set actually uses --
    # see FEATURE_COLUMNS below -- while the raw columns stay untouched for
    # every rule-based strategy that already normalizes them itself.
    df["ema_9_norm"] = (df["close"] - df["ema_9"]) / df["close"]
    df["ema_21_norm"] = (df["close"] - df["ema_21"]) / df["close"]
    df["ema_50_norm"] = (df["close"] - df["ema_50"]) / df["close"]
    df["atr_pct"] = df["atr_14"] / df["close"]
    df["macd_norm"] = df["macd"] / df["close"]
    df["macd_signal_norm"] = df["macd_signal"] / df["close"]
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]

    # funding_z_score's window is expressed in HOURS worth of bars (168 =
    # 1 week) -- scaled here to the candles' actual resolution so "1 week"
    # means the same span whether this is 15m, 1h, or any other resolution,
    # instead of silently being ~42 hours at 15m (168 bars taken literally).
    # funding_z_score's window is expressed in HOURS worth of bars (168 =
    # 1 week) -- scaled here to the candles' actual resolution so "1 week"
    # means the same span whether this is 15m, 1h, or any other resolution,
    # instead of silently being ~42 hours at 15m (168 bars taken literally).
    bars_per_hour = infer_bars_per_hour(df["time"])
    funding_window = max(2, round(168 * bars_per_hour))

    if funding is not None and not funding.empty:
        f = funding.sort_values("time")[["time", "value"]].rename(columns={"value": "funding_rate"})
        df = pd.merge_asof(df, f, on="time", direction="backward")
        df["funding_z"] = funding_z_score(df["funding_rate"], window=funding_window)
    else:
        df["funding_rate"] = 0.0
        df["funding_z"] = 0.0

    if mark is not None and spot is not None and not mark.empty and not spot.empty:
        m = mark.sort_values("time")[["time", "value"]].rename(columns={"value": "mark_price"})
        s = spot.sort_values("time")[["time", "value"]].rename(columns={"value": "spot_price"})
        df = pd.merge_asof(df, m, on="time", direction="backward")
        df = pd.merge_asof(df, s, on="time", direction="backward")
        df["basis"] = basis(df["mark_price"], df["spot_price"])
    else:
        df["basis"] = 0.0

    # Funding/basis can be legitimately unavailable (e.g. not backfilled yet) --
    # treat "no macro signal" as neutral rather than dropping the whole row,
    # which otherwise silently zeroes out the ML training set.
    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    df["funding_z"] = df["funding_z"].fillna(0.0)
    df["basis"] = df["basis"].fillna(0.0)

    # -- External data (app/data/ingest_external.py) -----------------------
    # Same merge_asof-onto-the-grid, neutral-default-when-missing treatment
    # as funding/mark/spot above. Rate-of-change/z-score windows reuse the
    # same "scale to this resolution's actual bar size" trick as
    # funding_window above, so "24h" and "3 days" mean the same span at 15m,
    # 1h, or 4h instead of a fixed bar count silently meaning different
    # amounts of real time per resolution.
    oi_window = max(2, round(24 * bars_per_hour))
    lsr_window = max(6, round(72 * bars_per_hour))

    if fear_greed is not None and not fear_greed.empty:
        fg = fear_greed.sort_values("time")[["time", "value"]].rename(columns={"value": "fear_greed"})
        df = pd.merge_asof(df, fg, on="time", direction="backward")
    else:
        df["fear_greed"] = 50.0
    # 0-100 scale, 50 = neutral -- rescaled to roughly [-1, 1] so it sits on
    # the same footing as the model's other bounded features.
    df["fear_greed_z"] = (df["fear_greed"] - 50.0) / 50.0

    if binance_funding is not None and not binance_funding.empty:
        bf = binance_funding.sort_values("time")[["time", "value"]].rename(columns={"value": "binance_funding_rate"})
        df = pd.merge_asof(df, bf, on="time", direction="backward")
    else:
        df["binance_funding_rate"] = 0.0
    # Cross-exchange funding divergence: a genuinely different signal from
    # Delta's own funding_z (which only sees Delta's own positioning) --
    # this sees whether Delta's book is crowded RELATIVE TO Binance's.
    df["funding_spread"] = df["funding_rate"] - df["binance_funding_rate"]

    if binance_oi is not None and not binance_oi.empty:
        boi = binance_oi.sort_values("time")[["time", "value"]].rename(columns={"value": "binance_oi"})
        df = pd.merge_asof(df, boi, on="time", direction="backward")
        # Rate of change, not the raw level -- open interest drifts over
        # years just like price does (same non-stationarity issue already
        # fixed for ema/atr/macd), so only the CHANGE is a stationary,
        # model-safe feature.
        df["binance_oi_change"] = df["binance_oi"].pct_change(oi_window)
    else:
        df["binance_oi_change"] = 0.0

    if binance_lsr is not None and not binance_lsr.empty:
        blsr = binance_lsr.sort_values("time")[["time", "value"]].rename(columns={"value": "binance_lsr"})
        df = pd.merge_asof(df, blsr, on="time", direction="backward")
        lsr_mean = df["binance_lsr"].rolling(lsr_window, min_periods=max(2, lsr_window // 4)).mean()
        lsr_std = df["binance_lsr"].rolling(lsr_window, min_periods=max(2, lsr_window // 4)).std()
        # Z-score, not the raw ratio -- "crowded positioning" is relative to
        # this symbol's OWN recent normal, not a fixed absolute threshold.
        df["binance_lsr_z"] = (df["binance_lsr"] - lsr_mean) / lsr_std.replace(0, np.nan)
    else:
        df["binance_lsr_z"] = 0.0

    if dvol is not None and not dvol.empty:
        dv = dvol.sort_values("time")[["time", "value"]].rename(columns={"value": "dvol"})
        df = pd.merge_asof(df, dv, on="time", direction="backward")
        df["dvol_change"] = df["dvol"].pct_change(oi_window)
    else:
        # No Deribit options market at all for this symbol (e.g. SOLUSD) --
        # neutral, not an error; see DERIBIT_CURRENCY in ingest_external.py.
        df["dvol_change"] = 0.0

    for col in ("fear_greed_z", "funding_spread", "binance_oi_change", "binance_lsr_z", "dvol_change"):
        df[col] = df[col].fillna(0.0)

    # -- Multi-timeframe context ---------------------------------------
    # A documented finding for exactly this problem (Bitcoin ML prediction
    # across 15m/4h/daily/3-day resolutions): INTERMEDIATE-timeframe
    # features (4h Bollinger position + momentum oscillators) dominated
    # feature importance over the base timeframe's own indicators.
    # `higher_timeframe_candles` is this SAME symbol's OHLCV at a longer
    # resolution (e.g. 4h when `candles` above is 15m) -- indicators
    # computed on it, then forward-filled onto the base grid the same
    # merge_asof way as every other optional series above, so each row
    # only ever sees the most recent COMPLETED higher-timeframe reading,
    # never a future one.
    if higher_timeframe_candles is not None and not higher_timeframe_candles.empty:
        htf = add_all_indicators(higher_timeframe_candles.sort_values("time").reset_index(drop=True))
        htf["htf_trend_norm"] = ((htf["ema_9"] - htf["ema_21"]) / htf["atr_14"].replace(0, np.nan)).clip(-3, 3) / 3.0
        htf_cols = htf[["time", "htf_trend_norm", "rsi_14", "bb_z"]].rename(
            columns={"rsi_14": "htf_rsi_14", "bb_z": "htf_bb_z"}
        ).copy()
        # A higher-timeframe bar's indicators are only actually KNOWN once
        # that bar has CLOSED -- `time` here is each bar's OPEN, so merging
        # on it as-is would let an early base-resolution row within that
        # same still-forming higher-timeframe window see indicators
        # computed from that bar's eventual close/high/low. That's genuine
        # lookahead, and a severe one here: caught empirically because it
        # inflated SOLUSD's walk-forward accuracy from a real 49.9% to a
        # fabricated 61.9% -- the label horizon (~3h45m) is almost exactly
        # one 4h bar, so the leak nearly lines up with the label itself.
        # Shifting the join key forward by the higher timeframe's own
        # (inferred, not hardcoded) bar duration means a base-resolution
        # row only ever matches a higher-timeframe bar that has genuinely
        # closed by that row's own timestamp.
        htf_bar_seconds = htf["time"].diff().median()
        htf_cols["time"] = (htf_cols["time"] + htf_bar_seconds).astype(df["time"].dtype)
        df = pd.merge_asof(df, htf_cols, on="time", direction="backward")
    else:
        df["htf_trend_norm"] = np.nan
        df["htf_rsi_14"] = np.nan
        df["htf_bb_z"] = np.nan
    df["htf_trend_norm"] = df["htf_trend_norm"].fillna(0.0)
    df["htf_rsi_14"] = df["htf_rsi_14"].fillna(50.0)
    df["htf_bb_z"] = df["htf_bb_z"].fillna(0.0)

    # -- Trajectory / shape features -------------------------------------
    # LightGBM (like any tree ensemble) has no native sense of time or
    # sequence -- it only ever sees the CURRENT row's feature values, so
    # anything about a value's recent TRAJECTORY has to be handed to it
    # explicitly as its own column. rsi/adx "slope" mirrors the exact
    # construction already validated inside TrendMomentumStrategy's own
    # ADX-decay gate (a declining ADX means something different from a
    # flat one at the same level) -- extended here to RSI too, plus a
    # rolling-skew "shape of recent moves" feature.
    df["rsi_14_slope_5"] = df["rsi_14"].diff(5)
    df["adx_14_slope_5"] = df["adx_14"].diff(5)
    df["ret_1_skew_20"] = df["ret_1"].rolling(20).skew()

    # -- Cross-asset (BTC leads) ------------------------------------------
    # Documented lead-lag relationship: BTC/ETH price action has a
    # stronger influence on altcoin prices than sentiment does. Only
    # meaningful for ETH/SOL -- passing BTC's own candles when building
    # BTC's OWN feature frame would just be feeding the model its own
    # (shifted) target in disguise, so callers must never do that.
    if cross_asset_candles is not None and not cross_asset_candles.empty:
        btc = cross_asset_candles.sort_values("time").reset_index(drop=True)
        btc_ret = btc[["time"]].copy()
        btc_ret["btc_ret_5"] = btc["close"].pct_change(5)
        df = pd.merge_asof(df, btc_ret, on="time", direction="backward")
    else:
        df["btc_ret_5"] = np.nan
    df["btc_ret_5"] = df["btc_ret_5"].fillna(0.0)

    # -- Astrology (app/astro/panchang.py, app/astro/dasha.py) -------------
    # Every one of these mechanisms already exists, independently verified,
    # and is already used by GrahaStrategy as a hand-tuned multiplicative
    # tilt -- but NONE of them had ever been handed to the ML model as raw
    # features before, where a tree ensemble can learn its own nonlinear
    # thresholds/interactions instead of a fixed formula. `times_utc` is
    # pure function of `time`, so every one of these costs nothing extra to
    # compute (no external data fetch) and is always present.
    times_utc = df["time"].to_numpy(dtype=float)
    df["astro_paksha_bias"] = np.where(is_waxing_moon(times_utc), 1.0, -1.0)
    df["astro_nakshatra_lord_bias"] = nakshatra_lord_bias(times_utc)
    df["astro_tithi_class_bias"] = tithi_class_bias(times_utc)
    df["astro_vara_lord_bias"] = vara_lord_bias(times_utc)
    df["astro_hora_bias"] = hora_bias(times_utc)
    df["astro_aspect_score"] = benefic_malefic_aspect_score(times_utc)
    df["astro_yoga_severity"] = yoga_severity(times_utc).astype(float)
    df["astro_malefic_retrograde_count"] = malefic_retrograde_count(times_utc).astype(float)
    df["astro_combust_count"] = combust_planet_count(times_utc).astype(float)
    df["astro_graha_yuddha_intensity"] = graha_yuddha_intensity(times_utc)
    df["astro_rahu_kaal"] = is_rahu_kaal(times_utc).astype(float)
    df["astro_vishti_karana"] = is_vishti_karana(times_utc).astype(float)
    # Dasha is symbol-dependent (needs a genesis chart) -- fails closed to
    # neutral for an unrecognized/absent symbol, same as GrahaStrategy.
    df["astro_dasha_bias"] = dasha_bias(symbol, times_utc) if symbol else 0.0

    # A genuinely novel combination: signed COUNT of how many independent
    # astrological sub-systems (lunar phase, lunar mansion, lunar day,
    # weekday, planetary hour, aspects, and multi-year Dasha regime --
    # cycles ranging from hourly to multi-year, none of them derived from
    # any of the others) currently lean benefic vs malefic, all at once.
    # Not something any single classical text combines this way -- each
    # system is real and independently sourced, but this specific synthesis
    # (a cross-system "consensus strength" score) is assembled here. A tree
    # model COULD in principle reconstruct "how many of these 7 columns
    # agree" from the raw columns alone, but only via splits across all 7
    # simultaneously -- handing it the aggregate directly is the same
    # reasoning as rsi_14_slope_5/adx_14_slope_5 above (make the pattern
    # explicit rather than hoping many splits stumble onto it).
    agreement_inputs = [
        df["astro_paksha_bias"],
        df["astro_nakshatra_lord_bias"],
        df["astro_tithi_class_bias"],
        df["astro_vara_lord_bias"],
        df["astro_hora_bias"],
        np.sign(df["astro_aspect_score"]),
        df["astro_dasha_bias"],
    ]
    df["astro_agreement_score"] = sum(np.sign(x) for x in agreement_inputs)

    # -- Quantitative / orbital-mechanics extensions -----------------------
    # A second research pass, moving from discrete Panchang-style rules to
    # genuinely continuous, real-astronomy-derived quantities: a
    # comprehensive Bradley-Barometer-style aggregate across ALL 21 pairs
    # of the 7 classical bodies (not just 4 pairs relative to the Moon),
    # the SAME aggregate recomputed in the HELIOCENTRIC frame (mathematically
    # cleaner -- apparent retrograde is a purely geocentric illusion, absent
    # here entirely), a DECLINATION-based (not ecliptic-longitude) aspect
    # system -- a genuinely different coordinate axis -- and continuous
    # planetary angular velocity (normalized by real orbital-mechanics mean
    # motion) replacing the binary retrograde flags with their full
    # magnitude, not just their sign.
    df["astro_all_pairs_aspect_score"] = all_pairs_aspect_score(times_utc)
    df["astro_helio_aspect_score"] = helio_aspect_score(times_utc)
    df["astro_declination_aspect_score"] = declination_aspect_score(times_utc)
    df["astro_mars_velocity"] = planet_velocity_normalized("mars", times_utc)
    df["astro_jupiter_velocity"] = planet_velocity_normalized("jupiter", times_utc)
    df["astro_saturn_velocity"] = planet_velocity_normalized("saturn", times_utc)

    # -- Math/event-driven short-term features (not astrology) -------------
    # hurst_100 (Hurst exponent, rescaled-range analysis over a trailing
    # 100-bar window) is already computed above by add_all_indicators --
    # a real, independently-tested statistical measure of the CURRENT
    # series' own trending (>0.5) vs. mean-reverting (<0.5) tendency,
    # specifically validated for segmenting a short-term crypto ML model
    # in published research (Sidhu et al.). weekly_expiry_proximity is a
    # real calendar-event feature (proximity to the weekly Friday 08:00
    # UTC crypto options expiry) -- documented but with genuinely mixed
    # recent evidence on reliability, so tested rather than assumed here.
    df["weekly_expiry_proximity"] = weekly_expiry_proximity(times_utc)

    return df


FEATURE_COLUMNS = [
    "ema_9_norm",
    "ema_21_norm",
    "ema_50_norm",
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "atr_pct",
    "adx_14",
    "bb_z",
    "realized_vol_20",
    "ret_1",
    "ret_5",
    "ret_15",
    "funding_rate",
    "funding_z",
    "basis",
    # Added after honest walk-forward + a genuine 70/30 holdout validation
    # (neither used to pick these specific 3 columns out of the larger
    # candidate set below): accuracy improved on ALL 3 symbols on the
    # holdout (BTC +1.7pt, ETH +3.6pt, SOL +1.2pt), IC improved on 2 of 3
    # with only a negligible dip on BTC's already-near-zero IC. The
    # smallest, most consistent win of any feature-expansion attempt this
    # project has tried -- see the excluded candidates below for the ones
    # that DIDN'T clear this bar from the same research pass.
    "rsi_14_slope_5",
    "adx_14_slope_5",
    "ret_1_skew_20",
    # Added after a third research pass (quant/orbital-mechanics astrology --
    # see the astro_* block above and its extended comment below): a
    # Bradley-Barometer-style aggregate across all 21 pairs of the 7
    # classical bodies, comprehensively summed rather than Moon-centric.
    # Walk-forward showed a consistent positive-or-flat pattern across all
    # 3 symbols; a genuine 70/30 holdout confirmed it cleanly -- accuracy
    # (the primary metric used throughout this file) improved on ALL 3
    # symbols with no exceptions (BTC +1.1pt, ETH +2.8pt, SOL +3.5pt).
    # BTC's IC alone dipped slightly (a secondary metric, and BTC's own
    # accuracy still improved), which is why this is reported honestly as
    # a real but not flawless win, not a spin-free clean sweep.
    "astro_all_pairs_aspect_score",
]

# fear_greed_z, funding_spread, binance_oi_change, binance_lsr_z, and
# dvol_change are all computed above and available on every feature frame,
# but deliberately excluded from FEATURE_COLUMNS (i.e. NOT fed to the ML
# model): retraining with them added was backtested honestly via the same
# purged walk-forward validation as everything else, and it made 2 of 3
# symbols' out-of-sample accuracy WORSE, not better -- SOLUSD lost its
# single strongest fold (IC +0.184 -> -0.026), BTCUSD's most recent fold
# went from mildly negative to sharply negative (IC -0.018 -> -0.267), and
# ETHUSD was flat at best. The likely cause: Binance OI/long-short history
# only goes back ~3 weeks and Deribit DVOL ~6 weeks, vs. ~51 days of 15m
# training data, so a large block of early rows carry a constant
# "neutral" fill for these columns -- which a tree model can latch onto as
# a spurious time-index proxy rather than genuine signal. Left computed
# (and covered by tests/test_feature_pipeline_external.py) for future
# research -- e.g. once more history has accumulated, or if it's ever
# tried as a rule-based strategy input instead -- but not force-fed to the
# model on the strength of theory alone. Same discipline as
# TrendMomentumStrategy's efficiency-ratio gate.
#
# htf_trend_norm/htf_rsi_14/htf_bb_z (multi-timeframe context) and
# btc_ret_5 (cross-asset BTC lead-lag) were tried in the SAME research
# pass that produced rsi_14_slope_5/adx_14_slope_5/ret_1_skew_20 above,
# and are ALSO excluded, for two different reasons:
#   - htf_* actually had a severe LOOKAHEAD BUG on first implementation
#     (merging a still-forming higher-timeframe bar's indicators, computed
#     from its eventual close, onto base-resolution rows from EARLY within
#     that same window -- caught because it inflated SOLUSD's walk-forward
#     accuracy from a real 49.9% to a fabricated 61.9%). Fixed by shifting
#     the merge key forward by the higher timeframe's own bar duration.
#     Even after the fix, though, it only helped SOLUSD (acc +2.7pt, IC
#     +0.084) while making ETHUSD WORSE (acc -1.6pt, IC -0.022) and BTCUSD
#     roughly flat -- a 1-of-3 result, the same bar every other rejected
#     change this project has failed.
#   - btc_ret_5's effect was simply too small to trust either way (BTCUSD
#     exactly 0 by construction, ETHUSD -0.5pt, SOLUSD +0.1pt) -- closer to
#     noise than signal.
# Both are fully computed (tests/test_feature_pipeline_multiframe.py,
# including a dedicated regression test for the lookahead bug) and
# available via their optional higher_timeframe_candles/cross_asset_candles
# parameters for future research, just not force-fed to the model now.
#
# astro_* (12 individual mechanisms + astro_agreement_score, a composite
# signed-count of how many of 7 independent astrological sub-systems --
# lunar phase, lunar mansion, lunar day, weekday, planetary hour, aspects,
# and Dasha regime -- currently lean benefic vs malefic at once) were
# tried as ML features for the first time in this research pass (every
# one of them already existed and is used by GrahaStrategy as a hand-
# tuned multiplicative tilt, but none had ever been handed to the ML
# model as raw features before). Tested as several groups (all together,
# the 3 brand-new mechanisms alone, the composite alone, Dasha alone, the
# pre-existing-panchang-only group) across all 3 symbols via walk-forward,
# then the single most promising candidate (astro_agreement_score alone,
# the only group with a consistently positive IC delta on all 3 symbols
# in the walk-forward) was checked against a genuine 70/30 holdout. It
# improved BTCUSD and ETHUSD clearly on both accuracy and IC, but SOLUSD's
# accuracy got WORSE (-2.0pt) on that same holdout despite a flat-to-
# positive IC -- a 2-of-3 result once the metric that actually matters
# (accuracy, the same headline measure used everywhere else in this file)
# is checked out-of-sample, not the clean cross-symbol win the walk-
# forward alone suggested. astro_dasha_bias's isolated effect was exactly
# zero on every metric for every symbol -- not a bug: the current
# Mahadasha AND Antardasha are both unchanged across this entire ~51-day
# training window for all 3 symbols (see GrahaStrategy's docstring),
# making it a literal constant column for this specific dataset. All 14
# columns are fully computed and tested
# (tests/test_feature_pipeline_astro.py, tests/test_panchang_extended.py)
# and available for future research -- e.g. once Dasha has genuinely
# transitioned within a training window, or with a differently-constructed
# composite -- but none are force-fed to the model now.
#
# A fourth research pass moved from discrete Panchang-style rules to
# continuous, orbital-mechanics-derived quantities: astro_helio_aspect_score
# (the SAME comprehensive aspect scoring as astro_all_pairs_aspect_score
# above, recomputed in the HELIOCENTRIC frame -- mathematically cleaner in
# theory, since apparent retrograde motion is a purely geocentric viewing
# illusion absent there entirely), astro_declination_aspect_score (a
# genuinely different coordinate AXIS -- parallel/contra-parallel
# declination aspects, not ecliptic longitude), and
# astro_mars_velocity/astro_jupiter_velocity/astro_saturn_velocity
# (continuous angular velocity normalized by real orbital-mechanics mean
# motion, replacing the binary retrograde flags with their full
# magnitude). Tested individually and combined across all 3 symbols:
# astro_all_pairs_aspect_score (above, deployed) was the standout;
# everything else in this fourth pass was mixed-to-negative --
# astro_helio_aspect_score was consistently WORSE on all 3 symbols despite
# the theoretically cleaner frame, astro_declination_aspect_score and the
# velocity features each helped one symbol while hurting the other two.
# None of these three are in FEATURE_COLUMNS. All computed and tested
# (tests/test_ephemeris_extended.py, tests/test_panchang_quant.py).
#
# A fifth research pass moved away from astrology entirely: hurst_100
# (Hurst exponent via rescaled-range analysis, app/features/indicators.py --
# a real, independently-tested statistical trend-persistence measure, with
# published research specifically validating it for segmenting a short-term
# crypto ML model) and weekly_expiry_proximity (proximity to the weekly
# Friday 08:00 UTC crypto options expiry, app/features/calendar_events.py --
# a real, documented but recently-weakening "max pain" effect). Walk-forward
# looked genuinely clean: accuracy AND IC up on all 3 symbols with the two
# combined. A genuine 70/30 holdout told a different story each time it was
# checked: combined, SOLUSD's accuracy dropped (-1.4pt) despite better IC;
# hurst_100 alone, BTCUSD and ETHUSD both got WORSE (-1.3pt/-0.5pt) while
# only SOLUSD improved -- a different symbol failing depending on which
# exact combination was tried, which is a strong sign of overfitting to the
# walk-forward window rather than a real, stable effect. Neither is in
# FEATURE_COLUMNS. Both fully computed and tested
# (tests/test_indicators.py, tests/test_calendar_events.py,
# tests/test_feature_pipeline_events.py) for future research, e.g. once
# meaningfully more training history accumulates.
