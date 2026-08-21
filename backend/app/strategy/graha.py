from __future__ import annotations

import numpy as np
import pandas as pd

from app.astro.dasha import dasha_bias
from app.astro.ephemeris import is_retrograde
from app.astro.natal import natal_transit_score
from app.astro.panchang import (
    benefic_malefic_aspect_score,
    combust_planet_count,
    hora_bias,
    is_rahu_kaal,
    is_vishti_karana,
    is_waxing_moon,
    malefic_retrograde_count,
    nakshatra_lord_bias,
    yoga_severity,
)
from app.strategy.base import Strategy


class GrahaStrategy(Strategy):
    """EXPERIMENTAL. Deep, full-Panchang Vedic-astrology-conditioned trend
    strategy: real, properly-computed planetary positions (skyfield + JPL
    DE421 ephemeris, Lahiri ayanamsa, plus the Meeus mean-lunar-node formula
    for Rahu/Ketu) feed traditional Panchang and Jyotish rules covering all
    9 Navagraha, not just Sun/Moon/Mercury, that gate/tilt an ordinary
    EMA-cross trend signal. There is no established empirical or causal link
    between planetary positions and crypto markets -- this is built for
    honest, transparent backtesting of that question. Kept enabled per an
    explicit, informed user override of that default (see registry.py):
    judged purely on backtest results, not on priors about plausibility.

    Every rule below is a real, verified traditional concept (not
    fabricated), each independently sourced during a research pass rather
    than assumed from general astrology knowledge:

      - Rahu Kaal (~90min "inauspicious" solar-time window, one of 8 equal
        divisions of that day's real sunrise-to-sunset span at New Delhi,
        weekday-dependent): flat, no new positions -- exactly like
        `RahuStrategy`'s daytime avoidance.
      - Vishti Karana / Bhadra (the most feared of the 11 Karanas -- half-
        Tithi periods -- the 7th of 7 repeating "chara" karanas): computed
        and available (`gate_vishti_karana=True` re-enables it), but OFF by
        default -- an 11-mechanism ablation study found this specific gate
        consistently hurt Sharpe on all 3 symbols, confirmed on a genuine
        holdout split, unlike everything else here.
      - Lunar paksha (waxing vs waning Moon): a soft tilt. Trend calls
        agreeing with the traditional "waxing=bullish/waning=bearish" bias
        keep full conviction; disagreeing calls are discounted, not flipped.
      - Nakshatra lord benefic/malefic (an INDEPENDENT lunar-mansion factor
        from paksha -- which of the 9 Navagraha rules the Moon's current
        one-of-27 nakshatra, classified benefic/malefic per classical
        Jyotish): a second soft tilt, on top of the paksha one.
      - Nitya Yoga severity: the 2 most severe of the 9 traditionally
        inauspicious Yogas (Vyatipata, Vaidhriti) gate flat, like Rahu
        Kaal/Vishti; the other 7 moderate ones (Vishkambha, Atiganda, Shula,
        Ganda, Vyaghata, Vajra, Parigha) only discount conviction.
      - Planetary aspects (drishti-adjacent, via angular separation to the
        Moon): tight trine/sextile from benefics (Jupiter, Venus) boosts
        conviction; tight square/opposition from malefics (Saturn, Mars)
        discounts it.
      - Retrograde caution: Mercury retrograde halves conviction (the
        traditional "avoid new commitments" rule); each of Mars/Saturn ALSO
        retrograde compounds additional caution on top of that.
      - Combustion (Surya Asta): each of Mercury/Venus/Mars/Jupiter/Saturn
        within its own classical degree-orb (Brihat Parasara Hora Shastra)
        of the Sun is "combust" -- dimmed -- and compounds additional
        caution, same shape as the retrograde discount.
      - Genesis/natal transit tilt (a SLOW, multi-month/year signal,
        distinct from every daily-Panchang rule above): each asset's real
        genesis-block timestamp (BTC's Genesis Block, ETH's block #0, SOL's
        Mainnet Beta launch) is treated as its "birth chart," the same
        reference point real financial astrologers use for an instrument's
        "first trade" chart. Current transiting Jupiter/Rahu in harmonious
        aspect to the asset's OWN natal Sun/Moon boosts conviction;
        transiting Saturn in challenging aspect discounts it.
      - Vimshottari Dasha lord tilt (app/astro/dasha.py): the CENTRAL
        predictive technique of Vedic astrology, not yet combined with a
        genesis/natal chart anywhere else -- where the natal transit tilt
        above asks "where are the planets now, relative to the chart,"
        Dasha asks "which planet's significations are IN FORCE over this
        extended chapter of the chart's life," with everything else here
        read as playing out WITHIN that. Antardasha (sub-period, changes
        every few months to ~3 years) is weighted primary; Mahadasha
        (major period, 6-20 years -- essentially constant across any one
        backtest window for these genesis dates) contributes at half
        weight as slow background context. Same agree/disagree tilt shape
        as paksha/nakshatra-lord/natal-transit above. Computed and
        available (`dasha_weight` re-enables it), but OFF (0.0) by
        default: backtested at multiple weights (0.05, 0.15) and both
        signs across all 3 symbols, it consistently HURT SOLUSD (Sharpe
        -0.31 to -1.56 in every configuration except one marginal +0.24
        with the sign flipped) while BTCUSD/ETHUSD moved by only small,
        inconsistent, noise-level amounts depending on weight/sign -- no
        single configuration cleared this codebase's cross-symbol bar.
      - Hora (planetary hour) tilt: the 7 classical grahas ruling in fixed
        "Chaldean order" through New Delhi's real sunrise-to-sunset and
        sunset-to-sunrise spans (12 unequal Horas each -- the genuine
        classical construction, not equal clock hours). A much finer clock
        than Rahu Kaal's single ~90min daily gate, and deliberately matched
        to this strategy's actual 1h/4h trading resolution, where a
        day-level-only signal can't move within a single bar. Computed and
        available (`hora_weight` re-enables it), but OFF (0.0) by default
        for the same reason as Dasha above: backtested at 0.05 and 0.10,
        both signs, it also consistently hurt SOLUSD (-0.36 to -0.58, or
        -0.62 sign-flipped) with only small, mixed effects on BTC/ETH.
      - Both mechanisms are real, precisely-implemented, independently
        verified traditional techniques (not the problem) -- the honest
        result is simply that neither shows a genuine, robust, cross-
        symbol edge for THIS strategy on THIS data, the same standard
        every other mechanism here was held to.
    """

    name = "graha"

    def __init__(
        self,
        symbol: str | None = None,
        ema_fast: str = "ema_9",
        ema_slow: str = "ema_21",
        disagreement_discount: float = 0.6,
        nakshatra_lord_weight: float = 0.2,
        aspect_weight: float = 0.12,
        mercury_retrograde_discount: float = 0.5,
        malefic_retrograde_discount: float = 0.8,
        moderate_yoga_discount: float = 0.7,
        combustion_discount: float = 0.85,
        natal_transit_weight: float = 0.1,
        # Both default 0.0 (off) -- see the class docstring: backtested at
        # several weights and both signs, both consistently hurt SOLUSD
        # with only small/inconsistent effects on BTC/ETH. Left toggleable
        # for future research.
        dasha_weight: float = 0.0,
        hora_weight: float = 0.0,
        gate_rahu_kaal: bool = True,
        # Default False, not True: an 11-mechanism ablation study (each
        # factor individually neutralized, backtested across all 3 symbols
        # at 4h) found this gate consistently HURT performance -- removing
        # it improved Sharpe on BTC/ETH/SOL alike (+0.61/+0.66/+0.17). A
        # genuine 70/30 train/holdout split confirmed it isn't just fitting
        # the ablation window: BTC improves on both slices, and ETH/SOL's
        # holdout Sharpe dips only slightly (remaining strongly positive,
        # 4.2+) while their holdout RETURN improves. Every other mechanism's
        # ablation signal was weaker or mixed across symbols and was left
        # alone rather than chased -- this is the one change with a clear,
        # consistent, holdout-validated case behind it.
        gate_vishti_karana: bool = False,
        gate_severe_yoga: bool = True,
    ) -> None:
        self.symbol = symbol
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.disagreement_discount = disagreement_discount
        self.nakshatra_lord_weight = nakshatra_lord_weight
        self.aspect_weight = aspect_weight
        self.mercury_retrograde_discount = mercury_retrograde_discount
        # Per additional malefic (Mars/Saturn) found retrograde -- compounds,
        # so both at once is malefic_retrograde_discount ** 2.
        self.malefic_retrograde_discount = malefic_retrograde_discount
        self.moderate_yoga_discount = moderate_yoga_discount
        # Per combust planet found (of Mercury/Venus/Mars/Jupiter/Saturn) --
        # also compounds.
        self.combustion_discount = combustion_discount
        self.natal_transit_weight = natal_transit_weight
        # Non-zero re-enables Dasha/Hora for further research -- see the
        # class docstring for why both default off.
        self.dasha_weight = dasha_weight
        self.hora_weight = hora_weight
        # Each of the 3 full gates can be individually switched off -- used
        # for ablation studies (which mechanisms actually earn their keep)
        # without needing a separate code path per combination. Defaults
        # preserve the full, all-gates-on behavior.
        self.gate_rahu_kaal = gate_rahu_kaal
        self.gate_vishti_karana = gate_vishti_karana
        self.gate_severe_yoga = gate_severe_yoga

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        if "time" not in df.columns:
            # Fail CLOSED (never trade), not open
            return pd.Series(0.0, index=df.index)

        ema_spread = (df[self.ema_fast] - df[self.ema_slow]) / df["atr_14"].replace(0, np.nan)
        direction = np.sign(ema_spread)
        macd_confirm = np.sign(df["macd_hist"])
        agreement = (direction == macd_confirm).astype(float)

        magnitude = ema_spread.clip(-3, 3).abs() / 3.0
        raw = direction * magnitude * (0.5 + 0.5 * agreement)

        times_utc = df["time"].astype(float).to_numpy()
        direction_np = direction.to_numpy()

        # Lunar paksha tilt.
        waxing = is_waxing_moon(times_utc)
        moon_bias = np.where(waxing, 1.0, -1.0)
        paksha_agree = direction_np == moon_bias
        paksha_multiplier = np.where(paksha_agree, 1.0, self.disagreement_discount)

        # Nakshatra-lord benefic/malefic tilt -- independent of paksha above.
        lord_bias = nakshatra_lord_bias(times_utc)
        lord_agree = np.sign(direction_np) == np.sign(lord_bias)
        nakshatra_multiplier = np.where(
            lord_bias == 0,
            1.0,
            np.where(lord_agree, 1.0 + self.nakshatra_lord_weight, 1.0 - self.nakshatra_lord_weight),
        )

        # Planetary aspects to the Moon: benefics in harmonious aspect boost,
        # malefics in challenging aspect discount. Score typically in [-2, 2].
        aspect_score = benefic_malefic_aspect_score(times_utc)
        aspect_multiplier = 1.0 + self.aspect_weight * aspect_score

        # Mercury retrograde -- the classical "avoid new commitments" rule.
        mercury_retro = is_retrograde("mercury", times_utc)
        mercury_multiplier = np.where(mercury_retro, self.mercury_retrograde_discount, 1.0)

        # Each additional malefic (Mars/Saturn) also retrograde compounds
        # further caution on top of Mercury's.
        malefic_retro_n = malefic_retrograde_count(times_utc)
        malefic_multiplier = self.malefic_retrograde_discount**malefic_retro_n

        # Moderate-severity Nitya Yoga discounts (not a full gate -- severe
        # ones are gated flat below, alongside Rahu Kaal/Vishti Karana).
        yoga_sev = yoga_severity(times_utc)
        yoga_multiplier = np.where(yoga_sev == 1, self.moderate_yoga_discount, 1.0)

        # Combustion: each combust planet (of Mercury/Venus/Mars/Jupiter/
        # Saturn) compounds additional caution, same shape as retrograde.
        combust_n = combust_planet_count(times_utc)
        combustion_multiplier = self.combustion_discount**combust_n

        # Genesis/natal transit tilt -- a SLOW signal (Jupiter/Saturn take
        # ~1/~2.5 years per sign), unlike everything else here which moves
        # daily-to-weekly. Direction-agreement works the same way paksha and
        # nakshatra-lord do: agreeing calls get a boost, disagreeing calls a
        # discount, never a flip.
        natal_score = natal_transit_score(self.symbol, times_utc) if self.symbol else np.zeros(len(times_utc))
        natal_multiplier = 1.0 + self.natal_transit_weight * natal_score * np.sign(direction_np)

        # Vimshottari Dasha lord tilt -- see app/astro/dasha.py. Fails
        # closed (zero score, neutral multiplier) when self.symbol is
        # unset/unrecognized, same as the natal transit tilt above.
        dasha_score = dasha_bias(self.symbol, times_utc) if self.symbol else np.zeros(len(times_utc))
        dasha_multiplier = 1.0 + self.dasha_weight * dasha_score * np.sign(direction_np)

        # Hora (planetary hour) tilt -- hourly-resolution ruling planet,
        # see hora_bias in app/astro/panchang.py.
        hora_score = hora_bias(times_utc)
        hora_multiplier = 1.0 + self.hora_weight * hora_score * np.sign(direction_np)

        raw = (
            raw
            * paksha_multiplier
            * nakshatra_multiplier
            * aspect_multiplier
            * mercury_multiplier
            * malefic_multiplier
            * yoga_multiplier
            * combustion_multiplier
            * natal_multiplier
            * dasha_multiplier
            * hora_multiplier
        )

        # Full gates: flat during Rahu Kaal, Vishti Karana, or a severe
        # (Vyatipata/Vaidhriti) Yoga -- any one of these forces 0.0, which
        # closes any existing position via the standard pipeline.
        false_mask = pd.Series(False, index=df.index)
        rahu_kaal_mask = pd.Series(is_rahu_kaal(times_utc), index=df.index) if self.gate_rahu_kaal else false_mask
        vishti_mask = pd.Series(is_vishti_karana(times_utc), index=df.index) if self.gate_vishti_karana else false_mask
        severe_yoga_mask = pd.Series(yoga_sev == 2, index=df.index) if self.gate_severe_yoga else false_mask
        gated = rahu_kaal_mask | vishti_mask | severe_yoga_mask
        raw = raw.where(~gated, 0.0)

        return raw.clip(-1, 1).fillna(0.0)

    def required_warmup_bars(self) -> int:
        return 30
