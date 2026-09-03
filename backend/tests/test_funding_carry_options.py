import datetime as dt

import pytest

from app.strategy.funding_carry_options import (
    CARRY_SYMBOLS,
    MIN_DAYS_TO_EXPIRY,
    MIN_NET_CARRY_PCT_PER_YEAR,
    ROUND_TRIP_COST_PCT,
    annualised_forward_rate,
    best_quote,
    evaluate_carry,
)

NOW = dt.datetime(2026, 9, 3, 12, 0, tzinfo=dt.timezone.utc)


def _quote(**kw):
    base = dict(
        symbol="BTCUSD", expiry="031226", strike=77_000.0, perp_mark=77_000.0,
        call_bid=1_000.0, call_ask=1_010.0, put_bid=500.0, put_ask=510.0,
        funding_rate_pct_per_year=10.95, open_interest=50.0,
        call_ask_size=5_000.0, put_bid_size=5_000.0, now=NOW,
    )
    base.update(kw)
    return evaluate_carry(**base)


def test_sol_is_excluded_from_carry_symbols():
    """SOL funding was positive in only 4 of 7 years with a 7-year mean of
    +0.17% and a -35.6% year -- it does not behave like the majors and must
    not be silently lumped in with them."""
    assert CARRY_SYMBOLS == ("BTCUSD", "ETHUSD")
    assert "SOLUSD" not in CARRY_SYMBOLS


def test_forward_rate_is_annualised_not_a_raw_gap():
    """A 1% premium at 3 months and a 1% premium at 1 month are completely
    different rates. Comparing a raw gap against an annual funding rate was
    the exact error that made 147/283 strikes look like arbitrage."""
    quarter = annualised_forward_rate(101.0, 100.0, years=0.25)
    annual = annualised_forward_rate(101.0, 100.0, years=1.0)
    assert quarter == pytest.approx(4.0)
    assert annual == pytest.approx(1.0)


def test_executable_prices_are_used_not_mids():
    """Buying the synthetic forward lifts the call ask and hits the put bid.
    Pricing off mids overstates the edge -- the median combined call+put
    spread on the live chain was 0.153% of notional."""
    wide = _quote(call_ask=1_100.0, put_bid=400.0)   # 700 wide
    tight = _quote(call_ask=1_010.0, put_bid=500.0)  # 510 wide
    # The wider spread makes the synthetic forward more expensive to buy,
    # so the implied rate rises and the carry shrinks.
    assert wide.implied_forward_rate > tight.implied_forward_rate
    assert wide.net_carry < tight.net_carry


def test_carry_is_funding_minus_forward_premium_minus_costs():
    q = _quote()
    assert q.gross_carry == pytest.approx(q.funding_rate - q.implied_forward_rate)
    assert q.net_carry == pytest.approx(q.gross_carry - q.cost_drag)


def test_short_tenors_are_rejected_because_cost_drag_dominates():
    """Measured: the same structure nets -0.95%/yr at 14 days and +4.11%/yr
    at 60, purely because the round trip amortises over fewer days."""
    near = _quote(expiry="130926")   # ~10 days out
    far = _quote(expiry="030327")    # ~6 months out
    assert near.cost_drag > far.cost_drag
    assert not near.tradable
    assert any("to expiry" in r for r in near.reasons)


def test_cost_drag_shrinks_with_holding_period():
    short = _quote(expiry="031026")  # ~1 month
    long = _quote(expiry="030327")   # ~6 months
    assert short.cost_drag > long.cost_drag > 0
    # Round trip is paid twice (open + unwind), annualised over the holding period.
    assert short.cost_drag == pytest.approx(ROUND_TRIP_COST_PCT * 2 * 365.25 / short.days_to_expiry)


def test_thin_strikes_are_rejected_on_QUOTE_DEPTH_not_open_interest():
    """Gating on open interest measures the wrong thing. Checked live: the
    best structure had OI of 1-5 contracts but 4,442 on the call ask and
    6,663 on the put bid. Low OI with deep quotes means makers are posting
    size at an untraded strike -- a liquidity signal, not a warning. What
    actually stops a fill is thin QUOTES."""
    thin = _quote(expiry="030327", funding_rate_pct_per_year=40.0,
                  call_ask_size=5.0, put_bid_size=5.0)
    assert thin.net_carry > MIN_NET_CARRY_PCT_PER_YEAR  # maths says yes
    assert not thin.tradable                            # depth says no
    assert any("too thin to fill" in r for r in thin.reasons)

    # Zero open interest but deep quotes must still be tradable.
    deep = _quote(expiry="030327", funding_rate_pct_per_year=12.0, open_interest=0.0,
                  call_ask_size=4_442.0, put_bid_size=6_663.0)
    assert deep.tradable, deep.reasons


def test_thin_net_carry_is_rejected():
    """Funding chosen so the arithmetic lands just UNDER the floor: with these
    quotes the implied forward is ~1.34%/yr and cost drag ~0.44%/yr, so 3.0%
    funding leaves ~1.2% net -- real, but inside the noise of three legs'
    bid/ask and not worth carrying."""
    q = _quote(expiry="030327", funding_rate_pct_per_year=3.0)
    assert 0 < q.net_carry < MIN_NET_CARRY_PCT_PER_YEAR  # positive but too thin
    assert not q.tradable
    assert any("floor" in r for r in q.reasons)


def test_a_genuinely_good_structure_passes_every_gate():
    q = _quote(expiry="030327", funding_rate_pct_per_year=12.0, open_interest=100.0)
    assert q.days_to_expiry > MIN_DAYS_TO_EXPIRY
    assert q.net_carry >= MIN_NET_CARRY_PCT_PER_YEAR
    assert q.tradable and q.reasons == []


def test_negative_funding_destroys_the_trade():
    """The whole structure is paid for by funding. If funding flips negative
    the short perp PAYS instead of receiving, and there is nothing to harvest
    -- this is the trade's main risk, not an edge case."""
    q = _quote(expiry="030327", funding_rate_pct_per_year=-5.0)
    assert q.gross_carry < 0
    assert not q.tradable


def test_best_quote_only_ever_returns_a_tradable_structure():
    good = _quote(expiry="030327", funding_rate_pct_per_year=12.0, open_interest=100.0)
    rich_but_illiquid = _quote(expiry="030327", funding_rate_pct_per_year=99.0,
                               call_ask_size=2.0, put_bid_size=2.0)
    pick = best_quote([rich_but_illiquid, good])
    assert pick is good  # never the untradable one, however good its number


def test_best_quote_returns_none_when_nothing_qualifies():
    assert best_quote([_quote(funding_rate_pct_per_year=-5.0)]) is None
    assert best_quote([]) is None
