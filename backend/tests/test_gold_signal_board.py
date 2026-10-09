import numpy as np
import pandas as pd
import pytest

from app.gold.ingest import find_gaps, incremental_start_ms
from app.gold.signal_board import BUY_LINE, build_board

DAY = 86_400


def _candles(closes, start=1_700_000_000 // DAY * DAY, step=DAY):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "time": start + np.arange(len(closes)) * step,
        "open": np.r_[closes[0], closes[:-1]],
        "high": closes * 1.004,
        "low": closes * 0.996,
        "close": closes,
    })


def _walk(n, drift=0.0, seed=3):
    rng = np.random.default_rng(seed)
    return 2000 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


def _board(closes, **kw):
    df = _candles(closes)
    now = float(df["time"].iloc[-1] + 2 * DAY)  # every bar closed
    return build_board(df, bar_seconds=DAY, horizon_bars=5, label="daily", now_ts=now, **kw)


def test_steady_uptrend_reads_buy_and_downtrend_reads_sell():
    up = _board(_walk(600, drift=0.004))
    down = _board(_walk(600, drift=-0.004))
    assert up["consensus"]["direction"] == "BUY" and up["consensus"]["score"] >= BUY_LINE
    assert down["consensus"]["direction"] == "SELL" and down["consensus"]["score"] <= -BUY_LINE


def test_board_is_direction_neutral_a_mirrored_market_mirrors_the_call():
    # The same path flipped upside down (in log space, so every % move flips
    # sign): momentum votes and the overall call must flip too. A board with a
    # built-in long bias would fail this.
    closes = _walk(600, drift=0.003)
    mirrored = np.exp(2 * np.log(closes).mean() - np.log(closes))
    a, b = _board(closes), _board(mirrored)
    votes_a = {i["key"]: i["vote"] for i in a["indicators"]}
    votes_b = {i["key"]: i["vote"] for i in b["indicators"]}
    for key in ("mom_20", "mom_120"):
        assert votes_a[key] == -votes_b[key] != 0
    assert {a["consensus"]["direction"], b["consensus"]["direction"]} == {"BUY", "SELL"}


def test_unfinished_bar_is_never_voted_on():
    df = _candles(_walk(400))
    now = float(df["time"].iloc[-1] + DAY / 2)  # last bar only half done
    board = build_board(df, bar_seconds=DAY, horizon_bars=5, label="daily", now_ts=now)
    assert board["price"] == df["close"].iloc[-2]


def test_signal_time_points_at_the_bar_where_the_current_vote_began():
    board = _board(_walk(500, drift=0.002))
    for ind in board["indicators"]:
        assert ind["bars_since"] >= 0
        assert ind["since"] <= board["bar_close_time"]


def test_track_record_compares_against_always_up_on_the_same_bars():
    board = _board(_walk(800))
    for ind in board["indicators"]:
        if ind["hit"] is not None:
            assert 0 <= ind["hit"] <= 1 and 0 <= ind["base"] <= 1 and ind["n"] >= 20


def test_gann_extras_present_with_levels_bracketing_price():
    board = _board(_walk(600))
    sq9 = board["gann"]["square_of_nine"]
    price = board["price"]
    assert any(l["price"] > price for l in sq9) and any(l["price"] < price for l in sq9)
    assert all(c["due"] > board["bar_close_time"] for c in board["gann"]["time_cycles"])


def test_too_little_history_returns_none():
    assert _board(_walk(100)) is None


def test_gap_finder_and_resume_point():
    assert find_gaps([0, 3600, 10800, 14400], 3600) == [(7200, 10800)]
    assert find_gaps([0, 3600, 7200], 3600) == []
    # Long outage: resume from the newest stored bar, not a fixed window.
    assert incremental_start_ms(newest_stored_s=1_000, now_ms=10_000_000, window_s=7_200) == 1_000_000
    # Short outage: the normal window already covers it.
    assert incremental_start_ms(newest_stored_s=9_999, now_ms=10_000_000, window_s=7_200) == 10_000_000 - 7_200_000


def _plan_frame(path):
    df = _candles(path)
    df["high"], df["low"] = df["close"] + 1.0, df["close"] - 1.0
    return df, df["time"].to_numpy(dtype=float) + DAY


def test_trade_plan_long_reaches_its_single_target():
    from app.gold.signal_board import trade_plan
    df, times = _plan_frame([100.0] * 30 + [110, 120, 140, 160])
    plan = trade_plan(df, 29, +1, times)
    assert plan["stop_loss"] < plan["entry"] < plan["take_profit"]
    assert plan["entry"] - plan["stop_loss"] == pytest.approx(plan["take_profit"] - plan["entry"])  # 1:1
    assert plan["status"] == "TP HIT" and plan["result"] == "TP" and "take_profit_2" not in plan


def test_trade_plan_short_is_stopped_when_price_rallies():
    from app.gold.signal_board import trade_plan
    df, times = _plan_frame([100.0] * 30 + [104, 110, 120])
    plan = trade_plan(df, 29, -1, times)
    assert plan["stop_loss"] > plan["entry"] > plan["take_profit"]
    assert plan["status"] == "STOPPED OUT" and plan["pnl_pct"] < 0


def test_bar_touching_stop_and_target_counts_as_stopped():
    from app.gold.signal_board import trade_plan
    df, times = _plan_frame([100.0] * 31)
    df.loc[30, ["high", "low"]] = [200.0, 0.0]  # one huge bar through both
    plan = trade_plan(df, 29, +1, times)
    assert plan["status"] == "STOPPED OUT"


def test_neutral_board_offers_breakout_triggers_instead_of_a_trade():
    from app.gold.signal_board import build_board
    closes = 2000 + 5 * np.sin(np.arange(500) / 3.0)  # flat, choppy
    df = _candles(closes)
    board = build_board(df, bar_seconds=DAY, horizon_bars=5, label="daily", now_ts=float(df["time"].iloc[-1] + 2 * DAY))
    # A trade closes only at its stop or TP2, so it can still be open on a
    # NEUTRAL board; the breakout triggers show exactly when no trade is open.
    if board["trade"] is None:
        assert board["triggers"]["buy_above"] > board["triggers"]["sell_below"]
    else:
        assert board["trade"]["open"] and board["triggers"] is None


# ------------------------------------------------------------ trade log

def _history_frame(closes):
    df = _candles(closes)
    df["high"], df["low"] = df["close"] + 1.0, df["close"] - 1.0
    times = df["time"].to_numpy(dtype=float) + DAY
    return df, times


def _run(direction_values):
    import pandas as pd
    return pd.Series(direction_values, dtype=float)


def test_history_trade_exits_at_the_target():
    from app.gold.signal_board import trade_history
    closes = [100.0] * 30 + [101, 104, 108, 112, 116] + [116] * 5
    df, times = _history_frame(closes)
    d = _run([0] * 29 + [1] * 11)
    [t] = trade_history(df, d, times, warmup=1)
    assert t["side"] == "BUY" and t["result"] == "TP" and not t["open"]
    assert t["exit_price"] == pytest.approx(t["take_profit"]) and t["pnl_pct"] > 0


def test_history_trade_exits_at_the_stop():
    from app.gold.signal_board import trade_history
    closes = [100.0] * 30 + [99, 96, 92, 88] + [88] * 6
    df, times = _history_frame(closes)
    [t] = trade_history(df, _run([0] * 29 + [1] * 11), times, warmup=1)
    assert t["result"] == "SL" and t["exit_price"] == pytest.approx(t["stop_loss"]) and t["pnl_pct"] < 0


def test_signal_change_does_not_close_the_trade_only_stop_or_target_do():
    from app.gold.signal_board import trade_history
    closes = [100.0] * 40
    df, times = _history_frame(closes)
    # long from bar 29 to 34, then the consensus goes neutral: the trade stays open
    trades = trade_history(df, _run([0] * 29 + [1] * 6 + [0] * 5), times, warmup=1)
    [t] = trades
    assert t["open"] and t["result"] is None and t["exit_time"] is None


def test_new_signals_are_skipped_while_a_trade_is_open():
    from app.gold.signal_board import trade_history
    df, times = _history_frame([100.0] * 60)
    # BUY at 29, flips SELL at 35, BUY again at 45 -- price never moves, so the
    # first trade is still open and the later signals open nothing
    trades = trade_history(df, _run([0] * 29 + [1] * 6 + [-1] * 10 + [1] * 15), times, warmup=1)
    assert len(trades) == 1 and trades[0]["side"] == "BUY" and trades[0]["open"]


def test_next_trade_opens_on_the_first_signal_after_the_previous_one_closed():
    from app.gold.signal_board import trade_history
    closes = [100.0] * 30 + [99, 96, 92, 88] + [88.0] * 26
    df, times = _history_frame(closes)
    # BUY at 29 (stopped at bar ~31), SELL signal at 40 opens the next trade
    trades = trade_history(df, _run([0] * 29 + [1] * 11 + [-1] * 20), times, warmup=1)
    assert [t["side"] for t in trades] == ["BUY", "SELL"]
    assert trades[0]["result"] == "SL" and trades[1]["entry_time"] == _iso_close(times, 40)


def _iso_close(times, i):
    from app.gold.signal_board import _iso
    return _iso(times[i])


def test_open_trade_when_nothing_has_hit_yet():
    from app.gold.signal_board import trade_history
    df, times = _history_frame([100.0] * 40)
    [t] = trade_history(df, _run([0] * 35 + [-1] * 5), times, warmup=1)
    assert t["open"] and t["side"] == "SELL" and t["status"] == "ACTIVE" and t["exit_time"] is None


def test_finished_trade_leaves_the_card_and_lands_in_the_log():
    # Steady uptrend: the BUY signal lasts far longer than it takes to hit TP2.
    board = _board(_walk(700, drift=0.006, seed=11))
    assert board["consensus"]["direction"] == "BUY"
    if board["trade"] is None:  # the current signal's trade already finished
        assert board["last_trade"]["result"] in ("TP", "SL")
        assert board["plan"] is not None  # card shows entry / SL / TP1 for a trade entered now
    hist, stats = board["history"], board["history_stats"]
    assert hist and all(not t["open"] for t in hist)
    assert [t["entry_time"] for t in hist] == sorted((t["entry_time"] for t in hist), reverse=True)
    assert stats["tp"] + stats["sl"] == stats["trades"]


def test_active_signal_without_open_trade_shows_entry_sl_tp1_for_now():
    board = _board(_walk(700, drift=0.006, seed=11))
    assert board["consensus"]["direction"] == "BUY"
    if board["trade"] is None:
        plan = board["plan"]
        assert plan["side"] == "BUY" and plan["entry"] == pytest.approx(board["price"])
        assert plan["stop_loss"] < plan["entry"] < plan["take_profit"]
        assert plan["entry"] - plan["stop_loss"] == pytest.approx(plan["take_profit"] - plan["entry"])  # 1:1
        assert board["triggers"] is None
    else:
        assert board["plan"] is None
