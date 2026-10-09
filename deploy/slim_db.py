"""Build a deploy-sized copy of the trading database.

The working database is ~594 MB, almost none of which the live loop reads. It
carries 3.4M trade ticks, 179k order-book snapshots with JSON blobs, 750k
1m/5m bars, and the whole gold research series -- all of that is RESEARCH data
kept for backtesting, and none of it is touched by a trading tick.

What the live loop actually needs:

    BINANCE:{sym} 1h     the signal itself (the ladder's 1600-span EMA needs
                         ~6,400 bars, so this cannot be trimmed by date)
    BFUNDHIST:{sym} 8h   real funding, charged as a cost
    Delta 15m/4h         only for the ATR that would place stops, plus a
                         fallback price
    orders/fills/positions/equity_snapshots/app_state/strategy_configs

This writes a NEW file and never modifies the original, so the research data
stays where it is.

    python deploy/slim_db.py backend/cryptoalgo.db deploy/live.db
"""
from __future__ import annotations

import os
import sqlite3
import sys

# Dropped: 1m/5m bars (no strategy, model or page reads them) and the raw
# websocket tables (written every 5s, read by nothing -- see
# app/data/retention.py). EVERYTHING ELSE IS KEPT, deliberately including
# PAXGUSDT: an earlier version of this script kept only an allow-list of the
# trading loop's own series, which silently deleted the gold page's entire
# price history (and its models' training data) from the deployed copy.
DROP_RESOLUTIONS = ("1m", "5m")
DROP_TABLES = ("trade_ticks", "orderbook_snapshots", "ticker_snapshots", "funding_rate_ticks", "model_predictions")


def main(src: str, dst: str) -> None:
    if not os.path.exists(src):
        sys.exit("source database not found: %s" % src)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)

    before = os.path.getsize(src) / 1e6
    print("copying %s (%.0f MB) -> %s" % (src, before, dst))
    # sqlite backup API, not a file copy: safe even while the bot is writing.
    with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
        source.backup(target)

    con = sqlite3.connect(dst)
    cur = con.cursor()

    # --- candles -----------------------------------------------------------
    n_before = cur.execute("select count(*) from candles").fetchone()[0]
    cur.execute("delete from candles where resolution in (%s)" % ",".join("'%s'" % r for r in DROP_RESOLUTIONS))
    n_after = cur.execute("select count(*) from candles").fetchone()[0]
    print("  candles: %d -> %d rows" % (n_before, n_after))
    gold = cur.execute("select count(*) from candles where symbol = 'PAXGUSDT'").fetchone()[0]
    print("  gold (PAXGUSDT) candles kept: %d" % gold)

    # --- research tables ---------------------------------------------------
    for table in DROP_TABLES:
        try:
            n = cur.execute("select count(*) from %s" % table).fetchone()[0]
            cur.execute("delete from %s" % table)
            print("  %-22s cleared (%d rows)" % (table, n))
        except sqlite3.OperationalError:
            pass

    # Equity history is the live track record -- keep it, but a year is plenty.
    try:
        cur.execute("delete from equity_snapshots where timestamp < date('now','-365 days')")
    except sqlite3.OperationalError:
        pass

    con.commit()
    print("  vacuuming...")
    cur.execute("VACUUM")
    con.commit()
    con.close()

    after = os.path.getsize(dst) / 1e6
    print("done: %.0f MB -> %.0f MB (%.0f%% smaller)" % (before, after, (1 - after / before) * 100))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "backend/cryptoalgo.db"
    dst = sys.argv[2] if len(sys.argv) > 2 else "deploy/live.db"
    main(src, dst)
