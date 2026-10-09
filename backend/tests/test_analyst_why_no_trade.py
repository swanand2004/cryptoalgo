"""The Analyst page's "why no trade?" and risk settings: the latest run's gates
are kept next to _latest and exposed read-only in analyst_snapshot."""
from __future__ import annotations

import types

from app.analyst import execution as ex
from app.analyst import service as svc
from app.config import Settings


class _A:
    def __init__(self, symbol, decision, score):
        self.d = {"symbol": symbol, "decision": decision, "score": score, "conviction": 2, "factors": [],
                  "notes": [], "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0, "families": {},
                  "size_factor": 1.0, "price": 100.0}

    def to_dict(self):
        return dict(self.d)


def _runtime():
    settings = Settings(delta_api_key="k", delta_api_secret="s", analyst_symbols="BTCUSD,ETHUSD")
    return types.SimpleNamespace(settings=settings, registry=types.SimpleNamespace(symbols=["BTCUSD", "ETHUSD"]),
                                 live_signals={"BTCUSD": -0.4, "ETHUSD": 0.0}, cache=None)


async def test_snapshot_exposes_latest_gates_book_and_risk(monkeypatch):
    monkeypatch.setattr(svc, "_latest", {})
    monkeypatch.setattr(svc, "_latest_gates", {})
    monkeypatch.setattr(svc, "_stable_refreshed", {"at": 1e18})

    async def fake_inputs(runtime, symbol):
        return symbol

    calls = {"BTCUSD": ("BUY", 0.3), "ETHUSD": ("WAIT", 0.1)}
    monkeypatch.setattr(svc, "gather_inputs", fake_inputs)
    monkeypatch.setattr(svc, "analyse", lambda sym, inp: _A(sym, *calls[sym]))

    async def fake_open(runtime, results, *, gates=None, **kw):
        gates.update({"BTCUSD": "book_disagrees", "ETHUSD": "wait"})

    monkeypatch.setattr(ex, "open_best_trade", fake_open)
    rt = _runtime()
    await svc.run_analysis(rt)
    snap = await svc.analyst_snapshot(rt)
    assert snap["gates"] == {"BTCUSD": "book_disagrees", "ETHUSD": "wait"}
    assert snap["book_signals"] == {"BTCUSD": -0.4, "ETHUSD": 0.0}
    rs = snap["risk_settings"]
    assert rs["risk_cap"] == ex.trade_risk_cap(rt.settings)
    assert rs["target_leverage"] == rt.settings.analyst_target_leverage
    assert rs["maker_execution"] == rt.settings.maker_execution
    assert rs["min_score"] == rt.settings.analyst_min_score
    # A copy: the page can never change what the next run sees.
    snap["gates"]["BTCUSD"] = "x"
    assert svc._latest_gates["BTCUSD"] == "book_disagrees"


def test_risk_settings_without_settings_is_empty():
    assert svc._risk_settings(None) == {}
