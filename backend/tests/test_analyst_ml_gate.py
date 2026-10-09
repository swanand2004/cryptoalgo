"""analyst_ml_gate "g2": the crypto Analyst enters only on coins the frozen ML book
filter rates 1.5x today; no fresh record -> the gate stays open; "off" = old rule."""
import pytest

from app.analyst import execution as ex
from tests.test_team_analyst_execution import _OM, _call, _live_and_running, _runtime  # noqa: F401  (autouse fixture)


def _patch_mult(monkeypatch, mult):
    calls = []

    async def fake(now=None):
        calls.append(1)
        return mult

    monkeypatch.setattr("app.ml.book_filter.current_multipliers", fake)
    return calls


def test_default_is_g2():
    from app.config import Settings

    assert Settings(delta_api_key="k", delta_api_secret="s").analyst_ml_gate == "g2"


@pytest.mark.asyncio
async def test_g2_blocks_a_coin_rated_below_1_5(monkeypatch):
    _patch_mult(monkeypatch, {"ETHUSD": 1.0, "BTCUSD": 1.5})
    om = _OM({"ETHUSD": 2700.0})
    gates = {}
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}, analyst_ml_gate="g2"),
                                 {"ETHUSD": _call("ETHUSD", "BUY", 0.7)}, gates=gates)
    assert t is None and om.orders == [] and gates["ETHUSD"] == "ml_gate"


@pytest.mark.asyncio
async def test_g2_lets_a_coin_rated_1_5_trade(monkeypatch):
    _patch_mult(monkeypatch, {"ETHUSD": 1.5})
    om = _OM({"ETHUSD": 2700.0})
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}, analyst_ml_gate="g2"),
                                 {"ETHUSD": _call("ETHUSD", "BUY", 0.7)})
    assert t is not None and t.symbol == "ETHUSD"


@pytest.mark.asyncio
async def test_no_fresh_record_leaves_the_gate_open(monkeypatch):
    _patch_mult(monkeypatch, {})
    om = _OM({"ETHUSD": 2700.0})
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}, analyst_ml_gate="g2"),
                                 {"ETHUSD": _call("ETHUSD", "BUY", 0.7)})
    assert t is not None


@pytest.mark.asyncio
async def test_a_failing_model_read_leaves_the_gate_open(monkeypatch):
    async def boom(now=None):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.ml.book_filter.current_multipliers", boom)
    om = _OM({"ETHUSD": 2700.0})
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}, analyst_ml_gate="g2"),
                                 {"ETHUSD": _call("ETHUSD", "BUY", 0.7)})
    assert t is not None


@pytest.mark.asyncio
async def test_off_never_reads_the_model(monkeypatch):
    calls = _patch_mult(monkeypatch, {"ETHUSD": 0.5})
    om = _OM({"ETHUSD": 2700.0})
    t = await ex.open_best_trade(_runtime(om, {"ETHUSD": 0.5}, analyst_ml_gate="off"),
                                 {"ETHUSD": _call("ETHUSD", "BUY", 0.7)})
    assert t is not None and calls == []


@pytest.mark.asyncio
async def test_gold_is_never_gated(monkeypatch):
    calls = _patch_mult(monkeypatch, {"XAUTUSD": 0.5})
    assert await ex._ml_gate_multipliers(types_ns(analyst_ml_gate="g2")) == {"XAUTUSD": 0.5}
    # open_best_trade for gold runs with require_book=False, which skips the gate entirely
    om = _OM({"ETHUSD": 2700.0})
    gates = {}
    await ex.open_best_trade(_runtime(om, {}, analyst_ml_gate="g2"), {"ETHUSD": _call("ETHUSD", "BUY", 0.7)},
                             paper_only=True, require_book=False, gates=gates)
    assert gates.get("ETHUSD") != "ml_gate"
    assert len(calls) == 1


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)
