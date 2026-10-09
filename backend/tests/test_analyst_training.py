import json
import types

import numpy as np
import pandas as pd
import pytest

import app.config
from app.analyst import training as tr
from app.analyst.factors import FAMILY_WEIGHTS, family_weights
from app.config import Settings

FAMS = tuple(FAMILY_WEIGHTS)


def _settings(**kw) -> Settings:
    return Settings(delta_api_key="k", delta_api_secret="s", **kw)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "STORE_DIR", tmp_path)
    tr._record_cache.update(key=None, record=None)
    return tmp_path


def _enable(monkeypatch, on: bool) -> None:
    s = _settings(analyst_training_enabled=on)
    monkeypatch.setattr(app.config, "get_settings", lambda: s)


def _weights(**over) -> dict:
    w = {f: 0.0 for f in FAMS}
    w.update(over)
    return w


# ------------------------------------------------------------ adopt / reject
def _m(calls=100, exp=0.10, rpr=0.02, seq=5.0):
    return {"calls": calls, "expectancy_r": exp, "r_per_row": rpr, "seq_total_r": seq}


def test_adopts_only_when_every_preset_check_passes():
    cur = _m(exp=0.05, rpr=0.01, seq=2.0)
    ok, reasons = tr.decide(_m(exp=0.05 + tr.MARGIN_R + 0.01, rpr=0.02, seq=3.0), cur)
    assert ok and all(r.startswith("PASS") for r in reasons)


@pytest.mark.parametrize("cand", [
    _m(calls=tr.MIN_HOLDOUT_CALLS - 1, exp=0.5, rpr=0.5, seq=9),  # too few calls
    _m(exp=-0.01, rpr=0.5, seq=9),  # not profitable after costs
    _m(exp=0.05 + tr.MARGIN_R - 0.01, rpr=0.5, seq=9),  # better, but inside the margin
    _m(exp=0.5, rpr=0.005, seq=9),  # wins only by trading less
    _m(exp=0.5, rpr=0.5, seq=1.0),  # the live one-at-a-time path gets worse
])
def test_rejects_when_any_check_fails(cand):
    ok, reasons = tr.decide(cand, _m(exp=0.05, rpr=0.01, seq=2.0))
    assert not ok and any(r.startswith("FAIL") for r in reasons)


def test_current_with_no_calls_counts_as_zero_expectancy():
    cur = {"calls": 0, "expectancy_r": None, "r_per_row": 0.0, "seq_total_r": 0.0}
    assert tr.decide(_m(exp=0.2), cur)[0]


def _synthetic_rows(n_per_symbol=1500, predictive="technical", seed=0):
    """Rows where one family's sign predicts the long/short outcome."""
    rng = np.random.default_rng(seed)
    out = []
    t0 = 1_600_000_000
    for sym in ("BTCUSD", "ETHUSD"):
        t = t0 + np.arange(n_per_symbol) * 4 * 3600
        fams = {f: rng.uniform(-1, 1, n_per_symbol) for f in FAMS}
        edge = np.sign(fams[predictive]) if predictive else np.zeros(n_per_symbol)
        noise = rng.normal(0, 0.3, n_per_symbol)
        df = pd.DataFrame({"symbol": sym, "time": t, "book": rng.choice([-0.5, 0.5], n_per_symbol)})
        for f in FAMS:
            df[f"fam_{f}"] = fams[f]
        df["r_long"] = 0.5 * edge + noise
        df["r_short"] = -0.5 * edge + noise
        df["exit_long"] = df["time"] + 3600 * 24
        df["exit_short"] = df["time"] + 3600 * 24
        out.append(df)
    return pd.concat(out, ignore_index=True)


def test_train_adopts_a_real_edge_and_moves_toward_it():
    rows = _synthetic_rows(predictive="sentiment")
    res = tr.train(rows, dict(FAMILY_WEIGHTS), min_conviction=4)
    assert res["decision"] == "adopted"
    assert res["candidate"]["sentiment"] > FAMILY_WEIGHTS["sentiment"]
    w = res["candidate"]
    assert all(v >= 0 for v in w.values()) and abs(sum(w.values()) - 1) < 1e-9
    # shrinkage: never further than halfway from the current weights
    assert all(abs(w[f] - FAMILY_WEIGHTS[f]) <= (1 - tr.SHRINK) + 1e-4 for f in FAMS)


def test_train_keeps_current_weights_on_pure_noise():
    rows = _synthetic_rows(predictive=None, seed=4)
    res = tr.train(rows, dict(FAMILY_WEIGHTS), min_conviction=4)
    assert res["decision"] == "kept"


def test_train_skips_with_too_little_data():
    rows = _synthetic_rows(n_per_symbol=100)
    assert tr.train(rows, dict(FAMILY_WEIGHTS), min_conviction=4)["decision"] == "skipped"


# ------------------------------------------------------------ no lookahead
def test_holdout_is_strictly_after_the_fit_and_fit_outcomes_close_before_it():
    rows = _synthetic_rows()
    rows.loc[rows.index % 7 == 0, "exit_long"] += 40 * 86400  # some long-running trades
    fit, hold, start = tr.split(rows, holdout_days=90)
    assert len(fit) and len(hold)
    assert fit["time"].max() < hold["time"].min()
    assert hold["time"].min() > start >= fit["time"].max()
    assert (np.fmax(fit["exit_long"], fit["exit_short"]) <= start).all()
    assert hold["time"].min() >= rows["time"].max() - 90 * 86400
    # a trade that would still be open at the holdout start is purged from the fit
    late = rows[(rows["time"] <= start) & (np.fmax(rows["exit_long"], rows["exit_short"]) > start)]
    assert len(late) and not late.index.isin(fit.index).any()


def _hourly(n_days=90, seed=1):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    close = 100 * np.exp(np.cumsum(0.0003 + rng.normal(0, 0.006, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close * 1.004, "low": close * 0.996,
                         "close": close}, index=idx)


def test_features_never_see_the_future_but_outcomes_do():
    h = _hourly()
    d = tr.SymbolData("BTCUSD", h, book_signal=pd.Series(0.5, index=h.index))
    i = len(h) - 24 * 10 - 1
    before = tr.feature_row(d, i)
    shocked = h.copy()
    shocked.iloc[i + 1:] *= 0.5  # rewrite everything after the decision bar
    d2 = tr.SymbolData("BTCUSD", shocked, book_signal=pd.Series(0.5, index=h.index))
    after = tr.feature_row(d2, i)
    for k, v in before.items():
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(after[k])
        else:
            assert after[k] == v, k
    s = _settings()
    rows = pd.DataFrame([before])
    o1 = tr.attach_outcomes(rows, {"BTCUSD": d}, s)
    o2 = tr.attach_outcomes(rows, {"BTCUSD": d2}, s)
    assert o1["r_long"].iloc[0] != o2["r_long"].iloc[0]  # outcomes come from the future bars
    assert o1["exit_long"].iloc[0] > before["time"]


def test_decision_steps_are_4h_closes():
    h = _hourly()
    d = tr.SymbolData("BTCUSD", h)
    steps = tr.decision_steps(d, h.index[0], h.index[-1])
    assert steps and all(((h.index[i].hour + 1) % 4 == 0) for i in steps)
    assert min(steps) >= 24 * 60


def test_simulated_trade_stops_first_and_charges_nothing_hidden():
    s = _settings()
    o = np.array([100.0, 100.0]); hi = np.array([110.0, 101.0]); lo = np.array([94.0, 99.0]); c = np.array([100.0, 100.0])
    r, held = tr.simulate_trade("BUY", 100.0, 95.0, 105.0, 3.0, o, hi, lo, c, s)
    assert (r, held) == (-1.0, 1)  # bar touches both: stopped
    r, held = tr.simulate_trade("BUY", 100.0, 95.0, 105.0, 3.0, o[1:], np.array([106.0]), lo[1:], c[1:], s)
    assert (r, held) == (1.0, 1)  # TP1 closes (no extension in training)
    r, _ = tr.simulate_trade("SELL", 100.0, 105.0, 95.0, 3.0, np.array([108.0]), np.array([109.0]),
                             np.array([107.0]), np.array([108.0]), s)
    assert r == pytest.approx(-1.6)  # gap through the stop fills at the open


# ------------------------------------------------------------ stored weights / fallback
def test_no_stored_weights_falls_back_to_the_constants(store, monkeypatch):
    _enable(monkeypatch, True)
    assert tr.active_weights("BTCUSD") is None
    assert family_weights("BTCUSD") == FAMILY_WEIGHTS


def test_stored_weights_apply_only_when_enabled_and_only_to_trained_symbols(store, monkeypatch):
    w = _weights(technical=0.5, structure=0.5)
    (store / tr.WEIGHTS_FILE).write_text(json.dumps({"weights": w, "symbols": ["BTCUSD"]}))
    _enable(monkeypatch, True)
    assert family_weights("BTCUSD") == w
    assert family_weights("XAUTUSD") == FAMILY_WEIGHTS  # gold never uses crypto-trained weights
    _enable(monkeypatch, False)
    assert tr.active_weights("BTCUSD") is None
    assert family_weights("BTCUSD") == FAMILY_WEIGHTS


@pytest.mark.parametrize("bad", [
    {"weights": {"technical": 1.0}, "symbols": ["BTCUSD"]},  # missing families
    {"weights": _weights(technical=0.7, macro=0.7), "symbols": ["BTCUSD"]},  # sum != 1
    {"weights": _weights(technical=1.2, macro=-0.2), "symbols": ["BTCUSD"]},  # negative
])
def test_malformed_weights_fall_back(store, monkeypatch, bad):
    (store / tr.WEIGHTS_FILE).write_text(json.dumps(bad))
    _enable(monkeypatch, True)
    assert family_weights("BTCUSD") == FAMILY_WEIGHTS


def test_record_run_writes_weights_only_on_adoption(store):
    base = {"run_at": "2026-10-06T02:40:00+00:00", "symbols": ["BTCUSD"], "current": dict(FAMILY_WEIGHTS),
            "candidate": _weights(technical=0.6, structure=0.4), "holdout": {}, "fit": {}}
    tr.record_run({**base, "decision": "kept", "reasons": ["FAIL x"]}, store)
    assert not (store / tr.WEIGHTS_FILE).exists()
    tr.record_run({**base, "decision": "adopted", "reasons": []}, store)
    assert tr.adopted_record(store)["weights"] == base["candidate"]
    assert [r["decision"] for r in tr.load_runs(store)] == ["kept", "adopted"]


def test_status_lists_latest_runs_first(store, monkeypatch):
    _enable(monkeypatch, False)
    for k in range(3):
        tr.record_run({"run_at": str(k), "decision": "kept", "symbols": []}, store)
    st = tr.training_status()
    assert st["enabled"] is False and [r["run_at"] for r in st["runs"]] == ["2", "1", "0"]
    assert st["weights_in_use"] == FAMILY_WEIGHTS


# ------------------------------------------------------------ the job is gated
@pytest.mark.asyncio
@pytest.mark.parametrize("on", [False, True])
async def test_scheduler_job_runs_only_when_enabled(monkeypatch, on):
    from app import scheduler

    calls = []

    async def fake_job(settings, **_):
        calls.append(settings)
        return {}

    monkeypatch.setattr(tr, "run_training_job", fake_job)
    runtime = types.SimpleNamespace(settings=_settings(analyst_training_enabled=on))
    await scheduler._analyst_training(runtime)
    assert len(calls) == (1 if on else 0)


def test_setting_defaults_off_and_job_is_scheduled_at_0240():
    from app.scheduler import build_scheduler

    assert _settings().analyst_training_enabled is False

    async def _noop(*_a, **_k):
        return None

    class _Stub:
        def __getattr__(self, _name):
            return _noop

    runtime = types.SimpleNamespace(settings=_settings(), cache=None, paper_broker=_Stub(), live_broker=_Stub(),
                                    order_manager=_Stub(), risk_manager=_Stub(),
                                    registry=types.SimpleNamespace(symbols=[]))
    job = build_scheduler(runtime).get_job("analyst_training")
    assert job is not None
    assert str(job.trigger.fields[5]) == "2" and str(job.trigger.fields[6]) == "40"  # hour, minute
