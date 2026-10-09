import types

import pytest

from app.config import Settings
from app.core import memory


def _job_ids(settings) -> set[str]:
    from app.scheduler import build_scheduler

    async def _noop(*_a, **_k):
        return None

    class _Stub:  # any method the scheduler wires up resolves to a no-op
        def __getattr__(self, _name):
            return _noop

    runtime = types.SimpleNamespace(
        settings=settings, cache=None, paper_broker=_Stub(), live_broker=_Stub(), order_manager=_Stub(),
        risk_manager=_Stub(), registry=types.SimpleNamespace(symbols=[]),
    )
    scheduler = build_scheduler(runtime)
    return {job.id for job in scheduler.get_jobs()}


def test_retrains_run_in_process_by_default():
    ids = _job_ids(Settings(delta_api_key="k", delta_api_secret="s"))
    assert {"ml_retrain", "gold_retrain"} <= ids
    assert "trading_tick" in ids


def test_server_mode_schedules_no_retrain_in_the_trading_process():
    ids = _job_ids(Settings(delta_api_key="k", delta_api_secret="s", retrain_in_process=False))
    assert not ({"ml_retrain", "gold_retrain"} & ids)
    assert {"trading_tick", "record_equity", "gold_research_tick"} <= ids  # everything else unchanged


@pytest.mark.asyncio
async def test_standalone_job_runs_the_chosen_retrains(monkeypatch):
    import app.jobs.retrain as job

    calls = []

    async def fake_refresh(self, client):
        self.__dict__.setdefault("_symbols", ["BTCUSD"])

    async def fake_crypto(runtime):
        calls.append("crypto")

    async def fake_gold(runtime):
        calls.append("gold")

    monkeypatch.setattr(job.SymbolRegistry, "refresh", fake_refresh)
    monkeypatch.setattr(job, "_retrain_models", fake_crypto)
    monkeypatch.setattr(job, "_gold_retrain", fake_gold)
    await job.run("all")
    await job.run("gold")
    assert calls == ["crypto", "gold", "gold"]


def test_release_free_memory_is_safe_everywhere():
    assert memory.release_free_memory() in (True, False)
