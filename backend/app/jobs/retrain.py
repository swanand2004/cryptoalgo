"""Nightly model retraining as its OWN process.

    python -m app.jobs.retrain [crypto|gold|all]

Run by the cryptoalgo-retrain systemd timer on the VPS (deploy/vps/), with
retrain_in_process=false so the trading process schedules none of it. Two
reasons it lives outside the trading process:

* Memory. The gold retrain peaks around +500 MB (1.1 GB total measured). In a
  long-lived process that memory is mostly kept afterwards; in this process it
  is all returned to the OS the moment the job exits.
* Priority. The unit runs at the lowest CPU and I/O priority, so on a 1-core
  server the trading loop always goes first, and if training ever ran out of
  memory only this job would die -- trading would not even notice.

Both trading and the dashboard pick up new models automatically: predictors
read the model store's *_latest.json pointer each time they are created.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
import types

from app.config import get_settings
from app.core.delta_client import DeltaRestClient
from app.core.symbols import SymbolRegistry
from app.scheduler import _gold_retrain, _retrain_models

logger = logging.getLogger("app.jobs.retrain")


async def _noop() -> None:
    return None


async def run(which: str) -> None:
    settings = get_settings()
    client = DeltaRestClient(settings)
    registry = SymbolRegistry(settings)
    try:
        await registry.refresh(client)
    finally:
        await client.aclose()
    # The retrain functions only need the symbol list; the trading process's
    # ensembles do not depend on these models, so there is nothing to refresh.
    runtime = types.SimpleNamespace(registry=registry, settings=settings, refresh_ensembles=_noop)

    if which in ("crypto", "all"):
        t = time.monotonic()
        await _retrain_models(runtime)
        logger.info("crypto ML retrain finished in %.0fs", time.monotonic() - t)
    if which in ("gold", "all"):
        t = time.monotonic()
        await _gold_retrain(runtime)
        logger.info("gold retrain finished in %.0fs", time.monotonic() - t)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("which", nargs="?", default="all", choices=("crypto", "gold", "all"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(args.which))


if __name__ == "__main__":
    main()
