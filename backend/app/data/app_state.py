from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.data.db import SessionLocal
from app.data.models import AppState


async def get_state(key: str, default: str | None = None) -> str | None:
    async with SessionLocal() as session:
        result = await session.execute(select(AppState).where(AppState.key == key))
        row = result.scalar_one_or_none()
        return row.value if row else default


async def set_state(key: str, value: str) -> None:
    async with SessionLocal() as session:
        stmt = sqlite_insert(AppState).values(key=key, value=value)
        stmt = stmt.on_conflict_do_update(index_elements=["key"], set_={"value": value})
        await session.execute(stmt)
        await session.commit()


async def get_trading_mode() -> str:
    return await get_state("trading_mode", "paper") or "paper"


async def set_trading_mode(mode: str) -> None:
    if mode not in ("paper", "live"):
        raise ValueError(f"invalid trading mode: {mode}")
    await set_state("trading_mode", mode)


async def is_kill_switch_on() -> bool:
    return (await get_state("kill_switch", "off")) == "on"


async def set_kill_switch(on: bool, reason: str = "manual (dashboard)") -> None:
    await set_state("kill_switch", "on" if on else "off")
    if on:
        # Records the calendar day (UTC) the switch was most recently
        # engaged -- whether by a daily breach auto-trigger or a manual
        # toggle -- so scheduler._daily_kill_switch_reset can tell "engaged
        # today" (leave it alone) apart from "engaged on some earlier day
        # and never reset" (stale, should be cleared) without needing its
        # own separate bookkeeping of when it last checked.
        await set_state("kill_switch_engaged_day", dt.datetime.now(dt.timezone.utc).date().isoformat())
        # Owner alert (Telegram, off unless configured): every halt passes
        # through here -- the dashboard button and all three risk supervisors.
        # Fire-and-forget; it can never raise or delay the halt.
        from app.notify.telegram import kill_switch_on

        kill_switch_on(reason)
