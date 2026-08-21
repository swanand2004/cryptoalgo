from __future__ import annotations

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


async def set_kill_switch(on: bool) -> None:
    await set_state("kill_switch", "on" if on else "off")
