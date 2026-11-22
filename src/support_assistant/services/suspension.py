"""Suspension-lift decision from live data (E3-4)."""

from __future__ import annotations

from datetime import date

from support_assistant import repositories as repo
from support_assistant.db import Pool
from support_assistant.domain.suspension import lift_allowed


async def can_lift_suspension(pool: Pool, customer_id: str, today: date) -> bool:
    async with pool.connection() as conn:
        overdue = await repo.count_overdue_bills(conn, customer_id, today)
        contract_end = await repo.latest_contract_end(conn, customer_id)
    return lift_allowed(overdue, contract_end, today)
