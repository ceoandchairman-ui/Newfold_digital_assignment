"""Persist the audit trail (E5-5) without slowing the turn down.

The agents return audit entries with each response; the graph collects them for the turn, and the
orchestrator hands them to an AuditSink after replying. Writing happens in the background, so the
customer never waits for it. A failed write is logged loudly and the structured logs (which also
carry every tool call) remain the fallback record.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import structlog

from support_assistant.db import Pool

log = structlog.get_logger()
AuditEntry = dict[str, Any]


class AuditSink(Protocol):
    async def write(self, session_id: str, customer_id: str, entries: list[AuditEntry]) -> None: ...


class PgAuditSink:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def write(self, session_id: str, customer_id: str, entries: list[AuditEntry]) -> None:
        if not entries:
            return
        rows = [
            (session_id, customer_id, str(e.get("agent")), str(e.get("tool")), bool(e.get("ok")))
            for e in entries
        ]
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cur = conn.cursor()
                await cur.executemany(
                    "INSERT INTO audit_events (session_id, customer_id, agent, tool, ok)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    rows,
                )


class BackgroundAudit:
    """Fire-and-forget writes that keep a reference to each task until it finishes."""

    def __init__(self, sink: AuditSink) -> None:
        self._sink = sink
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, session_id: str, customer_id: str, entries: list[AuditEntry]) -> None:
        if not entries:
            return
        task = asyncio.create_task(self._write(session_id, customer_id, entries))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _write(self, session_id: str, customer_id: str, entries: list[AuditEntry]) -> None:
        try:
            await self._sink.write(session_id, customer_id, entries)
        except Exception:  # noqa: BLE001
            log.error("audit_write_failed", session_id=session_id, entries=len(entries))

    async def drain(self) -> None:
        """Wait for pending writes (shutdown and tests)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
