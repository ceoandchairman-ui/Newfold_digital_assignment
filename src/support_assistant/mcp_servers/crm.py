"""MCP server `crm`: what support already knows about this customer. Read-only.

"CRM" here is the platform's own record of past support: resolved issues (`customer_episodes`) and
the last days of conversations (`session_conversations`). An external CRM would be another server
behind the same tool names; callers would not change.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant import repositories as repo
from support_assistant.context import summarize_session
from support_assistant.db import Pool
from support_assistant.mcp_catalog import CRM
from support_assistant.mcp_servers.common import plain, safe

MAX_CASES = 10
MAX_DAYS = 30
MAX_SESSIONS = 10


def build_crm_server(pool: Pool, *, host: str = "0.0.0.0", port: int = CRM.port) -> FastMCP:
    server = FastMCP(CRM.name, host=host, port=port)

    @server.tool()
    async def get_case_history(customer_id: str, limit: int = 5) -> dict[str, Any]:
        """The customer's past resolved issues, newest first: what happened and how it ended."""
        try:
            n = max(1, min(int(limit), MAX_CASES))
            async with pool.connection() as conn:
                rows = await repo.list_episodes(conn, customer_id, n)
            return {"ok": True, "cases": [plain(r) for r in rows]}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def get_recent_interactions(customer_id: str, days: int = 5) -> dict[str, Any]:
        """Conversations from the last `days` days (default 5, at most 30), across sessions."""
        try:
            window = max(1, min(int(days), MAX_DAYS))
            async with pool.connection() as conn:
                rows = await repo.recent_sessions(conn, customer_id, window, MAX_SESSIONS)
            sessions = [
                {
                    "session_id": r["session_id"],
                    "updated_at": str(r["updated_at"]),
                    "messages": len(r["messages"]),
                    "summary": summarize_session(r["messages"]),
                }
                for r in rows
            ]
            return {"ok": True, "days": window, "sessions": sessions}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
