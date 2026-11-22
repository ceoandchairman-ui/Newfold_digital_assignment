"""MCP server `database` (E8-1): the only path agents have to the ledger.

Reads are customer-scoped; `process_payment` is the single write and runs the idempotent,
row-locked payment flow. It is only reachable after the human-approval step in the orchestrator.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant import repositories as repo
from support_assistant.db import Pool
from support_assistant.mcp_catalog import DATABASE
from support_assistant.mcp_servers.common import plain, safe
from support_assistant.services.payments import PaymentGateway, pay_bill
from support_assistant.services.suspension import can_lift_suspension


def build_database_server(
    pool: Pool, gateway: PaymentGateway, *, host: str = "0.0.0.0", port: int = DATABASE.port
) -> FastMCP:
    server = FastMCP(DATABASE.name, host=host, port=port)

    @server.tool()
    async def get_account_summary(customer_id: str) -> dict[str, Any]:
        """Lines, plans and data usage for the verified customer."""
        try:
            async with pool.connection() as conn:
                lines = await repo.list_lines(conn, customer_id)
            return {"ok": True, "lines": [plain(r) for r in lines]}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def get_bills(customer_id: str) -> dict[str, Any]:
        """All bills with status."""
        try:
            async with pool.connection() as conn:
                bills = await repo.list_bills(conn, customer_id)
            return {"ok": True, "bills": [plain(r) for r in bills]}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def check_suspension_lift(customer_id: str) -> dict[str, Any]:
        """Whether a suspension may be lifted: no overdue bills and an unexpired contract."""
        try:
            allowed = await can_lift_suspension(pool, customer_id, date.today())
            return {"ok": True, "allowed": allowed}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def check_recent_payments(
        customer_id: str, bill_id: str, minutes: int = 10
    ) -> dict[str, Any]:
        """Velocity check used by the evaluator to catch duplicate payments."""
        try:
            async with pool.connection() as conn:
                n = await repo.recent_payment_count(conn, customer_id, bill_id, minutes)
            return {"ok": True, "recent_payments": n}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def process_payment(
        customer_id: str, bill_id: str, amount: str, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Pay a bill. Only reachable after the human-approval step in the orchestrator."""
        try:
            result = await pay_bill(
                pool,
                gateway,
                customer_id=customer_id,
                bill_id=bill_id,
                amount=Decimal(amount),
                idempotency_key=idempotency_key or str(uuid.uuid4()),
            )
            return {
                "ok": True,
                "transaction_id": result.transaction_id,
                "replayed": result.replayed,
            }
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
