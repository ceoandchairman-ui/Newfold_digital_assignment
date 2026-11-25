"""MCP server `account_management`: who the customer is and what they hold. Read-only.

Every tool is scoped to the verified `customer_id` the harness injects. Nothing here returns a
date of birth, a full IMEI, or a full email address: support needs to confirm them, not to read
them out.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant import repositories as repo
from support_assistant.config import business_today
from support_assistant.db import Pool
from support_assistant.mcp_catalog import ACCOUNT
from support_assistant.mcp_servers.common import mask_email, plain, safe


def build_account_server(pool: Pool, *, host: str = "0.0.0.0", port: int = ACCOUNT.port) -> FastMCP:
    server = FastMCP(ACCOUNT.name, host=host, port=port)

    @server.tool()
    async def get_customer_profile(customer_id: str) -> dict[str, Any]:
        """Name, phone and a masked email for the verified customer."""
        try:
            async with pool.connection() as conn:
                row = await repo.get_customer(conn, customer_id)
            return {
                "ok": True,
                "profile": {
                    "id": row["id"],
                    "full_name": row["full_name"],
                    "phone": row["phone"],
                    "email": mask_email(row.get("email")),
                },
            }
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def get_devices(customer_id: str) -> dict[str, Any]:
        """The customer's devices, with only the last four digits of each IMEI."""
        try:
            async with pool.connection() as conn:
                rows = await repo.list_devices(conn, customer_id)
            devices = [
                {
                    "id": r["id"],
                    "model": r["model"],
                    "line_id": r["line_id"],
                    "imei_last4": (r.get("imei") or "")[-4:] or None,
                }
                for r in rows
            ]
            return {"ok": True, "devices": devices}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def get_contract(customer_id: str) -> dict[str, Any]:
        """Per line: plan, status and contract end date, and whether the contract has expired."""
        try:
            async with pool.connection() as conn:
                lines = await repo.list_lines(conn, customer_id)
            today = business_today()
            contracts = []
            for ln in lines:
                end = ln.get("contract_end_date")
                contracts.append(
                    {
                        **plain(
                            {
                                "line_id": ln["id"],
                                "phone_number": ln["phone_number"],
                                "status": ln["status"],
                                "plan_name": ln["plan_name"],
                                "contract_end_date": end,
                            }
                        ),
                        "expired": bool(end and end < today),
                    }
                )
            return {"ok": True, "contracts": contracts}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
