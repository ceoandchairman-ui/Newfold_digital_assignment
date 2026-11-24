"""MCP server `gmail`: the company's support mailbox. Not a customer's mailbox.

`send_email` takes a template name and sends to the address on the customer's record, so a model
cannot choose the recipient or write the body. `get_support_thread` reads support conversations
with the customer's own address, for escalation hand-off.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant import repositories as repo
from support_assistant.db import Pool
from support_assistant.errors import ValidationError
from support_assistant.integrations.google_providers import EmailProvider
from support_assistant.mcp_catalog import GMAIL
from support_assistant.mcp_servers.common import safe
from support_assistant.worker import SUBJECTS, TEMPLATES

MAX_THREADS = 5


def build_gmail_server(
    pool: Pool, provider: EmailProvider, *, host: str = "0.0.0.0", port: int = GMAIL.port
) -> FastMCP:
    server = FastMCP(GMAIL.name, host=host, port=port)

    async def _address(customer_id: str) -> str:
        async with pool.connection() as conn:
            email = (await repo.get_customer(conn, customer_id)).get("email")
        if not email:
            raise ValidationError("no email address on file")
        return str(email)

    @server.tool()
    async def send_email(customer_id: str, template: str) -> dict[str, Any]:
        """Send a templated email to the customer's address on file."""
        try:
            if template not in TEMPLATES:
                raise ValidationError("unsupported template")
            message_id = await provider.send(
                await _address(customer_id), SUBJECTS[template], TEMPLATES[template]
            )
            return {"ok": True, "message_id": message_id}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def get_support_thread(customer_id: str, limit: int = 3) -> dict[str, Any]:
        """Recent support-mailbox conversations with this customer's address."""
        try:
            n = max(1, min(int(limit), MAX_THREADS))
            found = await provider.threads(await _address(customer_id), n)
            return {
                "ok": True,
                "threads": [
                    {"thread_id": t.thread_id, "subject": t.subject, "snippet": t.snippet}
                    for t in found
                ],
            }
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
