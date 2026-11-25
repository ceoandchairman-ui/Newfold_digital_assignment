"""MCP server `comms` (E8-3, E8-4, E8-6): customer notifications through the outbox.

A tool call only queues the message. A separate worker sends it, so a provider outage never fails
or blocks the conversation (notification failure isolation).
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant.errors import ValidationError
from support_assistant.mcp_catalog import COMMS
from support_assistant.mcp_servers.common import safe
from support_assistant.notifications import Outbox

CHANNELS = frozenset({"email", "sms"})


def build_comms_server(outbox: Outbox, *, host: str = "0.0.0.0", port: int = COMMS.port) -> FastMCP:
    server = FastMCP(COMMS.name, host=host, port=port)

    @server.tool()
    async def send_notification(
        customer_id: str, channel: str, template: str, event_id: str | None = None
    ) -> dict[str, Any]:
        """Queue an email or SMS (Service Bus in Azure; log-only while the stub flags are set)."""
        if channel not in CHANNELS:
            return safe(ValidationError("unsupported channel"))
        try:
            await outbox.enqueue(customer_id, channel, template, event_id)
            return {"ok": True, "queued": True}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
