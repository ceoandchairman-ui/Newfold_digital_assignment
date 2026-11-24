"""MCP server `twilio_sms`: send an SMS to the verified customer through Twilio, and check on it.

The message text is never written by a model: `send_sms` takes a template name and the server fills
in the text. The phone number comes from the customer's record, never from a tool argument, so a
model cannot send a message to a number of its choosing.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant import repositories as repo
from support_assistant.db import Pool
from support_assistant.errors import NotFoundError, ValidationError
from support_assistant.integrations.twilio_client import SmsProvider
from support_assistant.mcp_catalog import TWILIO
from support_assistant.mcp_servers.common import safe
from support_assistant.worker import TEMPLATES


def build_twilio_server(
    pool: Pool, provider: SmsProvider, *, host: str = "0.0.0.0", port: int = TWILIO.port
) -> FastMCP:
    server = FastMCP(TWILIO.name, host=host, port=port)

    @server.tool()
    async def send_sms(customer_id: str, template: str) -> dict[str, Any]:
        """Send a templated SMS to the customer's phone number on file."""
        try:
            if template not in TEMPLATES:
                raise ValidationError("unsupported template")
            async with pool.connection() as conn:
                phone = str((await repo.get_customer(conn, customer_id))["phone"])
            sent = await provider.send(phone, TEMPLATES[template])
            return {"ok": True, "message_sid": sent.sid, "status": sent.status}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def get_message_status(customer_id: str, message_sid: str) -> dict[str, Any]:
        """Delivery status of a message, only if it was sent to this customer's number."""
        try:
            async with pool.connection() as conn:
                phone = str((await repo.get_customer(conn, customer_id))["phone"])
            found = await provider.status(message_sid)
            if found is None or found.to != phone:  # someone else's message looks like none at all
                raise NotFoundError("message")
            return {"ok": True, "message_sid": found.sid, "status": found.status}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
