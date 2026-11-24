"""MCP server `stripe`: what happened to a payment, and resolving one that is stuck.

Charging is NOT here. The single write path to the ledger is `process_payment` (database server),
which calls the Stripe gateway behind the same row-lock and idempotency rules. This server only
reads a payment's state and runs the reconcile step that settles or releases a PENDING attempt.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant import repositories as repo
from support_assistant.db import Pool
from support_assistant.errors import NotFoundError
from support_assistant.mcp_catalog import STRIPE
from support_assistant.mcp_servers.common import plain, safe
from support_assistant.services.payments import PaymentGateway, reconcile_payment


def build_stripe_server(
    pool: Pool, gateway: PaymentGateway, *, host: str = "0.0.0.0", port: int = STRIPE.port
) -> FastMCP:
    server = FastMCP(STRIPE.name, host=host, port=port)

    @server.tool()
    async def get_payment_status(customer_id: str, idempotency_key: str) -> dict[str, Any]:
        """The state of one of this customer's payment attempts."""
        try:
            async with pool.connection() as conn:
                tx = await repo.find_transaction(conn, idempotency_key)
            if tx is None or tx["customer_id"] != customer_id:
                raise NotFoundError("payment")  # another customer's payment looks like none at all
            return {
                "ok": True,
                "payment": plain({k: tx[k] for k in ("bill_id", "amount", "status")}),
            }
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def reconcile_payment_attempt(customer_id: str, idempotency_key: str) -> dict[str, Any]:
        """Ask the gateway what happened to a PENDING attempt, then settle or release it."""
        try:
            result = await reconcile_payment(
                pool, gateway, customer_id=customer_id, idempotency_key=idempotency_key
            )
            return {"ok": True, "settled": result is not None}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
