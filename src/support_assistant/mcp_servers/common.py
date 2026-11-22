"""Helpers shared by every MCP server.

Callers (agent harnesses) inject `customer_id` from verified session state; a server never trusts
a customer id typed by a model, and never returns an exception message to the caller.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import structlog

from support_assistant.errors import SupportError, to_customer_message

log = structlog.get_logger()


def plain(row: dict[str, Any]) -> dict[str, Any]:
    """Make a database row JSON-safe (Decimal and date become strings)."""
    return {k: (str(v) if isinstance(v, (Decimal, date)) else v) for k, v in row.items()}


def safe(exc: Exception) -> dict[str, Any]:
    """Turn any exception into a customer-safe tool result. Unexpected errors are logged, not shown."""
    if not isinstance(exc, SupportError):
        log.exception("tool_failed")
    return {
        "ok": False,
        "error": getattr(exc, "code", "internal_error"),
        "message": to_customer_message(exc),
    }
