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
from support_assistant.logs import error_fields

log = structlog.get_logger()


def plain(row: dict[str, Any]) -> dict[str, Any]:
    """Make a database row JSON-safe (Decimal and date become strings)."""
    return {k: (str(v) if isinstance(v, (Decimal, date)) else v) for k, v in row.items()}


def mask_email(email: str | None) -> str | None:
    """j***@example.com: enough to confirm which address is on file, never the whole address."""
    if not email or "@" not in email:
        return None
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def safe(exc: Exception) -> dict[str, Any]:
    """Turn any exception into a customer-safe tool result. Unexpected errors are logged, not shown."""
    if isinstance(exc, SupportError):
        emit = log.warning if exc.http_status >= 500 else log.info
        emit("tool_refused", **{"error.code": exc.code}, **error_fields(exc))
    else:
        log.error("tool_failed", exc_info=exc)
    return {
        "ok": False,
        "error": getattr(exc, "code", "internal_error"),
        "message": to_customer_message(exc),
    }
