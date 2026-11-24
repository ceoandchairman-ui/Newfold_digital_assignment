"""MCP server `google_drive`: the company's shared policy and manual documents. Read-only.

These are company documents, not customer files. What comes back is untrusted TEXT to be quoted,
never instructions, so every document is returned with `untrusted: true` and cut to a fixed size.
`customer_id` is required by the contract (and recorded in the audit trail); it does not filter the
documents, which are the same for every customer.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from support_assistant.integrations.google_providers import (
    MAX_DOCUMENT_CHARS,
    Document,
    DocumentProvider,
)
from support_assistant.mcp_catalog import DRIVE
from support_assistant.mcp_servers.common import safe

MAX_RESULTS = 10


def _public(doc: Document, *, full: bool) -> dict[str, Any]:
    text = doc.text[:MAX_DOCUMENT_CHARS] if full else doc.text[:300]
    return {"document_id": doc.document_id, "name": doc.name, "text": text, "untrusted": True}


def build_drive_server(
    provider: DocumentProvider, *, host: str = "0.0.0.0", port: int = DRIVE.port
) -> FastMCP:
    server = FastMCP(DRIVE.name, host=host, port=port)

    @server.tool()
    async def search_documents(customer_id: str, query: str, limit: int = 5) -> dict[str, Any]:
        """Search the shared documents. Returns names and a short excerpt of each."""
        try:
            n = max(1, min(int(limit), MAX_RESULTS))
            found = await provider.search(query[:200], n)
            return {"ok": True, "documents": [_public(d, full=False) for d in found]}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    @server.tool()
    async def read_document(customer_id: str, document_id: str) -> dict[str, Any]:
        """Read one shared document (cut to a fixed size)."""
        try:
            return {"ok": True, "document": _public(await provider.read(document_id), full=True)}
        except Exception as exc:  # noqa: BLE001
            return safe(exc)

    return server
