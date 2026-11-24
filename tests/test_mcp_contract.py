"""MCP server contract tests (E8-7): each server exposes exactly what the shared catalog says.

The servers are built by factories with fakes, so nothing needs a database or environment.
"""

from contextlib import asynccontextmanager

import pytest

from support_assistant.errors import NotFoundError
from support_assistant.mcp_catalog import ACCOUNT, ALL_SERVERS, COMMS, CRM, DATABASE, DEVICE
from support_assistant.mcp_servers.account import build_account_server
from support_assistant.mcp_servers.common import plain, safe
from support_assistant.mcp_servers.comms import build_comms_server
from support_assistant.mcp_servers.crm import build_crm_server
from support_assistant.mcp_servers.database import build_database_server
from support_assistant.mcp_servers.device import build_device_server
from support_assistant.notifications import LogOutbox
from support_assistant.services.payments import StubGateway


class FakePool:
    @asynccontextmanager
    async def connection(self):
        raise AssertionError("tool listing must not touch the database")
        yield


@pytest.fixture
def servers():
    return {
        DATABASE.name: build_database_server(FakePool(), StubGateway()),
        COMMS.name: build_comms_server(LogOutbox()),
        DEVICE.name: build_device_server(),
        ACCOUNT.name: build_account_server(FakePool()),
        CRM.name: build_crm_server(FakePool()),
    }


async def _tools(server):
    return {t.name: t for t in await server.list_tools()}


async def test_every_server_exposes_exactly_the_tools_the_catalog_lists(servers):
    """If a tool is added to a server but not the catalog (or vice versa), this fails."""
    for spec in ALL_SERVERS:
        assert set(await _tools(servers[spec.name])) == set(spec.tools), spec.name


def test_the_catalog_has_no_duplicate_tools_or_ports():
    tools = [t for s in ALL_SERVERS for t in s.tools]
    ports = [s.port for s in ALL_SERVERS]
    assert len(tools) == len(set(tools)) and len(ports) == len(set(ports))


async def test_servers_listen_on_the_cataloged_ports(servers):
    for spec in ALL_SERVERS:
        assert servers[spec.name].settings.port == spec.port


async def test_every_tool_takes_a_required_customer_id(servers):
    """Tenant isolation at the contract level: no tool can be called without naming a customer."""
    for spec in ALL_SERVERS:
        for name, tool in (await _tools(servers[spec.name])).items():
            assert "customer_id" in tool.inputSchema["required"], name


async def test_payment_tool_requires_bill_and_amount_and_keeps_the_key_optional(servers):
    schema = (await _tools(servers[DATABASE.name]))["process_payment"].inputSchema
    assert {"customer_id", "bill_id", "amount"} <= set(schema["required"])
    assert "idempotency_key" not in schema["required"]


async def test_comms_rejects_an_unsupported_channel(servers):
    result = await servers[COMMS.name].call_tool(
        "send_notification", {"customer_id": "C1", "channel": "pigeon", "template": "x"}
    )
    payload = result[1] if isinstance(result, tuple) else result
    assert payload["ok"] is False and payload["error"] == "validation_error"


async def test_comms_queues_a_valid_notification(servers):
    result = await servers[COMMS.name].call_tool(
        "send_notification", {"customer_id": "C1", "channel": "sms", "template": "payment_received"}
    )
    payload = result[1] if isinstance(result, tuple) else result
    assert payload == {"ok": True, "queued": True}


# --- error handling is customer-safe ---------------------------------------------------------------------
def test_tool_errors_never_leak_internals():
    shown = safe(NotFoundError("bill B9 in table bills"))
    assert shown["ok"] is False and "bills" not in shown["message"]
    leaked = safe(RuntimeError("psycopg.OperationalError: table bills"))
    assert "psycopg" not in leaked["message"] and "bills" not in leaked["message"]


def test_rows_are_made_json_safe():
    from datetime import date
    from decimal import Decimal

    assert plain({"a": Decimal("1.50"), "d": date(2026, 1, 2), "n": 3}) == {
        "a": "1.50",
        "d": "2026-01-02",
        "n": 3,
    }
