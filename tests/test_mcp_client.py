"""MCP client registry (E8-2): routing by the shared catalog, results, readiness and the harness."""

from types import SimpleNamespace
from typing import Any

import pytest

from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.mcp_client import McpRegistry, StaticRegistry
from support_assistant.contracts import AgentRequest
from support_assistant.errors import AuthorizationError, NotFoundError
from support_assistant.mcp_catalog import ACCOUNT, ALL_SERVERS, COMMS, CRM, DATABASE, DEVICE

URLS = {
    DATABASE.name: "http://db",
    COMMS.name: "http://comms",
    DEVICE.name: "http://dev",
    ACCOUNT.name: "http://account",
    CRM.name: "http://crm",
}
for _s in ALL_SERVERS:  # a server the tests do not name still needs a URL
    URLS.setdefault(_s.name, f"http://{_s.name}")


class FakeConn:
    """Stands in for McpConnection: records calls, serves a configurable tool list."""

    def __init__(self, url: str, served: set[str] | None = None, down: bool = False) -> None:
        self.url, self.served, self.down = url, served, down
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.retry_flags: list[bool] = []
        self.closed = False

    async def call(self, tool: str, args: dict[str, Any], *, retry: bool = True) -> Any:
        self.calls.append((tool, args))
        self.retry_flags.append(retry)
        return SimpleNamespace(structuredContent={"ok": True, "tool": tool, "from": self.url})

    async def list_tools(self) -> Any:
        if self.down:
            raise ConnectionError("down")
        return SimpleNamespace(tools=[SimpleNamespace(name=n) for n in (self.served or set())])

    async def aclose(self) -> None:
        self.closed = True


def make_registry(**overrides: FakeConn):
    conns: dict[str, FakeConn] = {}

    def factory(url: str) -> FakeConn:
        conn = overrides.get(url) or FakeConn(url, served=_served_by(url))
        conns[url] = conn
        return conn

    return McpRegistry(URLS, connection_factory=factory), conns  # type: ignore[arg-type]


def _served_by(url: str) -> set[str]:
    return set(next(s.tools for s in ALL_SERVERS if URLS[s.name] == url))


# --- routing --------------------------------------------------------------------------------------------
async def test_each_tool_is_routed_to_the_server_that_hosts_it():
    reg, conns = make_registry()
    assert (await reg.call("get_bills", {"customer_id": "C1"}))["from"] == "http://db"
    assert (await reg.call("send_notification", {"customer_id": "C1"}))["from"] == "http://comms"
    assert (await reg.call("run_diagnostics", {"customer_id": "C1"}))["from"] == "http://dev"
    assert (await reg.call("get_customer_profile", {"customer_id": "C1"}))[
        "from"
    ] == "http://account"
    assert (await reg.call("get_case_history", {"customer_id": "C1"}))["from"] == "http://crm"
    for tool in ("send_sms", "get_payment_status", "send_email", "search_documents"):
        await reg.call(tool, {"customer_id": "C1"})  # the four provider servers
    assert [len(c.calls) for c in conns.values()] == [1] * len(ALL_SERVERS)


def test_the_registry_knows_exactly_the_cataloged_tools():
    reg, _ = make_registry()
    every_tool = {t for s in ALL_SERVERS for t in s.tools}
    assert all(reg.has(t) for t in every_tool) and not reg.has("drop_database")


async def test_an_unknown_tool_is_refused_before_any_network_call():
    reg, conns = make_registry()
    with pytest.raises(NotFoundError):
        await reg.call("drop_database", {})
    assert all(not c.calls for c in conns.values())


async def test_a_result_without_structured_content_is_reported_not_trusted():
    class Bare(FakeConn):
        async def call(self, tool, args, *, retry=True):
            return SimpleNamespace(structuredContent=None)

    reg, _ = make_registry(**{"http://db": Bare("http://db")})
    assert (await reg.call("get_bills", {}))["ok"] is False


async def test_close_closes_every_connection():
    reg, conns = make_registry()
    await reg.aclose()
    assert len(conns) == len(ALL_SERVERS) and all(c.closed for c in conns.values())


# --- readiness (verify) -----------------------------------------------------------------------------------
async def test_verify_is_empty_when_every_needed_tool_is_served():
    reg, _ = make_registry()
    assert await reg.verify(roles.DIRECTOR_TOOLS) == []


async def test_verify_names_tools_a_server_does_not_actually_serve():
    short = FakeConn("http://db", served={"get_bills"})  # server is missing the other tools
    reg, _ = make_registry(**{"http://db": short})
    missing = await reg.verify({"get_bills", "get_account_summary"})
    assert missing == ["get_account_summary"]


async def test_verify_treats_an_unreachable_server_as_all_its_tools_missing():
    down = FakeConn("http://db", down=True)
    reg, _ = make_registry(**{"http://db": down})
    assert await reg.verify({"get_bills", "process_payment"}) == ["get_bills", "process_payment"]


async def test_verify_flags_tools_the_catalog_does_not_know():
    reg, _ = make_registry()
    assert await reg.verify({"get_bills", "mystery_tool"}) == ["mystery_tool"]


# --- the harness goes through the registry ------------------------------------------------------------
def _req() -> AgentRequest:
    return AgentRequest(customer_id="C1", session_id="S1", identity_verified=True, messages=[])


async def test_the_harness_injects_the_customer_and_routes_through_the_registry():
    reg, conns = make_registry()
    h = Harness("director", roles.DIRECTOR_TOOLS, reg)
    await h.call(_req(), "get_bills", customer_id="SOMEONE-ELSE")
    assert conns["http://db"].calls == [("get_bills", {"customer_id": "C1"})]
    assert h.audit == [{"agent": "director", "tool": "get_bills", "ok": True}]


async def test_the_allow_list_is_enforced_even_though_the_registry_knows_the_tool():
    reg, conns = make_registry()
    h = Harness("evaluator", roles.EVALUATOR_TOOLS, reg)
    with pytest.raises(AuthorizationError):
        await h.call(_req(), "process_payment", amount="1")
    assert conns["http://db"].calls == []  # refused by the harness, never sent


async def test_the_harness_refuses_a_tool_the_registry_does_not_have():
    h = Harness("director", roles.DIRECTOR_TOOLS, StaticRegistry({}))
    with pytest.raises(AuthorizationError):
        await h.call(_req(), "get_bills")
