"""E8-2 resilience (no auto-retry of writes, timeouts, circuit breaker), E4-5 lazy retrieval,
E5-4 citations and E6-5 token budgets."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from fakes import FakeRetrieval
from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.llm import TemplateLLM
from support_assistant.agents.mcp_client import McpConnection, McpRegistry, StaticRegistry
from support_assistant.agents.mcp_client.registry import CircuitBreaker
from support_assistant.budget import TokenBudget, estimate_tokens
from support_assistant.contracts import AgentRequest
from support_assistant.domain.intent import CLOSING_REPLY, needs_retrieval
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.mcp_catalog import (
    ACCOUNT,
    ALL_SERVERS,
    COMMS,
    CRM,
    DATABASE,
    DEVICE,
    WRITE_TOOLS,
)


# --- connection: writes are never retried, calls time out ------------------------------------------
class FlakySession:
    """Fails the first call, as a dropped connection would."""

    def __init__(self, log):
        self.log = log

    async def call_tool(self, tool, args):
        self.log.append(tool)
        if len(self.log) == 1:
            raise ConnectionError("dropped")
        return {"tool": tool}


def opener(log, sessions=3):
    @asynccontextmanager
    async def open_(url):
        yield FlakySession(log)

    return open_


async def test_a_read_is_retried_once_after_a_dropped_connection():
    log: list[str] = []
    conn = McpConnection("u", opener(log))
    assert (await conn.call("get_bills", {}))["tool"] == "get_bills"
    assert log == ["get_bills", "get_bills"]
    await conn.aclose()


async def test_a_write_is_never_retried_so_an_sms_cannot_be_sent_twice():
    log: list[str] = []
    conn = McpConnection("u", opener(log))
    with pytest.raises(ExternalServiceUnavailable):
        await conn.call("send_notification", {}, retry=False)
    assert log == ["send_notification"]  # exactly one attempt
    await conn.aclose()


async def test_a_slow_server_times_out_instead_of_hanging_the_turn():
    class Slow:
        async def call_tool(self, tool, args):
            await asyncio.sleep(5)

    @asynccontextmanager
    async def open_(url):
        yield Slow()

    conn = McpConnection("u", open_, timeout_s=0.05)
    with pytest.raises(ExternalServiceUnavailable):
        await conn.call("get_bills", {}, retry=False)
    await conn.aclose()


def test_payments_and_every_outbound_message_are_write_tools():
    assert WRITE_TOOLS == {
        "process_payment",
        "reconcile_payment_attempt",
        "send_notification",
        "send_sms",
        "send_email",
    }


# --- registry: write flag and circuit breaker ----------------------------------------------------------
URLS = {
    DATABASE.name: "db",
    COMMS.name: "comms",
    DEVICE.name: "dev",
    ACCOUNT.name: "account",
    CRM.name: "crm",
}
for _s in ALL_SERVERS:  # a server the tests do not name still needs a URL
    URLS.setdefault(_s.name, f"http://{_s.name}")


class RecConn:
    def __init__(self, url, fail=False):
        self.url, self.fail, self.calls = url, fail, []

    async def call(self, tool, args, *, retry=True):
        self.calls.append((tool, retry))
        if self.fail:
            raise ExternalServiceUnavailable("down")
        return SimpleNamespace(structuredContent={"ok": True})

    async def aclose(self):
        pass


async def test_the_registry_disables_retry_for_write_tools_only():
    conns: dict[str, RecConn] = {}

    def factory(url):
        conns[url] = RecConn(url)
        return conns[url]

    reg = McpRegistry(URLS, connection_factory=factory)  # type: ignore[arg-type]
    await reg.call("get_bills", {})
    await reg.call("process_payment", {})
    await reg.call("send_notification", {})
    assert conns["db"].calls == [("get_bills", True), ("process_payment", False)]
    assert conns["comms"].calls == [("send_notification", False)]


async def test_an_open_circuit_fails_fast_without_calling_the_server():
    now = [0.0]
    conns: dict[str, RecConn] = {}

    def factory(url):
        conns[url] = RecConn(url, fail=(url == "db"))
        return conns[url]

    reg = McpRegistry(
        URLS,
        connection_factory=factory,  # type: ignore[arg-type]
        breaker_factory=lambda: CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: now[0]),
    )
    for _ in range(2):
        with pytest.raises(ExternalServiceUnavailable):
            await reg.call("get_bills", {})
    calls_before = len(conns["db"].calls)
    with pytest.raises(ExternalServiceUnavailable):
        await reg.call("get_bills", {})
    assert len(conns["db"].calls) == calls_before  # circuit open: the server was not called
    assert (await reg.call("run_diagnostics", {}))["ok"]  # other servers are unaffected


def test_the_breaker_allows_a_trial_after_cooldown_and_closes_on_success():
    now = [0.0]
    b = CircuitBreaker(threshold=1, cooldown_s=10, clock=lambda: now[0])
    b.failure()
    assert b.is_open
    now[0] = 11
    assert not b.is_open  # trial call allowed
    b.success()
    assert not b.is_open


# --- lazy retrieval and citations (E4-5, E5-4) -------------------------------------------------------------
@pytest.mark.parametrize(
    "ack", ["thanks", "Thank you!", "ok", "that worked", "it works now.", "bye"]
)
def test_acknowledgements_need_no_retrieval(ack):
    assert not needs_retrieval(ack)


@pytest.mark.parametrize(
    "problem", ["no signal", "ok so my data is slow", "thanks but it still fails"]
)
def test_problems_need_retrieval(problem):
    assert needs_retrieval(problem)


def _req(text: str, **ctx: Any) -> AgentRequest:
    return AgentRequest(
        customer_id="C1",
        session_id="S1",
        identity_verified=True,
        messages=[{"role": "user", "content": text}],
        context=ctx,
    )


async def test_an_acknowledgement_gets_a_closing_reply_without_any_search():
    retrieval = FakeRetrieval()
    h = Harness("troubleshooter", roles.TROUBLESHOOTER_TOOLS, StaticRegistry({}))
    res = await roles.troubleshooter(_req("thanks!"), h, TemplateLLM(), retrieval)
    assert res.reply == CLOSING_REPLY and retrieval.seen == [] and not res.escalate


async def test_troubleshooting_answers_cite_their_manual_sections():
    retrieval = FakeRetrieval(
        ("4.2", "Guide > Data > APN", "Reset the APN settings."),
        ("4.3", "Guide > Data > APN", "Restart afterwards."),
        ("2.1", "Guide > Signal", "Toggle airplane mode."),
    )
    h = Harness("troubleshooter", roles.TROUBLESHOOTER_TOOLS, StaticRegistry({}))
    res = await roles.troubleshooter(_req("my data fails"), h, TemplateLLM(), retrieval)
    assert res.reply.endswith("Source: Guide > Data > APN; Guide > Signal")  # deduplicated, max 2


# --- token budget (E6-5) ---------------------------------------------------------------------------------------
class Counter:
    def __init__(self, fail=False):
        self.data: dict[str, int] = {}
        self.expiry: dict[str, int] = {}
        self.fail = fail

    async def incrby(self, key, n):
        if self.fail:
            raise ConnectionError
        self.data[key] = self.data.get(key, 0) + n
        return self.data[key]

    async def expire(self, key, s):
        self.expiry[key] = s


DAY = datetime(2026, 11, 21, tzinfo=UTC)


async def test_budget_allows_up_to_the_limit_then_refuses():
    b = TokenBudget(Counter(), daily_limit=10)
    assert await b.consume("C1", 6, DAY) and await b.consume("C1", 4, DAY)
    assert not await b.consume("C1", 1, DAY)


async def test_budgets_are_per_customer_and_per_day():
    redis = Counter()
    b = TokenBudget(redis, daily_limit=5)
    assert not await b.consume("C1", 9, DAY)
    assert await b.consume("C2", 5, DAY)  # another customer is unaffected
    assert await b.consume("C1", 5, datetime(2026, 11, 22, tzinfo=UTC))  # a new day resets
    assert all(v == 2 * 24 * 3600 for v in redis.expiry.values())


async def test_budget_fails_open_when_redis_is_down():
    assert await TokenBudget(Counter(fail=True), daily_limit=1).consume("C1", 999, DAY)


def test_token_estimate_is_never_zero():
    assert estimate_tokens("") == 1 and estimate_tokens("x" * 400) == 100
