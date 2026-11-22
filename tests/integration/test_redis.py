"""Real Redis Stack: the LangGraph Redis checkpointer across a 'restart', the run lock, the tool
cache and the token budget. Skipped unless IT_REDIS_URL is set (CI only)."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command

from support_assistant.agents.mcp_client import StaticRegistry
from support_assistant.budget import TokenBudget
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.errors import ConflictError
from support_assistant.locks import RunLock
from support_assistant.orchestrator.graph import build_graph
from support_assistant.tool_cache import CachingRegistry, ToolResultCache

from .conftest import needs_redis

pytestmark = needs_redis


class Client:
    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        if agent == "director":
            return AgentResponse(
                proposed_action={"type": "pay_bill", "bill_id": "B1", "amount": "5"}
            )
        return AgentResponse(verdict="approve")


def _state() -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": "S-it",
        "identity_verified": True,
        "route": "billing",
        "messages": [{"role": "user", "content": "pay"}],
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
        "audit": None,
    }


async def test_a_paused_payment_survives_an_orchestrator_restart(redis):
    """Pause with one saver, resume with a brand-new one: the state lives in Redis (D9)."""
    paid: list[tuple] = []

    async def pay(*a):
        paid.append(a)
        return {"ok": True}

    cfg = {"configurable": {"thread_id": "S-it"}}
    saver1 = AsyncRedisSaver(redis_client=redis, ttl={"default_ttl": 60, "refresh_on_read": True})
    await saver1.asetup()
    out = await build_graph(Client(), pay, saver1).ainvoke(_state(), cfg, durability="sync")
    assert out["__interrupt__"] and paid == []

    saver2 = AsyncRedisSaver(redis_client=redis, ttl={"default_ttl": 60, "refresh_on_read": True})
    await saver2.asetup()  # a new pod: nothing in memory, only Redis
    done = await build_graph(Client(), pay, saver2).ainvoke(
        Command(resume={"approved": True}, update={"audit": None}), cfg, durability="sync"
    )
    assert len(paid) == 1 and "successful" in done["reply"]


async def test_the_run_lock_admits_one_run_per_session(redis):
    lock = RunLock(redis, ttl_ms=5000)
    async with lock.hold("S1"):
        try:
            async with lock.hold("S1"):
                raise AssertionError("second holder must be refused")
        except ConflictError:
            pass
    async with lock.hold("S1"):  # released, so it can be taken again
        pass


async def test_the_tool_cache_serves_repeats_and_invalidation_works(redis):
    calls: list[str] = []

    async def get_bills(customer_id: str) -> dict[str, Any]:
        calls.append(customer_id)
        return {"ok": True, "bills": []}

    cache = ToolResultCache(redis, {"get_account_summary": 60, "get_bills": 30})
    reg = CachingRegistry(StaticRegistry({"get_bills": get_bills}), cache)
    await reg.call("get_bills", {"customer_id": "C1"})
    await reg.call("get_bills", {"customer_id": "C1"})
    await cache.invalidate("C1")
    await reg.call("get_bills", {"customer_id": "C1"})
    assert calls == ["C1", "C1"]


async def test_the_token_budget_counts_in_redis(redis):
    budget = TokenBudget(redis, daily_limit=10)
    assert await budget.consume("C1", 10)
    assert not await budget.consume("C1", 1)
    assert await budget.consume("C2", 10)
