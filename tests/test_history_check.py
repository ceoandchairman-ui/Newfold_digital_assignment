"""The graph's history_check node: new or returning customer, first or later message."""

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.orchestrator.graph import build_graph
from support_assistant.services.customer_memory import CustomerMemory

NOTE = "Earlier conversations with this customer: sim not working"


class Agents:
    """Records the system notes the agents were given on each call."""

    def __init__(self) -> None:
        self.notes: list[list[str]] = []

    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        if req.context.get("mode") == "route":
            self.notes.append([m["content"] for m in req.messages if m["role"] == "system"])
            return AgentResponse(route="NORMAL")
        return AgentResponse(reply="hello")


class Lookup:
    def __init__(self, *results: CustomerMemory | Exception) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, customer_id: str, session_id: str, query: str) -> CustomerMemory:
        self.calls.append((customer_id, session_id, query))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


async def _pay(*a: Any) -> dict[str, Any]:
    raise AssertionError("must not pay")


class Session:
    def __init__(self, lookup: Lookup) -> None:
        self.agents, self.lookup = Agents(), lookup
        self.graph = build_graph(self.agents, _pay, MemorySaver(), history=lookup)

    async def say(self, text: str, customer: str = "C1", verified: bool = True) -> dict[str, Any]:
        # what the orchestrator sends each turn: no memory fields, so the checkpoint keeps them
        state = {
            "customer_id": customer,
            "session_id": "S1",
            "identity_verified": verified,
            "messages": [{"role": "user", "content": text}],
            "proposed_action": None,
            "eval_attempts": 0,
            "reply": "",
            "escalated": False,
        }
        return await self.graph.ainvoke(state, {"configurable": {"thread_id": "S1"}})  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "memory, expected_notes",
    [
        (CustomerMemory(returning=False), []),  # cases 01 then 00
        (CustomerMemory(returning=True, note=NOTE), [NOTE]),  # cases 11 then 10
    ],
    ids=["new_customer", "returning_customer"],
)
async def test_memory_is_looked_up_once_and_kept_for_the_whole_session(memory, expected_notes):
    s = Session(Lookup(memory))
    for text in ("first", "second", "third"):
        out = await s.say(text)
    assert s.lookup.calls == [("C1", "S1", "first")]
    assert s.agents.notes == [expected_notes] * 3  # exactly one copy, on every message
    assert out["returning_customer"] is memory.returning


async def test_an_unverified_request_never_reads_any_memory():
    s = Session(Lookup(CustomerMemory(returning=True, note=NOTE)))
    out = await s.say("hi", verified=False)
    assert s.lookup.calls == [] and s.agents.notes == [] and "verify" in out["reply"].lower()


async def test_saved_memory_is_never_given_to_another_customer():
    s = Session(Lookup(CustomerMemory(True, NOTE), CustomerMemory(False)))
    await s.say("first", customer="C1")
    await s.say("second", customer="C2")
    assert [c[0] for c in s.lookup.calls] == ["C1", "C2"] and s.agents.notes[1] == []


async def test_a_failed_lookup_does_not_stop_the_turn_and_is_tried_again():
    s = Session(Lookup(ConnectionError("db"), CustomerMemory(True, NOTE)))
    first = await s.say("first")
    await s.say("second")
    await s.say("third")
    assert first["reply"] == "hello" and first["returning_customer"] is None
    assert len(s.lookup.calls) == 2 and s.agents.notes == [[], [NOTE], [NOTE]]


async def test_a_partial_lookup_is_used_now_and_tried_again():
    s = Session(Lookup(CustomerMemory(True, "partial", complete=False), CustomerMemory(True, NOTE)))
    await s.say("first")
    await s.say("second")
    assert s.agents.notes == [["partial"], [NOTE]] and len(s.lookup.calls) == 2
