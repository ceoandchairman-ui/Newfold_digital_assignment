from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.llm import TemplateLLM
from support_assistant.agents.mcp_client import StaticRegistry
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.errors import AuthenticationError, AuthorizationError
from support_assistant.orchestrator.graph import build_graph
from support_assistant.services.retrieval import rrf


def _req(verified: bool = True, **ctx: Any) -> AgentRequest:
    return AgentRequest(
        customer_id="C1",
        session_id="S1",
        identity_verified=verified,
        messages=[{"role": "user", "content": "please pay my bill"}],
        context=ctx,
    )


def _tools(bills: list[dict], recent: int = 0):
    seen: list[dict] = []

    async def get_bills(**kw):
        seen.append(kw)
        return {"ok": True, "bills": bills}

    async def check_recent_payments(**kw):
        return {"ok": True, "recent_payments": recent}

    async def process_payment(**kw):
        return {"ok": True}

    registry = StaticRegistry(
        {
            "get_bills": get_bills,
            "check_recent_payments": check_recent_payments,
            "process_payment": process_payment,
        }
    )
    return registry, seen


BILL = {"id": "B1", "amount_due": "50.00", "due_date": "2026-11-01", "status": "ISSUED"}


# --- retrieval fusion ----------------------------------------------------------------
def test_rrf_rewards_items_ranked_by_both_lists():
    scores = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(scores, key=scores.__getitem__) == 1
    assert scores[1] > scores[2]


# --- harness ----------------------------------------------------------------------------
async def test_unverified_customer_cannot_call_tools():
    tools, _ = _tools([BILL])
    h = Harness("director", roles.DIRECTOR_TOOLS, tools)
    with pytest.raises(AuthenticationError):
        await h.call(_req(verified=False), "get_bills")


async def test_evaluator_cannot_call_write_tools():
    tools, _ = _tools([BILL])
    h = Harness("evaluator", roles.EVALUATOR_TOOLS, tools)
    with pytest.raises(AuthorizationError):
        await h.call(_req(), "process_payment")


async def test_customer_id_comes_from_state_not_model():
    tools, seen = _tools([BILL])
    h = Harness("director", roles.DIRECTOR_TOOLS, tools)
    await h.call(_req(), "get_bills", customer_id="SOMEONE-ELSE")
    assert seen[0]["customer_id"] == "C1"


# --- roles ----------------------------------------------------------------------------------
async def test_director_proposes_but_never_pays():
    tools, _ = _tools([BILL])
    h = Harness("director", roles.DIRECTOR_TOOLS, tools)
    res = await roles.director(_req(), h, TemplateLLM())
    assert res.proposed_action == {"type": "pay_bill", "bill_id": "B1", "amount": "50.00"}


async def test_evaluator_approves_clean_payment():
    tools, _ = _tools([BILL])
    h = Harness("evaluator", roles.EVALUATOR_TOOLS, tools)
    res = await roles.evaluator(
        _req(proposed_action={"type": "pay_bill", "bill_id": "B1", "amount": "50.00"}), h
    )
    assert res.verdict == "approve"


async def test_evaluator_rejects_duplicate_and_overpayment():
    action = {"type": "pay_bill", "bill_id": "B1", "amount": "50.00"}
    tools, _ = _tools([BILL], recent=1)
    res = await roles.evaluator(
        _req(proposed_action=action), Harness("e", roles.EVALUATOR_TOOLS, tools)
    )
    assert res.verdict == "reject"

    tools, _ = _tools([BILL])
    over = {**action, "amount": "99.00"}
    res = await roles.evaluator(
        _req(proposed_action=over), Harness("e", roles.EVALUATOR_TOOLS, tools)
    )
    assert res.verdict == "reject"


# --- graph: approval interrupt then payment -------------------------------------------------
class FakeClient:
    def __init__(self, verdict: str = "approve") -> None:
        self.verdict = verdict

    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        if agent == "director":
            # Turn separation (E4-8): a proposal carries no free text.
            return AgentResponse(
                proposed_action={"type": "pay_bill", "bill_id": "B1", "amount": "50.00"}
            )
        if agent == "evaluator":
            return AgentResponse(verdict=self.verdict, reply="Rejected.")  # type: ignore[arg-type]
        return AgentResponse(reply="Try restarting.")


def _state() -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": "S1",
        "identity_verified": True,
        "route": "billing",
        "messages": [{"role": "user", "content": "pay"}],
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
    }


async def test_payment_waits_for_approval_then_pays():
    paid: list[tuple] = []

    async def pay(cid, bill, amount, key):
        paid.append((cid, bill, amount, key))
        return {"ok": True}

    graph = build_graph(FakeClient(), pay, MemorySaver())
    cfg = {"configurable": {"thread_id": "S1"}}
    out = await graph.ainvoke(_state(), cfg)
    assert out["__interrupt__"] and not paid  # paused, nothing charged yet

    out = await graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert len(paid) == 1 and "successful" in out["reply"]
    assert paid[0][3] == "S1:B1"  # stable idempotency key


async def test_declined_confirmation_does_not_pay():
    paid: list[tuple] = []

    async def pay(*a):
        paid.append(a)
        return {"ok": True}

    graph = build_graph(FakeClient(), pay, MemorySaver())
    cfg = {"configurable": {"thread_id": "S2"}}
    await graph.ainvoke(_state(), cfg)
    out = await graph.ainvoke(Command(resume={"approved": False}), cfg)
    assert not paid and "not made any payment" in out["reply"]


async def test_evaluator_rejection_never_reaches_payment():
    paid: list[tuple] = []

    async def pay(*a):
        paid.append(a)
        return {"ok": True}

    graph = build_graph(FakeClient("reject"), pay, MemorySaver())
    out = await graph.ainvoke(_state(), {"configurable": {"thread_id": "S3"}})
    assert not paid and "__interrupt__" not in out


class MixedClient:
    """A misbehaving director: free text AND an action in the same turn."""

    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        return AgentResponse(
            reply="Paying now!",
            proposed_action={"type": "pay_bill", "bill_id": "B1", "amount": "50.00"},
        )


async def test_a_turn_with_both_a_reply_and_an_action_escalates_and_pays_nothing():
    paid: list[tuple] = []

    async def pay(*a):
        paid.append(a)
        return {"ok": True}

    graph = build_graph(MixedClient(), pay, MemorySaver())
    out = await graph.ainvoke(_state(), {"configurable": {"thread_id": "mixed"}})
    assert out["escalated"] and out["escalation_reason"] == "INVARIANT_VIOLATION"
    assert "__interrupt__" not in out and paid == []  # never reaches approval or the gateway
