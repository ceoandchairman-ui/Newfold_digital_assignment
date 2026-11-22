"""Thin LangGraph router (Fat Backend / Skinny Graph). Agents are called through AgentClient.

Flow:  auth_gate (deterministic, no LLM) -> unverified: stop with a verification prompt
       verified -> director | retrieve -> troubleshooter
       retrieve: a LangChain retriever called from a node, outside the agent harness (D8)
       director -> (proposed payment?) evaluate -> approve (interrupt) -> pay -> respond
       evaluator reject -> explain and stop; evaluator outage or repeated rejects -> escalate (E3-7)
Pauses use interrupt(); state is checkpointed by the official Postgres saver (D9).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from opentelemetry.trace import Status, StatusCode

from support_assistant.contracts import AgentRequest
from support_assistant.domain.escalation import (
    ESCALATION_MESSAGE,
    EscalationReason,
    troubleshoot_loop_exceeded,
)
from support_assistant.domain.intent import needs_retrieval
from support_assistant.errors import (
    AuthenticationError,
    ExternalServiceUnavailable,
    InvariantViolation,
    to_customer_message,
)
from support_assistant.guards import assert_turn_separation, sanitize_reply
from support_assistant.observability import PAYMENTS, tracer
from support_assistant.orchestrator.client import AgentClient

STATE_SCHEMA_VERSION = 1  # bump with a migration path when the state shape changes (E7-3)


def _audit_reducer(
    old: list[dict[str, Any]] | None, new: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """None resets (start of a turn or a resume); a list is appended to this turn's entries."""
    if new is None:
        return []
    return [*(old or []), *new]


def _traced(name: str, fn: Any) -> Any:
    """One span per graph node (E12-B4). An approval pause (GraphInterrupt) is not an error."""

    async def node(state: SessionState) -> dict[str, Any]:
        with tracer.start_as_current_span(
            f"graph.node {name}",
            attributes={"langgraph.node": name, "support.route": str(state.get("route", ""))},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                out: dict[str, Any] = await fn(state)
            except GraphInterrupt:
                span.set_attribute("support.awaiting_approval", True)
                raise
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            if out.get("escalated"):
                span.set_attribute("support.escalated", True)
                span.set_attribute("support.escalation_reason", str(out.get("escalation_reason")))
            if name == "pay":
                span.set_attribute("support.payment", True)  # tail sampling keeps all of these
            return out

    return node


class SessionState(TypedDict, total=False):
    schema_version: int
    audit: Annotated[list[dict[str, Any]], _audit_reducer]  # this turn's tool calls (E5-5)
    customer_id: str
    session_id: str
    identity_verified: bool
    route: Literal["billing", "troubleshoot"]
    messages: list[dict[str, Any]]  # plain dicts; the backend owns history (memory.py)
    search_query: str | None  # rewritten retrieval query from the backend (chains.py)
    kb_chunks: list[dict[str, str]]  # trimmed retrieval hits (payload hygiene, E7-3)
    payment_verified: bool  # true only after a confirmed PAID read; gates "payment successful" text
    proposed_action: dict[str, Any] | None
    eval_attempts: int
    troubleshoot_turns: int
    reply: str
    escalated: bool
    escalation_reason: str | None


PaymentCaller = Any  # async (customer_id, bill_id, amount, idempotency_key) -> dict


def build_graph(
    client: AgentClient,
    pay: PaymentCaller,
    checkpointer: BaseCheckpointSaver[Any],
    max_eval_retries: int = 2,
    max_troubleshoot_turns: int = 3,
    retriever: BaseRetriever | None = None,
) -> Any:
    def _req(state: SessionState, **ctx: Any) -> AgentRequest:
        return AgentRequest(
            customer_id=state["customer_id"],
            session_id=state["session_id"],
            identity_verified=state.get("identity_verified", False),
            messages=list(state.get("messages", [])),
            context={
                **ctx,
                **({"search_query": state["search_query"]} if state.get("search_query") else {}),
                **({"kb_chunks": state["kb_chunks"]} if state.get("kb_chunks") else {}),
            },
        )

    def _query(state: SessionState) -> str:
        if state.get("search_query"):
            return str(state["search_query"])
        for m in reversed(state.get("messages", [])):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    async def auth_gate(state: SessionState) -> dict[str, Any]:
        """Deterministic entry check (E4-2). No model decides whether someone is verified."""
        return {}

    async def egress(state: SessionState) -> dict[str, Any]:
        """Every path ends here: the customer only ever sees filtered text (E5-4)."""
        reply = sanitize_reply(
            state.get("reply", ""),
            customer_id=state.get("customer_id", ""),
            payment_verified=bool(state.get("payment_verified")),
        )
        return {"reply": reply}

    async def deny(state: SessionState) -> dict[str, Any]:
        return {"reply": AuthenticationError("unverified").customer_message}

    async def retrieve(state: SessionState) -> dict[str, Any]:
        """Retriever call from a node (D8). A retrieval failure degrades to the agent's own search."""
        if retriever is None or not needs_retrieval(_query(state)):
            return {}  # lazy (E4-5): an acknowledgement needs no search
        try:
            docs = await retriever.ainvoke(_query(state))
        except Exception:  # noqa: BLE001 - never block troubleshooting on the retriever
            return {"kb_chunks": []}
        return {
            "kb_chunks": [
                {
                    "section_id": str(d.metadata.get("section_id", "")),
                    "heading_path": str(d.metadata.get("heading_path", "")),
                    "content": d.page_content[:1500],
                }
                for d in docs
            ]
        }

    def _escalate(reason: EscalationReason) -> dict[str, Any]:
        return {"reply": ESCALATION_MESSAGE, "escalated": True, "escalation_reason": reason.value}

    async def director(state: SessionState) -> dict[str, Any]:
        try:
            res = await client.run("director", _req(state))
        except ExternalServiceUnavailable as exc:
            return {"reply": to_customer_message(exc), "proposed_action": None}
        try:
            assert_turn_separation(res)  # reply OR action, never both (E4-8)
        except InvariantViolation:
            return {"proposed_action": None, **_escalate(EscalationReason.INVARIANT_VIOLATION)}
        return {"reply": res.reply, "proposed_action": res.proposed_action, "audit": res.audit}

    async def troubleshooter(state: SessionState) -> dict[str, Any]:
        turns = state.get("troubleshoot_turns", 0) + 1
        if troubleshoot_loop_exceeded(turns, max_troubleshoot_turns + 1):
            return _escalate(EscalationReason.TROUBLESHOOT_LOOP)
        try:
            res = await client.run("troubleshooter", _req(state))
        except ExternalServiceUnavailable as exc:
            return {"reply": to_customer_message(exc), "troubleshoot_turns": turns}
        if res.escalate:
            return {
                "reply": res.reply,
                "escalated": True,
                "escalation_reason": res.escalation_reason,
            }
        return {"reply": res.reply, "troubleshoot_turns": turns, "audit": res.audit}

    async def evaluate(state: SessionState) -> dict[str, Any]:
        attempts = state.get("eval_attempts", 0) + 1
        try:
            res = await client.run(
                "evaluator", _req(state, proposed_action=state["proposed_action"])
            )
        except ExternalServiceUnavailable:
            return {
                "eval_attempts": attempts,
                "proposed_action": None,
                **_escalate(EscalationReason.EVAL_RETRIES_EXHAUSTED),
            }
        if res.verdict == "approve":
            return {"eval_attempts": attempts, "audit": res.audit}
        if attempts > max_eval_retries:
            return {
                "eval_attempts": attempts,
                "proposed_action": None,
                **_escalate(EscalationReason.EVAL_RETRIES_EXHAUSTED),
            }
        return {
            "eval_attempts": attempts,
            "proposed_action": None,
            "reply": res.reply,
            "audit": res.audit,
        }

    async def approve(state: SessionState) -> dict[str, Any]:
        action = state["proposed_action"] or {}
        decision = interrupt(
            {
                "question": f"Confirm payment of {action.get('amount')} for bill {action.get('bill_id')}?",
                "action": action,
            }
        )
        if decision is True or (isinstance(decision, dict) and decision.get("approved") is True):
            return {}
        return {"proposed_action": None, "reply": "Okay, I have not made any payment."}

    async def pay_node(state: SessionState) -> dict[str, Any]:
        action = state.get("proposed_action")
        if not action:
            return {}
        key = f"{state['session_id']}:{action['bill_id']}"  # stable across graph replays
        result = await pay(state["customer_id"], action["bill_id"], action["amount"], key)
        entry = [{"agent": "orchestrator", "tool": "process_payment", "ok": bool(result.get("ok"))}]
        PAYMENTS.add(
            1, {"outcome": "succeeded" if result.get("ok") else str(result.get("error", "failed"))}
        )
        if result.get("ok"):
            # The MCP payment tool already did a read-after-write PAID check before returning ok.
            return {
                "reply": f"Your payment of {action['amount']} was successful.",
                "proposed_action": None,
                "payment_verified": True,
                "audit": entry,
            }
        return {
            "reply": result.get("message", "Payment could not be completed."),
            "proposed_action": None,
            "audit": entry,
        }

    def route(state: SessionState) -> str:
        if not state.get("identity_verified"):
            return "deny"
        return "retrieve" if state.get("route") == "troubleshoot" else "billing"

    def after_director(state: SessionState) -> str:
        return "evaluate" if state.get("proposed_action") else "egress"

    def after_evaluate(state: SessionState) -> str:
        if state.get("escalated"):
            return "egress"
        return "approve" if state.get("proposed_action") else "egress"

    def after_approve(state: SessionState) -> str:
        return "pay" if state.get("proposed_action") else "egress"

    g = StateGraph(SessionState)
    g.add_node("director", _traced("director", director))
    g.add_node("troubleshooter", _traced("troubleshooter", troubleshooter))
    g.add_node("evaluate", _traced("evaluate", evaluate))
    g.add_node("approve", _traced("approve", approve))
    g.add_node("pay", _traced("pay", pay_node))
    g.add_node("auth_gate", _traced("auth_gate", auth_gate))
    g.add_node("deny", _traced("deny", deny))
    g.add_node("retrieve", _traced("retrieve", retrieve))
    g.add_edge(START, "auth_gate")
    g.add_conditional_edges(
        "auth_gate",
        route,
        {"deny": "deny", "billing": "director", "retrieve": "retrieve"},
    )
    g.add_edge("retrieve", "troubleshooter")
    g.add_node("egress", _traced("egress", egress))
    g.add_edge("deny", "egress")
    g.add_conditional_edges(
        "director", after_director, {"evaluate": "evaluate", "egress": "egress"}
    )
    g.add_conditional_edges("evaluate", after_evaluate, {"approve": "approve", "egress": "egress"})
    g.add_conditional_edges("approve", after_approve, {"pay": "pay", "egress": "egress"})
    g.add_edge("troubleshooter", "egress")
    g.add_edge("pay", "egress")
    g.add_edge("egress", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["SessionState", "build_graph", "Command"]
