"""Thin LangGraph router (Fat Backend / Skinny Graph). Agents are called through AgentClient.

The edge contract of backlog E4-10 / scope doc section 3:

    auth_gate -> unverified: deny -> egress
              -> verified:   support_agent   (routes: NORMAL | INSTRUCT | BILLING | TROUBLESHOOT)
        NORMAL       -> normal_resp_gen  (simple questions, small model)               -> egress
        INSTRUCT     -> retrieve -> instruct_gen (reasoning model, policy + facts)     -> egress
        BILLING      -> billing_gen (task-executing model, tools; PROPOSES a payment)
                          -> proposal?  eval_agent -> approve (interrupt) -> mcp_client_routing_node
        TROUBLESHOOT -> retrieve -> troubleshoot_agent -> eval_agent -> egress

`retrieve` is the independent, non-MCP retriever (a LangChain retriever called from a node, D8): the
support_agent's instruct_gen and the troubleshooter read its chunks, MCP is only for tools.
`mcp_client_routing_node` is the graph's MCP hub: it executes the approved payment (the only write).
eval_agent: payment verdicts are plain code; a troubleshooting reply gets an invariant check first.
Evaluator reject -> explain and stop; outage or repeated rejects -> escalate (E3-7).
Pauses use interrupt(); state is checkpointed by the Redis saver (D9).
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
            if name == "mcp_client_routing_node":
                span.set_attribute("support.payment", True)  # tail sampling keeps all of these
            return out

    return node


class SessionState(TypedDict, total=False):
    schema_version: int
    audit: Annotated[list[dict[str, Any]], _audit_reducer]  # this turn's tool calls (E5-5)
    customer_id: str
    session_id: str
    identity_verified: bool
    route: str  # legacy lowercase hint ("billing" | "troubleshoot"); only a fallback
    route_hint: str | None  # the fat backend's cheap keyword guess; only a fallback
    intent: Literal["NORMAL", "INSTRUCT", "BILLING", "TROUBLESHOOT"]  # decided by support_agent
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

    def _fallback_intent(state: SessionState) -> str:
        """Used only when support_agent gives no route (outage, or an agent that predates routing)."""
        hint = str(state.get("route_hint") or "").upper()
        if hint in ("NORMAL", "INSTRUCT", "BILLING", "TROUBLESHOOT"):
            return hint
        legacy = {"billing": "BILLING", "troubleshoot": "TROUBLESHOOT"}
        return legacy.get(str(state.get("route") or ""), "NORMAL")  # E4-3: unsure -> normal

    async def support_agent(state: SessionState) -> dict[str, Any]:
        """The Director (E4-3): one validated routing decision. No wording, no tools."""
        try:
            res = await client.run("director", _req(state, mode="route"))
            intent = res.route or _fallback_intent(state)
        except ExternalServiceUnavailable:
            intent = _fallback_intent(state)  # routing must never take the turn down
        return {"intent": intent}

    async def normal_resp_gen(state: SessionState) -> dict[str, Any]:
        try:
            res = await client.run("director", _req(state, mode="normal"))
        except ExternalServiceUnavailable as exc:
            return {"reply": to_customer_message(exc)}
        return {"reply": res.reply, "audit": res.audit}

    async def instruct_gen(state: SessionState) -> dict[str, Any]:
        try:
            res = await client.run("director", _req(state, mode="instruct"))
        except ExternalServiceUnavailable as exc:
            return {"reply": to_customer_message(exc)}
        if res.escalate:
            return {
                "reply": res.reply,
                "escalated": True,
                "escalation_reason": res.escalation_reason,
            }
        return {"reply": res.reply, "audit": res.audit}

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

    async def billing_gen(state: SessionState) -> dict[str, Any]:
        """The task-executing sub-agent (E4-4): account and bill answers, and payment PROPOSALS."""
        try:
            res = await client.run("director", _req(state, mode="billing"))
        except ExternalServiceUnavailable as exc:
            return {"reply": to_customer_message(exc), "proposed_action": None}
        try:
            assert_turn_separation(res)  # reply OR action, never both (E4-8)
        except InvariantViolation:
            return {"proposed_action": None, **_escalate(EscalationReason.INVARIANT_VIOLATION)}
        return {"reply": res.reply, "proposed_action": res.proposed_action, "audit": res.audit}

    async def troubleshoot_agent(state: SessionState) -> dict[str, Any]:
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

    async def eval_agent(state: SessionState) -> dict[str, Any]:
        """The checkpoint (E4-6). A payment proposal gets the deterministic payment verdict; a
        troubleshooting reply gets an invariant check (for example, it must cite the manual)."""
        if not state.get("proposed_action"):
            return await _check_reply(state)
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

    async def _check_reply(state: SessionState) -> dict[str, Any]:
        if state.get("escalated") or not state.get("reply"):
            return {}  # an escalation text is fixed; nothing to check
        try:
            res = await client.run(
                "evaluator",
                _req(state, mode="reply_check", reply=state["reply"], intent=state.get("intent")),
            )
        except ExternalServiceUnavailable:
            return {}  # fail open here only: the egress filter still runs on this reply
        if res.verdict == "reject":
            return _escalate(EscalationReason.INVARIANT_VIOLATION)
        return {"audit": res.audit}

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

    async def mcp_client_routing_node(state: SessionState) -> dict[str, Any]:
        """The graph's MCP hub. It executes the one approved write (process_payment)."""
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

    def after_auth(state: SessionState) -> str:
        return "support_agent" if state.get("identity_verified") else "deny"

    def after_support_agent(state: SessionState) -> str:
        """Dynamic routing on the customer's intent (E4-3, E4-10)."""
        return {
            "NORMAL": "normal_resp_gen",
            "BILLING": "billing_gen",
            "INSTRUCT": "retrieve",
            "TROUBLESHOOT": "retrieve",
        }.get(str(state.get("intent")), "normal_resp_gen")

    def after_retrieve(state: SessionState) -> str:
        return "troubleshoot_agent" if state.get("intent") == "TROUBLESHOOT" else "instruct_gen"

    def after_billing_gen(state: SessionState) -> str:
        return "eval_agent" if state.get("proposed_action") else "egress"

    def after_troubleshoot(state: SessionState) -> str:
        return "egress" if state.get("escalated") else "eval_agent"

    def after_eval(state: SessionState) -> str:
        if state.get("escalated"):
            return "egress"
        return "approve" if state.get("proposed_action") else "egress"

    def after_approve(state: SessionState) -> str:
        return "mcp_client_routing_node" if state.get("proposed_action") else "egress"

    g = StateGraph(SessionState)
    for name, fn in (
        ("auth_gate", auth_gate),
        ("deny", deny),
        ("support_agent", support_agent),
        ("normal_resp_gen", normal_resp_gen),
        ("instruct_gen", instruct_gen),
        ("billing_gen", billing_gen),
        ("retrieve", retrieve),
        ("troubleshoot_agent", troubleshoot_agent),
        ("eval_agent", eval_agent),
        ("approve", approve),
        ("mcp_client_routing_node", mcp_client_routing_node),
        ("egress", egress),
    ):
        g.add_node(name, _traced(name, fn))
    g.add_edge(START, "auth_gate")
    g.add_conditional_edges(
        "auth_gate", after_auth, {"deny": "deny", "support_agent": "support_agent"}
    )
    g.add_conditional_edges(
        "support_agent",
        after_support_agent,
        {
            "normal_resp_gen": "normal_resp_gen",
            "billing_gen": "billing_gen",
            "retrieve": "retrieve",
        },
    )
    g.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"troubleshoot_agent": "troubleshoot_agent", "instruct_gen": "instruct_gen"},
    )
    g.add_conditional_edges(
        "billing_gen", after_billing_gen, {"eval_agent": "eval_agent", "egress": "egress"}
    )
    g.add_conditional_edges(
        "troubleshoot_agent", after_troubleshoot, {"eval_agent": "eval_agent", "egress": "egress"}
    )
    g.add_conditional_edges("eval_agent", after_eval, {"approve": "approve", "egress": "egress"})
    g.add_conditional_edges(
        "approve",
        after_approve,
        {"mcp_client_routing_node": "mcp_client_routing_node", "egress": "egress"},
    )
    g.add_edge("normal_resp_gen", "egress")
    g.add_edge("instruct_gen", "egress")
    g.add_edge("mcp_client_routing_node", "egress")
    g.add_edge("deny", "egress")
    g.add_edge("egress", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["SessionState", "build_graph", "Command"]
