"""Thin LangGraph router (Fat Backend / Skinny Graph). Agents are called through AgentClient.

The main graph has three AGENT nodes, each a small subgraph of its own (backlog E4-10, scope doc
section 3), plus the plain-code gates around them:

    auth_gate -> unverified: deny -> egress
              -> verified:   support_agent
    support_agent   (subgraph)  route -> normal_resp_gen                    simple questions
                                       -> instruct_gen                      reasoning questions
                                       -> billing_gen                       task questions, tools
                                       -> (TROUBLESHOOT: leaves the subgraph)
        TROUBLESHOOT       -> troubleshoot_agent  (subgraph)  diagnose
        payment proposed   -> eval_agent
        otherwise          -> egress
    troubleshoot_agent -> eval_agent
    eval_agent: a payment proposal gets the plain-code verdict; a troubleshooting reply gets the rule
        check and then the LLM judge -> approve (interrupt) -> pay -> egress, or egress
    egress: the last filter on every path.

The graph does no retrieval. The agents call the standalone retrieval service themselves and send
back REFERENCES to the chunks they used (never the text), which the eval_agent uses to fetch the same
evidence. MCP is only for tools, and tools run inside the agent services behind their harness. `pay`
is the one graph-side write: it executes the approved payment. Evaluator reject -> explain and stop;
outage or repeated rejects -> escalate (E3-7). Pauses use interrupt(); state is checkpointed by the
Redis saver (D9).
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

import structlog
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
from support_assistant.errors import (
    AuthenticationError,
    ExternalServiceUnavailable,
    InvariantViolation,
    to_customer_message,
)
from support_assistant.guards import assert_turn_separation, sanitize_reply
from support_assistant.logs import error_fields
from support_assistant.observability import PAYMENTS, tracer
from support_assistant.orchestrator.client import AgentClient

log = structlog.get_logger()

STATE_SCHEMA_VERSION = 2  # bump with a migration path when the state shape changes (E7-3)


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
        started = time.perf_counter()
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
                log.info("graph_node_paused", node=name, reason="awaiting_approval")
                raise
            except Exception as exc:
                log.error("graph_node_failed", node=name, **error_fields(exc))
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            if out.get("escalated"):
                span.set_attribute("support.escalated", True)
                span.set_attribute("support.escalation_reason", str(out.get("escalation_reason")))
            if name == "pay":
                span.set_attribute("support.payment", True)  # tail sampling keeps all of these
            log.info(
                "graph_node_completed",
                node=name,
                duration_ms=round((time.perf_counter() - started) * 1000),
                intent=out.get("intent"),
                escalated=bool(out.get("escalated")),
                escalation_reason=out.get("escalation_reason"),
                proposed_payment=bool(out.get("proposed_action")),
            )
            return out

    return node


def _logged(branch: Callable[[SessionState], str]) -> Callable[[SessionState], str]:
    """Records which way each conditional edge went."""

    @functools.wraps(branch)
    def choose(state: SessionState) -> str:
        target = branch(state)
        log.info("graph_branch", branch=branch.__name__, next=target)
        return target

    return choose


class SessionState(TypedDict, total=False):
    schema_version: int
    audit: Annotated[list[dict[str, Any]], _audit_reducer]  # this turn's tool calls (E5-5)
    customer_id: str
    session_id: str
    identity_verified: bool
    route_hint: str | None  # the fat backend's cheap keyword guess; only a fallback
    intent: Literal["NORMAL", "INSTRUCT", "BILLING", "TROUBLESHOOT"]  # decided by support_agent
    messages: list[dict[str, Any]]  # plain dicts; the backend owns history (memory.py)
    turn_id: str | None  # for log correlation across services
    search_query: str | None  # rewritten retrieval query from the backend (chains.py)
    evidence_refs: list[dict[str, Any]]  # this turn's chunk references, no text (E7-3)
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
                **({"turn_id": state["turn_id"]} if state.get("turn_id") else {}),
            },
        )

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
        if reply != state.get("reply", ""):
            log.warning("egress_reply_replaced", intent=state.get("intent"))
        return {"reply": reply}

    async def deny(state: SessionState) -> dict[str, Any]:
        return {"reply": AuthenticationError("unverified").customer_message}

    def _fallback_intent(state: SessionState) -> str:
        """Used only when support_agent gives no route (an outage). The backend's keyword hint, or
        NORMAL when unsure (E4-3)."""
        hint = str(state.get("route_hint") or "").upper()
        return hint if hint in ("NORMAL", "INSTRUCT", "BILLING", "TROUBLESHOOT") else "NORMAL"

    async def route(state: SessionState) -> dict[str, Any]:
        """The support_agent's router (E4-3): one validated decision. No wording, no tools."""
        try:
            res = await client.run("director", _req(state, mode="route"))
            intent = res.route or _fallback_intent(state)
        except ExternalServiceUnavailable as exc:
            log.warning(
                "agent_unavailable", node="route", action="keyword_route", **error_fields(exc)
            )
            intent = _fallback_intent(state)  # routing must never take the turn down
        return {"intent": intent}

    async def normal_resp_gen(state: SessionState) -> dict[str, Any]:
        try:
            res = await client.run("director", _req(state, mode="normal"))
        except ExternalServiceUnavailable as exc:
            log.warning("agent_unavailable", node="normal", action="apologise", **error_fields(exc))
            return {"reply": to_customer_message(exc)}
        return {"reply": res.reply, "audit": res.audit}

    async def instruct_gen(state: SessionState) -> dict[str, Any]:
        try:
            res = await client.run("director", _req(state, mode="instruct"))
        except ExternalServiceUnavailable as exc:
            log.warning(
                "agent_unavailable", node="instruct", action="apologise", **error_fields(exc)
            )
            return {"reply": to_customer_message(exc)}
        if res.escalate:
            return {
                "reply": res.reply,
                "escalated": True,
                "escalation_reason": res.escalation_reason,
            }
        return {"reply": res.reply, "audit": res.audit}

    def _escalate(reason: EscalationReason) -> dict[str, Any]:
        return {"reply": ESCALATION_MESSAGE, "escalated": True, "escalation_reason": reason.value}

    async def billing_gen(state: SessionState) -> dict[str, Any]:
        """The task-executing sub-agent (E4-4): account and bill answers, and payment PROPOSALS."""
        try:
            res = await client.run("director", _req(state, mode="billing"))
        except ExternalServiceUnavailable as exc:
            log.warning(
                "agent_unavailable", node="billing", action="apologise", **error_fields(exc)
            )
            return {"reply": to_customer_message(exc), "proposed_action": None}
        try:
            assert_turn_separation(res)  # reply OR action, never both (E4-8)
        except InvariantViolation as exc:
            log.error("invariant_violated", node="billing_gen", **error_fields(exc))
            return {"proposed_action": None, **_escalate(EscalationReason.INVARIANT_VIOLATION)}
        action = res.proposed_action
        if action:
            # A new id for each proposal, saved in the checkpoint BEFORE the approval pause: a
            # replay of `pay` reuses it, while a later proposal for the same bill gets a new one.
            action = {**action, "proposal_id": uuid.uuid4().hex}
        return {"reply": res.reply, "proposed_action": action, "audit": res.audit}

    async def diagnose(state: SessionState) -> dict[str, Any]:
        """The troubleshoot_agent's step: answer from the manual, 3 turns at most (E4-5)."""
        # Every way out sets evidence_refs, so an earlier turn's references can never be reused.
        turns = state.get("troubleshoot_turns", 0) + 1
        if troubleshoot_loop_exceeded(turns, max_troubleshoot_turns + 1):
            return {"evidence_refs": [], **_escalate(EscalationReason.TROUBLESHOOT_LOOP)}
        try:
            res = await client.run("troubleshooter", _req(state))
        except ExternalServiceUnavailable as exc:
            log.warning(
                "agent_unavailable", node="diagnose", action="apologise", **error_fields(exc)
            )
            return {
                "reply": to_customer_message(exc),
                "troubleshoot_turns": turns,
                "evidence_refs": [],
            }
        if res.escalate:
            return {
                "reply": res.reply,
                "escalated": True,
                "escalation_reason": res.escalation_reason,
                "evidence_refs": [],
            }
        return {
            "reply": res.reply,
            "troubleshoot_turns": turns,
            "evidence_refs": [r.model_dump() for r in res.evidence],
            "audit": res.audit,
        }

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
        except ExternalServiceUnavailable as exc:
            log.error(
                "agent_unavailable", node="eval_agent", action="escalate", **error_fields(exc)
            )
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
                _req(
                    state,
                    mode="reply_check",
                    reply=state["reply"],
                    intent=state.get("intent"),
                    evidence_refs=state.get("evidence_refs", []),
                ),
            )
        except ExternalServiceUnavailable as exc:
            log.error(
                "agent_unavailable", node="reply_check", action="fail_open", **error_fields(exc)
            )
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
            log.info("payment_approval", approved=True)
            return {}
        log.info("payment_approval", approved=False)
        return {"proposed_action": None, "reply": "Okay, I have not made any payment."}

    async def pay_node(state: SessionState) -> dict[str, Any]:
        """Execute the approved payment: the one write the graph itself performs."""
        action = state.get("proposed_action")
        if not action:
            return {}
        # Stable across graph replays. A run paused before proposal ids existed keeps its old key.
        key = ":".join(
            str(part)
            for part in (state["session_id"], action["bill_id"], action.get("proposal_id"))
            if part
        )
        result = await pay(state["customer_id"], action["bill_id"], action["amount"], key)
        log.info(
            "payment_tool_result",
            ok=bool(result.get("ok")),
            **({} if result.get("ok") else {"error.code": str(result.get("error", "failed"))}),
        )
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

    def after_route(state: SessionState) -> str:
        """Dynamic routing on the customer's intent (E4-3). TROUBLESHOOT leaves this subgraph."""
        return {
            "NORMAL": "normal_resp_gen",
            "BILLING": "billing_gen",
            "INSTRUCT": "instruct_gen",
        }.get(str(state.get("intent")), "done")

    def after_support_agent(state: SessionState) -> str:
        if state.get("intent") == "TROUBLESHOOT" and not state.get("escalated"):
            return "troubleshoot_agent"
        return "eval_agent" if state.get("proposed_action") else "egress"

    def after_troubleshoot(state: SessionState) -> str:
        return "egress" if state.get("escalated") else "eval_agent"

    def after_eval(state: SessionState) -> str:
        if state.get("escalated"):
            return "egress"
        return "approve" if state.get("proposed_action") else "egress"

    def after_approve(state: SessionState) -> str:
        return "pay" if state.get("proposed_action") else "egress"

    # --- the support_agent: one agent made of several nodes (E4-3, E4-4) -------------------------
    support = StateGraph(SessionState)
    for name, fn in (
        ("route", route),
        ("normal_resp_gen", normal_resp_gen),
        ("instruct_gen", instruct_gen),
        ("billing_gen", billing_gen),
    ):
        support.add_node(name, _traced(name, fn))
    support.add_edge(START, "route")
    support.add_conditional_edges(
        "route",
        _logged(after_route),
        {
            "normal_resp_gen": "normal_resp_gen",
            "billing_gen": "billing_gen",
            "instruct_gen": "instruct_gen",
            "done": END,
        },
    )
    for name in ("normal_resp_gen", "instruct_gen", "billing_gen"):
        support.add_edge(name, END)

    # --- the troubleshoot_agent: a bounded diagnosis (E4-5) ---------------------------------------
    trouble = StateGraph(SessionState)
    trouble.add_node("diagnose", _traced("diagnose", diagnose))
    trouble.add_edge(START, "diagnose")
    trouble.add_edge("diagnose", END)

    # --- the main graph: three agents and the code around them -----------------------------------
    g = StateGraph(SessionState)
    for name, fn in (
        ("auth_gate", auth_gate),
        ("deny", deny),
        ("eval_agent", eval_agent),
        ("approve", approve),
        ("pay", pay_node),
        ("egress", egress),
    ):
        g.add_node(name, _traced(name, fn))
    g.add_node("support_agent", support.compile())  # subgraphs share the state and the checkpointer
    g.add_node("troubleshoot_agent", trouble.compile())
    g.add_edge(START, "auth_gate")
    g.add_conditional_edges(
        "auth_gate", _logged(after_auth), {"deny": "deny", "support_agent": "support_agent"}
    )
    g.add_conditional_edges(
        "support_agent",
        _logged(after_support_agent),
        {
            "troubleshoot_agent": "troubleshoot_agent",
            "eval_agent": "eval_agent",
            "egress": "egress",
        },
    )
    g.add_conditional_edges(
        "troubleshoot_agent",
        _logged(after_troubleshoot),
        {"eval_agent": "eval_agent", "egress": "egress"},
    )
    g.add_conditional_edges(
        "eval_agent", _logged(after_eval), {"approve": "approve", "egress": "egress"}
    )
    g.add_conditional_edges("approve", _logged(after_approve), {"pay": "pay", "egress": "egress"})
    g.add_edge("pay", "egress")
    g.add_edge("deny", "egress")
    g.add_edge("egress", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["SessionState", "build_graph", "Command"]
