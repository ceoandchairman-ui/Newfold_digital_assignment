"""The three agent behaviours. Safety decisions are deterministic code; the LLM only words replies."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from support_assistant.agents.harness import Harness
from support_assistant.agents.intent import (
    IntentClassifier,
    KeywordIntentClassifier,
    KeywordRouteClassifier,
    RouteClassifier,
)
from support_assistant.agents.llm import LLM
from support_assistant.agents.maf_tools import build_maf_tools
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.domain.escalation import EscalationReason, escalate
from support_assistant.domain.intent import CLOSING_REPLY, needs_retrieval
from support_assistant.guards import clean_untrusted, looks_like_injection, wrap_untrusted
from support_assistant.services.retrieval import Chunk

# The support_agent (all of its sub-agents) reads who the customer is, what they hold, what they
# owe and what support already knows. It has no write tool: payments are proposals only.
DIRECTOR_TOOLS = frozenset(
    {
        "get_account_summary",
        "get_bills",
        "check_suspension_lift",
        "get_customer_profile",
        "get_contract",
        "get_payment_methods",
        "get_case_history",
        "get_recent_interactions",
    }
)
TROUBLESHOOTER_TOOLS = frozenset(
    {
        "get_account_summary",
        "get_devices",
        "get_case_history",
        "get_recent_interactions",
        "run_diagnostics",
        "toggle_airplane_mode",
        "toggle_mobile_data",
        "toggle_roaming",
        "reseat_sim_card",
        "reset_apn_settings",
        "reboot_device",
        "can_send_mms",
    }
)
EVALUATOR_TOOLS = frozenset({"check_recent_payments", "get_bills"})  # read-only by construction
ALLOWED_TOOLS = {
    "director": DIRECTOR_TOOLS,
    "troubleshooter": TROUBLESHOOTER_TOOLS,
    "evaluator": EVALUATOR_TOOLS,
}


def _last_user(req: AgentRequest) -> str:
    for m in reversed(req.messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


Searcher = Callable[[str], Awaitable[list[Chunk]]]


def _account_facts(lines: list[dict[str, Any]]) -> str:
    out = []
    for ln in lines:
        limit = ln.get("data_limit_mb")
        allowance = (
            "unlimited data" if limit is None else f"{ln['data_used_mb']} of {limit} MB used"
        )
        out.append(
            f"line {ln['phone_number']} ({ln['status']}): plan {ln['plan_name']} "
            f"at {ln['monthly_price']}/month, {allowance}, contract ends {ln.get('contract_end_date') or 'n/a'}"
        )
    return "; ".join(out) or "No lines found."


TOOL_CALLING_INSTRUCTIONS = (
    "You are a careful telecom account and billing assistant. Use the tools to look up the "
    "customer's own data and answer only from what they return. You cannot make payments."
)


async def director(
    req: AgentRequest,
    h: Harness,
    llm: LLM,
    classifier: IntentClassifier | None = None,
    tool_calling: bool = False,
) -> AgentResponse:
    """Read-only account and billing help. Proposes a payment but never executes one.

    Intent comes from a validated classifier (E4-3), falling back to keywords.
    Turn separation (E4-8): a turn that proposes an action carries NO free text. The customer is
    asked to confirm by the approval step, so the model never narrates an action it does not own.
    """
    h.require_verified(req)
    text = clean_untrusted(_last_user(req))
    intent = await (classifier or KeywordIntentClassifier()).classify(text)

    # Model-driven tool calling (off by default; enable after the E4-0 end-to-end run). Reads only:
    # payment intents always take the deterministic proposal path below. Any failure falls back.
    if tool_calling and intent.kind != "payment" and hasattr(llm, "respond_with_tools"):
        try:
            tools = build_maf_tools(h, req, DIRECTOR_TOOLS)
            reply = await llm.respond_with_tools(TOOL_CALLING_INSTRUCTIONS, text, tools)
            return AgentResponse(reply=reply, audit=h.audit)
        except Exception:  # noqa: BLE001 - never lose the turn to the experimental path
            pass

    if intent.kind == "account":
        acct = await h.call(req, "get_account_summary")
        if not acct.get("ok"):
            return AgentResponse(reply=acct["message"], audit=h.audit)
        reply = await llm.respond(
            "You are a careful telecom account assistant.", _account_facts(acct["lines"]), text
        )
        return AgentResponse(reply=reply, audit=h.audit)

    bills = await h.call(req, "get_bills")
    if not bills.get("ok"):
        return AgentResponse(reply=bills["message"], audit=h.audit)
    unpaid = [b for b in bills["bills"] if b["status"] != "PAID"]

    if intent.kind == "payment" and unpaid:  # a proposal only: evaluator + approval decide
        target = unpaid[0]
        action = {"type": "pay_bill", "bill_id": target["id"], "amount": str(target["amount_due"])}
        return AgentResponse(proposed_action=action, audit=h.audit)

    facts = (
        "No unpaid bills."
        if not unpaid
        else "; ".join(
            f"bill {b['id']}: {b['amount_due']} due {b['due_date']} ({b['status']})" for b in unpaid
        )
    )
    reply = await llm.respond("You are a careful telecom billing assistant.", facts, text)
    return AgentResponse(reply=reply, audit=h.audit)


# billing_gen (E4-4): account and bill answers, tool-driven, and the payment PROPOSAL. It is the
# agentic, task-executing sub-agent. `director` is the same function under its older name.
billing_gen = director


async def support_route(
    req: AgentRequest, h: Harness, classifier: RouteClassifier | None = None
) -> AgentResponse:
    """support_agent in route mode (E4-3): one validated classification, no tools, no wording.

    An open workflow step is not re-classified: the graph checks it before calling this.
    """
    h.require_verified(req)
    text = clean_untrusted(_last_user(req))
    decision = await (classifier or KeywordRouteClassifier()).classify(text)
    return AgentResponse(route=decision.route, route_confidence=decision.confidence, audit=h.audit)


NORMAL_INSTRUCTIONS = (
    "You are a warm, concise telecom support assistant. This reply has no account data: do not "
    "state any plan, bill, balance or device fact. For anything about the customer's account, say "
    "you can look it up and ask what they would like to know."
)


async def normal_resp_gen(req: AgentRequest, h: Harness, llm: LLM) -> AgentResponse:
    """The simple-question sub-agent (E4-4): high-empathy dialogue on the small model, no tools."""
    h.require_verified(req)
    text = clean_untrusted(_last_user(req))
    if not needs_retrieval(text):  # "thanks", "ok": no model call at all
        return AgentResponse(reply=CLOSING_REPLY, audit=h.audit)
    return AgentResponse(reply=await llm.respond(NORMAL_INSTRUCTIONS, "", text), audit=h.audit)


INSTRUCT_INSTRUCTIONS = (
    "You are a careful telecom support assistant. Reason step by step over the verified account "
    "facts and the reference policy sections, then give a short, clear answer. Use only those "
    "facts and sections. If they do not cover the question, say so and offer a human agent. "
    "Never promise a refund, credit or exception."
)


def _bill_facts(bills: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"bill {b['id']}: {b['amount_due']} due {b['due_date']} ({b['status']})" for b in bills
    )


async def instruct_gen(req: AgentRequest, h: Harness, llm: LLM) -> AgentResponse:
    """The reasoning sub-agent (E4-4): policy and "why" questions over the customer's own facts.

    Facts come from read-only tools through the harness. Policy text comes from the independent
    retriever (the graph's `retrieve` node), never from an MCP tool, and is screened and wrapped as
    untrusted. It cannot propose or perform an action.
    """
    h.require_verified(req)
    text = clean_untrusted(_last_user(req))
    parts: list[str] = []
    acct = await h.call(req, "get_account_summary")
    if acct.get("ok"):
        parts.append(_account_facts(acct["lines"]))
    bills = await h.call(req, "get_bills")
    if bills.get("ok") and bills["bills"]:
        parts.append(_bill_facts(bills["bills"]))
    supplied = [
        (str(c.get("section_id", "")), str(c.get("heading_path", "")), str(c["content"]))
        for c in req.context.get("kb_chunks") or []
    ]
    used = [p for p in supplied if not looks_like_injection(p[2])][:3]
    parts += [wrap_untrusted(f"policy {sec}", body) for sec, _, body in used]
    if not parts:
        e = escalate(EscalationReason.OUT_OF_SCOPE)
        return AgentResponse(
            reply=e.message, escalate=True, escalation_reason=e.reason.value, audit=h.audit
        )
    reply = await llm.respond(INSTRUCT_INSTRUCTIONS, "\n".join(parts), text)
    if used:
        reply = f"{reply}\n\n{_citation(used)}"
    return AgentResponse(reply=reply, audit=h.audit)


async def troubleshooter(
    req: AgentRequest, h: Harness, llm: LLM, search: Searcher
) -> AgentResponse:
    h.require_verified(req)
    text = clean_untrusted(_last_user(req))
    # Lazy retrieval (E4-5): "thanks" or "that worked" needs no manual search and no model call.
    if not needs_retrieval(text):
        return AgentResponse(reply=CLOSING_REPLY, audit=h.audit)
    # Prefer the hits the graph's retrieve node already fetched; search ourselves only as a fallback.
    supplied = req.context.get("kb_chunks")
    if supplied:
        passages = [
            (str(c["section_id"]), str(c.get("heading_path", "")), str(c["content"]))
            for c in supplied
        ]
    else:
        found = await search(str(req.context.get("search_query") or text))
        passages = [(c.section_id, c.heading_path, c.content) for c in found]
    # Retrieved text is untrusted: drop anything that tries to give the model instructions.
    passages = [p for p in passages if not looks_like_injection(p[2])]
    if not passages:
        e = escalate(EscalationReason.OUT_OF_SCOPE)
        return AgentResponse(
            reply=e.message, escalate=True, escalation_reason=e.reason.value, audit=h.audit
        )
    used = passages[:3]
    facts = "\n".join(wrap_untrusted(f"section {sec}", body) for sec, _, body in used)
    reply = await llm.respond(
        "You are a telecom troubleshooting assistant. Answer only from the reference sections.",
        facts,
        text,
    )
    return AgentResponse(reply=f"{reply}\n\n{_citation(used)}", audit=h.audit)


def _citation(passages: list[tuple[str, str, str]]) -> str:
    """Manual citations appended to every troubleshooting answer (E5-4)."""
    seen: list[str] = []
    for sec, heading, _ in passages:
        label = heading or f"section {sec}"
        if label not in seen:
            seen.append(label)
    return "Source: " + "; ".join(seen[:2])


def reply_invariants_hold(reply: str, intent: str | None) -> bool:
    """Deterministic checks on a finished reply (E5-4): a troubleshooting answer must cite the
    manual, unless it is the fixed closing text. Anything else only has to be non-empty."""
    if not reply.strip():
        return False
    if intent == "TROUBLESHOOT" and reply != CLOSING_REPLY:
        return "Source:" in reply
    return True


async def evaluator(req: AgentRequest, h: Harness) -> AgentResponse:
    """Independent check of a proposed payment, or of a finished reply. Fully deterministic."""
    h.require_verified(req)
    if req.context.get("mode") == "reply_check":
        ok = reply_invariants_hold(str(req.context.get("reply", "")), req.context.get("intent"))
        return AgentResponse(
            verdict="approve" if ok else "reject",
            reply="" if ok else "Reply failed an invariant.",
            audit=h.audit,
        )
    action = req.context.get("proposed_action") or {}
    bill_id, amount = action.get("bill_id"), action.get("amount")
    if action.get("type") != "pay_bill" or not bill_id or amount is None:
        return AgentResponse(verdict="reject", reply="Unsupported action.", audit=h.audit)

    bills = await h.call(req, "get_bills")
    bill = next((b for b in bills.get("bills", []) if b["id"] == bill_id), None)
    if bill is None or bill["status"] == "PAID" or Decimal(amount) > Decimal(bill["amount_due"]):
        return AgentResponse(
            verdict="reject", reply="Payment does not match the bill.", audit=h.audit
        )

    recent = await h.call(req, "check_recent_payments", bill_id=bill_id, minutes=10)
    if not recent.get("ok") or recent.get("recent_payments", 1) > 0:
        return AgentResponse(
            verdict="reject", reply="A recent payment exists for this bill.", audit=h.audit
        )
    return AgentResponse(verdict="approve", audit=h.audit)
