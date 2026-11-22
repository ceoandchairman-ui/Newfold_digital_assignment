"""The three agent behaviours. Safety decisions are deterministic code; the LLM only words replies."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from support_assistant.agents.harness import Harness
from support_assistant.agents.intent import IntentClassifier, KeywordIntentClassifier
from support_assistant.agents.llm import LLM
from support_assistant.agents.maf_tools import build_maf_tools
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.domain.escalation import EscalationReason, escalate
from support_assistant.domain.intent import CLOSING_REPLY, needs_retrieval
from support_assistant.guards import clean_untrusted, looks_like_injection, wrap_untrusted
from support_assistant.services.retrieval import Chunk

DIRECTOR_TOOLS = frozenset({"get_account_summary", "get_bills", "check_suspension_lift"})
TROUBLESHOOTER_TOOLS = frozenset(
    {
        "get_account_summary",
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


async def evaluator(req: AgentRequest, h: Harness) -> AgentResponse:
    """Independent check of a proposed payment. Fully deterministic."""
    h.require_verified(req)
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
