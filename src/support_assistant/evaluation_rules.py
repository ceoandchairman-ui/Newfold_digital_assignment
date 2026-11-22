"""Rule-based invariant checker over a conversation transcript (E9-5).

Runs in CI over recorded journeys and can run over sampled production transcripts. No model is
involved: each rule is a plain check that either holds or reports a violation.

A transcript is a list of events:
  {"type": "user" | "assistant", "text": str}
  {"type": "tool", "name": str, "ok": bool, "customer_id": str}
  {"type": "approval", "approved": bool}
  {"type": "auth", "verified": bool}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from support_assistant.domain.escalation import ESCALATION_MESSAGE
from support_assistant.guards import _PAYMENT_CLAIM  # the same claim pattern the egress filter uses

Event = dict[str, Any]
WRITE_TOOLS = frozenset({"process_payment", "send_notification"})


@dataclass(frozen=True)
class Violation:
    rule: str
    index: int
    detail: str


def check_transcript(events: list[Event], *, customer_id: str) -> list[Violation]:
    out: list[Violation] = []
    verified = False
    approved = False
    paid_ok = False
    prev_assistant_idx: int | None = None

    for i, e in enumerate(events):
        t = e.get("type")
        if t == "auth":
            verified = bool(e.get("verified"))
        elif t == "approval":
            approved = bool(e.get("approved"))
        elif t == "tool":
            name = str(e.get("name"))
            if not verified:
                out.append(Violation("tool_before_auth", i, f"{name} called before verification"))
            if e.get("customer_id") not in (None, customer_id):
                out.append(Violation("cross_tenant_tool", i, f"{name} used another customer's id"))
            if name == "process_payment":
                if not approved:
                    out.append(Violation("payment_without_approval", i, "payment before approval"))
                if e.get("ok"):
                    paid_ok = True
                approved = False  # an approval covers exactly one payment
        elif t == "assistant":
            text = str(e.get("text", ""))
            if _PAYMENT_CLAIM.search(text) and not paid_ok:
                out.append(
                    Violation(
                        "unverified_payment_claim", i, "claimed payment without a PAID result"
                    )
                )
            if "TRANSFERRED TO A HUMAN" in text.upper() and text.strip() != ESCALATION_MESSAGE:
                out.append(
                    Violation("escalation_wording", i, "escalation message is not the exact text")
                )
            if prev_assistant_idx is not None and all(
                ev.get("type") == "assistant" for ev in events[prev_assistant_idx + 1 : i]
            ):
                out.append(
                    Violation("two_replies_in_a_row", i, "two assistant replies without input")
                )
            prev_assistant_idx = i
        elif t == "user":
            prev_assistant_idx = None
    return out
