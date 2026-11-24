"""Wire contracts between the orchestrator and the agent services (D17, E4-11)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["director", "troubleshooter", "evaluator"]

# The support_agent (the "director" service) runs in one of four modes, chosen by the graph and
# passed as `context["mode"]`: it routes, or it is one of the three sub-node generators (E4-3/E4-4).
Route = Literal["NORMAL", "INSTRUCT", "BILLING", "TROUBLESHOOT"]
SupportMode = Literal["route", "normal", "instruct", "billing"]


class AgentRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    identity_verified: bool
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)  # e.g. proposed payment for the evaluator


class AgentResponse(BaseModel):
    reply: str = ""
    escalate: bool = False
    escalation_reason: str | None = None
    proposed_action: dict[str, Any] | None = (
        None  # e.g. {"type": "pay_bill", ...} awaiting approval
    )
    verdict: Literal["approve", "reject"] | None = None  # evaluator only
    route: Route | None = None  # support_agent in "route" mode only
    route_confidence: float | None = None
    audit: list[dict[str, Any]] = Field(default_factory=list)
