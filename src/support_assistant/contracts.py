"""Wire contracts between the orchestrator and the agent services (D17, E4-11)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["director", "troubleshooter", "evaluator"]


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
    audit: list[dict[str, Any]] = Field(default_factory=list)
