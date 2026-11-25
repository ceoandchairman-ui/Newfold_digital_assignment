"""Wire contracts between the orchestrator and the agent services (D17, E4-11)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["director", "troubleshooter", "evaluator"]

# The support_agent (the "director" service) runs in one of four modes, chosen by the graph and
# passed as `context["mode"]`: it routes, or it is one of the three sub-node generators (E4-3/E4-4).
Route = Literal["NORMAL", "INSTRUCT", "BILLING", "TROUBLESHOOT"]
SupportMode = Literal["route", "normal", "instruct", "billing"]


# --- retrieval service wire contracts -------------------------------------------------------------
# One store today. A customer store (memory, user documents) must not be added here until the
# service verifies who the customer is: today it only trusts the network policy (values.yaml).
Store = Literal["company_knowledge"]


class EvidenceRef(BaseModel):
    """Points at one chunk exactly as it was when it was used. Safe to keep in graph state: no text."""

    store: Store
    id: int
    content_hash: str = Field(max_length=128)


class Evidence(BaseModel):
    ref: EvidenceRef
    section_id: str
    heading_path: str
    content: str
    score: float = 0.0


class SearchRequest(BaseModel):
    store: Store
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=10)


class SearchResponse(BaseModel):
    results: list[Evidence] = Field(default_factory=list)


class FetchRequest(BaseModel):
    refs: list[EvidenceRef] = Field(min_length=1, max_length=10)


class FetchResponse(BaseModel):
    results: list[Evidence] = Field(default_factory=list)
    # Refs whose chunk no longer exists or whose content has changed since it was used.
    missing: list[EvidenceRef] = Field(default_factory=list)


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
    evidence: list[EvidenceRef] = Field(default_factory=list)  # chunks the model was really given
    audit: list[dict[str, Any]] = Field(default_factory=list)
