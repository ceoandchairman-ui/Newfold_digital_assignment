"""Deterministic escalation triggers and the fixed transfer message (E3-7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

ESCALATION_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."


class EscalationReason(StrEnum):
    SIM_LOCKED = "SIM_LOCKED"
    TROUBLESHOOT_LOOP = "TROUBLESHOOT_LOOP"
    SUSPENSION_NOT_LIFTABLE = "SUSPENSION_NOT_LIFTABLE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNKNOWN_BILL_STATUS = "UNKNOWN_BILL_STATUS"
    EVAL_RETRIES_EXHAUSTED = "EVAL_RETRIES_EXHAUSTED"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"


@dataclass(frozen=True)
class Escalation:
    reason: EscalationReason
    message: str = ESCALATION_MESSAGE


def escalate(reason: EscalationReason) -> Escalation:
    return Escalation(reason=reason)


def troubleshoot_loop_exceeded(turns_taken: int, max_turns: int) -> bool:
    return turns_taken >= max_turns
