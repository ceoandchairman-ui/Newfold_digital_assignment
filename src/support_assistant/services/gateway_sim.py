"""Payment gateway simulator with fault injection (E3-5).

Modes:
  APPROVE      charge succeeds.
  DECLINE      charge is declined.
  UNAVAILABLE  request never reaches the gateway (clean failure, nothing recorded).
  TIMEOUT      request reaches the gateway and is recorded, but the reply is lost (outcome unknown).
  PARTIAL      same as TIMEOUT for an approval: money moved, the caller still sees a timeout.

The gateway dedupes on the idempotency key: a repeat returns the original outcome and never
charges twice.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from support_assistant.errors import ExternalServiceUnavailable, PaymentOutcomeUnknown
from support_assistant.services.payments import GatewayOutcome


class FaultMode(StrEnum):
    APPROVE = "APPROVE"
    DECLINE = "DECLINE"
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    PARTIAL = "PARTIAL"


class SimulatedGateway:
    def __init__(self, mode: FaultMode = FaultMode.APPROVE) -> None:
        self.mode = mode
        self.recorded: dict[str, GatewayOutcome] = {}
        self.charges_applied = 0  # real money movements, to assert "never double-charged"

    async def charge(
        self, customer_id: str, amount: Decimal, idempotency_key: str
    ) -> GatewayOutcome:
        if idempotency_key in self.recorded:  # dedupe: original outcome, no second charge
            return self.recorded[idempotency_key]
        if self.mode is FaultMode.UNAVAILABLE:
            raise ExternalServiceUnavailable("gateway unreachable")

        outcome = (
            GatewayOutcome.DECLINED if self.mode is FaultMode.DECLINE else GatewayOutcome.APPROVED
        )
        self.recorded[idempotency_key] = outcome
        if outcome is GatewayOutcome.APPROVED:
            self.charges_applied += 1
        if self.mode in (FaultMode.TIMEOUT, FaultMode.PARTIAL):
            raise PaymentOutcomeUnknown("gateway timed out after recording the charge")
        return outcome

    async def lookup(self, idempotency_key: str) -> GatewayOutcome | None:
        return self.recorded.get(idempotency_key)
