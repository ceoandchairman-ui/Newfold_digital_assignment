"""Bill status, guarded transitions and the single overdue predicate (E1-5, E3-3, E3-4)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from support_assistant.errors import ConflictError, UnknownBillStatus, ValidationError


class BillStatus(StrEnum):
    ISSUED = "ISSUED"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    PAID = "PAID"


# Legal transitions. A failed payment returns the bill to ISSUED.
_LEGAL: dict[BillStatus, frozenset[BillStatus]] = {
    BillStatus.ISSUED: frozenset({BillStatus.AWAITING_PAYMENT}),
    BillStatus.AWAITING_PAYMENT: frozenset({BillStatus.PAID, BillStatus.ISSUED}),
    BillStatus.PAID: frozenset(),
}


def parse_status(raw: str) -> BillStatus:
    """Fail closed: an unknown status raises so the caller can escalate."""
    try:
        return BillStatus(raw)
    except ValueError as exc:
        raise UnknownBillStatus(f"unknown bill status: {raw!r}") from exc


def assert_transition(current: BillStatus, new: BillStatus) -> None:
    if new not in _LEGAL[current]:
        raise ConflictError(f"illegal bill transition {current} -> {new}")


def is_overdue(due_date: date, status: BillStatus, today: date) -> bool:
    """The only overdue rule in the system. `today` is injected so tests control the clock."""
    return status is not BillStatus.PAID and due_date < today


def validate_payment_amount(requested: Decimal, outstanding: Decimal) -> Decimal:
    """Amount must be positive and never above the outstanding balance."""
    if requested <= 0:
        raise ValidationError("payment amount must be positive")
    if requested > outstanding:
        raise ValidationError("payment amount exceeds the outstanding balance")
    return requested
