"""Bill status, guarded transitions and the single overdue predicate (E1-5, E3-3, E3-4)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from support_assistant.errors import ConflictError, UnknownBillStatus, ValidationError


class BillStatus(StrEnum):
    """The five states of the supplied policy, plus none of our own."""

    DRAFT = "DRAFT"
    ISSUED = "ISSUED"
    OVERDUE = "OVERDUE"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    PAID = "PAID"
    DISPUTED = "DISPUTED"


# Legal transitions. Only an OVERDUE bill can enter payment (policy: "check the bill status to make
# sure it is overdue"), so a payment that fails puts it back to OVERDUE.
_LEGAL: dict[BillStatus, frozenset[BillStatus]] = {
    BillStatus.DRAFT: frozenset({BillStatus.ISSUED}),
    BillStatus.ISSUED: frozenset({BillStatus.OVERDUE}),
    BillStatus.OVERDUE: frozenset({BillStatus.AWAITING_PAYMENT}),
    BillStatus.AWAITING_PAYMENT: frozenset({BillStatus.PAID, BillStatus.OVERDUE}),
    BillStatus.PAID: frozenset(),
    BillStatus.DISPUTED: frozenset(),
}


def is_payable(status: str) -> bool:
    """The payment rule, from the RECORDED status. A date never makes a bill payable here: the
    supplied data records B1005 as overdue although its due date is after the policy's current
    date, and the record wins. Draft, disputed, issued and paid bills cannot be paid."""
    return status == BillStatus.OVERDUE


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
    """Does this bill count as overdue, for example against lifting a suspension?

    A bill recorded OVERDUE always does. So does an unsettled one past its due date, including a
    disputed one: the policy does not say, so this fails closed. A paid or draft bill never does.
    `today` is injected (config.business_today)."""
    if status in (BillStatus.PAID, BillStatus.DRAFT):
        return False
    return status is BillStatus.OVERDUE or due_date < today


def validate_payment_amount(requested: Decimal, outstanding: Decimal) -> Decimal:
    """Amount must be positive and never above the outstanding balance."""
    if requested <= 0:
        raise ValidationError("payment amount must be positive")
    if requested > outstanding:
        raise ValidationError("payment amount exceeds the outstanding balance")
    return requested
