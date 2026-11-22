from datetime import date
from decimal import Decimal

import pytest

from support_assistant.domain.billing import (
    BillStatus,
    assert_transition,
    is_overdue,
    parse_status,
    validate_payment_amount,
)
from support_assistant.domain.escalation import (
    ESCALATION_MESSAGE,
    EscalationReason,
    escalate,
    troubleshoot_loop_exceeded,
)
from support_assistant.domain.suspension import lift_allowed
from support_assistant.errors import (
    GENERIC_MESSAGE,
    ConflictError,
    UnknownBillStatus,
    ValidationError,
    to_customer_message,
)

TODAY = date(2026, 11, 20)


# --- billing ---------------------------------------------------------------
def test_unknown_status_fails_closed():
    with pytest.raises(UnknownBillStatus):
        parse_status("PENDING_REVIEW")


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (BillStatus.ISSUED, BillStatus.AWAITING_PAYMENT),
        (BillStatus.AWAITING_PAYMENT, BillStatus.PAID),
        (BillStatus.AWAITING_PAYMENT, BillStatus.ISSUED),
    ],
)
def test_legal_transitions(current, new):
    assert_transition(current, new)


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (BillStatus.ISSUED, BillStatus.PAID),  # cannot skip AWAITING_PAYMENT
        (BillStatus.PAID, BillStatus.ISSUED),
        (BillStatus.PAID, BillStatus.AWAITING_PAYMENT),
    ],
)
def test_illegal_transitions(current, new):
    with pytest.raises(ConflictError):
        assert_transition(current, new)


def test_overdue_rule():
    assert is_overdue(date(2026, 11, 1), BillStatus.ISSUED, TODAY)
    assert not is_overdue(date(2026, 11, 1), BillStatus.PAID, TODAY)  # paid is never overdue
    assert not is_overdue(TODAY, BillStatus.ISSUED, TODAY)  # due today is not yet overdue
    assert not is_overdue(date(2026, 12, 1), BillStatus.ISSUED, TODAY)


def test_payment_amount_bounds():
    assert validate_payment_amount(Decimal("10.00"), Decimal("10.00")) == Decimal("10.00")
    with pytest.raises(ValidationError):
        validate_payment_amount(Decimal("0"), Decimal("10.00"))
    with pytest.raises(ValidationError):
        validate_payment_amount(Decimal("10.01"), Decimal("10.00"))


# --- suspension: all four combinations -------------------------------------
@pytest.mark.parametrize(
    ("overdue", "contract_end", "expected"),
    [
        (0, date(2027, 1, 1), True),  # paid, contract live
        (2, date(2027, 1, 1), False),  # overdue, contract live
        (0, date(2026, 1, 1), False),  # paid but contract expired overrides payment
        (2, date(2026, 1, 1), False),  # both bad
        (0, None, True),  # open-ended contract
    ],
)
def test_lift_allowed_truth_table(overdue, contract_end, expected):
    assert lift_allowed(overdue, contract_end, TODAY) is expected


# --- escalation ------------------------------------------------------------
def test_escalation_message_is_exact():
    assert ESCALATION_MESSAGE == "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
    e = escalate(EscalationReason.SIM_LOCKED)
    assert e.message == ESCALATION_MESSAGE
    assert e.reason is EscalationReason.SIM_LOCKED


def test_troubleshoot_loop_limit():
    assert not troubleshoot_loop_exceeded(2, 3)
    assert troubleshoot_loop_exceeded(3, 3)


# --- errors ----------------------------------------------------------------
def test_unexpected_exceptions_are_never_leaked():
    assert (
        to_customer_message(RuntimeError("psycopg.OperationalError: table bills"))
        == GENERIC_MESSAGE
    )
    assert "bills" not in to_customer_message(UnknownBillStatus("unknown bill status: 'X'"))
