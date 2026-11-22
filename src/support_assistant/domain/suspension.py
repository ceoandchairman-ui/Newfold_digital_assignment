"""Suspension lift eligibility (E3-6)."""

from __future__ import annotations

from datetime import date


def lift_allowed(overdue_bill_count: int, contract_end_date: date | None, today: date) -> bool:
    """A suspension may be lifted only if no bill is overdue AND the contract has not ended.

    The contract condition overrides payment: an expired contract blocks the lift even when
    every bill is paid. A missing end date is treated as an open-ended contract.
    """
    contract_expired = contract_end_date is not None and contract_end_date < today
    return overdue_bill_count == 0 and not contract_expired
