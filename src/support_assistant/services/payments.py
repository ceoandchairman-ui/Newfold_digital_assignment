"""Payment flow (E3-4, E3-5): reserve -> gateway -> settle, idempotent, row-locked, verified.

Phase 1 (transaction): lock the bill, check state/amount, OVERDUE -> AWAITING_PAYMENT, insert PENDING.
The gateway call happens outside any DB transaction so no row lock is held across the network.
Phase 2 (transaction): PENDING -> SUCCEEDED and bill -> PAID, or FAILED and bill back to OVERDUE.
A read-after-write check confirms the bill is PAID before we tell the customer.

Outcomes the gateway can report:
- approved / declined: definite. Settle or release.
- ExternalServiceUnavailable: the request never reached the gateway. Safe to release.
- PaymentOutcomeUnknown (timeout): the charge may have happened. We do NOT release the bill or
  mark the attempt failed. It stays PENDING / AWAITING_PAYMENT until `reconcile_payment` asks the
  gateway what happened, so a retry can never double-charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

import structlog

from support_assistant import repositories as repo
from support_assistant.db import Pool
from support_assistant.domain.billing import (
    BillStatus,
    assert_transition,
    is_payable,
    parse_status,
    validate_payment_amount,
)
from support_assistant.errors import (
    ConflictError,
    DuplicateOperation,
    ExternalServiceUnavailable,
    PaymentDeclined,
    PaymentOutcomeUnknown,
)
from support_assistant.logs import error_fields
from support_assistant.resilience import db_conflict_retry

log = structlog.get_logger()


class GatewayOutcome(StrEnum):
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"


class PaymentGateway(Protocol):
    async def charge(
        self, customer_id: str, amount: Decimal, idempotency_key: str
    ) -> GatewayOutcome:
        """Raise ExternalServiceUnavailable if the request never arrived, PaymentOutcomeUnknown
        if it may have. Calling again with the same key must return the original outcome."""
        ...

    async def lookup(self, idempotency_key: str) -> GatewayOutcome | None:
        """What the gateway recorded for this key, or None if it never received it."""
        ...


class StubGateway:
    """Approves everything unless `decline` is set. Used until the real gateway is chosen."""

    def __init__(self, decline: bool = False) -> None:
        self.decline = decline
        self._seen: dict[str, GatewayOutcome] = {}

    async def charge(
        self, customer_id: str, amount: Decimal, idempotency_key: str
    ) -> GatewayOutcome:
        outcome = GatewayOutcome.DECLINED if self.decline else GatewayOutcome.APPROVED
        return self._seen.setdefault(idempotency_key, outcome)

    async def lookup(self, idempotency_key: str) -> GatewayOutcome | None:
        return self._seen.get(idempotency_key)


@dataclass(frozen=True)
class PaymentResult:
    transaction_id: int
    bill_id: str
    amount: Decimal
    replayed: bool = False


async def pay_bill(
    pool: Pool,
    gateway: PaymentGateway,
    *,
    customer_id: str,
    bill_id: str,
    amount: Decimal,
    idempotency_key: str,
) -> PaymentResult:
    # ---- Phase 1: reserve ---------------------------------------------------
    async with pool.connection() as conn:
        async with conn.transaction():
            existing = await repo.find_transaction(conn, idempotency_key)
            if existing is not None:
                if existing["customer_id"] != customer_id or existing["bill_id"] != bill_id:
                    raise ConflictError("idempotency key reused for a different request")
                if existing["status"] == "SUCCEEDED":
                    log.info("payment_replayed", transaction_id=existing["id"])
                    return PaymentResult(existing["id"], bill_id, existing["amount"], replayed=True)
                if existing["status"] == "PENDING":
                    raise PaymentOutcomeUnknown("an earlier attempt is still unresolved")
                raise DuplicateOperation("payment already failed; use a new request")

            bill = await repo.lock_bill(conn, customer_id, bill_id)
            status = parse_status(bill["status"])
            if status is BillStatus.PAID:
                raise DuplicateOperation("bill already paid")
            if not is_payable(status):
                raise ConflictError("only an overdue bill can be paid")
            validate_payment_amount(amount, bill["amount_due"])
            assert_transition(status, BillStatus.AWAITING_PAYMENT)
            await repo.set_bill_status(conn, bill_id, BillStatus.AWAITING_PAYMENT)
            tx_id = await repo.insert_transaction(
                conn, customer_id, bill_id, amount, "PENDING", idempotency_key
            )

    log.info("payment_reserved", transaction_id=tx_id)
    # ---- Gateway (no locks held) ---------------------------------------------
    try:
        outcome = await gateway.charge(customer_id, amount, idempotency_key)
    except ExternalServiceUnavailable as exc:
        log.warning("payment_gateway_not_reached", transaction_id=tx_id, **error_fields(exc))
        await _release(pool, bill_id, tx_id)  # never reached the gateway: safe to give up
        raise
    except PaymentOutcomeUnknown as exc:
        log.error("payment_outcome_unknown", transaction_id=tx_id, **error_fields(exc))
        raise  # leave PENDING / AWAITING_PAYMENT for reconcile_payment

    log.info("payment_gateway_answered", transaction_id=tx_id, outcome=outcome.value)
    if outcome is GatewayOutcome.DECLINED:
        await _release(pool, bill_id, tx_id)
        raise PaymentDeclined("gateway declined")
    return await _settle(pool, customer_id, bill_id, tx_id, amount)


async def reconcile_payment(
    pool: Pool, gateway: PaymentGateway, *, customer_id: str, idempotency_key: str
) -> PaymentResult | None:
    """Resolve an unresolved (PENDING) attempt by asking the gateway what it recorded.

    Returns the settled result, or None if the gateway never saw it (attempt released).
    """
    async with pool.connection() as conn:
        tx = await repo.find_transaction(conn, idempotency_key)
    if tx is None or tx["customer_id"] != customer_id:
        raise ConflictError("no such payment attempt")
    if tx["status"] == "SUCCEEDED":
        return PaymentResult(tx["id"], tx["bill_id"], tx["amount"], replayed=True)
    if tx["status"] != "PENDING":
        return None

    outcome = await gateway.lookup(idempotency_key)
    log.info(
        "payment_reconciled",
        transaction_id=tx["id"],
        outcome=outcome.value if outcome else "never_received",
    )
    if outcome is GatewayOutcome.APPROVED:
        return await _settle(pool, customer_id, tx["bill_id"], tx["id"], tx["amount"])
    await _release(pool, tx["bill_id"], tx["id"])  # declined, or the gateway never received it
    return None


@db_conflict_retry
async def _commit_settlement(pool: Pool, customer_id: str, bill_id: str, tx_id: int) -> None:
    async with pool.connection() as conn:
        async with conn.transaction():
            await repo.lock_bill(conn, customer_id, bill_id)
            await repo.set_transaction_status(conn, tx_id, "SUCCEEDED")
            await repo.set_bill_status(conn, bill_id, BillStatus.PAID)


async def _settle(
    pool: Pool, customer_id: str, bill_id: str, tx_id: int, amount: Decimal
) -> PaymentResult:
    await _commit_settlement(pool, customer_id, bill_id, tx_id)
    log.info("payment_settled", transaction_id=tx_id)
    async with pool.connection() as conn:  # read-after-write verification
        verified = await repo.get_bill(conn, customer_id, bill_id)
    if parse_status(verified["status"]) is not BillStatus.PAID:
        raise ConflictError("bill not PAID after settlement")
    return PaymentResult(tx_id, bill_id, amount)


@db_conflict_retry
async def _release(pool: Pool, bill_id: str, tx_id: int) -> None:
    async with pool.connection() as conn:
        async with conn.transaction():
            await repo.set_transaction_status(conn, tx_id, "FAILED")
            await repo.set_bill_status(conn, bill_id, BillStatus.OVERDUE)
