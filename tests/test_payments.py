"""Runs the real pay_bill / reconcile_payment logic against an in-memory ledger.

The repository functions are replaced with an in-memory implementation. This proves the state
machine and the double-charge protection, not the SQL (that needs a real Postgres).
"""

from contextlib import asynccontextmanager
from decimal import Decimal

import pytest

from support_assistant.errors import (
    ConflictError,
    DuplicateOperation,
    ExternalServiceUnavailable,
    NotFoundError,
    PaymentDeclined,
    PaymentOutcomeUnknown,
)
from support_assistant.services import payments
from support_assistant.services.gateway_sim import FaultMode, SimulatedGateway


class Ledger:
    def __init__(self) -> None:
        self.bills = {
            "B1": {"id": "B1", "customer": "C1", "amount_due": Decimal("50.00"), "status": "ISSUED"}
        }
        self.tx: dict[str, dict] = {}
        self._next = 1


class FakeConn:
    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    @asynccontextmanager
    async def connection(self):
        yield FakeConn()


@pytest.fixture
def ledger(monkeypatch) -> Ledger:
    led = Ledger()

    async def lock_bill(conn, customer_id, bill_id):
        b = led.bills.get(bill_id)
        if not b or b["customer"] != customer_id:
            raise NotFoundError(bill_id)
        return b

    async def get_bill(conn, customer_id, bill_id):
        return await lock_bill(conn, customer_id, bill_id)

    async def set_bill_status(conn, bill_id, status):
        led.bills[bill_id]["status"] = str(status)

    async def find_transaction(conn, key):
        return led.tx.get(key)

    async def insert_transaction(conn, customer_id, bill_id, amount, status, key):
        tx = {
            "id": led._next,
            "customer_id": customer_id,
            "bill_id": bill_id,
            "amount": amount,
            "status": status,
            "key": key,
        }
        led._next += 1
        led.tx[key] = tx
        return tx["id"]

    async def set_transaction_status(conn, tx_id, status):
        for tx in led.tx.values():
            if tx["id"] == tx_id:
                tx["status"] = status

    for fn in (
        lock_bill,
        get_bill,
        set_bill_status,
        find_transaction,
        insert_transaction,
        set_transaction_status,
    ):
        monkeypatch.setattr(payments.repo, fn.__name__, fn)
    return led


def _pay(gw, key="K1", amount="50.00"):
    return payments.pay_bill(
        FakePool(), gw, customer_id="C1", bill_id="B1", amount=Decimal(amount), idempotency_key=key
    )


async def test_happy_path_marks_paid(ledger):
    gw = SimulatedGateway()
    res = await _pay(gw)
    assert ledger.bills["B1"]["status"] == "PAID" and res.amount == Decimal("50.00")
    assert gw.charges_applied == 1


async def test_replay_with_same_key_never_charges_twice(ledger):
    gw = SimulatedGateway()
    await _pay(gw)
    again = await _pay(gw)
    assert again.replayed and gw.charges_applied == 1


async def test_decline_releases_the_bill(ledger):
    with pytest.raises(PaymentDeclined):
        await _pay(SimulatedGateway(FaultMode.DECLINE))
    assert ledger.bills["B1"]["status"] == "ISSUED" and ledger.tx["K1"]["status"] == "FAILED"


async def test_unreachable_gateway_releases_the_bill(ledger):
    with pytest.raises(ExternalServiceUnavailable):
        await _pay(SimulatedGateway(FaultMode.UNAVAILABLE))
    assert ledger.bills["B1"]["status"] == "ISSUED"


async def test_timeout_keeps_attempt_pending_and_blocks_retry(ledger):
    gw = SimulatedGateway(FaultMode.TIMEOUT)
    with pytest.raises(PaymentOutcomeUnknown):
        await _pay(gw)
    assert ledger.bills["B1"]["status"] == "AWAITING_PAYMENT"  # NOT released
    assert ledger.tx["K1"]["status"] == "PENDING"
    with pytest.raises(PaymentOutcomeUnknown):  # retry is refused, not re-charged
        await _pay(gw)
    assert gw.charges_applied == 1


async def test_reconcile_settles_when_gateway_recorded_the_charge(ledger):
    gw = SimulatedGateway(FaultMode.PARTIAL)
    with pytest.raises(PaymentOutcomeUnknown):
        await _pay(gw)
    res = await payments.reconcile_payment(FakePool(), gw, customer_id="C1", idempotency_key="K1")
    assert res is not None and ledger.bills["B1"]["status"] == "PAID"
    assert gw.charges_applied == 1


async def test_reconcile_releases_when_gateway_never_saw_it(ledger):
    gw = SimulatedGateway(FaultMode.TIMEOUT)
    with pytest.raises(PaymentOutcomeUnknown):
        await _pay(gw)
    gw.recorded.clear()  # the gateway has no record of the attempt
    assert (
        await payments.reconcile_payment(FakePool(), gw, customer_id="C1", idempotency_key="K1")
        is None
    )
    assert ledger.bills["B1"]["status"] == "ISSUED"


async def test_guards_overpayment_other_customer_and_reused_key(ledger):
    gw = SimulatedGateway()
    from support_assistant.errors import ValidationError

    with pytest.raises(ValidationError):
        await _pay(gw, amount="99.00")
    with pytest.raises(NotFoundError):  # another customer's bill is invisible
        await payments.pay_bill(
            FakePool(),
            gw,
            customer_id="C2",
            bill_id="B1",
            amount=Decimal("50.00"),
            idempotency_key="K9",
        )
    await _pay(gw, key="K1")
    with pytest.raises(ConflictError):  # same key, different customer
        await payments.pay_bill(
            FakePool(),
            gw,
            customer_id="C2",
            bill_id="B1",
            amount=Decimal("50.00"),
            idempotency_key="K1",
        )
    with pytest.raises(DuplicateOperation):  # already paid
        await _pay(gw, key="K2")
