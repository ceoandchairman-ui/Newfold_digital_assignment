"""In-memory world for journey tests: a ledger, a fake pool and the tool set agents call.

The repository functions are replaced by in-memory versions, so the REAL pay_bill state machine,
agent roles, harness, graph and guards run end to end. What this does not prove is the SQL.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

from support_assistant.errors import NotFoundError, SupportError, to_customer_message
from support_assistant.services import payments
from support_assistant.services.gateway_sim import SimulatedGateway


class FakeConn:
    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    @asynccontextmanager
    async def connection(self):
        yield FakeConn()


class World:
    def __init__(self) -> None:
        self.bills: dict[str, dict[str, Any]] = {
            "B1005": {
                "id": "B1005",
                "customer": "C1002",
                "amount_due": Decimal("42.50"),
                "due_date": "2026-11-01",
                "status": "ISSUED",
            },
            "B2001": {
                "id": "B2001",
                "customer": "C1001",
                "amount_due": Decimal("20.00"),
                "due_date": "2026-12-01",
                "status": "PAID",
            },
        }
        self.lines = {
            "C1001": [
                {
                    "id": "L1",
                    "phone_number": "+15550001",
                    "status": "ACTIVE",
                    "data_used_mb": 800,
                    "data_limit_mb": 5000,
                    "plan_name": "Basic",
                    "monthly_price": Decimal("20.00"),
                    "contract_end_date": "2027-06-01",
                }
            ],
            "C1002": [
                {
                    "id": "L2",
                    "phone_number": "+15550002",
                    "status": "ACTIVE",
                    "data_used_mb": 4900,
                    "data_limit_mb": 5000,
                    "plan_name": "Plus",
                    "monthly_price": Decimal("42.50"),
                    "contract_end_date": None,
                }
            ],
        }
        self.tx: dict[str, dict[str, Any]] = {}
        self._next = 1


def install_repo(monkeypatch: Any, w: World) -> None:
    async def lock_bill(conn, customer_id, bill_id):
        b = w.bills.get(bill_id)
        if not b or b["customer"] != customer_id:
            raise NotFoundError(bill_id)
        return b

    async def set_bill_status(conn, bill_id, status):
        w.bills[bill_id]["status"] = str(status)

    async def find_transaction(conn, key):
        return w.tx.get(key)

    async def insert_transaction(conn, customer_id, bill_id, amount, status, key):
        tx = {
            "id": w._next,
            "customer_id": customer_id,
            "bill_id": bill_id,
            "amount": amount,
            "status": status,
        }
        w._next += 1
        w.tx[key] = tx
        return tx["id"]

    async def set_transaction_status(conn, tx_id, status):
        for tx in w.tx.values():
            if tx["id"] == tx_id:
                tx["status"] = status

    for fn in (
        lock_bill,
        set_bill_status,
        find_transaction,
        insert_transaction,
        set_transaction_status,
    ):
        monkeypatch.setattr(payments.repo, fn.__name__, fn)
    monkeypatch.setattr(payments.repo, "get_bill", lock_bill)


def make_tools(w: World, gateway: SimulatedGateway) -> dict[str, Any]:
    """The tool callables an agent harness would get from the MCP servers."""

    def mine(customer_id: str) -> list[dict[str, Any]]:
        return [b for b in w.bills.values() if b["customer"] == customer_id]

    async def get_bills(customer_id: str) -> dict[str, Any]:
        rows = [
            {
                "id": b["id"],
                "amount_due": str(b["amount_due"]),
                "due_date": b["due_date"],
                "status": b["status"],
            }
            for b in mine(customer_id)
        ]
        return {"ok": True, "bills": rows}

    async def get_account_summary(customer_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "lines": [
                {k: str(v) if isinstance(v, Decimal) else v for k, v in ln.items()}
                for ln in w.lines.get(customer_id, [])
            ],
        }

    async def check_recent_payments(
        customer_id: str, bill_id: str, minutes: int = 10
    ) -> dict[str, Any]:
        n = sum(
            1 for t in w.tx.values() if t["customer_id"] == customer_id and t["bill_id"] == bill_id
        )
        return {"ok": True, "recent_payments": n}

    async def process_payment(
        customer_id: str, bill_id: str, amount: str, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        try:
            res = await payments.pay_bill(
                FakePool(),
                gateway,
                customer_id=customer_id,
                bill_id=bill_id,
                amount=Decimal(amount),
                idempotency_key=idempotency_key or "auto",
            )
            return {"ok": True, "transaction_id": res.transaction_id}
        except SupportError as exc:
            return {"ok": False, "error": exc.code, "message": to_customer_message(exc)}

    async def run_diagnostics(customer_id: str, device_id: str) -> dict[str, Any]:
        return {"ok": True, "signal": "good"}

    return {
        "get_bills": get_bills,
        "get_account_summary": get_account_summary,
        "check_recent_payments": check_recent_payments,
        "process_payment": process_payment,
        "run_diagnostics": run_diagnostics,
    }
