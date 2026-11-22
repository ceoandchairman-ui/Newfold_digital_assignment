"""Seed loader (E1-2): load reference and demo data from TOML, idempotently.

    python -m support_assistant.seed seed/example_seed.toml

The original seed file (db.toml) is lost (D1), so this loader defines the format it accepts
(see seed/example_seed.toml). When the real file is recovered, map it onto these tables.

Every row is validated before anything is written, then all rows are upserted in foreign-key order
in ONE transaction: a bad file changes nothing, and loading the same file twice changes nothing.
"""

from __future__ import annotations

import asyncio
import sys
import tomllib
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from support_assistant.db import Pool


class Customer(BaseModel):
    id: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    date_of_birth: date
    phone: str = Field(min_length=3)
    email: str | None = None


class Plan(BaseModel):
    id: str = Field(min_length=1)
    name: str
    monthly_price: Decimal = Field(ge=0)
    data_limit_mb: int | None = Field(None, ge=0)


class PaymentMethod(BaseModel):
    id: str
    customer_id: str
    kind: str
    last4: str | None = Field(None, min_length=4, max_length=4)


class Line(BaseModel):
    id: str
    customer_id: str
    plan_id: str
    phone_number: str
    status: str
    data_used_mb: int = Field(0, ge=0)
    contract_end_date: date | None = None


class Device(BaseModel):
    id: str
    line_id: str
    model: str
    imei: str | None = None


class Bill(BaseModel):
    id: str
    customer_id: str
    amount_due: Decimal = Field(ge=0)
    due_date: date
    status: Literal["ISSUED", "AWAITING_PAYMENT", "PAID"]


class BillLineItem(BaseModel):
    bill_id: str
    description: str
    amount: Decimal


class SeedError(Exception):
    """The seed file is invalid. Nothing was written."""


class Seed(BaseModel):
    customers: list[Customer] = []
    plans: list[Plan] = []
    payment_methods: list[PaymentMethod] = []
    lines: list[Line] = []
    devices: list[Device] = []
    bills: list[Bill] = []
    bill_line_items: list[BillLineItem] = []

    def check_references(self) -> None:
        """Every foreign key must point at a row in the same file."""
        customers = {c.id for c in self.customers}
        plans = {p.id for p in self.plans}
        lines = {ln.id for ln in self.lines}
        bills = {b.id for b in self.bills}
        problems = [
            *(
                f"payment_method {m.id}: unknown customer {m.customer_id}"
                for m in self.payment_methods
                if m.customer_id not in customers
            ),
            *(
                f"line {ln.id}: unknown customer {ln.customer_id}"
                for ln in self.lines
                if ln.customer_id not in customers
            ),
            *(
                f"line {ln.id}: unknown plan {ln.plan_id}"
                for ln in self.lines
                if ln.plan_id not in plans
            ),
            *(
                f"device {d.id}: unknown line {d.line_id}"
                for d in self.devices
                if d.line_id not in lines
            ),
            *(
                f"bill {b.id}: unknown customer {b.customer_id}"
                for b in self.bills
                if b.customer_id not in customers
            ),
            *(
                f"line item: unknown bill {i.bill_id}"
                for i in self.bill_line_items
                if i.bill_id not in bills
            ),
        ]
        awaiting = [b.customer_id for b in self.bills if b.status == "AWAITING_PAYMENT"]
        problems += [
            f"customer {c}: more than one AWAITING_PAYMENT bill"
            for c in sorted({c for c in awaiting if awaiting.count(c) > 1})
        ]
        if problems:
            raise SeedError("; ".join(problems))


def parse_seed(text: str) -> Seed:
    try:
        seed = Seed.model_validate(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise SeedError(str(exc)) from exc
    seed.check_references()
    return seed


_UPSERTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("customers", "customers", ("id", "full_name", "date_of_birth", "phone", "email")),
    ("plans", "plans", ("id", "name", "monthly_price", "data_limit_mb")),
    ("payment_methods", "payment_methods", ("id", "customer_id", "kind", "last4")),
    (
        "lines",
        "lines",
        (
            "id",
            "customer_id",
            "plan_id",
            "phone_number",
            "status",
            "data_used_mb",
            "contract_end_date",
        ),
    ),
    ("devices", "devices", ("id", "line_id", "model", "imei")),
    ("bills", "bills", ("id", "customer_id", "amount_due", "due_date", "status")),
]


def _upsert_sql(table: str, cols: tuple[str, ...]) -> str:
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id")
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
        f"ON CONFLICT (id) DO UPDATE SET {updates}"
    )


async def load_seed(pool: Pool, seed: Seed) -> dict[str, int]:
    """Upsert everything in one transaction. Line items are replaced per bill (no natural key)."""
    counts: dict[str, int] = {}
    async with pool.connection() as conn:
        async with conn.transaction():
            for attr, table, cols in _UPSERTS:
                rows: list[Any] = getattr(seed, attr)
                sql = _upsert_sql(table, cols)
                for row in rows:
                    data = row.model_dump()
                    await conn.execute(sql, tuple(data[c] for c in cols))
                counts[table] = len(rows)
            for bill_id in sorted({i.bill_id for i in seed.bill_line_items}):
                await conn.execute("DELETE FROM bill_line_items WHERE bill_id = %s", (bill_id,))
            for item in seed.bill_line_items:
                await conn.execute(
                    "INSERT INTO bill_line_items (bill_id, description, amount) VALUES (%s, %s, %s)",
                    (item.bill_id, item.description, item.amount),
                )
            counts["bill_line_items"] = len(seed.bill_line_items)
    return counts


async def _main(path: str) -> None:
    from support_assistant.config import get_settings
    from support_assistant.db import make_pool

    seed = parse_seed(Path(path).read_text(encoding="utf-8"))
    pool = make_pool(get_settings())
    await pool.open()
    try:
        print(await load_seed(pool, seed))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1]))
