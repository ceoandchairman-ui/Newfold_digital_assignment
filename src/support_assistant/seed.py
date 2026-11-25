"""Seed loader (E1-2): load reference and demo data from TOML, idempotently.

    python -m support_assistant.seed data/corpus/db.toml       # the supplied telecom data
    python -m support_assistant.seed seed/example_seed.toml    # the small example

Two formats are accepted: the supplied db.toml (converted by corpus.py) and our own.

Every row is validated before anything is written, then all rows are upserted in foreign-key order
in ONE transaction: a bad file changes nothing, and loading the same file twice changes nothing.
"""

from __future__ import annotations

import asyncio
import sys
import tomllib
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from support_assistant.corpus import CorpusError, convert_corpus, is_corpus
from support_assistant.db import Pool


class Customer(BaseModel):
    id: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    date_of_birth: date
    phone: str = Field(min_length=3)
    email: str | None = None
    account_status: str = "ACTIVE"
    last_extension_date: date | None = None
    goodwill_credit_used_this_year: Decimal = Field(Decimal(0), ge=0)


class Plan(BaseModel):
    id: str = Field(min_length=1)
    name: str
    monthly_price: Decimal = Field(ge=0)
    data_limit_mb: int | None = Field(None, ge=0)
    refuel_price_per_gb: Decimal | None = Field(None, ge=0)


class PaymentMethod(BaseModel):
    id: str
    customer_id: str
    kind: str
    last4: str | None = Field(None, min_length=4, max_length=4)
    expiration: str | None = None


class Line(BaseModel):
    id: str
    customer_id: str
    plan_id: str
    phone_number: str
    status: str
    data_used_mb: int = Field(0, ge=0)
    contract_end_date: date | None = None
    data_refueling_mb: int = Field(0, ge=0)
    roaming_enabled: bool = False
    last_plan_change_date: date | None = None
    last_sim_replacement_date: date | None = None
    suspension_start_date: date | None = None


class Device(BaseModel):
    id: str
    line_id: str
    model: str
    imei: str | None = None
    device_type: str | None = None
    esim_capable: bool | None = None
    activated: bool | None = None
    activation_date: datetime | None = None
    last_esim_transfer_date: datetime | None = None


BillState = Literal["DRAFT", "ISSUED", "OVERDUE", "AWAITING_PAYMENT", "PAID", "DISPUTED"]


class Bill(BaseModel):
    id: str
    customer_id: str
    amount_due: Decimal = Field(ge=0)
    due_date: date
    status: BillState
    period_start: date | None = None
    period_end: date | None = None
    issue_date: date | None = None


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
        data = tomllib.loads(text)
        if is_corpus(data):
            data = convert_corpus(data)
        seed = Seed.model_validate(data)
    except (tomllib.TOMLDecodeError, ValidationError, CorpusError) as exc:
        raise SeedError(str(exc)) from exc
    seed.check_references()
    return seed


def _columns(model: type[BaseModel]) -> tuple[str, ...]:
    return tuple(model.model_fields)  # each model's fields are its table's columns


_UPSERTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("customers", "customers", _columns(Customer)),
    ("plans", "plans", _columns(Plan)),
    ("payment_methods", "payment_methods", _columns(PaymentMethod)),
    ("lines", "lines", _columns(Line)),
    ("devices", "devices", _columns(Device)),
    ("bills", "bills", _columns(Bill)),
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
