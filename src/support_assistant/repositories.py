"""Parameterised SQL access (E1-3). No string-built SQL anywhere; every query is scoped by customer."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from support_assistant.db import Conn
from support_assistant.errors import NotFoundError

Row = dict[str, Any]


async def get_customer(conn: Conn, customer_id: str) -> Row:
    cur = await conn.execute(
        "SELECT id, full_name, date_of_birth, phone, email FROM customers WHERE id = %s",
        (customer_id,),
    )
    row = await cur.fetchone()
    if row is None:
        raise NotFoundError(f"customer {customer_id}")
    return row


async def find_customers_by_name_dob(conn: Conn, full_name: str, dob: date) -> list[Row]:
    """Case- and space-insensitive name match on an exact date of birth. Callers need exactly one."""
    cur = await conn.execute(
        "SELECT id, full_name, date_of_birth FROM customers"
        " WHERE lower(trim(full_name)) = lower(trim(%s)) AND date_of_birth = %s LIMIT 2",
        (full_name, dob),
    )
    return await cur.fetchall()


async def get_customer_by_phone(conn: Conn, phone: str) -> Row | None:
    cur = await conn.execute(
        "SELECT id, full_name, date_of_birth FROM customers WHERE phone = %s", (phone,)
    )
    return await cur.fetchone()


async def list_lines(conn: Conn, customer_id: str) -> list[Row]:
    cur = await conn.execute(
        """
        SELECT l.id, l.phone_number, l.status, l.data_used_mb, l.contract_end_date,
               p.name AS plan_name, p.monthly_price, p.data_limit_mb
          FROM lines l JOIN plans p ON p.id = l.plan_id
         WHERE l.customer_id = %s ORDER BY l.id
        """,
        (customer_id,),
    )
    return await cur.fetchall()


async def list_bills(conn: Conn, customer_id: str) -> list[Row]:
    cur = await conn.execute(
        "SELECT id, amount_due, due_date, status FROM bills WHERE customer_id = %s ORDER BY due_date",
        (customer_id,),
    )
    return await cur.fetchall()


async def count_overdue_bills(conn: Conn, customer_id: str, today: date) -> int:
    cur = await conn.execute(
        # The same rule as domain.billing.is_overdue.
        "SELECT count(*) AS n FROM bills WHERE customer_id = %s"
        " AND status NOT IN ('PAID', 'DRAFT') AND (status = 'OVERDUE' OR due_date < %s)",
        (customer_id, today),
    )
    row = await cur.fetchone()
    return int(row["n"]) if row else 0


async def latest_contract_end(conn: Conn, customer_id: str) -> date | None:
    cur = await conn.execute(
        "SELECT max(contract_end_date) AS d FROM lines WHERE customer_id = %s", (customer_id,)
    )
    row = await cur.fetchone()
    return row["d"] if row else None


async def list_devices(conn: Conn, customer_id: str) -> list[Row]:
    cur = await conn.execute(
        """
        SELECT d.id, d.model, d.line_id, d.imei FROM devices d JOIN lines l ON l.id = d.line_id
         WHERE l.customer_id = %s
        """,
        (customer_id,),
    )
    return await cur.fetchall()


async def list_payment_methods(conn: Conn, customer_id: str) -> list[Row]:
    """Kind and last four digits only. The table holds nothing more sensitive than that."""
    cur = await conn.execute(
        "SELECT id, kind, last4 FROM payment_methods WHERE customer_id = %s ORDER BY id",
        (customer_id,),
    )
    return await cur.fetchall()


async def list_episodes(conn: Conn, customer_id: str, limit: int) -> list[Row]:
    """The customer's past resolved issues, newest first. No embeddings leave the database."""
    cur = await conn.execute(
        "SELECT summary, outcome, created_at FROM customer_episodes"
        " WHERE customer_id = %s ORDER BY created_at DESC LIMIT %s",
        (customer_id, limit),
    )
    return await cur.fetchall()


async def recent_sessions(conn: Conn, customer_id: str, days: int, limit: int) -> list[Row]:
    """The customer's conversations touched in the last `days` days, across sessions, newest first."""
    cur = await conn.execute(
        "SELECT session_id, messages, updated_at FROM session_conversations"
        " WHERE customer_id = %s AND updated_at > now() - make_interval(days => %s)"
        " ORDER BY updated_at DESC LIMIT %s",
        (customer_id, days, limit),
    )
    return await cur.fetchall()


async def has_earlier_history(conn: Conn, customer_id: str, session_id: str) -> bool:
    """Is anything stored from before this session: another conversation or a saved episode?"""
    cur = await conn.execute(
        "SELECT EXISTS (SELECT 1 FROM session_conversations"
        " WHERE customer_id = %s AND session_id <> %s)"
        " OR EXISTS (SELECT 1 FROM customer_episodes WHERE customer_id = %s) AS found",
        (customer_id, session_id, customer_id),
    )
    row = await cur.fetchone()
    return bool(row and row["found"])


async def recent_payment_count(conn: Conn, customer_id: str, bill_id: str, minutes: int) -> int:
    """Velocity query for the evaluator's duplicate-payment check (E4-6)."""
    cur = await conn.execute(
        """
        SELECT count(*) AS n FROM transaction_history
         WHERE customer_id = %s AND bill_id = %s
           AND created_at > now() - make_interval(mins => %s)
        """,
        (customer_id, bill_id, minutes),
    )
    row = await cur.fetchone()
    return int(row["n"]) if row else 0


async def lock_bill(conn: Conn, customer_id: str, bill_id: str) -> Row:
    """SELECT ... FOR UPDATE inside the caller's transaction (E3-3)."""
    cur = await conn.execute(
        "SELECT id, amount_due, due_date, status FROM bills WHERE id = %s AND customer_id = %s FOR UPDATE",
        (bill_id, customer_id),
    )
    row = await cur.fetchone()
    if row is None:
        raise NotFoundError(f"bill {bill_id}")
    return row


async def set_bill_status(conn: Conn, bill_id: str, status: str) -> None:
    await conn.execute("UPDATE bills SET status = %s WHERE id = %s", (status, bill_id))


async def find_transaction(conn: Conn, idempotency_key: str) -> Row | None:
    cur = await conn.execute(
        "SELECT id, customer_id, bill_id, amount, status FROM transaction_history WHERE idempotency_key = %s",
        (idempotency_key,),
    )
    return await cur.fetchone()


async def insert_transaction(
    conn: Conn,
    customer_id: str,
    bill_id: str,
    amount: Decimal,
    status: str,
    idempotency_key: str,
) -> int:
    cur = await conn.execute(
        """
        INSERT INTO transaction_history (customer_id, bill_id, amount, status, idempotency_key)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """,
        (customer_id, bill_id, amount, status, idempotency_key),
    )
    row = await cur.fetchone()
    assert row is not None
    return int(row["id"])


async def set_transaction_status(conn: Conn, tx_id: int, status: str) -> None:
    await conn.execute("UPDATE transaction_history SET status = %s WHERE id = %s", (status, tx_id))


async def get_bill(conn: Conn, customer_id: str, bill_id: str) -> Row:
    cur = await conn.execute(
        "SELECT id, amount_due, due_date, status FROM bills WHERE id = %s AND customer_id = %s",
        (bill_id, customer_id),
    )
    row = await cur.fetchone()
    if row is None:
        raise NotFoundError(f"bill {bill_id}")
    return row
