"""Relational ledger (E1-1, E1-5, E1-6).

NOTE: columns are inferred from the backlog. The original seed file (db.toml) is lost (D1), so
review this schema against the restored data before relying on it.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE customers (
            id          TEXT PRIMARY KEY,
            full_name   TEXT NOT NULL,
            date_of_birth DATE NOT NULL,
            phone       TEXT NOT NULL UNIQUE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE plans (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            monthly_price NUMERIC(12,2) NOT NULL CHECK (monthly_price >= 0),
            data_limit_mb INTEGER CHECK (data_limit_mb >= 0)
        );

        CREATE TABLE payment_methods (
            id          TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
            kind        TEXT NOT NULL,
            last4       CHAR(4)
        );
        CREATE INDEX ix_payment_methods_customer ON payment_methods (customer_id);

        CREATE TABLE lines (
            id                TEXT PRIMARY KEY,
            customer_id       TEXT NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
            plan_id           TEXT NOT NULL REFERENCES plans(id) ON DELETE RESTRICT,
            phone_number      TEXT NOT NULL UNIQUE,
            status            TEXT NOT NULL,
            data_used_mb      INTEGER NOT NULL DEFAULT 0 CHECK (data_used_mb >= 0),
            contract_end_date DATE
        );
        CREATE INDEX ix_lines_customer ON lines (customer_id);
        CREATE INDEX ix_lines_plan ON lines (plan_id);

        CREATE TABLE devices (
            id         TEXT PRIMARY KEY,
            line_id    TEXT NOT NULL REFERENCES lines(id) ON DELETE RESTRICT,
            model      TEXT NOT NULL,
            imei       TEXT UNIQUE
        );
        CREATE INDEX ix_devices_line ON devices (line_id);

        CREATE TABLE bills (
            id          TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
            amount_due  NUMERIC(12,2) NOT NULL CHECK (amount_due >= 0),
            due_date    DATE NOT NULL,
            status      TEXT NOT NULL CHECK (status IN ('ISSUED', 'AWAITING_PAYMENT', 'PAID'))
        );
        CREATE INDEX ix_bills_customer ON bills (customer_id);
        -- At most one AWAITING_PAYMENT bill per customer, enforced by the database (E1-6).
        CREATE UNIQUE INDEX ux_bills_one_awaiting_per_customer
            ON bills (customer_id) WHERE status = 'AWAITING_PAYMENT';

        CREATE TABLE bill_line_items (
            id          BIGSERIAL PRIMARY KEY,
            bill_id     TEXT NOT NULL REFERENCES bills(id) ON DELETE RESTRICT,
            description TEXT NOT NULL,
            amount      NUMERIC(12,2) NOT NULL
        );
        CREATE INDEX ix_bill_line_items_bill ON bill_line_items (bill_id);

        CREATE TABLE transaction_history (
            id              BIGSERIAL PRIMARY KEY,
            customer_id     TEXT NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
            bill_id         TEXT NOT NULL REFERENCES bills(id) ON DELETE RESTRICT,
            amount          NUMERIC(12,2) NOT NULL CHECK (amount > 0),
            status          TEXT NOT NULL CHECK (status IN ('PENDING', 'SUCCEEDED', 'FAILED')),
            idempotency_key TEXT NOT NULL UNIQUE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        -- Supports the eval_agent duplicate-payment velocity query (E4-6).
        CREATE INDEX ix_transaction_history_customer_time
            ON transaction_history (customer_id, created_at);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS transaction_history;
        DROP TABLE IF EXISTS bill_line_items;
        DROP TABLE IF EXISTS bills;
        DROP TABLE IF EXISTS devices;
        DROP TABLE IF EXISTS lines;
        DROP TABLE IF EXISTS payment_methods;
        DROP TABLE IF EXISTS plans;
        DROP TABLE IF EXISTS customers;
        """
    )
