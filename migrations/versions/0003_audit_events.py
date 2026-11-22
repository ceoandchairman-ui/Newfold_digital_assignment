"""Audit trail of every tool call (E5-5).

Append-only by convention: the application only ever INSERTs. Revoke UPDATE/DELETE on this table
from the application role in production so the trail cannot be rewritten.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE audit_events (
            id          BIGSERIAL PRIMARY KEY,
            at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            session_id  TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            agent       TEXT NOT NULL,
            tool        TEXT NOT NULL,
            ok          BOOLEAN NOT NULL
        );
        CREATE INDEX ix_audit_events_customer_time ON audit_events (customer_id, at);
        CREATE INDEX ix_audit_events_session ON audit_events (session_id);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_events")
