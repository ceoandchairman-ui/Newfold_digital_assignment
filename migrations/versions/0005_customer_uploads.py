"""Customer-uploaded files (E13). The file itself lives in blob storage; this is its record.

status: PENDING_SCAN (just uploaded, quarantined) -> CLEAN (malware scan passed, may be used)
        or REJECTED (scan found a threat; the blob is deleted, the row is kept for audit).

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE customer_uploads (
            id            UUID PRIMARY KEY,
            customer_id   TEXT NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
            session_id    TEXT NOT NULL,
            blob_name     TEXT NOT NULL UNIQUE,
            original_name TEXT NOT NULL,
            content_type  TEXT NOT NULL,
            size_bytes    BIGINT NOT NULL CHECK (size_bytes > 0),
            sha256        CHAR(64) NOT NULL,
            status        TEXT NOT NULL CHECK (status IN ('PENDING_SCAN', 'CLEAN', 'REJECTED')),
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            scanned_at    TIMESTAMPTZ
        );
        -- Supports the per-customer daily limit and the customer's own list.
        CREATE INDEX ix_customer_uploads_customer_time ON customer_uploads (customer_id, created_at);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS customer_uploads")
