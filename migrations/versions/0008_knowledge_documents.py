"""One record per indexed knowledge document, so an unchanged document is skipped.

Written in the same transaction as the document's chunks. Only the two indexing workloads may use
it (dbroles.py).

Revision ID: 0008
Revises: 0007
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            source                text PRIMARY KEY,
            document_hash         text NOT NULL,
            indexing_fingerprint  text NOT NULL,
            chunk_hashes          text[] NOT NULL,
            size_bytes            bigint NOT NULL,
            indexed_at            timestamptz NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS knowledge_documents")
