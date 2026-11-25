"""Knowledge chunks are unique per DOCUMENT, not across the whole table.

With `content_hash` unique on its own, one document "owned" a chunk. Renaming a file then deleted
its unchanged chunks (the new name skipped them as known, and cleanup removed the old name's rows),
and two documents could not hold the same text.

ONE-WAY FOR THE INDEXER: the previous release inserts with ON CONFLICT (content_hash), which has no
matching constraint after this. Search and every other service keep working with either schema.
After a rollback, the OLD indexer cannot write until it is rolled forward again. Downgrading needs
duplicate chunks removed first.

Revision ID: 0006
Revises: 0005
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- contract-ok: only the indexer depends on the old constraint; see the note above
    op.execute(
        """
        ALTER TABLE knowledge_base_chunks
            DROP CONSTRAINT IF EXISTS knowledge_base_chunks_content_hash_key;
        ALTER TABLE knowledge_base_chunks
            ADD CONSTRAINT ux_kb_chunks_source_hash UNIQUE (source, content_hash);
        CREATE INDEX IF NOT EXISTS ix_kb_chunks_content_hash
            ON knowledge_base_chunks (content_hash);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS ix_kb_chunks_content_hash;
        ALTER TABLE knowledge_base_chunks DROP CONSTRAINT IF EXISTS ux_kb_chunks_source_hash;
        ALTER TABLE knowledge_base_chunks
            ADD CONSTRAINT knowledge_base_chunks_content_hash_key UNIQUE (content_hash);
        """
    )
