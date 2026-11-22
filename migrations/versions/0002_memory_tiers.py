"""Memory tiers: vector tables, session windows (E2-1, E2-6, E2-7).

The embedding dimension is taken from the EMBEDDING_DIM environment variable so that `vector(N)`
matches the chosen embedder model (an open decision). Changing the model later means a new
migration and re-embedding.

Revision ID: 0002
Revises: 0001
"""

import os

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _dim() -> int:
    raw = os.environ.get("EMBEDDING_DIM")
    if not raw or not raw.isdigit() or int(raw) <= 0:
        raise RuntimeError("EMBEDDING_DIM must be set to a positive integer to run this migration")
    return int(raw)


def upgrade() -> None:
    dim = _dim()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"""
        CREATE TABLE knowledge_base_chunks (
            id                BIGSERIAL PRIMARY KEY,
            source            TEXT NOT NULL,
            section_id        TEXT NOT NULL,
            heading_path      TEXT NOT NULL,
            language          TEXT NOT NULL DEFAULT 'und',
            content           TEXT NOT NULL,
            content_hash      CHAR(64) NOT NULL UNIQUE,
            -- 'simple' is language-neutral; per-language configs arrive with E2-9.
            content_tsv       tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
            embedding         vector({dim}) NOT NULL,
            embedding_model   TEXT NOT NULL,
            embedding_version TEXT NOT NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX ix_kb_chunks_embedding
            ON knowledge_base_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        CREATE INDEX ix_kb_chunks_tsv ON knowledge_base_chunks USING gin (content_tsv);

        CREATE TABLE customer_episodes (
            id                BIGSERIAL PRIMARY KEY,
            customer_id       TEXT NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
            summary           TEXT NOT NULL,
            outcome           TEXT NOT NULL,
            content_hash      CHAR(64) NOT NULL UNIQUE,
            embedding         vector({dim}) NOT NULL,
            embedding_model   TEXT NOT NULL,
            embedding_version TEXT NOT NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX ix_episodes_customer ON customer_episodes (customer_id);
        CREATE INDEX ix_episodes_embedding
            ON customer_episodes USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);

        CREATE TABLE session_conversations (
            session_id  TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
            messages    JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX ix_session_conversations_customer ON session_conversations (customer_id);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS session_conversations;
        DROP TABLE IF EXISTS customer_episodes;
        DROP TABLE IF EXISTS knowledge_base_chunks;
        """
    )
