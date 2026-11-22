"""Alembic environment. Migrations are plain SQL; SQLAlchemy is used only to run them.

With DATABASE_ENTRA_AUTH=true (Azure: Postgres is Entra-only), a fresh Entra access token is used
as the password, because there is no static password to put in DATABASE_URL.
"""

import os

from alembic import context
from sqlalchemy import create_engine, pool

POSTGRES_ENTRA_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL must be set to run migrations")
    # Use the psycopg 3 driver.
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def _connect_args() -> dict[str, str]:
    if os.environ.get("DATABASE_ENTRA_AUTH", "").lower() not in {"1", "true", "yes"}:
        return {}
    from azure.identity import DefaultAzureCredential

    token = DefaultAzureCredential().get_token(POSTGRES_ENTRA_SCOPE).token
    return {"password": token, "sslmode": "require"}


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool, connect_args=_connect_args())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
