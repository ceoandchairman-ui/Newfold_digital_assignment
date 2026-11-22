"""Create a Postgres role per workload identity, with least-privilege grants (E10-6).

    python -m support_assistant.dbroles        # run by the db-bootstrap pre-deploy job, after migrations

Under Entra auth a service connects as its identity's NAME (id-<env>-<service>), and that role must
exist inside Postgres. This job runs as the db-bootstrap identity, which is a Postgres Entra
administrator, and is safe to run on every deploy:
- a missing role is created and mapped to the identity's object id;
- every role's grants are REVOKED and re-applied from the table below, so a grant removed here is
  removed in the database (no drift), and a service gets nothing it is not listed for.

The grants are the access each service really needs, derived from the repositories and services it
calls. agent-director and agent-evaluator reach the data only through MCP, so they get nothing.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

from support_assistant.db import Pool

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,63}$")

# workload -> list of (privileges, tables)
GRANTS: dict[str, list[tuple[str, list[str]]]] = {
    # The only path to the ledger. lock_bill's SELECT ... FOR UPDATE needs UPDATE on bills.
    "mcp-servers": [
        (
            "SELECT",
            ["customers", "plans", "payment_methods", "lines", "devices", "bill_line_items"],
        ),
        ("SELECT, UPDATE", ["bills"]),
        ("SELECT, INSERT, UPDATE", ["transaction_history"]),
    ],
    # Identity, conversation memory, episodes, and the retention job / erasure (DELETE).
    "fat-backend": [
        ("SELECT", ["customers"]),
        ("SELECT, INSERT, UPDATE, DELETE", ["session_conversations", "customer_episodes"]),
    ],
    # Graph-node retrieval and the audit trail.
    "orchestrator": [
        ("SELECT", ["knowledge_base_chunks"]),
        ("INSERT", ["audit_events"]),
    ],
    "agent-troubleshooter": [("SELECT", ["knowledge_base_chunks"])],
    # Looks up a customer's phone and email to send a notification.
    "notification-worker": [("SELECT", ["customers"])],
    # Agents without direct database access, and the bootstrap admin itself:
    "agent-director": [],
    "agent-evaluator": [],
    "db-bootstrap": [],
}


def quote_ident(name: str) -> str:
    if not _SAFE.match(name):
        raise ValueError(f"unsafe identifier: {name!r}")
    return '"' + name + '"'


def parse_principals(raw: str) -> dict[str, str]:
    """'mcp-servers=<oid>,fat-backend=<oid>' -> {workload: object id}."""
    out: dict[str, str] = {}
    for item in filter(None, (p.strip() for p in raw.split(","))):
        workload, _, oid = item.partition("=")
        if not workload or not oid:
            raise ValueError(f"bad principal entry: {item!r}")
        out[workload.strip()] = oid.strip()
    return out


def grant_statements(role: str, workload: str) -> list[str]:
    """Idempotent statements: wipe the role's table grants, then apply exactly what it is listed for."""
    if workload not in GRANTS:
        raise KeyError(f"no grants defined for workload {workload!r}")
    r = quote_ident(role)
    stmts = [
        f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {r}",
        f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {r}",
    ]
    wrote = False
    for privileges, tables in GRANTS[workload]:
        stmts.append(f"GRANT {privileges} ON {', '.join(quote_ident(t) for t in tables)} TO {r}")
        wrote = wrote or any(p in privileges for p in ("INSERT", "UPDATE"))
    if wrote:  # inserting into a BIGSERIAL table also needs its sequence
        stmts.append(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {r}")
    return stmts


async def apply_roles(pool: Pool, prefix: str, principals: dict[str, str]) -> dict[str, str]:
    """Create missing roles and apply grants, in one transaction. Returns {role: 'created'|'kept'}."""
    outcome: dict[str, str] = {}
    async with pool.connection() as conn:
        async with conn.transaction():
            for workload, oid in sorted(principals.items()):
                role = f"id-{prefix}-{workload}"
                quote_ident(role)  # validate before it goes anywhere near SQL
                row = await (
                    await conn.execute(
                        "SELECT 1 AS present FROM pg_roles WHERE rolname = %s", (role,)
                    )
                ).fetchone()
                if row is None:
                    await conn.execute(
                        "SELECT pgaadauth_create_principal_with_oid(%s, %s, 'service', false, false)",
                        (role, oid),
                    )
                    outcome[role] = "created"
                else:
                    outcome[role] = "kept"
                for stmt in grant_statements(role, workload):
                    await conn.execute(stmt)  # identifiers are validated and quoted above
    return outcome


async def _main() -> int:
    from support_assistant.config import get_settings
    from support_assistant.db import make_pool

    prefix = os.environ["IDENTITY_PREFIX"]
    principals = parse_principals(os.environ["DB_PRINCIPALS"])
    pool = make_pool(get_settings())
    await pool.open()
    try:
        for role, what in (await apply_roles(pool, prefix, principals)).items():
            print(f"{what:8} {role}")
    finally:
        await pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
