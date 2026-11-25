"""Per-workload Postgres roles and grants (E10-6): least privilege, idempotent, injection-safe."""

from contextlib import asynccontextmanager

import pytest

from support_assistant import dbroles
from support_assistant.db import connection_kwargs


# --- connection user ---------------------------------------------------------------------------------
async def test_entra_connections_use_the_identity_name_as_the_postgres_user():
    async def token() -> str:
        return "t"

    kw = await connection_kwargs(token, "id-support-dev-orchestrator")()
    assert kw["user"] == "id-support-dev-orchestrator" and kw["password"] == "t"
    assert "user" not in await connection_kwargs(token)()  # no user given: the URL's user is used


# --- grants -----------------------------------------------------------------------------------------------
def _joined(workload: str) -> str:
    return " | ".join(dbroles.grant_statements(f"id-support-dev-{workload}", workload))


def test_the_agents_without_database_access_get_nothing():
    for workload in ("agent-director", "agent-troubleshooter", "agent-evaluator", "db-bootstrap"):
        stmts = dbroles.grant_statements(f"id-x-{workload}", workload)
        assert all(s.startswith("REVOKE") for s in stmts)  # wiped, nothing granted


def test_only_the_mcp_servers_can_change_the_ledger():
    for workload in dbroles.GRANTS:
        text = _joined(workload)
        touches_ledger = any(f'"{t}"' in text for t in ("bills", "transaction_history"))
        writes = "UPDATE" in text or "INSERT" in text
        assert touches_ledger == (workload == "mcp-servers") or not (touches_ledger and writes)
    mcp = _joined("mcp-servers")
    assert (
        'SELECT, UPDATE ON "bills"' in mcp
        and 'SELECT, INSERT, UPDATE ON "transaction_history"' in mcp
    )
    assert "DELETE" not in mcp  # the ledger is never deleted from


def test_the_retrieval_service_can_only_read_the_manual():
    text = _joined("retrieval")
    assert (
        'GRANT SELECT ON "knowledge_base_chunks"' in text
        and "INSERT" not in text
        and "UPDATE" not in text
    )


def test_the_orchestrator_can_append_audit_but_not_rewrite_it():
    text = _joined("orchestrator")
    assert 'GRANT INSERT ON "audit_events"' in text
    assert "UPDATE" not in text and "DELETE" not in text  # an audit trail is append-only


def test_the_backend_may_delete_conversations_for_retention_but_cannot_touch_bills():
    text = _joined("fat-backend")
    assert 'DELETE ON "session_conversations", "customer_episodes"' in text
    assert '"bills"' not in text and '"transaction_history"' not in text


def test_a_role_that_writes_also_gets_the_sequences_and_a_reader_does_not():
    assert (
        "ALL SEQUENCES IN SCHEMA public TO"
        in _joined("fat-backend").split("REVOKE ALL ON ALL SEQUENCES")[1].split(" | ", 1)[1]
    )
    assert "GRANT USAGE" not in _joined("retrieval")


def test_every_grant_starts_by_revoking_so_removed_access_does_not_linger():
    for workload in dbroles.GRANTS:
        first, second = dbroles.grant_statements(f"id-x-{workload}", workload)[:2]
        assert first.startswith("REVOKE ALL ON ALL TABLES") and second.startswith(
            "REVOKE ALL ON ALL SEQUENCES"
        )


def test_an_unknown_workload_gets_no_grants_rather_than_default_ones():
    with pytest.raises(KeyError):
        dbroles.grant_statements("id-x-surprise", "surprise")


@pytest.mark.parametrize("bad", ['x"; DROP TABLE bills; --', "a b", "", "x" * 64, "id;x"])
def test_unsafe_role_names_never_reach_sql(bad):
    with pytest.raises(ValueError):
        dbroles.quote_ident(bad)


def test_principal_parsing():
    assert dbroles.parse_principals("a=1, b=2,") == {"a": "1", "b": "2"}
    with pytest.raises(ValueError):
        dbroles.parse_principals("a")


# --- applying ----------------------------------------------------------------------------------------------
class Cur:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


class Conn:
    def __init__(self, existing: set[str], log: list):
        self.existing, self.log = existing, log

    async def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))
        if sql.lstrip().startswith("SELECT 1 AS present"):
            return Cur({"present": 1} if params[0] in self.existing else None)
        return Cur(None)

    @asynccontextmanager
    async def transaction(self):
        yield


class Pool:
    def __init__(self, existing=()):
        self.log: list = []
        self.existing = set(existing)

    @asynccontextmanager
    async def connection(self):
        yield Conn(self.existing, self.log)


async def test_missing_roles_are_created_with_their_object_id_and_existing_ones_kept():
    pool = Pool(existing={"id-support-dev-orchestrator"})
    out = await dbroles.apply_roles(
        pool, "support-dev", {"orchestrator": "oid-1", "mcp-servers": "oid-2"}
    )
    assert out == {"id-support-dev-orchestrator": "kept", "id-support-dev-mcp-servers": "created"}
    creates = [p for sql, p in pool.log if "pgaadauth_create_principal_with_oid" in sql]
    assert creates == [("id-support-dev-mcp-servers", "oid-2")]  # parameters, never interpolated


async def test_grants_are_applied_for_every_role_even_when_it_already_exists():
    pool = Pool(existing={"id-support-dev-retrieval"})
    await dbroles.apply_roles(pool, "support-dev", {"retrieval": "oid"})
    sqls = [sql for sql, _ in pool.log]
    assert any(s.startswith("GRANT SELECT ON") for s in sqls) and any(
        s.startswith("REVOKE ALL ON ALL TABLES") for s in sqls
    )


async def test_a_bad_prefix_is_rejected_before_any_sql_runs():
    pool = Pool()
    with pytest.raises(ValueError):
        await dbroles.apply_roles(pool, 'x"; DROP', {"orchestrator": "oid"})
    assert pool.log == []
