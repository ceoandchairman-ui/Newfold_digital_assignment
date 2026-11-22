"""Smoke test (E11-A5) and migration safety lint (E11-A7)."""

import importlib.util
from pathlib import Path

import httpx
import pytest

from support_assistant.smoke import check_endpoints, check_journey

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_migrations", ROOT / "scripts" / "check_migrations.py"
)
assert _spec and _spec.loader
check_migrations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_migrations)


# --- smoke test ---------------------------------------------------------------------------------------
async def test_smoke_passes_only_when_every_service_answers_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503 if "evaluator" in request.url.host else 200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        results = await check_endpoints(http)
    failed = [r.name for r in results if not r.ok]
    assert failed == ["agent-evaluator"] and len(results) == 5


async def test_smoke_journey_needs_a_real_reply():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/chat":
            return httpx.Response(200, json={"reply": "You have used 800 MB."})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert (await check_journey(http, "http://b", "+1", "1990-05-01")).ok

    def unverified(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(unverified)) as http:
        res = await check_journey(http, "http://b", "+1", "1990-05-01")
    assert not res.ok and "verify" in res.detail


# --- migration lint ----------------------------------------------------------------------------------------
def test_the_repositorys_migrations_are_all_safe():
    assert check_migrations.main([str(ROOT / "migrations" / "versions")]) == 0


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE bills",
        "ALTER TABLE customers DROP COLUMN email",
        "ALTER TABLE bills RENAME COLUMN status TO state",
        "ALTER TABLE bills ALTER COLUMN amount_due TYPE TEXT",
        "ALTER TABLE customers ALTER COLUMN email SET NOT NULL",
        "TRUNCATE audit_events",
    ],
)
def test_destructive_changes_are_rejected(tmp_path, sql):
    f = tmp_path / "0099_bad.py"
    f.write_text(f'def upgrade():\n    op.execute("{sql}")\n\n\ndef downgrade():\n    pass\n')
    assert check_migrations.problems_in(f)


def test_an_explicit_reviewed_marker_allows_a_contract_step(tmp_path):
    f = tmp_path / "0100_contract.py"
    f.write_text(
        'def upgrade():\n    # -- contract-ok: column unused since release 42\n    op.execute("ALTER TABLE customers DROP COLUMN legacy")\n'
    )
    assert check_migrations.problems_in(f) == []


def test_downgrades_are_not_checked(tmp_path):
    f = tmp_path / "0101_ok.py"
    f.write_text(
        'def upgrade():\n    op.execute("ALTER TABLE x ADD COLUMN y TEXT")\n\n\ndef downgrade():\n    op.execute("ALTER TABLE x DROP COLUMN y")\n'
    )
    assert check_migrations.problems_in(f) == []
