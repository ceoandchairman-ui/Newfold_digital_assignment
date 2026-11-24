"""The user-data and CRM contracts: account_management, crm and get_payment_methods."""

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

import pytest

from support_assistant.agents import roles
from support_assistant.mcp_catalog import ACCOUNT, ALL_SERVERS, CRM, DATABASE
from support_assistant.mcp_servers.account import build_account_server
from support_assistant.mcp_servers.common import mask_email
from support_assistant.mcp_servers.crm import MAX_CASES, MAX_DAYS, build_crm_server
from support_assistant.mcp_servers.database import build_database_server
from support_assistant.services.payments import StubGateway


class Cursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)


class Conn:
    """Answers by the table named in the SQL and records every statement with its parameters."""

    def __init__(self, tables, log):
        self._tables, self._log = tables, log

    async def execute(self, sql, params=None):
        self._log.append((sql, params))
        for name, rows in self._tables.items():
            if f"FROM {name}" in sql or f"FROM {name} " in sql:
                return Cursor(rows)
        return Cursor([])


class Pool:
    def __init__(self, **tables):
        self.tables, self.log = tables, []

    @asynccontextmanager
    async def connection(self):
        yield Conn(self.tables, self.log)


async def _call(server, tool, args):
    result = await server.call_tool(tool, args)
    return result[1] if isinstance(result, tuple) else result


# --- the contract itself ---------------------------------------------------------------------
def test_the_new_namespaces_are_in_the_catalog_with_their_own_ports():
    assert ACCOUNT.tools == {"get_customer_profile", "get_devices", "get_contract"}
    assert CRM.tools == {"get_case_history", "get_recent_interactions"}
    assert "get_payment_methods" in DATABASE.tools
    assert {s.port for s in ALL_SERVERS} == {8001, 8002, 8003, 8004, 8005}


def test_every_new_tool_is_read_only_so_none_is_a_write_tool():
    from support_assistant.mcp_catalog import WRITE_TOOLS

    new = ACCOUNT.tools | CRM.tools | {"get_payment_methods"}
    assert not (new & WRITE_TOOLS)


def test_the_support_agent_may_read_all_of_it_and_the_evaluator_none_of_it():
    new = ACCOUNT.tools | CRM.tools | {"get_payment_methods"}
    assert {"get_customer_profile", "get_contract", "get_payment_methods"} <= roles.DIRECTOR_TOOLS
    assert not (new & roles.EVALUATOR_TOOLS)  # the evaluator stays on its two read tools
    assert "process_payment" not in roles.DIRECTOR_TOOLS
    assert {"get_devices", "get_case_history"} <= roles.TROUBLESHOOTER_TOOLS
    assert not ({"get_payment_methods", "get_bills"} & roles.TROUBLESHOOTER_TOOLS)


def test_every_catalog_tool_is_allowed_to_at_least_one_agent_or_is_the_payment_write():
    allowed = roles.DIRECTOR_TOOLS | roles.TROUBLESHOOTER_TOOLS | roles.EVALUATOR_TOOLS
    every = {t for s in ALL_SERVERS for t in s.tools}
    assert every - allowed == {"process_payment", "send_notification"}


# --- account_management ----------------------------------------------------------------------
async def test_the_profile_masks_the_email_and_never_returns_a_date_of_birth():
    pool = Pool(
        customers=[
            {
                "id": "C1",
                "full_name": "Ada Lovelace",
                "date_of_birth": date(1990, 1, 1),
                "phone": "+15550001",
                "email": "ada@example.com",
            }
        ]
    )
    out = await _call(build_account_server(pool), "get_customer_profile", {"customer_id": "C1"})
    assert out["ok"] and out["profile"]["email"] == "a***@example.com"
    assert "date_of_birth" not in out["profile"] and "ada@" not in str(out)
    assert pool.log[0][1] == ("C1",)  # scoped to the customer, parameterised


async def test_devices_expose_only_the_last_four_of_the_imei():
    pool = Pool(
        devices=[{"id": "D1", "model": "Phone X", "line_id": "L1", "imei": "356938035643809"}]
    )
    out = await _call(build_account_server(pool), "get_devices", {"customer_id": "C1"})
    assert out["devices"] == [
        {"id": "D1", "model": "Phone X", "line_id": "L1", "imei_last4": "3809"}
    ]
    assert "356938035643809" not in str(out)


async def test_the_contract_flags_an_expired_line():
    past, future = date.today() - timedelta(days=1), date.today() + timedelta(days=90)
    pool = Pool(
        lines=[
            {"id": "L1", "phone_number": "+1", "status": "SUSPENDED", "plan_name": "Plus",
             "monthly_price": 1, "data_limit_mb": 1, "data_used_mb": 0, "contract_end_date": past},
            {"id": "L2", "phone_number": "+2", "status": "ACTIVE", "plan_name": "Plus",
             "monthly_price": 1, "data_limit_mb": 1, "data_used_mb": 0, "contract_end_date": future},
            {"id": "L3", "phone_number": "+3", "status": "ACTIVE", "plan_name": "Basic",
             "monthly_price": 1, "data_limit_mb": 1, "data_used_mb": 0, "contract_end_date": None},
        ]
    )  # fmt: skip
    out = await _call(build_account_server(pool), "get_contract", {"customer_id": "C1"})
    assert [c["expired"] for c in out["contracts"]] == [True, False, False]
    assert out["contracts"][2]["contract_end_date"] is None


async def test_an_unknown_customer_gets_a_safe_error_not_a_stack_trace():
    out = await _call(build_account_server(Pool()), "get_customer_profile", {"customer_id": "ZZ"})
    assert out["ok"] is False and "customers" not in out["message"]


def test_email_masking():
    assert mask_email("jo@x.com") == "j***@x.com"
    assert mask_email(None) is None and mask_email("nope") is None


# --- crm --------------------------------------------------------------------------------------
async def test_case_history_is_capped_and_returns_no_embeddings():
    rows = [{"summary": "SIM reseated", "outcome": "resolved", "created_at": datetime(2026, 11, 1)}]
    pool = Pool(customer_episodes=rows)
    out = await _call(
        build_crm_server(pool), "get_case_history", {"customer_id": "C1", "limit": 99}
    )
    assert out["ok"] and out["cases"][0]["summary"] == "SIM reseated"
    assert "embedding" not in str(out)
    assert pool.log[0][1] == ("C1", MAX_CASES)  # the 99 was clamped


async def test_recent_interactions_default_to_five_days_and_clamp_the_window():
    msgs = [
        {"role": "user", "content": "my data is slow"},
        {"role": "assistant", "content": "Reset the APN. Source: APN guide"},
    ]
    pool = Pool(
        session_conversations=[
            {"session_id": "S1", "messages": msgs, "updated_at": datetime(2026, 11, 22)}
        ]
    )
    crm = build_crm_server(pool)
    out = await _call(crm, "get_recent_interactions", {"customer_id": "C1"})
    assert out["days"] == 5 and out["sessions"][0]["messages"] == 2
    assert "my data is slow" in out["sessions"][0]["summary"]
    assert pool.log[0][1][:2] == ("C1", 5)
    await _call(crm, "get_recent_interactions", {"customer_id": "C1", "days": 9999})
    assert pool.log[1][1][1] == MAX_DAYS  # an agent cannot pull the whole history


async def test_recent_interactions_query_is_scoped_to_the_customer_and_the_window():
    pool = Pool()
    await _call(build_crm_server(pool), "get_recent_interactions", {"customer_id": "C1"})
    sql, _ = pool.log[0]
    assert "customer_id = %s" in sql and "make_interval(days => %s)" in sql
    assert "LIMIT" in sql


# --- financial_operations (the database server) -----------------------------------------------
async def test_payment_methods_return_kind_and_last_four_only():
    pool = Pool(payment_methods=[{"id": "PM1", "kind": "card", "last4": "4242"}])
    out = await _call(
        build_database_server(pool, StubGateway()), "get_payment_methods", {"customer_id": "C1"}
    )
    assert out["payment_methods"] == [{"id": "PM1", "kind": "card", "last4": "4242"}]


@pytest.mark.parametrize("server", [ACCOUNT, CRM])
def test_a_new_server_listens_on_its_own_cataloged_port(server):
    builder = {ACCOUNT.name: build_account_server, CRM.name: build_crm_server}[server.name]
    assert builder(Pool()).settings.port == server.port
