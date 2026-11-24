"""Redis Entra credentials, seed loader (E1-2), device simulation (E3-8), email (E8-4) and the
Director's flagged tool calling (E4-3/E5-7)."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from support_assistant.config import Settings
from support_assistant.mcp_catalog import DEVICE
from support_assistant.mcp_servers.device import build_device_server
from support_assistant.redis_factory import EntraRedisCredentials, make_redis
from support_assistant.seed import SeedError, load_seed, parse_seed
from support_assistant.services.device_sim import (
    DEFAULT_APN,
    DeviceSimulation,
    DeviceState,
    InMemoryDeviceStore,
)
from support_assistant.worker import Outcome, handle_message

ROOT = Path(__file__).resolve().parents[1]


# --- Redis Entra credentials (E10-6) ------------------------------------------------------------------------
async def test_redis_token_is_cached_and_refreshed_before_it_expires():
    now = [1000.0]
    fetched: list[int] = []

    async def fetch():
        fetched.append(1)
        return f"tok-{len(fetched)}", now[0] + 3600

    creds = EntraRedisCredentials("oid-1", fetch, refresh_margin_s=300, clock=lambda: now[0])
    assert await creds.get_credentials_async() == ("oid-1", "tok-1")
    now[0] += 3000  # still more than 5 minutes left
    assert await creds.get_credentials_async() == ("oid-1", "tok-1") and len(fetched) == 1
    now[0] += 400  # inside the refresh margin
    assert await creds.get_credentials_async() == ("oid-1", "tok-2")


def test_entra_redis_needs_an_object_id():
    s = Settings(
        _env_file=None, database_url="postgresql://x", redis_url="redis://x", redis_entra_auth=True
    )
    with pytest.raises(RuntimeError):
        make_redis("redis://x", s)


def test_without_entra_redis_uses_the_url_credentials():
    s = Settings(_env_file=None, database_url="postgresql://x", redis_url="redis://x")
    client = make_redis("redis://u:p@h:6379/0", s)
    assert client.connection_pool.connection_kwargs.get("credential_provider") is None


# --- seed loader (E1-2) ---------------------------------------------------------------------------------------
EXAMPLE = (ROOT / "seed" / "example_seed.toml").read_text(encoding="utf-8")


def test_the_example_seed_is_valid_and_matches_the_journeys():
    seed = parse_seed(EXAMPLE)
    assert {b.id for b in seed.bills} == {"B1005", "B2001"} and len(seed.customers) == 2


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        (EXAMPLE.replace('plan_id = "P-PLUS"', 'plan_id = "P-NOPE"'), "unknown plan"),
        (EXAMPLE.replace('status = "ISSUED"', 'status = "MAYBE"'), "status"),
        (EXAMPLE.replace('amount_due = "42.50"', 'amount_due = "-1"'), "amount_due"),
        ("not = [valid", "toml"),
    ],
)
def test_an_invalid_seed_is_rejected_before_anything_is_written(bad, why):
    with pytest.raises(SeedError):
        parse_seed(bad)


def test_two_awaiting_bills_for_one_customer_are_rejected():
    doubled = EXAMPLE.replace('status = "ISSUED"', 'status = "AWAITING_PAYMENT"') + (
        '\n[[bills]]\nid = "B9"\ncustomer_id = "C1002"\namount_due = "1.00"\n'
        'due_date = 2026-11-02\nstatus = "AWAITING_PAYMENT"\n'
    )
    with pytest.raises(SeedError, match="AWAITING_PAYMENT"):
        parse_seed(doubled)


class RecConn:
    def __init__(self, log):
        self.log = log

    async def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    @asynccontextmanager
    async def transaction(self):
        yield


class RecPool:
    def __init__(self):
        self.log: list = []

    @asynccontextmanager
    async def connection(self):
        yield RecConn(self.log)


async def test_load_upserts_in_foreign_key_order_and_replaces_line_items():
    pool = RecPool()
    counts = await load_seed(pool, parse_seed(EXAMPLE))
    tables = [sql.split()[2] for sql, _ in pool.log if sql.startswith("INSERT")]
    order = [t for i, t in enumerate(tables) if t not in tables[:i]]
    assert order == [
        "customers",
        "plans",
        "payment_methods",
        "lines",
        "devices",
        "bills",
        "bill_line_items",
    ]
    assert all(
        "ON CONFLICT (id) DO UPDATE" in sql
        for sql, _ in pool.log
        if sql.startswith("INSERT INTO customers")
    )
    assert any(sql.startswith("DELETE FROM bill_line_items") for sql, _ in pool.log)
    assert counts["customers"] == 2 and counts["bill_line_items"] == 1


# --- device simulation (E3-8) ------------------------------------------------------------------------------------
async def test_wrong_apn_breaks_data_and_reset_fixes_it():
    sim = DeviceSimulation(InMemoryDeviceStore())
    await sim.set_apn_settings("C1", "D1", "wrong.apn")
    assert (await sim.can_send_mms("C1", "D1")) == {
        "can_send_mms": False,
        "reason": "APN settings are wrong",
    }
    fixed = await sim.reset_apn_settings("C1", "D1")
    assert fixed["apn"] == DEFAULT_APN and fixed["data_works"]


async def test_toggles_are_idempotent_so_a_retry_cannot_undo_them():
    sim = DeviceSimulation(InMemoryDeviceStore())
    await sim.toggle_airplane_mode("C1", "D1", enabled=True)
    await sim.toggle_airplane_mode("C1", "D1", enabled=True)  # a retried call
    assert (await sim.check_status("C1", "D1"))["airplane_mode"] is True


async def test_reboot_clears_airplane_mode_but_not_a_locked_sim():
    store = InMemoryDeviceStore({("C1", "D1"): DeviceState(airplane_mode=True, sim_locked=True)})
    after = await DeviceSimulation(store).reboot_device("C1", "D1")
    assert after["airplane_mode"] is False and after["has_service"] is False  # locked SIM: escalate


async def test_device_state_is_per_customer():
    sim = DeviceSimulation(InMemoryDeviceStore())
    await sim.toggle_mobile_data("C1", "D1", enabled=False)
    assert (await sim.check_status("C2", "D1"))["mobile_data"] is True


async def test_the_device_server_serves_exactly_the_catalog():
    names = {t.name for t in await build_device_server().list_tools()}
    assert names == set(DEVICE.tools)


# --- email (E8-4) -------------------------------------------------------------------------------------------------
async def _phone(_):
    return "+15550001"


async def _no_sms(*_):
    raise AssertionError("not sms")


def _msg(channel: str) -> str:
    return json.dumps({"customer_id": "C1", "channel": channel, "template": "payment_received"})


async def test_email_is_sent_to_the_address_on_file():
    sent: list[tuple] = []

    async def lookup_email(_):
        return "ada@example.invalid"

    async def send_email(addr, subject, text):
        sent.append((addr, subject))

    out = await handle_message(
        _msg("email"), _phone, _no_sms, lookup_email=lookup_email, send_email=send_email
    )
    assert out is Outcome.COMPLETE and sent == [("ada@example.invalid", "Payment received")]


async def test_email_without_an_address_or_sender_is_dead_lettered():
    async def none(_):
        return None

    async def send_email(*_):
        raise AssertionError("must not send")

    assert (
        await handle_message(
            _msg("email"), _phone, _no_sms, lookup_email=none, send_email=send_email
        )
        is Outcome.DEAD_LETTER
    )
    assert await handle_message(_msg("email"), _phone, _no_sms) is Outcome.DEAD_LETTER
