"""The SUPPLIED data and policy (data/corpus/), used exactly as given.

The agents, graph and payment state machine are real. The database is in memory and the payment
gateway is simulated. The SQL itself needs the integration tests.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from langgraph.types import Command

import test_journeys
from fakes import FakePool, World
from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.mcp_client import StaticRegistry
from support_assistant.config import business_today
from support_assistant.contracts import AgentRequest
from support_assistant.corpus import CorpusError, convert_corpus
from support_assistant.domain.billing import BillStatus, is_overdue, is_payable
from support_assistant.errors import ConflictError, DuplicateOperation
from support_assistant.seed import SeedError, parse_seed
from support_assistant.services import payments
from support_assistant.services.gateway_sim import FaultMode, SimulatedGateway

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
TEXT = (CORPUS / "db.toml").read_text(encoding="utf-8")
SEED = parse_seed(TEXT)
POLICY_DAY = date(2025, 2, 25)  # "The current time is 2025-02-25" (main_policy.md)


def _by_id(rows):
    return {r.id: r for r in rows}


class CorpusWorld(World):
    """The journey world, filled from the supplied file."""

    def __init__(self) -> None:
        super().__init__()
        plans = _by_id(SEED.plans)
        self.bills = {
            b.id: {
                "id": b.id,
                "customer": b.customer_id,
                "amount_due": b.amount_due,
                "due_date": b.due_date.isoformat(),
                "status": b.status,
            }
            for b in SEED.bills
        }
        self.lines = {}
        for ln in SEED.lines:
            plan = plans[ln.plan_id]
            self.lines.setdefault(ln.customer_id, []).append(
                {
                    "id": ln.id,
                    "phone_number": ln.phone_number,
                    "status": ln.status,
                    "data_used_mb": ln.data_used_mb,
                    "data_limit_mb": plan.data_limit_mb,
                    "plan_name": plan.name,
                    "monthly_price": plan.monthly_price,
                    "contract_end_date": str(ln.contract_end_date),
                }
            )


@pytest.fixture
def env_for(monkeypatch):
    monkeypatch.setattr(test_journeys, "World", CorpusWorld)

    def make(mode: FaultMode = FaultMode.APPROVE) -> test_journeys.Env:
        return test_journeys.Env(monkeypatch, mode)

    return make


# --- the data ---------------------------------------------------------------------------------
def test_every_record_in_the_supplied_file_is_loaded():
    counts = {name: len(getattr(SEED, name)) for name in type(SEED).model_fields}
    assert counts == {
        "customers": 4,
        "plans": 5,
        "payment_methods": 4,
        "lines": 9,
        "devices": 9,
        "bills": 6,
        "bill_line_items": 12,
    }


def test_the_supplied_statuses_units_and_links_survive_the_conversion():
    assert {b.id: b.status for b in SEED.bills} == {
        "B1001": "PAID",
        "B1002": "ISSUED",
        "B1003": "DRAFT",
        "B1004": "PAID",
        "B1005": "OVERDUE",
        "B1006": "DISPUTED",
    }
    lines, plans, devices = _by_id(SEED.lines), _by_id(SEED.plans), _by_id(SEED.devices)
    assert lines["L1001"].data_used_mb == 3200 and plans["P1001"].data_limit_mb == 5000
    assert plans["P1002"].refuel_price_per_gb == Decimal("2.0")
    assert devices["D1004"].line_id == "L1004" and lines["L1004"].customer_id == "C1002"
    assert lines["L1003"].status == "SUSPENDED" and lines["L1002"].roaming_enabled is True
    assert _by_id(SEED.customers)["C1004"].account_status == "PENDING_VERIFICATION"


def test_each_bills_line_items_add_up_to_its_total():
    for bill in SEED.bills:
        total = sum((i.amount for i in SEED.bill_line_items if i.bill_id == bill.id), Decimal(0))
        assert total == bill.amount_due, bill.id


def test_no_postal_address_is_imported():
    assert "Oak Ave" not in SEED.model_dump_json()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('status = "Overdue"', 'status = "Late"'),  # an unknown status
        ('bill_ids = ["B1004", "B1005"]', 'bill_ids = ["B1004"]'),  # nobody lists the bill
        ('line_ids = ["L1009"]', "line_ids = []"),  # a line with no owner
        (  # one device on two lines
            'plan_id = "P1002"\ndevice_id = "D1009"',
            'plan_id = "P1002"\ndevice_id = "D1008"',
        ),
    ],
)
def test_inconsistent_data_stops_the_load_instead_of_being_guessed(old, new):
    assert TEXT.count(old) == 1
    with pytest.raises(SeedError):
        parse_seed(TEXT.replace(old, new))


def test_a_missing_required_field_is_reported():
    with pytest.raises(CorpusError, match="missing"):
        convert_corpus({"customers": [{}], "plans": []})


# --- dates: the record wins, and the policy fixes "today" ---------------------------------------
def test_b1005_is_overdue_by_its_record_although_its_due_date_has_not_passed():
    bill = _by_id(SEED.bills)["B1005"]
    assert bill.due_date > POLICY_DAY  # the conflict in the supplied data
    assert is_payable(bill.status)
    assert is_overdue(bill.due_date, BillStatus(bill.status), POLICY_DAY)  # still blocks a lift


def test_only_a_recorded_overdue_bill_is_payable():
    assert [s.value for s in BillStatus if is_payable(s)] == ["OVERDUE"]


def test_on_the_policy_date_the_suspended_lines_contracts_have_not_ended(monkeypatch):
    monkeypatch.setenv("BUSINESS_DATE", "2025-02-25")
    assert business_today() == POLICY_DAY
    for line_id in ("L1003", "L1009"):
        assert _by_id(SEED.lines)[line_id].contract_end_date > business_today()
    monkeypatch.delenv("BUSINESS_DATE")
    assert business_today() == date.today()


# --- the assignment's payment case --------------------------------------------------------------
async def test_c1002_pays_overdue_bill_b1005_after_confirming(env_for):
    env = env_for()
    cfg = env.cfg("C1002")
    paused = await env.graph.ainvoke(env.state("C1002", "I want to pay my overdue bill"), cfg)
    action = paused["__interrupt__"][0].value["action"]
    assert (action["bill_id"], Decimal(action["amount"])) == ("B1005", Decimal("425"))
    assert env.world.bills["B1005"]["status"] == "OVERDUE" and env.gateway.charges_applied == 0

    done = await env.graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert env.world.bills["B1005"]["status"] == "PAID" and env.gateway.charges_applied == 1
    assert "successful" in done["reply"]
    assert env.world.bills["B1004"]["status"] == "PAID"  # untouched


async def test_a_declined_payment_returns_b1005_to_overdue_without_claiming_success(env_for):
    env = env_for(FaultMode.DECLINE)
    cfg = env.cfg("C1002")
    await env.graph.ainvoke(env.state("C1002", "pay my bill"), cfg)
    out = await env.graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert env.world.bills["B1005"]["status"] == "OVERDUE" and "declined" in out["reply"].lower()
    assert "successful" not in out["reply"].lower() and len(env.pay_calls) == 1


async def test_an_immediate_retry_after_a_decline_is_held_back_by_the_velocity_check(env_for):
    """Known limitation: the evaluator counts EVERY attempt in the last 10 minutes, declined ones
    included. No second charge is attempted until that window has passed."""
    env = env_for(FaultMode.DECLINE)
    cfg = env.cfg("C1002")
    await env.graph.ainvoke(env.state("C1002", "pay my bill"), cfg)
    await env.graph.ainvoke(Command(resume={"approved": True}), cfg)
    again = await env.graph.ainvoke(env.state("C1002", "pay my bill"), cfg)
    assert "__interrupt__" not in again and len(env.pay_calls) == 1
    assert "recent payment" in again["reply"].lower()


async def test_an_unknown_payment_outcome_keeps_b1005_awaiting_and_offers_no_second_payment(
    env_for,
):
    env = env_for(FaultMode.TIMEOUT)
    cfg = env.cfg("C1002")
    await env.graph.ainvoke(env.state("C1002", "pay my bill"), cfg)
    await env.graph.ainvoke(Command(resume={"approved": True}), cfg)
    assert env.world.bills["B1005"]["status"] == "AWAITING_PAYMENT"
    again = await env.graph.ainvoke(env.state("C1002", "pay my bill"), cfg)
    assert "__interrupt__" not in again
    assert again["reply"] == roles.PAYMENT_IN_PROGRESS_REPLY


@pytest.mark.parametrize(
    "customer",
    [
        "C1001",  # one bill paid, one issued, one draft: none overdue
        "C1003",  # disputed
        "C1004",  # no bills
    ],
)
async def test_a_customer_with_no_overdue_bill_is_never_offered_a_payment(env_for, customer):
    env = env_for()
    out = await env.graph.ainvoke(env.state(customer, "pay my bill"), env.cfg(customer))
    assert "__interrupt__" not in out and out["reply"] == roles.NO_OVERDUE_BILL_REPLY
    assert env.gateway.charges_applied == 0


@pytest.mark.parametrize(
    ("customer", "bill"),
    [("C1001", "B1001"), ("C1001", "B1002"), ("C1001", "B1003"), ("C1003", "B1006")],
)
async def test_the_ledger_itself_refuses_every_bill_that_is_not_overdue(env_for, customer, bill):
    """Even if a proposal and the evaluator were both wrong, the locked transaction says no."""
    env = env_for()
    amount = env.world.bills[bill]["amount_due"] or Decimal(1)
    before = env.world.bills[bill]["status"]
    # a paid bill is a duplicate payment; every other status is simply not payable
    with pytest.raises((ConflictError, DuplicateOperation)):
        await payments.pay_bill(
            FakePool(),
            SimulatedGateway(),
            customer_id=customer,
            bill_id=bill,
            amount=amount,
            idempotency_key=f"k-{bill}",
        )
    assert env.world.bills[bill]["status"] == before and not env.world.tx


@pytest.mark.parametrize("bill_status", ["ISSUED", "DRAFT", "DISPUTED", "PAID"])
async def test_the_evaluator_rejects_a_proposal_for_a_bill_that_is_not_overdue(bill_status):
    async def get_bills(**_):
        bills = [{"id": "B", "amount_due": "10", "due_date": "d", "status": bill_status}]
        return {"ok": True, "bills": bills}

    harness = Harness("evaluator", roles.EVALUATOR_TOOLS, StaticRegistry({"get_bills": get_bills}))
    request = AgentRequest(
        customer_id="C1001",
        session_id="S",
        identity_verified=True,
        context={"proposed_action": {"type": "pay_bill", "bill_id": "B", "amount": "10"}},
    )
    assert (await roles.evaluator(request, harness)).verdict == "reject"


async def test_billing_facts_never_present_a_draft_bill_as_owed(env_for):
    env = env_for()
    out = await env.graph.ainvoke(env.state("C1001", "what do I owe"), env.cfg("C1001"))
    assert "B1002" in out["reply"] and "B1003" not in out["reply"]
