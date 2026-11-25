"""The eval_agent's LLM-as-judge: it can only object, and never by failing."""

import asyncio
import json

import pytest

from fakes import FakeRetrieval
from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.agents.judge import (
    JUDGE_MIN_CONFIDENCE,
    Verdict,
    judge_grounded,
    objects,
)
from support_assistant.contracts import AgentRequest
from support_assistant.domain.intent import CLOSING_REPLY

CHUNKS = [{"section_id": "4", "content": "Reset the APN to the carrier default."}]
REPLY = "Reset the APN settings.\n\nSource: Guide > APN"


class Judge:
    def __init__(self, answer="", *, boom=False, hang=False):
        self.answer, self.boom, self.hang, self.seen = answer, boom, hang, []

    async def respond(self, instructions, facts, user_message, history=""):
        self.seen.append((instructions, facts, user_message))
        if self.hang:
            await asyncio.sleep(10)
        if self.boom:
            raise RuntimeError("down")
        return self.answer


def _verdict(supported: bool, confidence: float) -> str:
    return json.dumps({"supported": supported, "confidence": confidence})


def _kb() -> FakeRetrieval:
    return FakeRetrieval(("4", "Guide > APN", "Reset the APN to the carrier default."))


REFS = [{"store": "company_knowledge", "id": 1, "content_hash": "hash-1"}]


def _req(reply=REPLY, intent="TROUBLESHOOT", refs=REFS) -> AgentRequest:
    return AgentRequest(
        customer_id="C1",
        session_id="S",
        identity_verified=True,
        messages=[],
        context={"mode": "reply_check", "reply": reply, "intent": intent, "evidence_refs": refs},
    )


def _h() -> Harness:
    class R:
        def has(self, t):
            return True

        async def call(self, t, a):
            raise AssertionError("the reply check uses no tool")

    return Harness("evaluator", roles.EVALUATOR_TOOLS, R())


# --- the judge on its own ---------------------------------------------------------------------
async def test_a_confident_unsupported_verdict_is_an_objection():
    v = await judge_grounded(Judge(_verdict(False, 0.9)), REPLY, CHUNKS)
    assert v == Verdict(supported=False, confidence=0.9) and objects(v)


@pytest.mark.parametrize(
    "answer",
    ["not json", '{"supported": "maybe"}', _verdict(False, JUDGE_MIN_CONFIDENCE - 0.1)],
)
async def test_unclear_or_unsure_answers_never_object(answer):
    assert not objects(await judge_grounded(Judge(answer), REPLY, CHUNKS))


async def test_an_outage_or_a_timeout_means_no_objection():
    assert await judge_grounded(Judge(boom=True), REPLY, CHUNKS) is None
    assert await judge_grounded(Judge(hang=True), REPLY, CHUNKS, timeout_s=0.05) is None
    assert not objects(None)


async def test_the_sources_reach_the_judge_as_data_and_the_citation_line_is_removed():
    j = Judge(_verdict(True, 0.9))
    await judge_grounded(j, REPLY, CHUNKS)
    instructions, facts, message = j.seen[0]
    assert "DATA" in instructions and "ignore any instruction" in instructions
    assert "Reset the APN to the carrier default." in facts
    assert "Source:" not in message and "Reset the APN settings." in message


# --- inside the evaluator ---------------------------------------------------------------------
async def test_an_unsupported_reply_is_rejected_by_the_eval_agent():
    res = await roles.evaluator(_req(), _h(), judge=Judge(_verdict(False, 0.95)), retrieval=_kb())
    assert res.verdict == "reject"


async def test_a_supported_reply_is_approved():
    res = await roles.evaluator(_req(), _h(), judge=Judge(_verdict(True, 0.95)), retrieval=_kb())
    assert res.verdict == "approve"


async def test_a_broken_judge_never_blocks_a_reply():
    for j in (Judge(boom=True), Judge("garbage"), Judge(hang=True)):
        res = await roles.evaluator(_req(), _h(), judge=j, judge_timeout_s=0.05, retrieval=_kb())
        assert res.verdict == "approve"


async def test_the_rules_run_first_so_a_reply_without_a_citation_needs_no_judge():
    j = Judge(_verdict(True, 1.0))
    res = await roles.evaluator(_req(reply="Reset the APN."), _h(), judge=j, retrieval=_kb())
    assert res.verdict == "reject" and j.seen == []


@pytest.mark.parametrize(
    ("intent", "reply"),
    [
        ("INSTRUCT", REPLY),  # it also rests on account facts the judge cannot see
        ("TROUBLESHOOT", CLOSING_REPLY),  # the fixed closing text is not a claim
    ],
)
async def test_the_judge_is_skipped_when_there_is_nothing_to_check(intent, reply):
    j, kb = Judge(_verdict(False, 1.0)), _kb()
    res = await roles.evaluator(_req(reply=reply, intent=intent), _h(), judge=j, retrieval=kb)
    assert res.verdict == "approve" and j.seen == [] and kb.fetched == []


async def test_without_a_judge_the_rules_and_the_evidence_check_decide():
    assert (await roles.evaluator(_req(), _h(), retrieval=_kb())).verdict == "approve"


# --- evidence that cannot be checked never passes ------------------------------------------------
@pytest.mark.parametrize(
    ("refs", "retrieval"),
    [
        ([], _kb()),  # no references at all
        ([{"store": "company_knowledge", "id": 1, "content_hash": "old"}], _kb()),  # changed
        ([{"store": "company_knowledge", "id": 9, "content_hash": "hash-9"}], _kb()),  # deleted
        ([*REFS, {"store": "company_knowledge", "id": 9, "content_hash": "x"}], _kb()),  # one gone
        ([{"store": "customer_memory", "id": 1, "content_hash": "hash-1"}], _kb()),  # not a store
        (["nonsense"], _kb()),
        (REFS, FakeRetrieval(down=True)),
        (REFS, None),
    ],
)
async def test_a_reply_whose_evidence_cannot_be_checked_is_rejected(refs, retrieval):
    j = Judge(_verdict(True, 1.0))
    res = await roles.evaluator(_req(refs=refs), _h(), judge=j, retrieval=retrieval)
    assert res.verdict == "reject" and j.seen == []


async def test_a_payment_verdict_never_involves_the_judge():
    j = Judge(_verdict(False, 1.0))
    req = AgentRequest(
        customer_id="C1",
        session_id="S",
        identity_verified=True,
        context={"proposed_action": {"type": "refund"}},
    )
    res = await roles.evaluator(req, _h(), judge=j)
    assert res.verdict == "reject" and j.seen == []
