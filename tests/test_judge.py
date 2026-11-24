"""The eval_agent's LLM-as-judge: it can only object, and never by failing."""

import asyncio
import json

import pytest

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

    async def respond(self, instructions, facts, user_message):
        self.seen.append((instructions, facts, user_message))
        if self.hang:
            await asyncio.sleep(10)
        if self.boom:
            raise RuntimeError("down")
        return self.answer


def _verdict(supported: bool, confidence: float) -> str:
    return json.dumps({"supported": supported, "confidence": confidence})


def _req(reply=REPLY, intent="TROUBLESHOOT", chunks=CHUNKS) -> AgentRequest:
    return AgentRequest(
        customer_id="C1",
        session_id="S",
        identity_verified=True,
        messages=[],
        context={"mode": "reply_check", "reply": reply, "intent": intent, "kb_chunks": chunks},
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
    res = await roles.evaluator(_req(), _h(), judge=Judge(_verdict(False, 0.95)))
    assert res.verdict == "reject"


async def test_a_supported_reply_is_approved():
    res = await roles.evaluator(_req(), _h(), judge=Judge(_verdict(True, 0.95)))
    assert res.verdict == "approve"


async def test_a_broken_judge_never_blocks_a_reply():
    for j in (Judge(boom=True), Judge("garbage"), Judge(hang=True)):
        res = await roles.evaluator(_req(), _h(), judge=j, judge_timeout_s=0.05)
        assert res.verdict == "approve"


async def test_the_rules_run_first_so_a_reply_without_a_citation_needs_no_judge():
    j = Judge(_verdict(True, 1.0))
    res = await roles.evaluator(_req(reply="Reset the APN."), _h(), judge=j)
    assert res.verdict == "reject" and j.seen == []


@pytest.mark.parametrize(
    ("intent", "chunks", "reply"),
    [
        ("INSTRUCT", CHUNKS, REPLY),  # the judge is for troubleshooting replies
        ("TROUBLESHOOT", [], REPLY),  # nothing to ground against
        ("TROUBLESHOOT", CHUNKS, CLOSING_REPLY),  # the fixed closing text is not a claim
    ],
)
async def test_the_judge_is_skipped_when_there_is_nothing_to_check(intent, chunks, reply):
    j = Judge(_verdict(False, 1.0))
    res = await roles.evaluator(_req(reply=reply, intent=intent, chunks=chunks), _h(), judge=j)
    assert res.verdict == "approve" and j.seen == []


async def test_without_a_judge_the_rules_alone_decide():
    assert (await roles.evaluator(_req(), _h())).verdict == "approve"


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
