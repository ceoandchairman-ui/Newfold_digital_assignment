"""Tier routing, the LangChain retriever, and the graph's auth_gate and retrieve nodes."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import MemorySaver

from support_assistant.agents.llm import Tier, TieredLLM, pick_tier
from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.embeddings import FakeEmbedder
from support_assistant.orchestrator.graph import build_graph


# --- tier routing -----------------------------------------------------------------------
class Recorder:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name, self.fail, self.calls = name, fail, 0

    async def respond(self, instructions: str, facts: str, user_message: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return self.name


def test_easy_turns_pick_small_and_hard_turns_pick_large():
    assert pick_tier("No unpaid bills.", "what is my plan") is Tier.SMALL
    assert pick_tier("bill B1 overdue", "hi") is Tier.LARGE
    assert pick_tier("x" * 700, "hi") is Tier.LARGE
    assert pick_tier("Proposed: pay bill B1", "pay it") is Tier.LARGE


async def test_router_uses_the_cheap_model_for_easy_turns():
    small, large = Recorder("small"), Recorder("large")
    llm = TieredLLM(small, large)
    assert await llm.respond("i", "No unpaid bills.", "what is my plan") == "small"
    assert (small.calls, large.calls) == (1, 0)


async def test_router_uses_large_for_hard_and_falls_back_when_small_fails():
    small, large = Recorder("small"), Recorder("large")
    llm = TieredLLM(small, large)
    assert await llm.respond("i", "bill B1 overdue", "why") == "large"
    assert small.calls == 0

    broken_small = Recorder("small", fail=True)
    llm2 = TieredLLM(broken_small, large)
    assert await llm2.respond("i", "No unpaid bills.", "plan?") == "large"
    assert llm2.last_tier is Tier.LARGE


# --- optional reasoning tier --------------------------------------------------------------
def test_reasoning_tier_is_never_picked_unless_a_reasoning_model_is_configured():
    assert pick_tier("customer disputes bill B1", "refund me") is Tier.LARGE
    assert (
        pick_tier("customer disputes bill B1", "refund me", reasoning_available=True)
        is Tier.REASONING
    )
    # ordinary hard turns (overdue, long) stay on the large model even when reasoning exists
    assert pick_tier("bill B1 overdue", "hi", reasoning_available=True) is Tier.LARGE
    assert pick_tier("No unpaid bills.", "plan?", reasoning_available=True) is Tier.SMALL


async def test_reasoning_model_answers_disputes_and_large_covers_for_it_when_it_fails():
    small, large, reasoning = Recorder("small"), Recorder("large"), Recorder("reasoning")
    llm = TieredLLM(small, large, reasoning)
    assert await llm.respond("i", "bill B1", "I want to dispute this charge") == "reasoning"
    assert (small.calls, large.calls) == (0, 0)

    llm2 = TieredLLM(small, large, Recorder("reasoning", fail=True))
    assert await llm2.respond("i", "bill B1", "refund please") == "large"
    assert llm2.last_tier is Tier.LARGE


async def test_without_a_reasoning_model_disputes_use_large_as_before():
    small, large = Recorder("small"), Recorder("large")
    assert await TieredLLM(small, large).respond("i", "bill B1", "dispute this") == "large"


def test_query_rewrite_runs_on_the_small_model_with_a_short_timeout_by_default():
    from support_assistant.config import Settings

    fields = Settings.model_fields
    assert fields["azure_openai_rewrite_deployment"].default == "chat-small"
    assert fields["rewrite_timeout_seconds"].default <= 2
    assert fields["azure_openai_reasoning_deployment"].default is None


# --- graph: auth_gate and retrieve node -----------------------------------------------------
class FakeRetriever(BaseRetriever):
    hits: list[Document] = []
    fail: bool = False
    seen: list[str] = []

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        self.seen.append(query)
        if self.fail:
            raise RuntimeError("db down")
        return self.hits

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise NotImplementedError


async def test_real_hybrid_retriever_works_through_ainvoke(monkeypatch):
    """Regression: the sync method's signature decides whether run_manager is passed (LangChain)."""
    from support_assistant import retrievers
    from support_assistant.services.retrieval import Chunk

    async def fake_search(pool, embedder, query, **kw):
        assert kw["top_k"] == 5 and kw["candidates"] == 20
        return [Chunk(1, "2.0", "G > No signal", "Toggle airplane mode.", 0.03)]

    monkeypatch.setattr(retrievers, "search_kb", fake_search)
    # Built through the normal constructor, as the orchestrator does, not model_construct.
    retr = retrievers.HybridPgRetriever(
        pool=object(), embedder=FakeEmbedder(dim=4), top_k=5, candidates=20
    )
    docs = await retr.ainvoke("no signal")
    assert docs[0].page_content == "Toggle airplane mode."
    assert docs[0].metadata["heading_path"] == "G > No signal"


class CapturingClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, AgentRequest]] = []

    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        self.requests.append((agent, req))
        return AgentResponse(reply="ok")


def _state(verified: bool = True, route: str = "troubleshoot") -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": "S1",
        "identity_verified": verified,
        "route": route,
        "messages": [{"role": "user", "content": "no signal"}],
        "search_query": "mobile no signal",
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
    }


async def _no_pay(*a: Any) -> dict[str, Any]:
    raise AssertionError("must not pay")


async def test_unverified_session_never_reaches_any_agent_or_retriever():
    client, retr = CapturingClient(), FakeRetriever(seen=[])
    graph = build_graph(client, _no_pay, MemorySaver(), retriever=retr)
    out = await graph.ainvoke(_state(verified=False), {"configurable": {"thread_id": "A"}})
    assert client.requests == [] and retr.seen == []
    assert "verify" in out["reply"].lower()


async def test_retrieve_node_feeds_chunks_to_the_troubleshooter():
    doc = Document(
        page_content="Toggle airplane mode.",
        metadata={"section_id": "2.0", "heading_path": "G > No signal"},
    )
    client, retr = CapturingClient(), FakeRetriever(hits=[doc], seen=[])
    graph = build_graph(client, _no_pay, MemorySaver(), retriever=retr)
    await graph.ainvoke(_state(), {"configurable": {"thread_id": "B"}})
    assert retr.seen == ["mobile no signal"]  # the rewritten query, not the raw message
    (req,) = [r for a, r in client.requests if a == "troubleshooter"]
    assert req.context["kb_chunks"][0]["content"] == "Toggle airplane mode."


async def test_retriever_failure_degrades_instead_of_blocking():
    client, retr = CapturingClient(), FakeRetriever(fail=True, seen=[])
    graph = build_graph(client, _no_pay, MemorySaver(), retriever=retr)
    out = await graph.ainvoke(_state(), {"configurable": {"thread_id": "C"}})
    (req,) = [r for a, r in client.requests if a == "troubleshooter"]
    assert "kb_chunks" not in req.context  # the agent searches for itself
    assert out["reply"] == "ok"
