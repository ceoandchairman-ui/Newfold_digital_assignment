"""The LangChain retriever, and the graph's auth_gate and retrieve nodes."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import MemorySaver

from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.embeddings import FakeEmbedder
from support_assistant.orchestrator.graph import build_graph


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


def _state(verified: bool = True, route: str = "TROUBLESHOOT") -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": "S1",
        "identity_verified": verified,
        "route_hint": route,
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
