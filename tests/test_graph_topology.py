"""The edge contract of backlog E4-10: the compiled graph has exactly these nodes and edges, and each
intent reaches only its own branch."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import MemorySaver

from support_assistant.contracts import AgentRequest, AgentResponse
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.orchestrator.graph import build_graph

NODES = {
    "auth_gate",
    "deny",
    "support_agent",
    "normal_resp_gen",
    "instruct_gen",
    "billing_gen",
    "retrieve",
    "troubleshoot_agent",
    "eval_agent",
    "approve",
    "mcp_client_routing_node",
    "egress",
}
EDGES = {
    ("auth_gate", "deny"),
    ("auth_gate", "support_agent"),
    ("support_agent", "normal_resp_gen"),
    ("support_agent", "billing_gen"),
    ("support_agent", "retrieve"),
    ("retrieve", "troubleshoot_agent"),
    ("retrieve", "instruct_gen"),
    ("billing_gen", "eval_agent"),
    ("billing_gen", "egress"),
    ("troubleshoot_agent", "eval_agent"),
    ("troubleshoot_agent", "egress"),
    ("eval_agent", "approve"),
    ("eval_agent", "egress"),
    ("approve", "mcp_client_routing_node"),
    ("approve", "egress"),
    ("normal_resp_gen", "egress"),
    ("instruct_gen", "egress"),
    ("mcp_client_routing_node", "egress"),
    ("deny", "egress"),
}


class Retriever(BaseRetriever):
    seen: list[str] = []

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return []

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        self.seen.append(query)
        return [Document(page_content="Reset the APN.", metadata={"section_id": "4"})]


class Scripted:
    """Answers the router with a chosen route and records which agent modes were called."""

    def __init__(
        self, route: str | None, *, reply_check: str = "approve", route_down: bool = False
    ) -> None:
        self.route, self.reply_check, self.route_down = route, reply_check, route_down
        self.calls: list[tuple[str, str]] = []

    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        mode = str(req.context.get("mode", "-"))
        self.calls.append((agent, mode))
        if mode == "route":
            if self.route_down:
                raise ExternalServiceUnavailable("router down")
            return AgentResponse(route=self.route)  # type: ignore[arg-type]
        if agent == "evaluator":
            return AgentResponse(verdict=self.reply_check)  # type: ignore[arg-type]
        return AgentResponse(reply=f"{agent}:{mode} Source: x")


def _state(route: str | None = None, hint: str | None = None) -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": "S",
        "identity_verified": True,
        "messages": [{"role": "user", "content": "hello"}],
        **({"route": route} if route else {}),
        **({"route_hint": hint} if hint else {}),
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
    }


async def _pay(*a: Any) -> dict[str, Any]:
    raise AssertionError("must not pay")


async def _run(client: Scripted, state: dict[str, Any], retriever: Any = None) -> dict[str, Any]:
    g = build_graph(client, _pay, MemorySaver(), retriever=retriever)
    return await g.ainvoke(state, {"configurable": {"thread_id": "t"}})  # type: ignore[no-any-return]


def test_the_compiled_graph_has_exactly_the_contracted_nodes_and_edges():
    g = build_graph(Scripted("NORMAL"), _pay, MemorySaver()).get_graph()
    assert set(g.nodes) - {"__start__", "__end__"} == NODES
    edges = {(e.source, e.target) for e in g.edges}
    inner = {(s, t) for s, t in edges if "__start__" != s and "__end__" != t}
    assert inner == EDGES - {("egress", "__end__")}
    assert ("__start__", "auth_gate") in edges and ("egress", "__end__") in edges


def test_the_unverified_path_reaches_no_agent_tool_or_retriever():
    g = build_graph(Scripted("NORMAL"), _pay, MemorySaver()).get_graph()
    assert {e.target for e in g.edges if e.source == "deny"} == {"egress"}


async def test_normal_goes_to_the_simple_generator_only():
    c = Scripted("NORMAL")
    out = await _run(c, _state())
    assert c.calls == [("director", "route"), ("director", "normal")]
    assert out["reply"].startswith("director:normal")


async def test_instruct_retrieves_policy_then_reasons():
    c, r = Scripted("INSTRUCT"), Retriever(seen=[])
    out = await _run(c, _state(), r)
    assert c.calls == [("director", "route"), ("director", "instruct")] and r.seen == ["hello"]
    assert out["reply"].startswith("director:instruct")


async def test_billing_goes_to_the_task_generator_and_never_retrieves():
    c, r = Scripted("BILLING"), Retriever(seen=[])
    out = await _run(c, _state(), r)
    assert c.calls == [("director", "route"), ("director", "billing")] and r.seen == []
    assert out["reply"].startswith("director:billing")


async def test_troubleshoot_retrieves_then_the_eval_agent_checks_the_reply():
    c, r = Scripted("TROUBLESHOOT"), Retriever(seen=[])
    out = await _run(c, _state(), r)
    assert c.calls == [
        ("director", "route"),
        ("troubleshooter", "-"),
        ("evaluator", "reply_check"),
    ]
    assert r.seen == ["hello"] and out["reply"].startswith("troubleshooter")


async def test_a_reply_that_fails_the_eval_agent_is_escalated_not_shown():
    out = await _run(Scripted("TROUBLESHOOT", reply_check="reject"), _state(), Retriever(seen=[]))
    assert out["escalated"] and out["escalation_reason"] == "INVARIANT_VIOLATION"
    assert "Source: x" not in out["reply"]


async def test_when_the_router_is_down_the_backend_hint_decides_and_unsure_means_normal():
    c = Scripted(None, route_down=True)
    await _run(c, _state(hint="TROUBLESHOOT"), Retriever(seen=[]))
    assert ("troubleshooter", "-") in c.calls
    c2 = Scripted(None, route_down=True)
    await _run(c2, _state())
    assert c2.calls == [("director", "route"), ("director", "normal")]


async def test_the_legacy_lowercase_route_still_selects_a_branch():
    c = Scripted(None)  # the agent gives no route (an older agent)
    await _run(c, _state(route="billing"))
    assert c.calls[-1] == ("director", "billing")
