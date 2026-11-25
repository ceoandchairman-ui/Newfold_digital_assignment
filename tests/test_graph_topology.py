"""The edge contract of backlog E4-10: the compiled graph has exactly these nodes and edges, and each
intent reaches only its own branch."""

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from support_assistant.contracts import AgentRequest, AgentResponse, EvidenceRef
from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.orchestrator.graph import build_graph

# The main graph: three agents (each a subgraph) and the plain-code gates around them.
NODES = {
    "auth_gate", "deny", "support_agent", "troubleshoot_agent", "eval_agent", "approve", "pay",
    "egress",
}  # fmt: skip
EDGES = {
    ("auth_gate", "deny"), ("auth_gate", "support_agent"),
    ("support_agent", "troubleshoot_agent"), ("support_agent", "eval_agent"),
    ("support_agent", "egress"),
    ("troubleshoot_agent", "eval_agent"), ("troubleshoot_agent", "egress"),
    ("eval_agent", "approve"), ("eval_agent", "egress"),
    ("approve", "pay"), ("approve", "egress"),
    ("pay", "egress"), ("deny", "egress"),
}  # fmt: skip
# Inside the agents (LangGraph names a subgraph's nodes "<agent>:<node>").
SUPPORT_NODES = {"route", "normal_resp_gen", "instruct_gen", "billing_gen"}
SUPPORT_EDGES = {
    ("route", "normal_resp_gen"), ("route", "billing_gen"), ("route", "instruct_gen"),
}  # fmt: skip
TROUBLE_NODES = {"diagnose"}
REF = EvidenceRef(store="company_knowledge", id=7, content_hash="h")


class Scripted:
    """Answers the router with a chosen route and records which agent modes were called."""

    def __init__(
        self, route: str | None, *, reply_check: str = "approve", route_down: bool = False
    ) -> None:
        self.route, self.reply_check, self.route_down = route, reply_check, route_down
        self.calls: list[tuple[str, str]] = []
        self.contexts: list[dict[str, Any]] = []

    async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
        mode = str(req.context.get("mode", "-"))
        self.calls.append((agent, mode))
        self.contexts.append(dict(req.context))
        if mode == "route":
            if self.route_down:
                raise ExternalServiceUnavailable("router down")
            return AgentResponse(route=self.route)  # type: ignore[arg-type]
        if agent == "evaluator":
            return AgentResponse(verdict=self.reply_check)  # type: ignore[arg-type]
        refs = [REF] if agent == "troubleshooter" else []
        return AgentResponse(reply=f"{agent}:{mode} Source: x", evidence=refs)


def _state(hint: str | None = None) -> dict[str, Any]:
    return {
        "customer_id": "C1",
        "session_id": "S",
        "identity_verified": True,
        "messages": [{"role": "user", "content": "hello"}],
        **({"route_hint": hint} if hint else {}),
        "proposed_action": None,
        "eval_attempts": 0,
        "reply": "",
        "escalated": False,
    }


async def _pay(*a: Any) -> dict[str, Any]:
    raise AssertionError("must not pay")


async def _run(client: Scripted, state: dict[str, Any]) -> dict[str, Any]:
    g = build_graph(client, _pay, MemorySaver())
    return await g.ainvoke(state, {"configurable": {"thread_id": "t"}})  # type: ignore[no-any-return]


def _graph(xray: bool = False):
    return build_graph(Scripted("NORMAL"), _pay, MemorySaver()).get_graph(xray=xray)


def test_the_main_graph_has_exactly_the_three_agents_and_the_code_around_them():
    g = _graph()
    assert set(g.nodes) - {"__start__", "__end__"} == NODES
    edges = {(e.source, e.target) for e in g.edges}
    assert {(s, t) for s, t in edges if s != "__start__" and t != "__end__"} == EDGES
    assert ("__start__", "auth_gate") in edges and ("egress", "__end__") in edges


def test_the_support_agent_is_one_agent_made_of_a_router_and_three_generators():
    g = _graph(xray=True)
    inner = {n.split(":", 1)[1] for n in g.nodes if n.startswith("support_agent:")}
    assert inner - {"__end__"} == SUPPORT_NODES
    edges = {
        (e.source.split(":")[-1], e.target.split(":")[-1])
        for e in g.edges
        if e.source.startswith("support_agent:") and e.target.startswith("support_agent:")
    }
    assert SUPPORT_EDGES <= edges
    # every generator ends the agent: none of them calls another generator
    assert not {(a, b) for a, b in edges if a in SUPPORT_NODES - {"route"} and b != "__end__"}


def test_neither_agent_has_a_retrieval_node():
    g = _graph(xray=True)
    inner = {n.split(":", 1)[1] for n in g.nodes if n.startswith("troubleshoot_agent:")}
    assert inner - {"__end__"} == TROUBLE_NODES
    assert not [n for n in g.nodes if "retrieve" in n]


def test_there_is_no_mcp_hub_node_the_one_write_is_the_pay_step():
    assert "mcp_client_routing_node" not in _graph(xray=True).nodes and "pay" in _graph().nodes


def test_the_unverified_path_reaches_no_agent_or_tool():
    g = _graph()
    assert {e.target for e in g.edges if e.source == "deny"} == {"egress"}


async def test_normal_goes_to_the_simple_generator_only():
    c = Scripted("NORMAL")
    out = await _run(c, _state())
    assert c.calls == [("director", "route"), ("director", "normal")]
    assert out["reply"].startswith("director:normal")


async def test_instruct_goes_straight_to_the_reasoning_generator():
    c = Scripted("INSTRUCT")
    out = await _run(c, _state())
    assert c.calls == [("director", "route"), ("director", "instruct")]
    assert out["reply"].startswith("director:instruct")


async def test_billing_goes_to_the_task_generator_only():
    c = Scripted("BILLING")
    out = await _run(c, _state())
    assert c.calls == [("director", "route"), ("director", "billing")]
    assert out["reply"].startswith("director:billing")


async def test_the_eval_agent_gets_the_references_the_troubleshooter_used_and_no_text():
    c = Scripted("TROUBLESHOOT")
    out = await _run(c, _state())
    assert c.calls == [
        ("director", "route"),
        ("troubleshooter", "-"),
        ("evaluator", "reply_check"),
    ]
    assert c.contexts[-1]["evidence_refs"] == [REF.model_dump()]
    assert out["evidence_refs"] == [REF.model_dump()] and "kb_chunks" not in out
    assert out["reply"].startswith("troubleshooter")


async def test_references_never_carry_over_into_the_next_turn():
    class Down(Scripted):
        async def run(self, agent: str, req: AgentRequest) -> AgentResponse:
            if agent == "troubleshooter":
                raise ExternalServiceUnavailable("down")
            return await super().run(agent, req)

    g = build_graph(Down("TROUBLESHOOT", reply_check="reject"), _pay, MemorySaver())
    cfg = {"configurable": {"thread_id": "carry"}}
    stale = {**_state(), "evidence_refs": [REF.model_dump()]}
    out = await g.ainvoke(stale, cfg)
    assert out["evidence_refs"] == []


async def test_a_reply_that_fails_the_eval_agent_is_escalated_not_shown():
    out = await _run(Scripted("TROUBLESHOOT", reply_check="reject"), _state())
    assert out["escalated"] and out["escalation_reason"] == "INVARIANT_VIOLATION"
    assert "Source: x" not in out["reply"]


async def test_when_the_router_is_down_the_backend_hint_decides_and_unsure_means_normal():
    c = Scripted(None, route_down=True)
    await _run(c, _state(hint="TROUBLESHOOT"))
    assert ("troubleshooter", "-") in c.calls
    c2 = Scripted(None, route_down=True)
    await _run(c2, _state())
    assert c2.calls == [("director", "route"), ("director", "normal")]
