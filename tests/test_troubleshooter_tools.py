"""The troubleshooter's bounded device-check loop: one validated action at a time, every call through
the harness, a locked SIM escalates with no retry, and a broken loop never costs the turn."""

import json

from support_assistant.agents import roles
from support_assistant.agents.harness import Harness
from support_assistant.contracts import AgentRequest

CHUNKS = [{"section_id": "4", "heading_path": "Guide > APN", "content": "Reset the APN settings."}]


class Registry:
    def __init__(self, devices=("D1",), diagnostics=None, fail=()):
        self.calls: list[tuple[str, dict]] = []
        self.devices, self.fail = list(devices), set(fail)
        self.diagnostics = diagnostics or {"sim_locked": False, "apn": "default"}

    def has(self, tool):
        return True

    async def call(self, tool, args):
        self.calls.append((tool, dict(args)))
        if tool in self.fail:
            return {"ok": False, "message": "down"}
        if tool == "get_devices":
            return {"ok": True, "devices": [{"id": d} for d in self.devices]}
        if tool == "run_diagnostics":
            return {"ok": True, **self.diagnostics}
        return {"ok": True, "done": tool}


class ScriptedLLM:
    """Answers the tool router from a script and everything else with a fixed sentence."""

    def __init__(self, *actions):
        self.actions, self.answers, self.router_calls = list(actions), [], 0

    async def respond(self, instructions, facts, user_message):
        if instructions == roles.TOOL_ROUTER_INSTRUCTIONS:
            self.router_calls += 1
            nxt = self.actions.pop(0) if self.actions else '{"tool": null}'
            return nxt if isinstance(nxt, str) else json.dumps(nxt)
        self.answers.append(facts)
        return "Reset your APN."


def _req(msg="my data does not work") -> AgentRequest:
    return AgentRequest(
        customer_id="C1",
        session_id="S",
        identity_verified=True,
        messages=[{"role": "user", "content": msg}],
        context={"kb_chunks": CHUNKS},
    )


def _run(llm, reg, steps=3):
    h = Harness("troubleshooter", roles.TROUBLESHOOTER_TOOLS, reg)

    async def search(_):
        return []

    return roles.troubleshooter(_req(), h, llm, search, tool_steps=steps)


async def test_it_checks_the_device_first_then_answers_with_what_it_found():
    reg = Registry()
    llm = ScriptedLLM({"tool": "run_diagnostics", "device_id": "D1"}, {"tool": None})
    res = await _run(llm, reg)
    assert [c[0] for c in reg.calls] == ["get_devices", "run_diagnostics"]
    assert reg.calls[1][1] == {
        "device_id": "D1",
        "customer_id": "C1",
    }  # the harness adds the customer
    assert "run_diagnostics on D1" in llm.answers[0] and "Source:" in res.reply


async def test_a_fix_with_a_toggle_needs_an_explicit_on_or_off():
    reg = Registry()
    llm = ScriptedLLM({"tool": "toggle_airplane_mode", "device_id": "D1"})  # no "enabled"
    await _run(llm, reg)
    assert [c[0] for c in reg.calls] == ["get_devices"]  # refused before any call
    reg = Registry()
    llm = ScriptedLLM({"tool": "toggle_airplane_mode", "device_id": "D1", "enabled": False})
    await _run(llm, reg)
    assert reg.calls[-1] == (
        "toggle_airplane_mode",
        {"device_id": "D1", "enabled": False, "customer_id": "C1"},
    )


async def test_a_device_that_is_not_the_customers_is_never_touched():
    reg = Registry(devices=("D1", "D2"))
    await _run(ScriptedLLM({"tool": "reboot_device", "device_id": "D9"}), reg)
    assert [c[0] for c in reg.calls] == ["get_devices"]


async def test_only_device_tools_can_be_chosen_never_a_payment_or_a_message():
    for tool in ("process_payment", "send_sms", "get_bills", "drop_database"):
        reg = Registry()
        await _run(ScriptedLLM({"tool": tool, "device_id": "D1"}), reg)
        assert [c[0] for c in reg.calls] == ["get_devices"], tool


async def test_a_single_device_is_assumed_when_the_model_names_none():
    reg = Registry()
    await _run(ScriptedLLM({"tool": "reset_apn_settings"}), reg)
    assert reg.calls[-1] == ("reset_apn_settings", {"device_id": "D1", "customer_id": "C1"})


async def test_the_loop_is_bounded_and_never_repeats_the_same_action():
    reg = Registry()
    same = {"tool": "run_diagnostics", "device_id": "D1"}
    llm = ScriptedLLM(same, same, same, same, same)
    await _run(llm, reg, steps=3)
    assert [c[0] for c in reg.calls].count("run_diagnostics") == 1  # the repeat ended the loop
    reg = Registry()
    steps = [
        {"tool": t, "device_id": "D1"}
        for t in ("reboot_device", "reset_apn_settings", "reseat_sim_card", "can_send_mms")
    ]
    await _run(ScriptedLLM(*steps), reg, steps=3)
    assert len(reg.calls) == 1 + 3  # get_devices, then at most three actions


async def test_a_locked_sim_escalates_at_once_and_runs_nothing_more():
    reg = Registry(diagnostics={"sim_locked": True})
    llm = ScriptedLLM(
        {"tool": "run_diagnostics", "device_id": "D1"}, {"tool": "reboot_device", "device_id": "D1"}
    )
    res = await _run(llm, reg)
    assert res.escalate is True and res.escalation_reason == "SIM_LOCKED"
    assert [c[0] for c in reg.calls] == ["get_devices", "run_diagnostics"]  # no retry, no reboot
    assert llm.answers == []  # the model is not even asked to word an answer


async def test_garbage_from_the_model_means_no_tool_and_the_answer_still_comes():
    reg = Registry()
    res = await _run(ScriptedLLM("not json at all"), reg)
    assert [c[0] for c in reg.calls] == ["get_devices"] and "Source:" in res.reply


async def test_a_failing_tool_ends_the_loop_but_not_the_turn():
    reg = Registry(fail={"run_diagnostics"})
    res = await _run(ScriptedLLM({"tool": "run_diagnostics", "device_id": "D1"}), reg)
    assert "Source:" in res.reply


async def test_a_customer_with_no_devices_just_gets_the_manual_answer():
    reg = Registry(devices=())
    llm = ScriptedLLM({"tool": "run_diagnostics", "device_id": "D1"})
    res = await _run(llm, reg)
    assert [c[0] for c in reg.calls] == ["get_devices"] and llm.router_calls == 0
    assert "Source:" in res.reply


async def test_with_the_loop_turned_off_no_device_tool_is_called_at_all():
    reg = Registry()
    res = await _run(ScriptedLLM({"tool": "run_diagnostics", "device_id": "D1"}), reg, steps=0)
    assert reg.calls == [] and "Source:" in res.reply


async def test_a_template_model_without_a_real_llm_never_runs_a_tool():
    from support_assistant.agents.llm import TemplateLLM

    reg = Registry()
    res = await _run(TemplateLLM(), reg)
    assert [c[0] for c in reg.calls] == ["get_devices"] and "Source:" in res.reply
