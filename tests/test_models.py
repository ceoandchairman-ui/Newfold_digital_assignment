"""One model per job: the role-to-deployment map, its defaults and the reasoning options."""

import pytest
from pydantic import ValidationError

from support_assistant.agents.llm import (
    MafLLM,
    TemplateLLM,
    build_role_llm,
    build_support_llms,
    reasoning_options,
)
from support_assistant.config import Settings
from support_assistant.models import Role, model_for


def _settings(**kw) -> Settings:
    return Settings(database_url="postgresql://x", redis_url="redis://x", **kw)


def test_every_role_has_its_own_deployment_by_default():
    s = _settings()
    names = {role: model_for(s, role).deployment for role in Role}
    assert names == {
        Role.CHAT: "chat-support",
        Role.PLANNER: "chat-planner",
        Role.REWRITE: "chat-rewrite",
        Role.TROUBLESHOOTER: "chat-troubleshooter",
        Role.JUDGE: "chat-judge",
    }
    assert len(set(names.values())) == len(Role)  # no two jobs share a deployment


def test_default_effort_follows_the_job_cheap_and_fast_where_latency_matters():
    s = _settings()
    effort = {role: model_for(s, role).effort for role in Role}
    assert effort[Role.REWRITE] == "minimal"  # a 64-token job on the customer's wait
    assert effort[Role.CHAT] == "low" and effort[Role.JUDGE] == "low"
    assert effort[Role.PLANNER] == "medium" and effort[Role.TROUBLESHOOTER] == "medium"


def test_a_model_swap_is_configuration_only(monkeypatch):
    monkeypatch.setenv("LLM_PLANNER_DEPLOYMENT", "chat-planner-v2")
    monkeypatch.setenv("LLM_PLANNER_EFFORT", "high")
    spec = model_for(_settings(), Role.PLANNER)
    assert (spec.deployment, spec.effort) == ("chat-planner-v2", "high")


def test_a_model_without_reasoning_effort_can_be_configured(monkeypatch):
    monkeypatch.setenv("LLM_CHAT_EFFORT", "")
    # an empty value is rejected rather than silently meaning "default": say what you mean
    with pytest.raises(ValidationError):
        _settings()
    assert model_for(_settings(llm_chat_effort=None), Role.CHAT).effort is None


def test_an_unknown_effort_is_rejected_at_startup():
    with pytest.raises(ValidationError):
        _settings(llm_planner_effort="extreme")


def test_the_rewrite_and_judge_have_short_timeouts_and_the_judge_can_be_turned_off():
    s = _settings()
    assert s.rewrite_timeout_seconds <= 2 and s.judge_timeout_seconds <= 5
    assert s.online_judge_enabled is True
    assert _settings(online_judge_enabled=False).online_judge_enabled is False


# --- reasoning options ------------------------------------------------------------------------
def test_reasoning_options_carry_the_effort_and_never_a_temperature():
    assert reasoning_options("medium") == {"reasoning": {"effort": "medium"}}
    assert reasoning_options(None) is None
    assert "temperature" not in str(reasoning_options("low"))


# --- building the models ----------------------------------------------------------------------
def test_without_an_openai_endpoint_every_role_is_a_template_not_a_model():
    s = _settings()
    assert all(isinstance(build_role_llm(s, r), TemplateLLM) for r in Role)


def test_the_support_agent_has_two_models_chat_and_planner():
    llms = build_support_llms(_settings())
    assert llms["route"] is llms["normal"]  # the conversation model
    assert llms["instruct"] is llms["billing"]  # the reasoning and task model
    assert llms["route"] is not llms["instruct"]


def test_with_an_endpoint_each_role_gets_its_own_deployment_and_effort(monkeypatch):
    seen: list[tuple[str, str | None]] = []

    def fake_init(self, settings, deployment, effort=None):
        seen.append((deployment, effort))

    monkeypatch.setattr(MafLLM, "__init__", fake_init)
    s = _settings(azure_openai_endpoint="https://oai.example")
    build_support_llms(s)
    assert seen == [("chat-support", "low"), ("chat-planner", "medium")]
