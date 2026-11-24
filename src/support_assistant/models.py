"""Which model does which job. One deployment per role, so any one can change without the others.

    CHAT            support_agent: runs the conversation and answers generic questions
    PLANNER         support_agent: reasoning and task questions (plans, tool use, payment proposals)
    REWRITE         fat backend: turns the customer's message into a search query
    TROUBLESHOOTER  troubleshoot_agent: multi-step diagnosis
    JUDGE           eval_agent: LLM-as-judge groundedness check (the payment verdict stays plain code)

The deployment names and the reasoning effort come from settings (environment). Terraform creates
the deployments and pins the model behind each name, so swapping a model is a tfvars change, not a
code change. The names here are only defaults that match `infra/terraform/envs/*.tfvars`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from support_assistant.config import Settings


class Role(StrEnum):
    CHAT = "chat"
    PLANNER = "planner"
    REWRITE = "rewrite"
    TROUBLESHOOTER = "troubleshooter"
    JUDGE = "judge"


@dataclass(frozen=True)
class ModelSpec:
    deployment: str
    effort: str | None  # reasoning effort for reasoning models; None for a model that has none
    fallback: str | None = None  # answers instead when this deployment fails or times out


def model_for(settings: Settings, role: Role) -> ModelSpec:
    """The deployment, reasoning effort and fallback configured for a role."""
    s = settings
    return {
        Role.CHAT: ModelSpec(
            s.llm_chat_deployment, s.llm_chat_effort, s.llm_chat_fallback_deployment
        ),
        Role.PLANNER: ModelSpec(
            s.llm_planner_deployment, s.llm_planner_effort, s.llm_planner_fallback_deployment
        ),
        Role.REWRITE: ModelSpec(s.llm_rewrite_deployment, s.llm_rewrite_effort),
        Role.TROUBLESHOOTER: ModelSpec(
            s.llm_troubleshooter_deployment,
            s.llm_troubleshooter_effort,
            s.llm_troubleshooter_fallback_deployment,
        ),
        Role.JUDGE: ModelSpec(s.llm_judge_deployment, s.llm_judge_effort),
    }[role]
