"""LangChain chains, used only in the fat backend and orchestrator nodes (D8), never in agent services.

Query rewrite: turns a conversational message into a standalone search query for retrieval.
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import AzureChatOpenAI

from support_assistant.config import Settings

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the customer's last message as one standalone search query for a telecom "
            "support knowledge base. Keep the language the customer used. Output only the query.",
        ),
        ("human", "Conversation:\n{history}\n\nLast message: {message}"),
    ]
)


def build_rewrite_chain(
    settings: Settings, deployment: str | None = None
) -> Runnable[dict[str, Any], str]:
    """Query rewrite runs on the small model (a 64-token job) with a hard timeout and no retries.

    It sits in series before the graph starts, so a slow answer is worse than none: past the
    timeout the caller falls back to the customer's own words.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=deployment or settings.azure_openai_rewrite_deployment,
        api_version="2024-10-21",
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        ),
        temperature=0,
        max_tokens=64,
        timeout=settings.rewrite_timeout_seconds,
        max_retries=0,
    )
    return _PROMPT | llm | StrOutputParser()


async def rewrite_query(
    chain: Runnable[dict[str, Any], str] | None, history: list[dict[str, Any]], message: str
) -> str:
    """Falls back to the raw message if no chain is configured or the model call fails."""
    if chain is None:
        return message
    text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    try:
        out = (await chain.ainvoke({"history": text, "message": message})).strip()
    except Exception:  # noqa: BLE001 - retrieval must still work on the original text
        return message
    return out or message
