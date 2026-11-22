"""Compact conversation history into a bounded context (E6-2).

Deterministic and model-free, so it adds no latency or cost: recent turns stay verbatim, older
turns are shortened, and the whole history is capped. This is what the graph receives.
"""

from __future__ import annotations

from typing import Any

Message = dict[str, Any]


def compact_history(
    messages: list[Message], *, keep_recent: int = 6, old_chars: int = 200, max_chars: int = 6000
) -> list[Message]:
    """Last `keep_recent` messages verbatim; older ones truncated; oldest dropped past `max_chars`."""
    if keep_recent < 0 or max_chars < 1:
        raise ValueError("keep_recent must be >= 0 and max_chars >= 1")
    split = max(len(messages) - keep_recent, 0)
    older = [{**m, "content": str(m.get("content", ""))[:old_chars]} for m in messages[:split]]
    recent = list(messages[split:])
    out = [*older, *recent]
    total = sum(len(str(m.get("content", ""))) for m in out)
    while out and total > max_chars and len(out) > len(recent):
        total -= len(str(out.pop(0).get("content", "")))  # drop the oldest summarised turn first
    return out


def episode_context(episodes: list[dict[str, str]], max_items: int = 3) -> str:
    """Short text block of relevant past episodes for a system message."""
    if not episodes:
        return ""
    lines = [f"- {e['summary']} (outcome: {e['outcome']})" for e in episodes[:max_items]]
    return "Previous interactions with this customer:\n" + "\n".join(lines)


def summarize_session(messages: list[Message], max_chars: int = 600) -> str:
    """Deterministic episode summary: the customer's first request and the final answer."""
    users = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
    replies = [str(m.get("content", "")) for m in messages if m.get("role") == "assistant"]
    if not users:
        return ""
    summary = f"Customer asked: {users[0][:250]}"
    if replies:
        summary += f" | Final answer: {replies[-1][:250]}"
    return summary[:max_chars]
