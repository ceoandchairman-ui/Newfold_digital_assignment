"""Compact conversation history into a bounded context (E6-2).

Deterministic and model-free, so it adds no latency or cost: recent turns stay verbatim, older
turns are shortened, and the whole history is capped. This is what the graph receives.
"""

from __future__ import annotations

from typing import Any

from support_assistant.guards import clean_untrusted, looks_like_injection

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


# --- the history a model is shown ----------------------------------------------------------------
HISTORY_MAX_MESSAGES = 8  # the most recent user and assistant messages
HISTORY_MESSAGE_CHARS = 500  # each one is cut to this
HISTORY_MAX_CHARS = 3000  # all of them together; whole messages are dropped, oldest first
NOTES_MAX_CHARS = 1200  # memory notes the backend added (past episodes, recent conversations)

_SPEAKERS = {"user": "Customer", "assistant": "Assistant"}


def _safe_text(value: Any, limit: int) -> str:
    """Old text is as untrusted as new text: cleaned, capped, and unable to close its own block."""
    text = clean_untrusted(str(value), limit).replace("<<<", "").replace(">>>", "")
    return "" if looks_like_injection(text) else text.strip()


def history_for_model(messages: list[Message]) -> str:
    """What came before the latest customer message, trimmed to a fixed size. Empty if nothing did.

    The agents receive `compact_history`'s output, whose newest messages are not length capped, so
    the limits here are what actually bound the prompt. Newest messages win. A message that tries to
    instruct the model is left out.
    """
    last_user = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None
    )
    if last_user is None:
        return ""
    earlier = messages[:last_user]

    notes = [
        _safe_text(m.get("content", ""), NOTES_MAX_CHARS)
        for m in earlier
        if m.get("role") == "system"
    ]
    note_block = "\n".join(n for n in notes if n)[:NOTES_MAX_CHARS]

    kept: list[tuple[str, str]] = []
    total = 0
    for m in reversed([m for m in earlier if m.get("role") in _SPEAKERS]):
        text = _safe_text(m.get("content", ""), HISTORY_MESSAGE_CHARS)
        if not text:
            continue
        role = str(m["role"])
        line = f"{_SPEAKERS[role]}: {text}"
        if len(kept) == HISTORY_MAX_MESSAGES or total + len(line) > HISTORY_MAX_CHARS:
            break
        kept.append((role, line))
        total += len(line)
    kept.reverse()
    while kept and kept[0][0] == "assistant":  # an answer without its question misleads
        kept.pop(0)

    parts = [note_block] if note_block else []
    parts += [line for _, line in kept]
    return "\n".join(parts)


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


def history_context(sessions: list[dict[str, Any]], max_chars: int = 1200) -> str:
    """A short note about the customer's recent conversations (other sessions), newest first.

    Each session is summarised deterministically (what was asked, how it ended), and the whole note is
    capped, so five days of history can never crowd out the current turn.
    """
    lines: list[str] = []
    for row in sessions:
        summary = summarize_session(row.get("messages") or [], max_chars=240)
        if summary:
            lines.append(f"- {str(row.get('updated_at', ''))[:10]}: {summary}")
    if not lines:
        return ""
    header = "Earlier conversations with this customer (most recent first):"
    return (header + "\n" + "\n".join(lines))[:max_chars]
