"""All-or-nothing text edits: nothing is written unless every match is found exactly once."""

from pathlib import Path

PENDING: dict[str, str] = {}
CRLF: dict[str, bool] = {}


def get(path):
    if path not in PENDING:
        raw = Path(path).read_bytes()
        CRLF[path] = b"\r\n" in raw
        PENDING[path] = raw.decode("utf-8").replace("\r\n", "\n")
    return PENDING[path]


def sub(path, old, new, count=1):
    s = get(path)
    assert s.count(old) == count, (path, old[:90], s.count(old))
    PENDING[path] = s.replace(old, new)


def cut(path, start, end, new=""):
    """Replace from `start` up to (not including) `end`."""
    s = get(path)
    a = s.index(start)
    b = s.index(end, a)
    PENDING[path] = s[:a] + new + s[b:]


def save():
    for path, s in PENDING.items():
        Path(path).write_bytes((s.replace("\n", "\r\n") if CRLF[path] else s).encode("utf-8"))
    print("applied", len(PENDING), "files")
