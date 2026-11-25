"""Where the indexer gets its documents from (E2-2): the team's `knowledge-base` container.

Any BlobStore works (Azure in production, in-memory in tests). A local folder works too, for
indexing a directory of files from a job image or a developer shell.

Robust by design: one bad file never stops a batch. Unsupported types, oversized files and files that
are not valid UTF-8 text are skipped and reported, never half-read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import structlog

from support_assistant.blobstore import BlobStore

log = structlog.get_logger()

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")
MAX_DOC_BYTES = 5_000_000


@dataclass
class LoadedDocs:
    documents: dict[str, str] = field(default_factory=dict)  # name -> text
    skipped: dict[str, str] = field(default_factory=dict)  # name -> reason

    @property
    def names(self) -> list[str]:
        return sorted(self.documents)


class DocumentSource(Protocol):
    async def load(self) -> LoadedDocs: ...


def _decode(name: str, raw: bytes, out: LoadedDocs) -> None:
    if len(raw) > MAX_DOC_BYTES:
        out.skipped[name] = f"larger than {MAX_DOC_BYTES} bytes"
        return
    try:
        out.documents[name] = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        log.warning("document_undecodable", **{"error.type": type(exc).__name__})
        out.skipped[name] = "not valid UTF-8 text"


def _supported(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_SUFFIXES)


class StoreDocumentSource:
    """Documents in a blob container (production) or an in-memory store (tests)."""

    def __init__(self, store: BlobStore) -> None:
        self._store = store

    async def load(self) -> LoadedDocs:
        out = LoadedDocs()
        for name in await self._store.list():
            if not _supported(name):
                out.skipped[name] = "unsupported file type (only .md, .markdown, .txt)"
                continue
            _decode(name, await self._store.read(name), out)
        for name, why in out.skipped.items():
            log.warning("document_skipped", name=name, reason=why)
        return out


class LocalDirSource:
    """A directory (or a single file) on disk. Names are paths relative to the root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def load(self) -> LoadedDocs:
        out = LoadedDocs()
        files = (
            [self._root]
            if self._root.is_file()
            else sorted(p for p in self._root.rglob("*") if p.is_file())
        )
        base = self._root.parent if self._root.is_file() else self._root
        for path in files:
            name = path.relative_to(base).as_posix()
            if not _supported(name):
                out.skipped[name] = "unsupported file type (only .md, .markdown, .txt)"
                continue
            _decode(name, path.read_bytes(), out)
        return out
