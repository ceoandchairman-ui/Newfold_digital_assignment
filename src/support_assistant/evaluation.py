"""Retrieval evaluation (E2-8): recall@k and MRR over a labelled question set.

Used to decide whether the reranker (D18) earns its latency and to catch regressions when the
embedder, chunking or index parameters change.

Eval set format (JSON list): [{"query": "...", "relevant_sections": ["Guide > No signal"]}, ...]
`relevant_sections` are heading paths. A hit is any retrieved chunk at or under one of them.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

Retriever = Callable[[str], Awaitable[Sequence[str]]]  # query -> ranked heading paths


@dataclass(frozen=True)
class Case:
    query: str
    relevant_sections: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    recall_at_k: float
    mrr: float
    cases: int
    k: int


def _is_hit(heading_path: str, relevant: Sequence[str]) -> bool:
    """Match on heading path (stable across re-chunking), at a ' > ' boundary."""
    return any(heading_path == r or heading_path.startswith(f"{r} > ") for r in relevant)


def load_cases(path: Path) -> list[Case]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Case(c["query"], tuple(c["relevant_sections"])) for c in raw]


async def evaluate(cases: Sequence[Case], retrieve: Retriever, k: int = 5) -> Report:
    if not cases:
        raise ValueError("evaluation set is empty")
    recalls, rranks = [], []
    for case in cases:
        ranked = list(await retrieve(case.query))[:k]
        hits = [i for i, s in enumerate(ranked) if _is_hit(s, case.relevant_sections)]
        recalls.append(1.0 if hits else 0.0)  # recall@k as "any relevant chunk in the top k"
        rranks.append(1.0 / (hits[0] + 1) if hits else 0.0)
    n = len(cases)
    return Report(sum(recalls) / n, sum(rranks) / n, n, k)


async def _main(eval_path: str) -> None:
    import os

    from support_assistant.config import get_settings
    from support_assistant.db import make_pool
    from support_assistant.embeddings import HttpEmbedder
    from support_assistant.reranker import HttpReranker
    from support_assistant.services.retrieval import search_kb

    s = get_settings()
    pool = make_pool(s)
    await pool.open()
    key = os.environ.get("EMBEDDER_API_KEY", "")

    async def token() -> str:
        return key

    embedder = HttpEmbedder(
        s.embedder_url, s.embedding_model, dim=s.embedding_dim, token_provider=token
    )
    reranker = HttpReranker(s.reranker_url, s.reranker_model, key) if s.reranker_url else None
    cases = load_cases(Path(eval_path))

    async def plain(q: str) -> list[str]:
        return [
            c.heading_path
            for c in await search_kb(
                pool,
                embedder,
                q,
                candidates=s.rerank_candidates,
                embedding=(s.embedding_model, s.embedding_version),
            )
        ]

    out = {"hybrid": await evaluate(cases, plain)}
    if reranker:

        async def reranked(q: str) -> list[str]:
            hits = await search_kb(
                pool,
                embedder,
                q,
                candidates=s.rerank_candidates,
                reranker=reranker,
                embedding=(s.embedding_model, s.embedding_version),
            )
            return [c.heading_path for c in hits]

        out["hybrid+rerank"] = await evaluate(cases, reranked)
    for name, r in out.items():
        print(f"{name:15} recall@{r.k}={r.recall_at_k:.3f}  mrr={r.mrr:.3f}  cases={r.cases}")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1]))
