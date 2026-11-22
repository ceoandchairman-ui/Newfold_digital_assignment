"""Post-deploy smoke test (E11-A5): run inside the cluster right after a Helm upgrade.

    python -m support_assistant.smoke

1. Every service is healthy and every agent is READY (its MCP tools are really served).
2. Optionally, one real journey: verify a seeded test customer and ask an account question
   (set SMOKE_PHONE and SMOKE_DOB, e.g. the seeded C1001). It must answer without an error.
Exits non-zero on any failure, which fails the pipeline and blocks promotion.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass

import httpx

CHECKS: tuple[tuple[str, str], ...] = (
    ("fat-backend", "http://fat-backend:8000/healthz"),
    ("orchestrator", "http://orchestrator:8000/healthz"),
    ("agent-director", "http://agent-director:8000/readyz"),
    ("agent-troubleshooter", "http://agent-troubleshooter:8000/readyz"),
    ("agent-evaluator", "http://agent-evaluator:8000/readyz"),
)


@dataclass(frozen=True)
class Result:
    name: str
    ok: bool
    detail: str


async def check_endpoints(
    http: httpx.AsyncClient, checks: tuple[tuple[str, str], ...] = CHECKS
) -> list[Result]:
    async def one(name: str, url: str) -> Result:
        try:
            r = await http.get(url)
            return Result(name, r.status_code == 200, f"HTTP {r.status_code}")
        except httpx.HTTPError as exc:
            return Result(name, False, type(exc).__name__)

    return list(await asyncio.gather(*(one(n, u) for n, u in checks)))


async def check_journey(http: httpx.AsyncClient, backend: str, phone: str, dob: str) -> Result:
    session = f"smoke-{uuid.uuid4().hex[:8]}"
    try:
        v = await http.post(
            f"{backend}/verify", json={"session_id": session, "phone": phone, "date_of_birth": dob}
        )
        if v.status_code != 200:
            return Result("journey", False, f"verify HTTP {v.status_code}")
        c = await http.post(
            f"{backend}/chat", json={"session_id": session, "message": "how much data have I used?"}
        )
        reply = c.json().get("reply", "") if c.status_code == 200 else ""
        await http.post(f"{backend}/end", json={"session_id": session, "outcome": "resolved"})
        return Result("journey", c.status_code == 200 and bool(reply), f"chat HTTP {c.status_code}")
    except httpx.HTTPError as exc:
        return Result("journey", False, type(exc).__name__)


async def run() -> int:
    async with httpx.AsyncClient(timeout=30.0) as http:
        results = await check_endpoints(http)
        phone, dob = os.environ.get("SMOKE_PHONE"), os.environ.get("SMOKE_DOB")
        if phone and dob:
            results.append(await check_journey(http, "http://fat-backend:8000", phone, dob))
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.name:22} {r.detail}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
