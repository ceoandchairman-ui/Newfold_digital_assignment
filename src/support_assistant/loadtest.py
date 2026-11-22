"""Load and latency baseline (E9-8):  python -m support_assistant.loadtest URL PHONE DOB [N] [CONCURRENCY]

Runs N chat turns with the given concurrency against a deployed backend and prints latency
percentiles. Run it against the Azure dev environment, never locally. The numbers are the baseline
the latency targets in docs/infra_requirements.md are checked against.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
import uuid

import httpx


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    if not sorted_values:
        raise ValueError("no samples")
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    return sorted_values[max(math.ceil(p / 100 * len(sorted_values)) - 1, 0)]


def summarize(latencies_s: list[float], failures: int, wall_s: float) -> dict[str, float]:
    ok = sorted(latencies_s)
    total = len(ok) + failures
    return {
        "requests": float(total),
        "failures": float(failures),
        "error_rate": failures / total if total else 0.0,
        "p50_ms": percentile(ok, 50) * 1000 if ok else 0.0,
        "p95_ms": percentile(ok, 95) * 1000 if ok else 0.0,
        "p99_ms": percentile(ok, 99) * 1000 if ok else 0.0,
        "throughput_rps": total / wall_s if wall_s > 0 else 0.0,
    }


async def run(base_url: str, phone: str, dob: str, n: int, concurrency: int) -> dict[str, float]:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as http:

        async def one() -> None:
            nonlocal failures
            session = uuid.uuid4().hex
            async with sem:
                try:
                    v = await http.post(
                        "/verify",
                        json={"session_id": session, "phone": phone, "date_of_birth": dob},
                    )
                    v.raise_for_status()
                    t0 = time.perf_counter()
                    r = await http.post(
                        "/chat",
                        json={"session_id": session, "message": "how much data have I used?"},
                    )
                    r.raise_for_status()
                    latencies.append(time.perf_counter() - t0)
                except httpx.HTTPError:
                    failures += 1

        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(n)))
        return summarize(latencies, failures, time.perf_counter() - start)


if __name__ == "__main__":
    url, phone, dob = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    conc = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    for key, value in asyncio.run(run(url, phone, dob, n, conc)).items():
        print(f"{key:15} {value:10.2f}")
