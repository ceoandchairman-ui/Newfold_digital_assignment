"""Scheduled maintenance jobs, run as Kubernetes CronJobs (E7-2, E5-8).

    python -m support_assistant.jobs prune-sessions --days 90

Checkpoints expire on their own through the Redis saver's TTL (D9). This job prunes the durable
conversation history in Postgres. The retention period is configuration supplied by the compliance
owner (E5-8), not a decision made here.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

from support_assistant.config import get_settings
from support_assistant.db import Pool, make_pool
from support_assistant.governance import prune_sessions

log = structlog.get_logger()


async def run_prune(pool: Pool, days: int) -> int:
    deleted = await prune_sessions(pool, days)
    log.info("sessions_pruned", older_than_days=days, deleted=deleted)
    return deleted


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="support_assistant.jobs")
    sub = parser.add_subparsers(dest="job", required=True)
    prune = sub.add_parser("prune-sessions", help="delete conversation rows older than --days")
    prune.add_argument("--days", type=int, required=True)
    args = parser.parse_args(argv)

    pool = make_pool(get_settings())
    await pool.open()
    try:
        if args.job == "prune-sessions":
            await run_prune(pool, args.days)
    finally:
        await pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
