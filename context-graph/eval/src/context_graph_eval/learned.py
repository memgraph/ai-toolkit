"""Benchmarks on a learned ontology: derive each user's model after reconciling (map #431).

Off by default (`--ontology fixed`): benchmarks normally stay on the hand
vocabulary so runs compare (#436). `--ontology learned` instead reconciles
every session under its user's adopted version -- hygm's default model to
start -- then runs `sessions-graph derive --user U --force` once per user,
which gates a candidate on that user's own sessions, adopts it if it passes
and re-extracts them under it. That is the product path, run as the CLI runs
it, a few users at a time in separate processes.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from dataclasses import dataclass, field

_OUTCOME = re.compile(r"^(?P<user>\S+): (?P<outcome>adopted|rejected|nothing to do|claimed elsewhere|derive off)")


@dataclass
class Derived:
    """How each user's derivation ended, for the run's report."""

    outcomes: dict[str, str] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        counted: dict[str, int] = {}
        for outcome in [*self.outcomes.values(), *("error" for _ in self.failures)]:
            counted[outcome] = counted.get(outcome, 0) + 1
        return counted


async def derive_users(db, *, memgraph_url: str, seeds: int, workers: int, progress: bool = True) -> Derived:
    """Run one forced derivation per user with reconciled sessions, `workers` at a time.

    Args:
        db: The eval graph; only read here, to list its users.
        memgraph_url: Handed to each derive process, which connects on its own.
        seeds: Candidates per derivation (`--seeds`); each costs its own LLM calls.
        workers: Derivations at once; each loads its own GLiNER2 checkpoint.
    """
    users = [
        row["user_id"]
        for row in db.query(
            "MATCH (u:User)-[:HAD_SESSION]->(:Session {reconciliation_status: 'completed'}) "
            "RETURN DISTINCT u.user_id AS user_id ORDER BY user_id"
        )
    ]
    executable = shutil.which("sessions-graph")
    command = [executable] if executable else [sys.executable, "-m", "sessions_graph.cli"]
    env = {**os.environ, "MEMGRAPH_URL": memgraph_url}
    env.setdefault("MEMGRAPH_USER", "")
    env.setdefault("MEMGRAPH_PASSWORD", "")
    env.setdefault("MEMGRAPH_DATABASE", "memgraph")
    derived, done = Derived(), 0
    limiter = asyncio.Semaphore(workers)

    async def _one(user_id: str) -> None:
        nonlocal done
        async with limiter:
            process = await asyncio.create_subprocess_exec(
                *command, "derive", "--user", user_id, "--force", "--seeds", str(seeds),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
            )  # fmt: skip
            out, err = await process.communicate()
        lines = out.decode(errors="replace").splitlines()
        match = next((m for line in lines if (m := _OUTCOME.match(line)) and m["user"] == user_id), None)
        if process.returncode == 0 and match:
            derived.outcomes[user_id] = match["outcome"]
        else:
            derived.failures[user_id] = (err.decode(errors="replace").strip().splitlines() or ["no output"])[-1]
        done += 1
        if progress:
            print(f"  derived {done}/{len(users)} {derived.counts()}", flush=True)

    await asyncio.gather(*(_one(user_id) for user_id in users))
    return derived
