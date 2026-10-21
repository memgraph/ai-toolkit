"""Run one real headless Claude Code session and assert the installed plugin's hooks stored it in Memgraph.

Run with the memgraph-context-graph uv tool's Python, so the Memgraph settings come from the config file
setup wrote, exactly as the hooks read them:

    "$(uv tool dir)/memgraph-context-graph/bin/python" session_lands.py --user ci-test-user

It calls a model, so it needs ANTHROPIC_API_KEY; without one it says so and exits 0 (fork PRs get no secrets).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

from memgraph_context_graph import capture_log
from memgraph_context_graph.adapters._identity import resolve_memgraph_env
from memgraph_context_graph.hooks.runner import memgraph_kwargs
from memgraph_toolbox.api.memgraph import Memgraph

PROMPT = "Reply with the single word: ok"
# The SessionEnd hook runs as claude exits, but give a slow runner a moment.
WAIT_SECONDS = 30


def _sessions(db: Memgraph, user: str) -> set[str]:
    rows = db.query(
        "MATCH (:User {user_id: $user})-[:HAD_SESSION]->(s:Session)-[:HAS_ACTION]->(:Action) "
        "RETURN DISTINCT s.session_id AS id",
        params={"user": user},
    )
    return {row["id"] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", required=True, help="identity.user_id setup was given")
    parser.add_argument("--model", default="haiku", help="claude --model (default: haiku)")
    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("::notice::ANTHROPIC_API_KEY is not set; skipping the real Claude Code session")
        return 0

    db = Memgraph(**memgraph_kwargs(resolve_memgraph_env()))
    before = _sessions(db, args.user)
    with tempfile.TemporaryDirectory() as project:
        subprocess.run(["git", "init", "-q", project], check=True)
        result = subprocess.run(
            ["claude", "-p", PROMPT, "--model", args.model, "--output-format", "text"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=300,
        )
    print(f"claude exited {result.returncode}: {result.stdout.strip()}")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return 1

    deadline = time.monotonic() + WAIT_SECONDS
    while not (new := _sessions(db, args.user) - before) and time.monotonic() < deadline:
        time.sleep(1)
    failures = capture_log.read_recent()
    for failure in failures:
        print(f"capture failure: {failure}", file=sys.stderr)
    if not new:
        print(f"no new session with actions for {args.user}: the plugin's hooks never stored one", file=sys.stderr)
        return 1
    print(f"stored session(s) {sorted(new)} for {args.user}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
