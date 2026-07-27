"""CLI for Sessions Graph session-content enrichment.

    sessions-graph enrich --session SESSION_ID
    sessions-graph enrich --pending [--limit N]

Batch-extracts entities from a session's Action/Memory content via
unstructured2graph's chunk + LightRAG pipeline (see enrich_session() in
core.py). This is the intended way to run enrichment — deliberately a
separate process from the SESSION_END hook, since LLM-backed entity
extraction is slow and hook subprocesses run under a runtime timeout.

Requires the ``sessions-graph[enrichment]`` extra and an LLM API key
(``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``) for LightRAG.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: sessions-graph enrich (--session SESSION_ID | --pending) [--limit N] [--working-dir DIR]

Batch-extract entities from session Action/Memory content via
unstructured2graph + LightRAG. Requires an LLM API key (OPENAI_API_KEY or
ANTHROPIC_API_KEY) -- see the lightrag-memgraph README.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        print(_HELP)
        return 2
    if args[0] in {"-h", "--help"}:
        print(_HELP)
        return 0

    command, rest = args[0], args[1:]
    if command == "enrich":
        return _enrich(rest)

    print(f"Unknown command: {command}", file=sys.stderr)
    print(_HELP)
    return 2


def _enrich(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sessions-graph enrich",
        description="Batch-extract entities from session Action/Memory content.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--session", help="Enrich a single session by ID.")
    target.add_argument(
        "--pending",
        action="store_true",
        help="Sweep all sessions with enrichment_status='pending'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max sessions to process with --pending (default: 100).",
    )
    parser.add_argument(
        "--working-dir",
        default="./lightrag_storage",
        help="LightRAG working_dir fallback for stores not backed by Memgraph (default: ./lightrag_storage).",
    )
    parsed = parser.parse_args(argv)

    return asyncio.run(_run_enrich(parsed))


async def _run_enrich(parsed: argparse.Namespace) -> int:
    from lightrag_memgraph import MemgraphLightRAGWrapper
    from sessions_graph import SessionsGraph

    graph = SessionsGraph()
    graph.setup()

    session_ids = [parsed.session] if parsed.session else graph.get_pending_enrichment_sessions(limit=parsed.limit)
    if not session_ids:
        print("No sessions to enrich.")
        return 0

    lightrag_wrapper = MemgraphLightRAGWrapper()
    await lightrag_wrapper.initialize(working_dir=parsed.working_dir)
    try:
        exit_code = 0
        for session_id in session_ids:
            summary = await graph.enrich_session(session_id, lightrag_wrapper=lightrag_wrapper)
            if summary.status == "completed":
                print(f"OK {session_id}: {summary.texts_deduped}/{summary.texts_considered} unique texts enriched")
            else:
                print(f"FAILED {session_id}: {summary.error}", file=sys.stderr)
                exit_code = 1
        return exit_code
    finally:
        await lightrag_wrapper.afinalize()


if __name__ == "__main__":
    raise SystemExit(main())
