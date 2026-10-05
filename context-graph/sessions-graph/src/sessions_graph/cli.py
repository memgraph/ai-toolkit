"""CLI for Sessions Graph session-content reconciliation and embedding.

    sessions-graph reconcile --session SESSION_ID
    sessions-graph reconcile --pending [--limit N]
    sessions-graph embed --session SESSION_ID [--model MODEL]
    sessions-graph embed --pending [--limit N] [--model MODEL]

Batch-extracts entities from a session's Action/Memory content via
unstructured2graph's chunk + LightRAG pipeline (see reconcile_session() in
core.py). This is the intended way to run reconciliation — deliberately a
separate process from the SESSION_END hook, since LLM-backed entity
extraction is slow and hook subprocesses run under a runtime timeout.

Requires the ``sessions-graph[reconciliation]`` extra and an LLM API key
(``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``) for LightRAG.

``embed`` computes recall's vectors inside Memgraph (see ``embeddings.py``):
no LLM and no extra, but a Memgraph with MAGE. The SESSION_END hook runs it
for every session; ``--pending`` retries sessions that failed or never ran.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: sessions-graph reconcile (--session SESSION_ID | --pending) [--limit N] [--working-dir DIR]
       sessions-graph embed (--session SESSION_ID | --pending) [--limit N] [--model MODEL]

reconcile: batch-extract entities from session Action/Memory content via
unstructured2graph + LightRAG. Requires an LLM API key (OPENAI_API_KEY or
ANTHROPIC_API_KEY) -- see the lightrag-memgraph README.

embed: embed a session's messages, entities and edges for recall, inside
Memgraph (needs MAGE). No LLM.
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
    if command == "reconcile":
        return _reconcile(rest)
    if command == "embed":
        return _embed(rest)

    print(f"Unknown command: {command}", file=sys.stderr)
    print(_HELP)
    return 2


def _reconcile(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sessions-graph reconcile",
        description="Batch-extract entities from session Action/Memory content.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--session", help="Reconcile a single session by ID.")
    target.add_argument(
        "--pending",
        action="store_true",
        help="Sweep all sessions with reconciliation_status='pending'.",
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

    return asyncio.run(_run_reconcile(parsed))


def _embed(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sessions-graph embed",
        description="Embed session messages, entities and edges for recall, inside Memgraph.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--session", help="Embed a single session by ID.")
    target.add_argument(
        "--pending",
        action="store_true",
        help="Embed every session whose embedding failed, never ran, or used another model.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max sessions with --pending (default: 100).")
    parser.add_argument(
        "--model",
        help="Embedding model (default: recall.embedding_model from the config file, else bge-small-en-v1.5).",
    )
    parsed = parser.parse_args(argv)

    _fill_env_from_context_graph_config()

    from sessions_graph import SessionsGraph
    from sessions_graph.embeddings import EmbeddingUnavailableError

    model = parsed.model or _configured_embedding_model()
    graph = SessionsGraph()
    graph.setup()
    session_ids = (
        [parsed.session] if parsed.session else graph.get_pending_embedding_sessions(model=model, limit=parsed.limit)
    )
    if not session_ids:
        print("No sessions to embed.")
        return 0

    for session_id in session_ids:
        try:
            embedded = graph.embed_session(session_id, model=model)
        except EmbeddingUnavailableError as exc:
            # The same cause fails every session; stop rather than repeat it.
            print(f"FAILED {session_id}: {exc}", file=sys.stderr)
            return 1
        print(
            f"OK {session_id}: {embedded.messages} messages, {embedded.entities} entities, "
            f"{embedded.edges} edges embedded with {model}"
        )
    return 0


def _configured_embedding_model() -> str:
    from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL

    try:
        from agent_context_graph.adapters._identity import resolve_embedding_model
    except ImportError:
        return DEFAULT_EMBEDDING_MODEL
    return resolve_embedding_model() or DEFAULT_EMBEDDING_MODEL


def _fill_env_from_context_graph_config() -> None:
    """Best-effort fallback for standalone (manual/cron) ``reconcile`` runs.

    When spawned by the SESSION_END hook, the parent process already overlays
    context-graph's config.toml onto this subprocess's environment (see
    sessions_graph.connector._child_env). Run standalone, there's no
    such parent, so fill in the same values here -- only for keys not already
    set, so explicit ambient env always wins. agent-context-graph is an
    optional extra; silently skip if it isn't installed.
    """
    try:
        from agent_context_graph.adapters._identity import resolve_llm_env, resolve_memgraph_env
    except ImportError:
        return
    for key, value in {**resolve_memgraph_env(), **resolve_llm_env()}.items():
        if value:
            os.environ.setdefault(key, value)


async def _run_reconcile(parsed: argparse.Namespace) -> int:
    _fill_env_from_context_graph_config()

    from lightrag_memgraph import MemgraphLightRAGWrapper
    from sessions_graph import SessionsGraph

    graph = SessionsGraph()
    graph.setup()

    session_ids = [parsed.session] if parsed.session else graph.get_pending_reconciliation_sessions(limit=parsed.limit)
    if not session_ids:
        print("No sessions to reconcile.")
        return 0

    lightrag_wrapper = MemgraphLightRAGWrapper()
    await lightrag_wrapper.initialize(working_dir=parsed.working_dir)
    try:
        exit_code = 0
        for session_id in session_ids:
            summary = await graph.reconcile_session(
                session_id,
                lightrag_wrapper=lightrag_wrapper,
                enforce_ontology=True,
                embedding_model=_configured_embedding_model(),
            )
            if summary.status == "completed":
                summarized = " (summary written)" if summary.summary_written else ""
                print(
                    f"OK {session_id}: {summary.texts_deduped}/{summary.texts_considered} "
                    f"unique texts reconciled{summarized}"
                )
            else:
                print(f"FAILED {session_id}: {summary.error}", file=sys.stderr)
                exit_code = 1
        return exit_code
    finally:
        await lightrag_wrapper.afinalize()


if __name__ == "__main__":
    raise SystemExit(main())
