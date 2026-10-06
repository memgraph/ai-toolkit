"""CLI for Sessions Graph session-content reconciliation and embedding.

    sessions-graph reconcile --session SESSION_ID
    sessions-graph reconcile --pending [--limit N]
    sessions-graph embed --session SESSION_ID [--model MODEL]
    sessions-graph embed --pending [--limit N] [--model MODEL]
    sessions-graph ontology load --file SCHEMA.yaml [--user USER_ID] [--derive extend|off]
    sessions-graph ontology show [--user USER_ID]

Batch-extracts entities from a session's Action/Memory content via
unstructured2graph's GLiNER2 backend over hygm's default model, and writes
the session's summary with an LLM through LightRAG (see reconcile_session()
in core.py). This is the intended way to run reconciliation — deliberately a
separate process from the SESSION_END hook, since local extraction and the
summary call are slow and hook subprocesses run under a runtime timeout.

Requires the ``sessions-graph[reconciliation]`` extra and an LLM API key
(``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``) for the summary.

``ontology`` supplies a schema as a user's next ontology version, or shows
the adopted one (see ``ontology.py``). ``reconcile`` also applies the config
file's ``[ontology] path`` to the configured user whenever the file changes.

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

    from sessions_graph import SessionsGraph
    from sessions_graph.ontology import OntologyVersion

_HELP = """usage: sessions-graph reconcile (--session SESSION_ID | --pending) [--limit N] [--working-dir DIR]
       sessions-graph embed (--session SESSION_ID | --pending) [--limit N] [--model MODEL]
       sessions-graph ontology load --file SCHEMA.yaml [--user USER_ID] [--derive extend|off]
       sessions-graph ontology show [--user USER_ID]

reconcile: extract entities from session Action/Memory content with GLiNER2
(local) and summarize the session with an LLM. Requires an LLM API key
(OPENAI_API_KEY or ANTHROPIC_API_KEY) -- see the lightrag-memgraph README.

embed: embed a session's messages, entities and edges for recall, inside
Memgraph (needs MAGE). No LLM.

ontology: supply a schema as a user's next ontology version (load), or print
the version their sessions are extracted under (show). USER_ID defaults to
the config file's identity.user_id.
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
    if command == "ontology":
        return _ontology(rest)

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


def _ontology(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="sessions-graph ontology", description="A user's ontology versions.")
    actions = parser.add_subparsers(dest="action", required=True)
    load = actions.add_parser("load", help="Supply a schema as the user's next ontology version.")
    load.add_argument("--file", required=True, help="A ManualStrategy YAML schema.")
    load.add_argument("--user", help="Whose model (default: the config file's identity.user_id).")
    load.add_argument(
        "--derive",
        choices=("extend", "off"),
        default="extend",
        help="extend: add the fixed core and let derivation add types beside the schema's (default). "
        "off: use the schema exactly as given and never derive.",
    )
    show = actions.add_parser("show", help="Print the version the user's sessions are extracted under.")
    show.add_argument("--user", help="Whose model (default: the config file's identity.user_id).")
    parsed = parser.parse_args(argv)

    _fill_env_from_context_graph_config()
    user_id = parsed.user or _configured_user_id()
    if not user_id:
        print("No user: pass --user or set identity.user_id in the config file.", file=sys.stderr)
        return 2

    from sessions_graph import SessionsGraph

    graph = SessionsGraph()
    graph.setup()
    if parsed.action == "load":
        try:
            version = graph.supply_ontology_file(user_id, parsed.file, derive=parsed.derive)
        except ValueError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(
            f"Adopted version {version.version} for {user_id}: {len(version.model.node_types)} node types, "
            f"{len(version.model.relation_types)} relation types, derive={version.derive}"
        )
        return 0

    print(_describe(graph.adopted_ontology(user_id)))
    return 0


def _describe(version: OntologyVersion) -> str:
    pinned = set(version.pinned)
    lines = [
        f"{version.user_id}: version {version.version} ({version.source}), derive={version.derive}"
        + (f", adopted {version.created_at}" if version.created_at else ""),
        "node types:",
        *(
            f"  {t.label}{' (pinned)' if t.label in pinned else ''} [{t.identity}] {t.description}"
            for t in version.model.node_types
        ),
        "relation types:",
        *(
            f"  {r.label}: {'|'.join(r.start_labels) or '*'} -> {'|'.join(r.end_labels) or '*'}"
            for r in version.model.relation_types
        ),
    ]
    return "\n".join(lines)


def _configured_user_id() -> str | None:
    try:
        from agent_context_graph.adapters._identity import load_config
    except ImportError:
        return None
    return load_config().user_id


def _sync_configured_ontology(graph: SessionsGraph) -> None:
    """Apply the config file's ``[ontology] path`` to the configured user when the file changed.

    A broken schema file is reported and the adopted version kept, so a typo
    in it never stops reconciliation.
    """
    try:
        from agent_context_graph.adapters._identity import load_config, resolve_ontology
    except ImportError:
        return
    path, derive = resolve_ontology()
    user_id = load_config().user_id
    if not path or not user_id:
        return

    try:
        version = graph.sync_ontology_file(user_id, path, derive=derive or "extend")
    except ValueError as exc:
        print(f"WARNING: ontology file {path} not applied, keeping the adopted version: {exc}", file=sys.stderr)
        return
    if version is not None:
        print(f"Adopted ontology version {version.version} for {user_id} from {path}")


async def _run_reconcile(parsed: argparse.Namespace) -> int:
    _fill_env_from_context_graph_config()

    from lightrag_memgraph import MemgraphLightRAGWrapper
    from sessions_graph import SessionsGraph

    graph = SessionsGraph()
    graph.setup()
    _sync_configured_ontology(graph)

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
