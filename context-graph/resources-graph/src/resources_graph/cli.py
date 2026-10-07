"""``resources-graph``: run the Sweep, or create the schema.

Memgraph comes from the Context Graph config file, the same file hooks read.
The Sweep fetches with the user's own ``gh`` login, or ``github.token`` from
that file when it is set.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from .core import ResourcesGraph

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``resources-graph`` command. Returns the exit code."""
    parser = argparse.ArgumentParser(prog="resources-graph", description="Remembered public GitHub resources.")
    commands = parser.add_subparsers(dest="command", required=True)
    sweep_parser = commands.add_parser("sweep", help="Fetch and store the resources pending Touches name.")
    sweep_parser.add_argument("--limit", type=int, default=None, help="Resolve at most this many Touches.")
    sweep_parser.add_argument("--quiet", action="store_true", help="Print only the summary.")
    commands.add_parser("setup", help="Create the indexes and constraints.")
    args = parser.parse_args(argv)

    try:
        from agent_context_graph.adapters._identity import load_config
    except ImportError:
        print(
            "resources-graph needs agent-context-graph for its config: pip install 'resources-graph[agent-context-graph]'"
        )
        return 2
    config = load_config()
    graph = ResourcesGraph(
        url=config.memgraph_url,
        username=config.memgraph_user,
        password=config.memgraph_password,
        database=config.memgraph_database,
    )
    graph.setup()
    if args.command == "setup":
        print("resources-graph schema ready")
        return 0
    from .github import resolve_token

    return _sweep(graph, resolve_token(config.github_token), limit=args.limit, quiet=args.quiet)


def _sweep(graph: ResourcesGraph, token: str | None, *, limit: int | None, quiet: bool) -> int:
    from .github import GitHubAuthError, GitHubSource, http_transport
    from .sweep import sweep

    if not token:
        print(
            "No GitHub credentials, so nothing can be fetched; Touches stay pending.\n"
            "Log in with `gh auth login`, or set a token: agent-context-graph config set github.token",
            file=sys.stderr,
        )
        return 1
    try:
        report = sweep(
            graph, GitHubSource(http_transport(token)), limit=limit, log=(lambda _: None) if quiet else print
        )
    except GitHubAuthError as exc:
        print(f"{exc}; Touches stay pending.", file=sys.stderr)
        return 1
    unresolved = ", ".join(f"{reason}={count}" for reason, count in sorted(report.unresolved.items())) or "none"
    print(f"resolved {report.resolved} · unresolved {unresolved} · {report.fetched} fetches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
