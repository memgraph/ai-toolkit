"""Generic command hook CLI for memgraph-context-graph runtimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: memgraph-context-graph hook <command> [options]

Bridge agent command hooks to memgraph-context-graph.

Commands:
  init <runtime>   Generate a runtime's private hook config, if it supports one.
  run <runtime>    Run a runtime's command-hook adapter.

Runtimes are discovered via the `memgraph_context_graph.runtimes` entry point --
run `memgraph-context-graph hook run --help` (or see `runtime_plugin.py`) to see
what's registered in this environment.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to a runtime-specific command hook adapter."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        print(_HELP)
        return 2
    if args[0] in {"-h", "--help"}:
        print(_HELP)
        return 0

    command = args[0]
    if command == "init":
        return _init(args[1:])

    if command == "run":
        if len(args) == 1:
            print("usage: memgraph-context-graph hook run <runtime> [options]", file=sys.stderr)
            return _hook_misconfigured(None, "MissingRuntime", "hook run was given no runtime")
        return _run_runtime(args[1], args[2:])

    print(f"Unknown hook command: {command}", file=sys.stderr)
    print(_HELP, file=sys.stderr)
    return _hook_misconfigured(None, "UnknownHookCommand", f"unknown hook command: {command}")


def _run_runtime(runtime: str, runtime_args: list[str]) -> int:
    from memgraph_context_graph.hooks.runner import run_hook
    from memgraph_context_graph.hooks.runtime_plugin import UnknownRuntimeError, get_runtime_plugin

    try:
        plugin = get_runtime_plugin(runtime)
    except Exception as exc:
        # An unknown runtime, or a registered one that fails to import (a broken install, version skew).
        print(str(exc), file=sys.stderr)
        error_type = "UnknownRuntime" if isinstance(exc, UnknownRuntimeError) else None
        return _hook_misconfigured(runtime, error_type, str(exc), exc)

    return run_hook(plugin, runtime_args)


def _hook_misconfigured(runtime: str | None, error_type: str | None, error: str, exc: Exception | None = None) -> int:
    """Log a hook command the runtime can't even dispatch, and exit 1.

    Never 2: on Claude Code's ``PreToolUse`` exit 2 blocks the tool call, and a
    capture problem must never block the agent. 1 is a non-blocking error the
    runtime shows as a hook notice, which is the right signal for a hooks
    config that names no runtime or one that isn't installed.
    """
    from memgraph_context_graph import capture_log

    capture_log.record_failure("args", exc, runtime=runtime, error_type=error_type, error=error)
    return 1


def _init(argv: list[str]) -> int:
    if not argv:
        print("usage: memgraph-context-graph hook init <runtime> [options]", file=sys.stderr)
        return 2

    from memgraph_context_graph.hooks.runtime_plugin import UnknownRuntimeError, get_runtime_plugin

    runtime, init_argv = argv[0], argv[1:]
    try:
        plugin = get_runtime_plugin(runtime)
    except UnknownRuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    init = getattr(plugin, "init", None)
    if init is None:
        print(f"{plugin.name} hook setup is not implemented yet.", file=sys.stderr)
        return 2

    return _run_generic_init(plugin.name, init, init_argv)


def _run_generic_init(runtime_name: str, init: Any, argv: list[str]) -> int:
    """Parse the shared ``hook init`` flags and delegate to *init*.

    The flag set is generic across runtimes. Each plugin's ``init`` writes its
    own config from the project dir, connectors, hook command, timeout, and
    force flag; ``--setup-schema`` (with the ``--memgraph-*`` overrides) is
    handled here, identically for every runtime.
    """
    parser = argparse.ArgumentParser(description=f"Generate a private {runtime_name} hook config.")
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Defaults to skills-graph, actions-graph, sessions-graph.",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory where the runtime's config should be generated.",
    )
    parser.add_argument(
        "--hook-command",
        default=None,
        help="Full command to place in the runtime's hook config. Defaults to this installed CLI.",
    )
    parser.add_argument("--memgraph-url", default=None, help="Memgraph Bolt URL for --setup-schema.")
    parser.add_argument("--memgraph-user", default=None, help="Memgraph username for --setup-schema.")
    parser.add_argument(
        "--memgraph-password", default=None, help="Memgraph password for --setup-schema. Never written to disk."
    )
    parser.add_argument("--memgraph-database", default=None, help="Memgraph database for --setup-schema.")
    parser.add_argument(
        "--setup-schema",
        action="store_true",
        help="Connect to Memgraph now and initialize enabled graph connector schemas.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Hook timeout in seconds.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated config.")
    args = parser.parse_args(argv)

    connectors = args.connector or ["skills-graph", "actions-graph", "sessions-graph"]
    project_dir = Path(args.project_dir).expanduser().resolve()

    try:
        init(project_dir, connectors, hook_command=args.hook_command, timeout=args.timeout, force=args.force)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        print("Re-run with --force to replace generated files.", file=sys.stderr)
        return 1

    if args.setup_schema:
        from memgraph_context_graph.adapters._identity import resolve_memgraph_env
        from memgraph_context_graph.hooks.runner import setup_connector_schemas

        memgraph_env = resolve_memgraph_env(
            url=args.memgraph_url,
            user=args.memgraph_user,
            password=args.memgraph_password,
            database=args.memgraph_database,
        )
        setup_connector_schemas(connectors, memgraph_env=memgraph_env)
        print(f"Initialized schema on {memgraph_env['MEMGRAPH_URL']} ({memgraph_env['MEMGRAPH_DATABASE']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
