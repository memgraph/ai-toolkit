"""Generic command hook CLI for agent-context-graph runtimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: agent-context-graph hook <command> [options]

Bridge agent command hooks to agent-context-graph.

Commands:
  init <runtime>   Generate a runtime's private hook config, if it supports one.
  run <runtime>    Run a runtime's command-hook adapter.

Runtimes are discovered via the `agent_context_graph.runtimes` entry point --
run `agent-context-graph hook run --help` (or see `runtime_plugin.py`) to see
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
            print("usage: agent-context-graph hook run <runtime> [options]", file=sys.stderr)
            return 2
        runtime = args[1]
        runtime_args = args[2:]
        return _run_runtime(runtime, runtime_args)

    # Backward-compatible module form:
    # python -m agent_context_graph.hooks.cli codex --connector skills-graph
    runtime = command
    runtime_args = args[1:]
    return _run_runtime(runtime, runtime_args)


def _run_runtime(runtime: str, runtime_args: list[str]) -> int:
    from agent_context_graph.hooks.runner import run_hook
    from agent_context_graph.hooks.runtime_plugin import UnknownRuntimeError, get_runtime_plugin

    try:
        plugin = get_runtime_plugin(runtime)
    except UnknownRuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return run_hook(plugin, runtime_args)


def _init(argv: list[str]) -> int:
    if not argv:
        print("usage: agent-context-graph hook init <runtime> [options]", file=sys.stderr)
        return 2

    from agent_context_graph.hooks.runtime_plugin import UnknownRuntimeError, get_runtime_plugin

    runtime, init_argv = argv[0], argv[1:]
    try:
        plugin = get_runtime_plugin(runtime)
    except UnknownRuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if any(arg.split("=", 1)[0] == "--memory-backend" for arg in init_argv):
        return _switch_memory_backend(plugin, init_argv)

    init = getattr(plugin, "init", None)
    if init is None:
        print(f"{plugin.name} hook setup is not implemented yet.", file=sys.stderr)
        return 2

    return _run_generic_init(plugin.name, init, init_argv)


def _switch_memory_backend(plugin: Any, argv: list[str]) -> int:
    """``setup <runtime> --memory-backend context-graph|native``: whose memory the harness's model uses."""
    from agent_context_graph.adapters._identity import MEMORY_BACKEND_GRAPH, MEMORY_BACKEND_NATIVE
    from agent_context_graph.memory_backend import MemoryBackendError, use_graph_memory, use_native_memory

    parser = argparse.ArgumentParser(
        prog=f"agent-context-graph setup {plugin.name}",
        description=(
            f"Make Context Graph {plugin.name}'s memory (turning its built-in memory off and importing it), "
            "or hand memory back to the built-in one."
        ),
    )
    parser.add_argument("--memory-backend", required=True, choices=[MEMORY_BACKEND_GRAPH, MEMORY_BACKEND_NATIVE])
    args = parser.parse_args(argv)
    switch = use_graph_memory if args.memory_backend == MEMORY_BACKEND_GRAPH else use_native_memory
    try:
        lines = switch(plugin)
    except MemoryBackendError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


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
        from agent_context_graph.adapters._identity import resolve_memgraph_env
        from agent_context_graph.hooks.runner import setup_connector_schemas

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
