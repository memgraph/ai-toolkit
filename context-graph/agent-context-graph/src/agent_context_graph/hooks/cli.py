"""Generic command hook CLI for agent-context-graph runtimes."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: agent-context-graph hook <command> [options]

Bridge agent command hooks to agent-context-graph.

Commands:
  init codex   Generate private Codex hook config.
  run codex    Run the OpenAI Codex command-hook adapter.

Runtimes:
  claude-code  Reserved for a future Claude Code hook adapter.
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
    if runtime == "codex":
        from agent_context_graph.adapters.codex import main as codex_main

        return codex_main(runtime_args)

    if runtime in {"claude-code", "claude_code"}:
        print("Claude Code hooks are not implemented yet.", file=sys.stderr)
        return 2

    print(f"Unknown hook runtime: {runtime}", file=sys.stderr)
    print(_HELP)
    return 2


def _init(argv: list[str]) -> int:
    if not argv:
        print("usage: agent-context-graph hook init <runtime> [options]", file=sys.stderr)
        return 2

    runtime = argv[0]
    if runtime == "codex":
        return _init_codex(argv[1:])

    if runtime in {"claude-code", "claude_code"}:
        print("Claude Code hook setup is not implemented yet.", file=sys.stderr)
        return 2

    print(f"Unknown hook runtime for init: {runtime}", file=sys.stderr)
    return 2


def _init_codex(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate private OpenAI Codex hook config.")
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Defaults to skills-graph.",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory where .codex config should be generated.",
    )
    parser.add_argument(
        "--hook-command",
        default=None,
        help="Full command to place in hooks.json. Defaults to this installed CLI.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Codex hook timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .codex/config.toml or .codex/hooks.json.",
    )
    args = parser.parse_args(argv)

    connectors = args.connector or ["skills-graph"]
    project_dir = Path(args.project_dir).expanduser().resolve()
    codex_dir = project_dir / ".codex"
    config_path = codex_dir / "config.toml"
    hooks_path = codex_dir / "hooks.json"

    existing = [path for path in (config_path, hooks_path) if path.exists()]
    if existing and not args.force:
        names = ", ".join(str(path) for path in existing)
        print(f"Refusing to overwrite existing Codex config: {names}", file=sys.stderr)
        print("Re-run with --force to replace generated files.", file=sys.stderr)
        return 1

    from agent_context_graph.adapters.codex import build_hooks_config

    hook_command = args.hook_command or _default_hook_command("codex", connectors)
    codex_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("[features]\ncodex_hooks = true\n", encoding="utf-8")
    hooks_path.write_text(
        json.dumps({"hooks": build_hooks_config(hook_command, timeout=args.timeout)}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {config_path}")
    print(f"Wrote {hooks_path}")
    print(f"Hook command: {hook_command}")
    return 0


def _default_hook_command(runtime: str, connectors: list[str]) -> str:
    executable = shutil.which("agent-context-graph")
    base = [executable] if executable else [sys.executable, "-m", "agent_context_graph.cli"]
    args = [*base, "hook", "run", runtime]
    for connector in connectors:
        args.extend(["--connector", connector])
    return shlex.join(args)


if __name__ == "__main__":
    raise SystemExit(main())
