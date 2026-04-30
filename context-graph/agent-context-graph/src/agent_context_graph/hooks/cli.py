"""Generic command hook CLI for agent-context-graph runtimes."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: agent-context-graph-hook <runtime> [runtime options]

Bridge agent command hooks to agent-context-graph.

Runtimes:
  codex        Run the OpenAI Codex command-hook adapter.
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

    runtime = args[0]
    runtime_args = args[1:]
    if runtime == "codex":
        from agent_context_graph.adapters.codex import main as codex_main

        return codex_main(runtime_args)

    if runtime in {"claude-code", "claude_code"}:
        print("Claude Code hooks are not implemented yet.", file=sys.stderr)
        return 2

    print(f"Unknown hook runtime: {runtime}", file=sys.stderr)
    print(_HELP)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
