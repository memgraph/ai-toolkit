"""Top-level CLI for agent-context-graph."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELP = """usage: agent-context-graph <command> [options]

Commands:
  hook <command>  Configure or run command hooks.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch top-level agent-context-graph commands."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        print(_HELP)
        return 2
    if args[0] in {"-h", "--help"}:
        print(_HELP)
        return 0

    command = args[0]
    if command == "hook":
        from agent_context_graph.hooks.cli import main as hook_main

        return hook_main(args[1:])

    print(f"Unknown command: {command}", file=sys.stderr)
    print(_HELP)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
