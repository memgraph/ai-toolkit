"""``memgraph-context-graph ui`` / ``context-graph-viewer``: open the viewer on the configured graph."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, resolve identity from the config file, and run the viewer until the user quits.

    Returns:
        0 on a normal exit, 2 when the config names an edition that can't be resolved.
    """
    parser = argparse.ArgumentParser(
        prog="memgraph-context-graph ui",
        description=(
            "Read-only terminal viewer for your Context Graph: capture health, pipeline state, sessions and the "
            "knowledge graph. Reads the same config file as the hooks (CONTEXT_GRAPH_CONFIG or "
            "~/.config/context-graph/config.toml)."
        ),
    )
    parser.add_argument("--poll", type=float, default=3.0, help="Seconds between refreshes (default 3).")
    parser.add_argument("--version", action="version", version=metadata.version("context-graph-viewer"))
    args = parser.parse_args(argv)

    from memgraph_context_graph.spaces import SpaceError
    from memgraph_context_graph.terminal import color_support

    if color_support() != "truecolor":
        # Textual would otherwise believe COLORTERM and send 24-bit escapes to a terminal that can't read them.
        # It reads this once, when it is imported, so it has to be set before the app is.
        os.environ.setdefault("TEXTUAL_COLOR_SYSTEM", "256")

    from .app import ViewerApp
    from .source import resolve_identity

    try:
        identity = resolve_identity()
    except SpaceError as exc:
        print(f"memgraph-context-graph ui: {exc}", file=sys.stderr)
        return 2
    ViewerApp(identity, poll_seconds=max(args.poll, 0.5)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
