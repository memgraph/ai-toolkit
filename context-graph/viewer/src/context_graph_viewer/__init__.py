"""Read-only terminal viewer for a Context Graph (map #374).

Run it with ``memgraph-context-graph ui``. It reads the hooks' config file, polls Memgraph, and never writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .cli import main

if TYPE_CHECKING:
    from .app import ViewerApp

__all__ = ["ViewerApp", "main"]


def __getattr__(name: str) -> Any:
    """Import the app on first use: Textual reads its colour setting when imported, and ``main`` sets it first."""
    if name == "ViewerApp":
        from .app import ViewerApp

        return ViewerApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
