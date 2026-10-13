"""Read-only terminal viewer for a Context Graph (map #374).

Run it with ``memgraph-context-graph ui``. It reads the hooks' config file, polls Memgraph, and never writes.
"""

from .app import ViewerApp
from .cli import main

__all__ = ["ViewerApp", "main"]
