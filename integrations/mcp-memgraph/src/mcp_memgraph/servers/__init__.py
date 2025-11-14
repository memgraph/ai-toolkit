"""Server registry for MCP Memgraph servers.

This module provides a registry system for discovering and loading different
MCP server implementations. Each server is a separate module with its own
tools, but all share the same configuration system.
"""

import importlib
from typing import Dict, Any

# Server registry - maps server names to their module paths and metadata
AVAILABLE_SERVERS: Dict[str, Dict[str, Any]] = {
    "server": {
        "module": "mcp_memgraph.servers.server",
        "emoji": "🚀",
        "description": "MCP server with stable tools",
    },
    "memgraph-experimental": {
        "module": "mcp_memgraph.servers.memgraph_experimental",
        "emoji": "🔬",
        "description": (
            "Memgraph experimental server with sampling and elicitation "
            "support for autonomous index management"
        ),
    },
    # Future servers can be added here:
    # "hygm": {
    #     "module": "mcp_memgraph.servers.hygm",
    #     "emoji": "🧬",
    #     "description": "HyGM experimental server",
    # },
}


def get_server(name: str):
    """Load and return the specified server module.

    Args:
        name: The name of the server to load (e.g., 'production',
              'experimental')

    Returns:
        The server module containing 'mcp' and 'logger' attributes

    Raises:
        ValueError: If the server name is not in the registry
        ImportError: If the server module cannot be imported
        AttributeError: If the module doesn't have required attributes
    """
    if name not in AVAILABLE_SERVERS:
        available = ", ".join(f"'{s}'" for s in AVAILABLE_SERVERS.keys())
        raise ValueError(f"Unknown server '{name}'. Available servers: {available}")

    server_info = AVAILABLE_SERVERS[name]
    module_path = server_info["module"]

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import server '{name}' from '{module_path}': {e}"
        ) from e

    # Verify the module has required attributes
    if not hasattr(module, "mcp"):
        raise AttributeError(
            f"Server module '{module_path}' missing required 'mcp' attribute"
        )
    if not hasattr(module, "logger"):
        raise AttributeError(
            f"Server module '{module_path}' missing required 'logger' " "attribute"
        )

    return module


def get_server_info(name: str) -> Dict[str, Any]:
    """Get metadata about a server without loading it.

    Args:
        name: The name of the server

    Returns:
        Dictionary with server metadata (module, emoji, description)

    Raises:
        ValueError: If the server name is not in the registry
    """
    if name not in AVAILABLE_SERVERS:
        available = ", ".join(f"'{s}'" for s in AVAILABLE_SERVERS.keys())
        raise ValueError(f"Unknown server '{name}'. Available servers: {available}")

    return AVAILABLE_SERVERS[name].copy()


def list_servers() -> Dict[str, Dict[str, Any]]:
    """Get a dictionary of all available servers with their metadata.

    Returns:
        Dictionary mapping server names to their metadata
    """
    return AVAILABLE_SERVERS.copy()


__all__ = [
    "AVAILABLE_SERVERS",
    "get_server",
    "get_server_info",
    "list_servers",
]
