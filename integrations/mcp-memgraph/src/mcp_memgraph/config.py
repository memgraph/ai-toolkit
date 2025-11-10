"""Environment configuration for the MCP Memgraph server.

This module handles all environment variable configuration with sensible defaults
and type conversion.
"""

from dataclasses import dataclass
import os
from typing import Optional
from enum import Enum


class TransportType(str, Enum):
    """Supported MCP server transport types."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"

    @classmethod
    def values(cls) -> list[str]:
        """Get all valid transport values."""
        return [transport.value for transport in cls]


@dataclass
class MemgraphConfig:
    """Configuration for Memgraph connection settings.

    This class handles all environment variable configuration with sensible defaults
    and type conversion. It provides typed methods for accessing each configuration value.

    Optional environment variables (with defaults):
        MEMGRAPH_URL: The connection URL for Memgraph (default: bolt://localhost:7687)
        MEMGRAPH_USER: The username for authentication (default: "")
        MEMGRAPH_PASSWORD: The password for authentication (default: "")
        MEMGRAPH_DATABASE: The database name (default: memgraph)
    """

    @property
    def url(self) -> str:
        """Get the Memgraph connection URL.

        Default: bolt://localhost:7687
        """
        return os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")

    @property
    def username(self) -> str:
        """Get the Memgraph username.

        Default: "" (empty string)
        """
        return os.getenv("MEMGRAPH_USER", "")

    @property
    def password(self) -> str:
        """Get the Memgraph password.

        Default: "" (empty string)
        """
        return os.getenv("MEMGRAPH_PASSWORD", "")

    @property
    def database(self) -> str:
        """Get the Memgraph database name.

        Default: memgraph
        """
        return os.getenv("MEMGRAPH_DATABASE", "memgraph")

    def get_client_config(self) -> dict:
        """Get the configuration dictionary for Memgraph client.

        Returns:
            dict: Configuration ready to be passed to Memgraph client
        """
        return {
            "url": self.url,
            "username": self.username,
            "password": self.password,
            "database": self.database,
        }


@dataclass
class MCPServerConfig:
    """Configuration for MCP server-level settings.

    These settings control the server transport, logging, and tool behavior.

    Optional environment variables (with defaults):
        MCP_TRANSPORT: "stdio" or "streamable-http" (default: stdio)
        MCP_HOST: Bind host for HTTP transport (default: 127.0.0.1)
        MCP_PORT: Bind port for HTTP transport (default: 8000)
        MCP_READ_ONLY: Enable read-only mode to prevent write operations (default: true)
        # TODO(antejavor): Implement log file handling
        MCP_LOG_FILE: Path to log file (default: None, disables file logging)
        MCP_LOG_LEVEL: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
    """

    @property
    def transport(self) -> str:
        """Get the MCP server transport type.

        Default: stdio
        """
        transport = os.getenv("MCP_TRANSPORT", TransportType.STDIO.value).lower()
        if transport not in TransportType.values():
            valid_options = ", ".join(f'"{t}"' for t in TransportType.values())
            raise ValueError(
                f"Invalid transport '{transport}'. Valid options: {valid_options}"
            )
        return transport

    @property
    def host(self) -> str:
        """Get the MCP server bind host.

        Default: 127.0.0.1
        """
        return os.getenv("MCP_HOST", "127.0.0.1")

    @property
    def port(self) -> int:
        """Get the MCP server bind port.

        Default: 8000
        """
        return int(os.getenv("MCP_PORT", "8000"))

    @property
    def read_only(self) -> bool:
        """Get the read-only mode setting.

        When enabled, write operations (CREATE, MERGE, DELETE, etc.) are blocked.

        Default: True
        """
        return os.getenv("MCP_READ_ONLY", "true").lower() in ("true", "1", "yes")

    @property
    def log_file(self) -> Optional[str]:
        """Get the log file path.

        Default: None (no file logging)
        """
        return os.getenv("MCP_LOG_FILE")

    @property
    def log_level(self) -> str:
        """Get the logging level.

        Default: INFO
        """
        return os.getenv("MCP_LOG_LEVEL", "INFO").upper()


# Global instance placeholders for the singleton pattern
_MEMGRAPH_CONFIG_INSTANCE = None
_MCP_CONFIG_INSTANCE = None


def get_memgraph_config() -> MemgraphConfig:
    """Gets the singleton instance of MemgraphConfig.

    Instantiates it on the first call.

    Returns:
        MemgraphConfig: The Memgraph configuration instance
    """
    global _MEMGRAPH_CONFIG_INSTANCE
    if _MEMGRAPH_CONFIG_INSTANCE is None:
        _MEMGRAPH_CONFIG_INSTANCE = MemgraphConfig()
    return _MEMGRAPH_CONFIG_INSTANCE


def get_mcp_config() -> MCPServerConfig:
    """Gets the singleton instance of MCPServerConfig.

    Instantiates it on the first call.

    Returns:
        MCPServerConfig: The MCP server configuration instance
    """
    global _MCP_CONFIG_INSTANCE
    if _MCP_CONFIG_INSTANCE is None:
        _MCP_CONFIG_INSTANCE = MCPServerConfig()
    return _MCP_CONFIG_INSTANCE
