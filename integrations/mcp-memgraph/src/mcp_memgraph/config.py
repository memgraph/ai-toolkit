"""Environment configuration for the MCP Memgraph server.

This module handles all environment variable configuration with sensible defaults
and type conversion.
"""

import os
from dataclasses import dataclass
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
            "user_agent": "mcp-memgraph",
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
            raise ValueError(f"Invalid transport '{transport}'. Valid options: {valid_options}")
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
    def log_file(self) -> str | None:
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


@dataclass
class MCPAuthConfig:
    """Configuration for Keycloak / OIDC-based JWT authentication.

    All of these are no-ops when ``enabled`` is False (the default). When
    enabled, the streamable-http transport gates every request behind a
    Keycloak-issued JWT and uses the ``tenants`` claim to route each session
    to a Memgraph logical database.

    Environment variables:
        MCP_AUTH_ENABLED: Master feature flag (default: false).
        MCP_AUTH_ISSUER: Keycloak realm issuer URL, e.g.
            http://keycloak.keycloak.svc:8080/realms/memgraph
        MCP_AUTH_AUDIENCE: Expected ``aud`` claim, e.g.
            http://mcp-memgraph.default.svc:8000
        MCP_AUTH_JWKS_URL: JWKS endpoint; derived from issuer if unset.
        MCP_AUTH_TENANTS_CLAIM: Claim holding the user's allowed tenant list
            (default: ``tenants``).
        MCP_AUTH_DEFAULT_TENANT_CLAIM: Optional claim for the user's preferred
            starting tenant (default: ``default_tenant``).
        MCP_AUTH_REQUIRED_SCOPE: Scope that must be present (default:
            ``mcp:tools``).
        MCP_TENANT_CATALOG: Comma-separated set of tenants this MCP deployment
            serves. Names must match Keycloak group names AND Memgraph
            database names. Required when auth is enabled.
        MCP_AUTH_STATIC_CLIENT_ID: When set, the middleware intercepts every
            Dynamic Client Registration request and returns this single
            pre-registered client_id verbatim. This implements a "shared
            public client" pattern: every MCP-aware IDE install (Claude,
            Cursor, VS Code, Codex, JetBrains, Python CLI/Jupyter, …) ends up
            using the same client_id, because per-user access control flows
            through the JWT's tenants claim — the client identity carries no
            authorization signal here. It's also a workaround for Claude
            Code's current behaviour of forcing DCR even when oauth.clientId
            is configured (see anthropics/claude-code#26675). Default empty
            disables the intercept and lets clients DCR against Keycloak
            directly.
    """

    @property
    def enabled(self) -> bool:
        return os.getenv("MCP_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")

    @property
    def issuer(self) -> str:
        return os.getenv("MCP_AUTH_ISSUER", "")

    @property
    def audience(self) -> str:
        return os.getenv("MCP_AUTH_AUDIENCE", "")

    @property
    def jwks_url(self) -> str:
        explicit = os.getenv("MCP_AUTH_JWKS_URL")
        if explicit:
            return explicit
        if not self.issuer:
            return ""
        return self.issuer.rstrip("/") + "/protocol/openid-connect/certs"

    @property
    def tenants_claim(self) -> str:
        return os.getenv("MCP_AUTH_TENANTS_CLAIM", "tenants")

    @property
    def default_tenant_claim(self) -> str:
        return os.getenv("MCP_AUTH_DEFAULT_TENANT_CLAIM", "default_tenant")

    @property
    def required_scope(self) -> str:
        return os.getenv("MCP_AUTH_REQUIRED_SCOPE", "mcp:tools")

    @property
    def tenant_catalog(self) -> frozenset[str]:
        raw = os.getenv("MCP_TENANT_CATALOG", "")
        return frozenset(t.strip() for t in raw.split(",") if t.strip())

    @property
    def static_client_id(self) -> str:
        return os.getenv("MCP_AUTH_STATIC_CLIENT_ID", "").strip()

    @property
    def dcr_intercept_enabled(self) -> bool:
        return self.enabled and bool(self.static_client_id)

    def validate(self) -> None:
        """Fail fast at startup when auth is enabled but config is incomplete."""
        if not self.enabled:
            return
        missing = []
        if not self.issuer:
            missing.append("MCP_AUTH_ISSUER")
        if not self.audience:
            missing.append("MCP_AUTH_AUDIENCE")
        if not self.tenant_catalog:
            missing.append("MCP_TENANT_CATALOG")
        if missing:
            raise ValueError(f"MCP_AUTH_ENABLED=true but required env vars are missing: {', '.join(missing)}")


# Global instance placeholders for the singleton pattern
_MEMGRAPH_CONFIG_INSTANCE = None
_MCP_CONFIG_INSTANCE = None
_AUTH_CONFIG_INSTANCE = None


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


def get_auth_config() -> MCPAuthConfig:
    """Gets the singleton instance of MCPAuthConfig.

    Instantiates it on the first call.

    Returns:
        MCPAuthConfig: The auth configuration instance
    """
    global _AUTH_CONFIG_INSTANCE
    if _AUTH_CONFIG_INSTANCE is None:
        _AUTH_CONFIG_INSTANCE = MCPAuthConfig()
    return _AUTH_CONFIG_INSTANCE
