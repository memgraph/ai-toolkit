"""Per-tenant Memgraph client registry.

Holds one ``Memgraph`` client per tenant in the static catalog. Clients are
created lazily on first use; each one inherits the shared
URL/username/password from ``MemgraphConfig`` and connects to the per-tenant
logical database whose name equals the tenant id.

When auth is disabled the registry falls back to a single default client built
from ``MemgraphConfig`` exactly the way the original mcp-memgraph server did,
so existing tests and stdio usage are unaffected.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from memgraph_toolbox.api.memgraph import Memgraph

if TYPE_CHECKING:
    from mcp_memgraph.config import MCPAuthConfig, MemgraphConfig


class UnknownTenantError(Exception):
    """Raised when a tenant name isn't in the server's MCP_TENANT_CATALOG.

    Intentionally generic — callers translate this into a uniform "database not
    available" error so the existence of catalog entries isn't leaked.
    """


class TenantClientRegistry:
    """Lazy per-tenant client cache, plus a legacy default client."""

    def __init__(self, memgraph_cfg: MemgraphConfig, auth_cfg: MCPAuthConfig) -> None:
        self._memgraph_cfg = memgraph_cfg
        self._auth_cfg = auth_cfg
        self._catalog: frozenset[str] = auth_cfg.tenant_catalog if auth_cfg.enabled else frozenset()
        self._clients: dict[str, Memgraph] = {}
        self._default: Memgraph | None = None
        self._lock = threading.Lock()

    @property
    def catalog(self) -> frozenset[str]:
        return self._catalog

    def get_for(self, tenant: str) -> Memgraph:
        """Return the client for *tenant*. Raises if it isn't in the catalog."""
        if tenant not in self._catalog:
            raise UnknownTenantError(tenant)
        with self._lock:
            client = self._clients.get(tenant)
            if client is None:
                client = self._build_client_for(tenant)
                self._clients[tenant] = client
            return client

    def get_default(self) -> Memgraph:
        """Return the legacy single-DB client used when auth is disabled."""
        with self._lock:
            if self._default is None:
                self._default = Memgraph(**self._memgraph_cfg.get_client_config())
            return self._default

    def _build_client_for(self, tenant: str) -> Memgraph:
        cfg = self._memgraph_cfg.get_client_config()
        cfg["database"] = tenant
        return Memgraph(**cfg)


_REGISTRY: TenantClientRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry(memgraph_cfg: MemgraphConfig, auth_cfg: MCPAuthConfig) -> TenantClientRegistry:
    """Process-wide singleton registry."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = TenantClientRegistry(memgraph_cfg, auth_cfg)
        return _REGISTRY


def reset_registry_for_tests() -> None:
    """Drop the cached registry. Used by tests that mutate env vars."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = None
